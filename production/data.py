"""Isolated, clock-guarded public data for the local forecasting service.

The source repository database is read-only.  Only this store's SQLite database,
metadata copies and content-addressed response snapshots are written.  Returned
curves are detached objects: callers never obtain a database handle or a future
suffix through this interface.
"""
from __future__ import annotations

import ast
from contextlib import closing, contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Iterable
import uuid

import httpx
import numpy as np

from analysis.backtest import Curve
from analysis.data_policy import MIN_EVENT_START_MS
from download_ranks import (DEFAULT_RATE, Limiter, RANKS, SERVER, integer,
                            merge_points, normalize_cutoffs, normalize_points,
                            normalize_top, request)


ROOT = Path(__file__).resolve().parents[1]
DAY_MS = 86400000
RECENT_MS = 14 * DAY_MS
METADATA_URLS = {
    "events": "https://bestdori.com/api/events/all.5.json",
    "characters": "https://bestdori.com/api/characters/all.5.json",
    "bands": "https://bestdori.com/api/bands/all.1.json",
}


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_bytes(value)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _ranks(ranks: Iterable[int] | None) -> list[int]:
    values = list(RANKS if ranks is None else ranks)
    if not values or any(isinstance(v, bool) for v in values) or len(set(values)) != len(values) or not set(values) <= set(RANKS):
        raise ValueError("Invalid requested production ranks")
    return [int(v) for v in values]


def _complete(points: list[dict], end: int, before: int) -> bool:
    return bool(end < before and len(points) >= 5 and points[-1]["ep"] > 0
                and end - 60000 <= points[-1]["time"] <= end + 300000)


def _clock_points(points: list[dict], start: int, end: int, before: int) -> list[dict]:
    # Tracker snapshots can continue long after an event has ended. Those rows
    # are not that event's final score; retain the same five-minute end window
    # as the frozen historical population before checking final quality.
    upper = min(before, end + 300000)
    return [point for point in points if start <= point["time"] <= upper]


def _event_rows(metadata: dict) -> list[dict]:
    if not isinstance(metadata, dict):
        raise ValueError("Event metadata must be an object")
    rows = []
    for key, value in metadata.items():
        event = integer(key, "event", 1)
        if not isinstance(value, dict):
            raise ValueError("Invalid event metadata record")
        starts, ends = value.get("startAt", []), value.get("endAt", [])
        if not isinstance(starts, list) or not isinstance(ends, list):
            raise ValueError("Invalid event metadata dates")
        if len(starts) <= SERVER or starts[SERVER] is None:
            continue
        if len(ends) <= SERVER or ends[SERVER] is None:
            raise ValueError("CN event starts without an end")
        start, end = integer(starts[SERVER], "start", 1), integer(ends[SERVER], "end", 1)
        if end <= start:
            raise ValueError("Invalid CN event duration")
        names = value.get("eventName", [])
        name = next((str(names[i]) for i in [SERVER, 0] if isinstance(names, list)
                     and len(names) > i and names[i]), str(event))
        rows.append({"event": event, "start_ms": start, "end_ms": end,
                     "kind": str(value.get("eventType") or "unknown"), "name": name})
    return sorted(rows, key=lambda row: (row["start_ms"], row["event"]))


class Store:
    """Own the production data store below ``root``; bootstrap at most once.

    ``load_curves(before_ms)`` exposes only verified complete history whose end
    is strictly before the clock. With ``complete_only=False`` it exposes every
    started, nonempty prefix, truncated inclusively at that clock instead. Both
    paths exclude tracker tails after the event's five-minute final window.
    ``events(now_ms)`` includes started dense CN events, including ended events.
    """

    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()
        protected = (ROOT / "data").resolve()
        if self.root == protected or protected in self.root.parents:
            raise ValueError("Production data must be separate from repository data")
        self.database = self.root / "data.sqlite3"
        self.metadata_directory = self.root / "metadata"
        self.root.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS events(
                    event INTEGER PRIMARY KEY,start_ms INTEGER NOT NULL,end_ms INTEGER NOT NULL,
                    kind TEXT NOT NULL,name TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS curves(
                    event INTEGER NOT NULL,rank INTEGER NOT NULL,points_json TEXT NOT NULL,
                    updated_ms INTEGER NOT NULL,verified_closed INTEGER NOT NULL DEFAULT 0,
                    source_sha256 TEXT,quality_json TEXT NOT NULL DEFAULT '{}',PRIMARY KEY(event,rank));
                CREATE TABLE IF NOT EXISTS fetch_log(
                    id INTEGER PRIMARY KEY,at_ms INTEGER NOT NULL,category TEXT NOT NULL,
                    event INTEGER,rank INTEGER,url TEXT,status TEXT NOT NULL,source_sha256 TEXT,
                    raw_path TEXT,error TEXT,details_json TEXT NOT NULL DEFAULT '{}');
                CREATE TABLE IF NOT EXISTS metadata_state(
                    name TEXT PRIMARY KEY,refreshed_ms INTEGER NOT NULL,source_sha256 TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            """)

    @contextmanager
    def _connect(self):
        with closing(sqlite3.connect(self.database, timeout=60)) as con:
            con.row_factory = sqlite3.Row
            with con:
                yield con

    @staticmethod
    def _upsert_events(con, rows):
        for row in rows:
            prior = con.execute("SELECT start_ms,end_ms FROM events WHERE event=?", (row["event"],)).fetchone()
            if prior is not None and (prior["start_ms"], prior["end_ms"]) != (row["start_ms"], row["end_ms"]):
                con.execute("UPDATE curves SET verified_closed=0 WHERE event=?", (row["event"],))
            con.execute("INSERT INTO events VALUES(:event,:start_ms,:end_ms,:kind,:name) "
                        "ON CONFLICT(event) DO UPDATE SET start_ms=excluded.start_ms,end_ms=excluded.end_ms,"
                        "kind=excluded.kind,name=excluded.name", row)

    @staticmethod
    def _validate_event_refresh(con, rows, now):
        if not rows:
            raise ValueError("Event refresh contains no valid CN activities")
        # A full-catalog response must retain the activities already supporting
        # historical features and current inference. Allow legitimate date
        # corrections, but do not accept missing IDs or missing CN dates as a
        # successful refresh: that would strand the metadata consumer.
        required = {row[0] for row in con.execute("SELECT event FROM events WHERE start_ms>=? AND start_ms<=?",
                                                (MIN_EVENT_START_MS, now))}
        missing = sorted(required - {row["event"] for row in rows})
        if missing:
            raise ValueError("Event refresh omits known CN activities or valid dates: " +
                             ",".join(map(str, missing)))

    def _archive(self, raw: bytes) -> tuple[str, str]:
        digest = hashlib.sha256(raw).hexdigest()
        relative = f"snapshots/{digest}.json"
        path = self.root / relative
        if path.exists():
            if _sha256(path) != digest:
                raise ValueError("Changed production response snapshot: " + relative)
        else:
            _atomic_bytes(path, raw)
        return digest, relative

    @staticmethod
    def _log(con, now, category, status, *, event=None, rank=None, url=None,
             digest=None, raw_path=None, error=None, details=None):
        con.execute("INSERT INTO fetch_log(at_ms,category,event,rank,url,status,source_sha256,raw_path,error,details_json) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?)", (now, category, event, rank, url, status, digest,
                                                     raw_path, error, _json(details or {})))

    def bootstrap(self, database=ROOT / "data/db.sqlite3", metadata_directory=ROOT / "data/metadata", *, now_ms: int) -> dict:
        """Copy local dense sources once, opening the existing database read-only."""
        now = integer(now_ms, "now_ms", 1)
        source, source_metadata = Path(database).resolve(), Path(metadata_directory).resolve()
        if source == self.database:
            raise ValueError("Cannot bootstrap from the production destination")
        with self._connect() as con:
            prior = con.execute("SELECT value FROM settings WHERE key='bootstrap'").fetchone()
            if prior is not None:
                return json.loads(prior["value"]) | {"status": "already_bootstrapped"}
        metadata = {}
        for name in ["events", "characters", "bands", "cards"]:
            raw = (source_metadata / f"{name}.json").read_bytes()
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError("Invalid local metadata: " + name)
            metadata[name] = (raw, value)
        event_rows = _event_rows(metadata["events"][1])
        source_digest = _sha256(source)
        imported, invalid = 0, 0
        with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as original, self._connect() as con:
            original.row_factory = sqlite3.Row
            original.execute("BEGIN")
            self._upsert_events(con, event_rows)
            for row in original.execute("SELECT * FROM event WHERE Country=? AND StartAt>=? AND StartAt<=? "
                                        "ORDER BY StartAt,EventID,Rank", ("中国大陆", MIN_EVENT_START_MS, now)):
                try:
                    event, rank = integer(row["EventID"], "event", 1), integer(row["Rank"], "rank", 1)
                    start, end = integer(row["StartAt"], "start", 1), integer(row["EndAt"], "end", 1)
                    if end <= start or rank not in RANKS:
                        raise ValueError("Invalid local event/rank")
                    payload = row["PointRank"]
                    try:
                        payload = json.loads(payload)
                    except (ValueError, TypeError):
                        payload = ast.literal_eval(payload)
                    values, quality = normalize_points(payload)
                    quality["future_points_excluded"] = sum(point["time"] > now for point in values)
                    # Metadata dates are refreshed independently; missing local metadata
                    # does not discard a valid public history stored in the source DB.
                    if con.execute("SELECT 1 FROM events WHERE event=?", (event,)).fetchone() is None:
                        self._upsert_events(con, [{"event": event, "start_ms": start, "end_ms": end,
                                                  "kind": str(row["EventType"] or "unknown"),
                                                  "name": str(row["EventName"] or event)}])
                    clock = con.execute("SELECT start_ms,end_ms FROM events WHERE event=?", (event,)).fetchone()
                    if clock["start_ms"] < MIN_EVENT_START_MS:
                        continue
                    quality["after_event_points_excluded"] = sum(p["time"] > clock["end_ms"] + 300000 for p in values)
                    values = _clock_points(values, clock["start_ms"], clock["end_ms"], now)
                    con.execute("INSERT INTO curves VALUES(?,?,?,?,?,?,?) ON CONFLICT(event,rank) DO NOTHING",
                                (event, rank, _json(values), now, int(_complete(values, clock["end_ms"], now)),
                                 source_digest, _json(quality)))
                    imported += 1
                except (ValueError, TypeError, SyntaxError, OverflowError) as exc:
                    invalid += 1
                    self._log(con, now, "bootstrap_curve", "invalid", error=str(exc),
                              details={"event": row["EventID"], "rank": row["Rank"]})
            for name, (raw, _) in metadata.items():
                digest, relative = self._archive(raw)
                _atomic_bytes(self.metadata_directory / f"{name}.json", raw)
                self._log(con, now, "bootstrap_metadata", "ok", digest=digest, raw_path=relative,
                          details={"name": name, "source": str(source_metadata / f"{name}.json")})
            result = {"status": "bootstrapped", "as_of_ms": now, "imported_curves": imported,
                      "invalid_curves": invalid, "source": str(source), "source_sha256": source_digest}
            con.execute("INSERT INTO settings VALUES('bootstrap',?)", (_json(result),))
            self._log(con, now, "bootstrap", "ok", digest=source_digest, details=result)
        return result

    def events(self, now_ms: int) -> list[dict]:
        now = integer(now_ms, "now_ms", 1)
        with self._connect() as con:
            return [dict(row) for row in con.execute("SELECT * FROM events WHERE start_ms>=? AND start_ms<=? "
                                                     "ORDER BY start_ms,event", (MIN_EVENT_START_MS, now))]

    def load_curves(self, before_ms: int, ranks=None, complete_only=True) -> list[Curve]:
        before, selected = integer(before_ms, "before_ms", 1), _ranks(ranks)
        with self._connect() as con:
            rows = con.execute("SELECT c.*,e.start_ms,e.end_ms,e.kind FROM curves c JOIN events e USING(event) "
                               "WHERE e.start_ms>=? AND e.start_ms<=? ORDER BY e.start_ms,e.event,c.rank",
                               (MIN_EVENT_START_MS, before)).fetchall()
        result = []
        for row in rows:
            if row["rank"] not in selected or (complete_only and row["end_ms"] >= before):
                continue
            points, _ = normalize_points(json.loads(row["points_json"]))
            points = _clock_points(points, row["start_ms"], row["end_ms"], before)
            if not points or (complete_only and not _complete(points, row["end_ms"], before)):
                continue
            result.append(Curve(row["event"], row["rank"], row["kind"], row["start_ms"], row["end_ms"],
                                np.array([p["time"] for p in points], dtype=np.int64),
                                np.array([p["ep"] for p in points], dtype=float)))
        return result


async def sync_data(store: Store, now_ms: int, ranks=None, transport=None) -> dict:
    """Refresh event dates each cycle and fetch active/recent unfinished pairs.

    Character and band dictionaries retain a daily refresh interval.
    Closed quality-verified pairs are not fetched repeatedly. Missing, empty,
    failed or incomplete pairs remain retryable for fourteen days after event
    end. Metadata/network errors are recorded and returned; good prior data is
    retained. Calls share the existing downloader's retry policy and 2 rps cap.
    """
    now, selected = integer(now_ms, "now_ms", 1), _ranks(ranks)
    result = {"as_of_ms": now, "ranks": selected, "requested_pairs": 0,
              "updated_curves": 0, "empty": 0, "verified_closed": 0,
              "metadata_refreshed": [], "errors": []}
    limiter = Limiter(DEFAULT_RATE)
    async with httpx.AsyncClient(timeout=httpx.Timeout(60., connect=15.), transport=transport,
                                 follow_redirects=True) as client:
        for name, url in METADATA_URLS.items():
            with store._connect() as con:
                state = con.execute("SELECT refreshed_ms FROM metadata_state WHERE name=?", (name,)).fetchone()
            if name != "events" and state is not None and 0 <= now - state["refreshed_ms"] < DAY_MS:
                continue
            digest = relative = None
            try:
                response, attempts = await request(client, url, None, limiter)
                digest, relative = store._archive(response.content)
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("Metadata must be an object")
                rows = _event_rows(payload) if name == "events" else None
                with store._connect() as con:
                    if rows is not None:
                        store._validate_event_refresh(con, rows, now)
                    _atomic_bytes(store.metadata_directory / f"{name}.json", response.content)
                    if rows is not None:
                        store._upsert_events(con, rows)
                    con.execute("INSERT INTO metadata_state VALUES(?,?,?) ON CONFLICT(name) DO UPDATE "
                                "SET refreshed_ms=excluded.refreshed_ms,source_sha256=excluded.source_sha256",
                                (name, now, digest))
                    store._log(con, now, "metadata", "ok", url=url, digest=digest, raw_path=relative,
                               details={"name": name, "attempts": attempts})
                result["metadata_refreshed"].append(name)
            except (httpx.HTTPError, ValueError, TypeError, KeyError, OSError) as exc:
                error = {"category": "metadata", "name": name, "error": f"{type(exc).__name__}: {exc}"}
                result["errors"].append(error)
                with store._connect() as con:
                    store._log(con, now, "metadata", "failed", url=url, digest=digest,
                               raw_path=relative, error=error["error"], details={"name": name})
        candidates = [row for row in store.events(now) if row["end_ms"] >= now - RECENT_MS]
        for event in candidates:
            for rank in selected:
                with store._connect() as con:
                    known = con.execute("SELECT * FROM curves WHERE event=? AND rank=?", (event["event"], rank)).fetchone()
                if known is not None and known["verified_closed"] and event["end_ms"] < now:
                    continue
                result["requested_pairs"] += 1
                url = "https://bestdori.com/api/eventtop/data" if rank == 10 else "https://bestdori.com/api/tracker/data"
                params = {"server": SERVER, "event": event["event"]}
                params.update({"mid": 0, "interval": 0} if rank == 10 else {"tier": rank})
                digest = relative = None
                try:
                    response, attempts = await request(client, url, params, limiter)
                    digest, relative = store._archive(response.content)
                    points, quality = (normalize_top if rank == 10 else normalize_cutoffs)(response.json())
                    quality["future_points_excluded"] = sum(p["time"] > now for p in points)
                    quality["after_event_points_excluded"] = sum(p["time"] > event["end_ms"] + 300000 for p in points)
                    points = _clock_points(points, event["start_ms"], event["end_ms"], now)
                    old = json.loads(known["points_json"]) if known is not None else []
                    merged, changes = merge_points(old, points)
                    merged = _clock_points(merged, event["start_ms"], event["end_ms"], now)
                    complete = _complete(merged, event["end_ms"], now)
                    quality.update(changes)
                    status = "ok" if points else "empty"
                    with store._connect() as con:
                        if points:
                            con.execute("INSERT INTO curves VALUES(?,?,?,?,?,?,?) ON CONFLICT(event,rank) DO UPDATE "
                                        "SET points_json=excluded.points_json,updated_ms=excluded.updated_ms,"
                                        "verified_closed=excluded.verified_closed,source_sha256=excluded.source_sha256,"
                                        "quality_json=excluded.quality_json",
                                        (event["event"], rank, _json(merged), now, int(complete), digest, _json(quality)))
                        store._log(con, now, "curve", status, event=event["event"], rank=rank, url=url,
                                   digest=digest, raw_path=relative,
                                   details={"params": params, "quality": quality, "attempts": attempts,
                                            "verified_closed": complete})
                    result["updated_curves"] += int(bool(points))
                    result["empty"] += int(not points)
                    result["verified_closed"] += int(complete and bool(points))
                except (httpx.HTTPError, ValueError, TypeError, KeyError, OSError) as exc:
                    error = {"category": "curve", "event": event["event"], "rank": rank,
                             "error": f"{type(exc).__name__}: {exc}"}
                    result["errors"].append(error)
                    with store._connect() as con:
                        store._log(con, now, "curve", "failed", event=event["event"], rank=rank, url=url,
                                   digest=digest, raw_path=relative, error=error["error"], details={"params": params})
    result["status"] = "degraded" if result["errors"] else "ok"
    return result
