"""Clock and recovery contracts for the isolated production source store."""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import httpx

from production.data import Store, sync_data


MIN_START = 1706504400000
DAY = 86400000


def event_info(start, end, name="CN activity", kind="story"):
    return {"startAt": [None, None, None, str(start)],
            "endAt": [None, None, None, str(end)], "eventType": kind,
            "eventName": [name, None, None, name], "characters": [], "members": []}


def points(start, end, final=500):
    return [{"time": start + (end - start) * i // 4, "ep": final * i // 4}
            for i in range(5)]


class StoreFixture:
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.database = self.base / "source.sqlite3"
        self.metadata = self.base / "source_metadata"
        self.metadata.mkdir()
        self.info = {}
        for name in ["events", "characters", "bands", "cards"]:
            (self.metadata / f"{name}.json").write_text("{}")
        with closing(sqlite3.connect(self.database)) as con, con:
            con.execute("CREATE TABLE event(EventID INTEGER, Rank INTEGER, Country TEXT, "
                        "StartAt INTEGER, EndAt INTEGER, EventType TEXT, EventName TEXT, PointRank TEXT)")
        self.store = Store(self.base / "live")

    def seed(self, event, rank, start, end, values, *, literal=False):
        self.info[str(event)] = event_info(start, end)
        (self.metadata / "events.json").write_text(json.dumps(self.info))
        with closing(sqlite3.connect(self.database)) as con, con:
            con.execute("INSERT INTO event VALUES(?,?,?,?,?,?,?,?)",
                        (event, rank, "中国大陆", start, end, "story", "CN activity",
                         repr(values) if literal else json.dumps(values)))

    def bootstrap(self, now):
        return self.store.bootstrap(database=self.database,
                                    metadata_directory=self.metadata, now_ms=now)


class ProductionStoreTests(StoreFixture, unittest.TestCase):
    def test_bootstrap_uses_actual_start_and_strict_completed_clock(self):
        start, end = MIN_START, MIN_START + DAY
        self.seed(1, 1000, start, end, points(start, end), literal=True)
        self.seed(9999, 1000, start - DAY, start - 1, points(start - DAY, start - 1))
        self.seed(2, 1000, end + 1, end + DAY, points(end + 1, end + DAY))
        original = self.database.read_bytes()
        result = self.bootstrap(end + DAY + 1)
        self.assertEqual([c.event for c in self.store.load_curves(end)], [])
        self.assertEqual([c.event for c in self.store.load_curves(end + 1)], [1])
        self.assertEqual([c.event for c in self.store.load_curves(end + DAY + 1)], [1, 2])
        self.assertEqual(self.database.read_bytes(), original)
        self.assertGreaterEqual(result["imported_curves"], 2)
        self.assertEqual([e["event"] for e in self.store.events(end)], [1])

    def test_bootstrap_is_idempotent_and_read_results_are_detached(self):
        end = MIN_START + DAY
        self.seed(7, 1000, MIN_START, end, points(MIN_START, end))
        self.bootstrap(end + 1)
        curve = self.store.load_curves(end + 1)[0]
        curve.scores[-1] = 99999
        self.seed(8, 1000, end + 1, end + DAY, points(end + 1, end + DAY))
        again = self.bootstrap(end + DAY + 1)
        self.assertEqual(again["status"], "already_bootstrapped")
        self.assertEqual([c.event for c in self.store.load_curves(end + DAY + 1)], [7])
        self.assertEqual(self.store.load_curves(end + 1)[0].final, 500)
        self.assertTrue((self.store.metadata_directory / "cards.json").exists())

    def test_invalid_sparse_and_missing_final_curves_do_not_become_labels(self):
        end = MIN_START + DAY
        valid = points(MIN_START, end)
        self.seed(1, 1000, MIN_START, end, valid)
        self.seed(2, 1000, MIN_START, end, valid[:4])
        self.seed(3, 1000, MIN_START, end, points(MIN_START, end - 60001))
        self.seed(4, 1000, MIN_START, end, points(MIN_START, end + 300001))
        self.seed(5, 1000, MIN_START, end, points(MIN_START, end, final=0))
        broken = points(MIN_START, end)
        broken[2]["ep"] = -1
        self.seed(6, 1000, MIN_START, end, broken)
        self.bootstrap(end + DAY)
        self.assertEqual([c.event for c in self.store.load_curves(end + DAY)], [1])

    def test_prefix_clock_hides_future_observations_and_filters_ranks(self):
        end = MIN_START + DAY
        self.seed(1, 1000, MIN_START, end, points(MIN_START, end))
        self.seed(1, 2000, MIN_START, end, points(MIN_START, end))
        now = MIN_START + DAY // 2
        self.bootstrap(now)
        curve = self.store.load_curves(now, ranks=[2000], complete_only=False)[0]
        self.assertEqual(curve.rank, 2000)
        self.assertEqual(curve.times.tolist(), [MIN_START, MIN_START + DAY // 4, now])
        self.assertEqual(self.store.load_curves(now), [])

    def test_complete_history_discards_postevent_tracker_tail(self):
        end = MIN_START + DAY
        values = points(MIN_START, end) + [{"time": end + 38 * 3600000, "ep": 99999999}]
        self.seed(1, 10, MIN_START, end, values)
        self.bootstrap(end + 3 * DAY)
        curves = self.store.load_curves(end + 3 * DAY, ranks=[10])
        self.assertEqual(len(curves), 1)
        self.assertEqual(curves[0].final, 500)
        self.assertEqual(int(curves[0].times[-1]), end)


class ProductionSyncTests(StoreFixture, unittest.IsolatedAsyncioTestCase):
    async def test_empty_incomplete_and_late_final_retry_then_skip_verified(self):
        start, end = MIN_START + 30 * DAY, MIN_START + 31 * DAY
        self.seed(8, 1000, start, end, [])
        self.bootstrap(end + 1)
        batches = [[], points(start, end - 120000), points(start, end)]
        requested = []

        def serve(request):
            if request.url.path.endswith("all.5.json"):
                return httpx.Response(200, json=self.info if "/events/" in request.url.path else {})
            if "/bands/" in request.url.path:
                return httpx.Response(200, json={})
            requested.append(str(request.url))
            return httpx.Response(200, json={"result": True, "cutoffs": batches.pop(0)})

        transport = httpx.MockTransport(serve)
        first = await sync_data(self.store, now_ms=end + 1, ranks=[1000], transport=transport)
        self.assertEqual(first["empty"], 1)
        self.assertEqual(self.store.load_curves(end + 1), [])
        await sync_data(self.store, now_ms=end + 2, ranks=[1000], transport=transport)
        self.assertEqual(self.store.load_curves(end + 2), [])
        await sync_data(self.store, now_ms=end + 3, ranks=[1000], transport=transport)
        self.assertEqual(self.store.load_curves(end + 3)[0].final, 500)
        final = await sync_data(self.store, now_ms=end + 4, ranks=[1000], transport=transport)
        self.assertEqual(final["requested_pairs"], 0)
        self.assertEqual(len(requested), 3)

    async def test_failed_or_malformed_fetch_preserves_known_prefix_and_provenance(self):
        start, end = MIN_START + 30 * DAY, MIN_START + 31 * DAY
        now = start + DAY // 2
        self.seed(8, 1000, start, end, points(start, end)[:3])
        self.bootstrap(now)
        response_kind = ["failure", "invalid", "success"]

        def serve(request):
            if "/events/" in request.url.path:
                return httpx.Response(200, json=self.info)
            if "/characters/" in request.url.path or "/bands/" in request.url.path:
                return httpx.Response(200, json={})
            mode = response_kind.pop(0)
            if mode == "failure":
                return httpx.Response(404)
            if mode == "invalid":
                return httpx.Response(200, json={"result": True, "cutoffs": [{"time": now, "ep": -1}]})
            return httpx.Response(200, json={"result": True, "cutoffs": [{"time": now - 1, "ep": 300},
                                                                          {"time": now + 1, "ep": 900}]})

        transport = httpx.MockTransport(serve)
        for index in range(3):
            status = await sync_data(self.store, now_ms=now, ranks=[1000], transport=transport)
            curve = self.store.load_curves(now, complete_only=False)[0]
            self.assertEqual(curve.final, 250)
            self.assertLessEqual(int(curve.times[-1]), now)
            if index < 2:
                self.assertEqual(len(status["errors"]), 1)
        with closing(sqlite3.connect(self.store.database)) as con:
            records = con.execute("SELECT raw_path,source_sha256 FROM fetch_log WHERE category='curve' "
                                  "AND raw_path IS NOT NULL").fetchall()
        self.assertEqual(len(records), 2)
        for relative, digest in records:
            self.assertEqual(hashlib.sha256((self.store.root / relative).read_bytes()).hexdigest(), digest)

    async def test_each_cycle_discovers_new_events_with_daily_dictionary_cache(self):
        now = MIN_START + 40 * DAY
        self.bootstrap(now)
        fresh = {"5": event_info(now - DAY, now + DAY, "new event", "live_try"),
                 "10000": event_info(MIN_START - DAY, MIN_START - 1)}
        calls = []

        def serve(request):
            calls.append(request.url.path)
            if "/events/" in request.url.path:
                return httpx.Response(200, json=fresh)
            if "/characters/" in request.url.path or "/bands/" in request.url.path:
                return httpx.Response(200, json={})
            return httpx.Response(200, json={"result": True, "cutoffs": [{"time": now, "ep": 12}]})

        transport = httpx.MockTransport(serve)
        await sync_data(self.store, now_ms=now, ranks=[1000], transport=transport)
        event = self.store.events(now)[0]
        self.assertEqual((event["event"], event["kind"], event["name"]), (5, "live_try", "new event"))
        self.assertEqual(self.store.load_curves(now, complete_only=False)[0].final, 12)
        fresh = fresh | {"6": event_info(now, now + 2 * DAY, "announced next cycle")}
        await sync_data(self.store, now_ms=now + 1, ranks=[1000], transport=transport)
        self.assertEqual([event["event"] for event in self.store.events(now + 1)], [5, 6])
        self.assertEqual(sum("/events/" in url for url in calls), 2)
        self.assertEqual(sum("/characters/" in url for url in calls), 1)
        self.assertEqual(sum("/bands/" in url for url in calls), 1)

    async def test_incomplete_event_refresh_preserves_good_metadata_and_retries(self):
        now = MIN_START + 40 * DAY
        self.seed(1, 1000, MIN_START, MIN_START + DAY, points(MIN_START, MIN_START + DAY))
        self.seed(2, 1000, now - DAY, now + 3 * DAY, [{"time": now, "ep": 12}])
        self.bootstrap(now)
        current_payload = self.info

        def serve(request):
            if "/events/" in request.url.path:
                return httpx.Response(200, json=current_payload)
            if "/characters/" in request.url.path or "/bands/" in request.url.path:
                return httpx.Response(200, json={})
            return httpx.Response(200, json={"result": True, "cutoffs": []})

        transport = httpx.MockTransport(serve)
        await sync_data(self.store, now_ms=now, ranks=[1000], transport=transport)
        path = self.store.metadata_directory / "events.json"
        good_file = path.read_bytes()
        missing_history = {"2": event_info(now - DAY, now + 3 * DAY, "wrong replacement", "versus")}
        missing_cn_date = {"1": self.info["1"], "2": dict(self.info["2"], startAt=[None] * 4)}
        malformed = {"1": self.info["1"], "2": dict(self.info["2"], startAt={"invalid": now})}
        for offset, payload in enumerate([{}, missing_history, missing_cn_date, malformed]):
            with self.subTest(payload=payload):
                current_payload = payload
                result = await sync_data(self.store, now_ms=now + DAY + offset,
                                         ranks=[1000], transport=transport)
                self.assertEqual(result["status"], "degraded")
                self.assertEqual(path.read_bytes(), good_file)
                self.assertNotIn("events", result["metadata_refreshed"])
                self.assertTrue(any(error.get("name") == "events" for error in result["errors"]))
                with closing(sqlite3.connect(self.store.database)) as con:
                    refreshed = con.execute("SELECT refreshed_ms FROM metadata_state WHERE name='events'").fetchone()[0]
                self.assertEqual(refreshed, now)
                self.assertEqual(self.store.events(now)[1]["name"], "CN activity")
        current_payload = self.info | {"3": event_info(now - DAY, now + 4 * DAY, "new activity")}
        recovered = await sync_data(self.store, now_ms=now + DAY + 10, ranks=[1000], transport=transport)
        self.assertEqual(recovered["status"], "ok")
        self.assertEqual(set(json.loads(path.read_text())), {"1", "2", "3"})

    async def test_empty_event_catalog_is_rejected_without_prior_activities(self):
        now = MIN_START + DAY
        self.bootstrap(now)

        def serve(request):
            return httpx.Response(200, json={})

        result = await sync_data(self.store, now_ms=now, ranks=[1000], transport=httpx.MockTransport(serve))
        self.assertEqual(result["status"], "degraded")
        self.assertNotIn("events", result["metadata_refreshed"])
        with closing(sqlite3.connect(self.store.database)) as con:
            self.assertIsNone(con.execute("SELECT refreshed_ms FROM metadata_state WHERE name='events'").fetchone())

    async def test_local_metadata_survives_network_failure_and_old_pending_is_bounded(self):
        now = MIN_START + 50 * DAY
        self.seed(1, 1000, MIN_START, MIN_START + DAY, [])
        self.seed(2, 1000, now - DAY, now + DAY, [{"time": now, "ep": 12}])
        self.bootstrap(now)

        def serve(request):
            return httpx.Response(404)

        result = await sync_data(self.store, now_ms=now, ranks=[1000], transport=httpx.MockTransport(serve))
        self.assertEqual(result["requested_pairs"], 1)
        self.assertEqual([e["event"] for e in self.store.events(now)], [1, 2])
        self.assertEqual(self.store.load_curves(now, complete_only=False)[0].final, 12)
        self.assertEqual(len(result["errors"]), 4)

    async def test_corrected_cn_dates_cannot_leave_prepolicy_history_eligible(self):
        now = MIN_START + 40 * DAY
        self.seed(1, 1000, MIN_START, MIN_START + DAY, points(MIN_START, MIN_START + DAY))
        self.bootstrap(now)
        self.assertEqual(len(self.store.load_curves(now)), 1)

        def serve(request):
            payload = {"1": event_info(MIN_START - DAY, MIN_START - 1)} if "/events/" in request.url.path else {}
            return httpx.Response(200, json=payload)

        await sync_data(self.store, now_ms=now, ranks=[1000], transport=httpx.MockTransport(serve))
        self.assertEqual(self.store.load_curves(now), [])
        self.assertEqual(self.store.events(now), [])

    async def test_top10_is_tenth_distinct_user_and_transport_retry_recovers(self):
        now = MIN_START + 10 * DAY
        self.info = {"1": event_info(now - DAY, now + DAY)}
        (self.metadata / "events.json").write_text(json.dumps(self.info))
        self.bootstrap(now)
        attempts = 0

        def serve(request):
            nonlocal attempts
            if "/events/" in request.url.path:
                return httpx.Response(200, json=self.info)
            if "/characters/" in request.url.path or "/bands/" in request.url.path:
                return httpx.Response(200, json={})
            self.assertEqual(request.url.path, "/api/eventtop/data")
            self.assertEqual(request.url.params["interval"], "0")
            attempts += 1
            if attempts == 1:
                return httpx.Response(503)
            return httpx.Response(200, json={"result": True, "points": [
                {"time": now, "uid": uid, "value": uid * 10} for uid in range(1, 11)]})

        result = await sync_data(self.store, now_ms=now, ranks=[10], transport=httpx.MockTransport(serve))
        self.assertEqual(result["errors"], [])
        self.assertEqual(self.store.load_curves(now, complete_only=False)[0].final, 10)

    async def test_sync_clips_postevent_tail_before_verifying_final(self):
        start, end = MIN_START + 30 * DAY, MIN_START + 31 * DAY
        self.seed(8, 1000, start, end, [])
        now = end + 3 * DAY
        self.bootstrap(now)
        received = []

        def serve(request):
            if "/events/" in request.url.path:
                return httpx.Response(200, json=self.info)
            if "/characters/" in request.url.path or "/bands/" in request.url.path:
                return httpx.Response(200, json={})
            received.append(request.url)
            return httpx.Response(200, json={"result": True, "cutoffs": points(start, end) + [
                {"time": end + 38 * 3600000, "ep": 99999999}]})

        transport = httpx.MockTransport(serve)
        first = await sync_data(self.store, now_ms=now, ranks=[1000], transport=transport)
        self.assertEqual(first["verified_closed"], 1)
        self.assertEqual(self.store.load_curves(now)[0].final, 500)
        second = await sync_data(self.store, now_ms=now + 1, ranks=[1000], transport=transport)
        self.assertEqual(second["requested_pairs"], 0)
        self.assertEqual(len(received), 1)

