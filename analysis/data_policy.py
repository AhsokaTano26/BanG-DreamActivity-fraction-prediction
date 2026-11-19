"""Shared population policy for new datasets and production forecasting.

Legacy frozen datasets and their readers retain their original populations.
Callers must apply this policy before fitting, constructing historical features,
or accepting a production model/calibration artifact.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Protocol, TypeVar


MIN_EVENT_START_MS = 1706504400000
MIN_EVENT_START_ISO = '2024-01-29T13:00:00+08:00'
POLICY = {
    'version': 'dense_start_v1',
    'country': '中国大陆',
    'minimum_event_start_ms': MIN_EVENT_START_MS,
    'minimum_event_start_iso': MIN_EVENT_START_ISO,
    'boundary': 'actual CN start >= minimum; never EventID order',
    'scope': 'new modeling populations, complete histories, and production artifacts',
}
POLICY_ID = hashlib.sha256(json.dumps(POLICY, sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()

ROOT = Path(__file__).resolve().parents[1]
DENSE_V1 = ROOT / 'data/holdout_v1_dense_20261115'
DENSE_V2 = ROOT / 'data/temporal_holdout_v2_dense_20261115'
DENSE_V2_TWO_RANK = ROOT / 'data/temporal_holdout_v2_dense_t1000_t2000_20261115'
DENSE_V3 = ROOT / 'data/rolling_holdout_v3_dense_t1000_t2000_20261115'


class HasEventStart(Protocol):
    start: int


T = TypeVar('T', bound=HasEventStart)


def eligible_history(curves: Iterable[T]) -> list[T]:
    """Retain the input order and objects, using the inclusive CN date boundary."""
    return [curve for curve in curves if curve.start >= MIN_EVENT_START_MS]


def require_eligible_start(start: int) -> None:
    """Reject an old target explicitly instead of treating it as new modeling."""
    if start < MIN_EVENT_START_MS:
        raise ValueError(f'活动开始时间不符合当前数据政策：需要 >= {MIN_EVENT_START_ISO}。')
