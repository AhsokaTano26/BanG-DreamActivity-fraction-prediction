"""Context for fixed holdout: train-only outcomes and explicit peer prefixes."""
from __future__ import annotations

import pandas as pd


class HoldoutFeatureBuilder:
    def __init__(self, history, metadata):
        self.history = list(history)
        self.metadata = metadata
        self.static_cache = {}
        self.history_cache = {}

    def frame(self, samples, mode):
        if mode not in ('baseline','band','static','history','peer'):
            raise ValueError(mode)
        rows = []
        for sample in samples:
            curve = sample['curve']
            key = (curve.event,curve.rank,sample['now'])
            features = sample['features'].copy()
            if mode == 'band':
                features.update(self.metadata.band_only(curve))
            elif mode != 'baseline':
                if key not in self.static_cache:
                    self.static_cache[key] = self.metadata.static(curve,sample['now'])
                features.update(self.static_cache[key])
            if mode in ('history','peer'):
                if key not in self.history_cache:
                    self.history_cache[key] = self.metadata.historical(sample,self.history)
                features.update(self.history_cache[key])
            if mode == 'peer':
                # Never look up a peer in a pool of full validation/test curves.
                peer = sample.get('peer_curve')
                features.update(self.metadata.peer(sample,[peer] if peer is not None else []))
            rows.append(features)
        return pd.DataFrame(rows)
