"""Boundary tests for local foundation models; torch stays in subprocesses."""
from collections import Counter
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from analysis import v2_next_foundation as foundation


def prefix(times=(0, 2, 4), scores=(0, 20, 40), end=7, cutoff=5):
    curve = SimpleNamespace(event=1, rank=1000, kind="normal", start=0,
                            end=end * 3_600_000,
                            times=np.asarray(times, dtype=np.int64) * 3_600_000,
                            scores=np.asarray(scores, dtype=float))
    return {"curve": curve, "now": int(curve.times[-1]), "cutoff": cutoff * 3_600_000,
            "score": float(curve.scores[-1]), "case_id": "fixture"}


class FoundationBoundaryTests(unittest.TestCase):
    def test_balancing_keeps_every_prefix_with_equal_event_mass(self):
        cases = [{"event": e, "case_id": str(i)} for i, e in enumerate([1, 2, 2, 2, 3, 3])]
        indices, audit = foundation.balanced_context_indices(cases, seed=42)
        self.assertEqual(set(indices), set(range(6)))
        self.assertEqual(Counter(cases[i]["event"] for i in indices), {1: 3, 2: 3, 3: 3})
        self.assertEqual(audit["original_cases"], 6)
        self.assertEqual(audit["duplicated_rows"], 3)
        self.assertEqual(audit["independent_events"], 3)
        np.testing.assert_array_equal(indices, foundation.balanced_context_indices(cases, seed=42)[0])

    def test_resampling_stops_at_actual_observation_and_forecasts_to_end(self):
        result = foundation.resample_prefix(prefix(), step_hours=2, max_context=256, max_horizon=256)
        np.testing.assert_array_equal(result["context"], [0, .5, 1])
        self.assertEqual(result["horizon"], 2)
        self.assertEqual(result["end_steps"], 1.5)
        self.assertEqual(result["diagnostics"]["observation_age_hours"], 1)
        self.assertEqual(result["diagnostics"]["observed_at"], 4 * 3_600_000)
        self.assertEqual(foundation.decode_endpoint(result, [1.5, 2.0]), 70.0)

    def test_short_horizon_zero_growth_and_negative_forecast_keep_cases(self):
        result = foundation.resample_prefix(prefix(end=4.5), step_hours=2, max_context=256, max_horizon=256)
        self.assertEqual(foundation.decode_endpoint(result, [2]), 50)
        self.assertEqual(foundation.decode_endpoint(result, [-2]), 40)
        flat = foundation.resample_prefix(prefix(scores=(40, 40, 40)), step_hours=2,
                                         max_context=256, max_horizon=256)
        np.testing.assert_array_equal(flat["context"], [1, 1, 1])

    def test_sparse_prefix_is_retained_and_missing_gap_is_audited(self):
        result = foundation.resample_prefix(prefix(times=(1, 9, 10), scores=(10, 90, 100), end=15, cutoff=11),
                                           step_hours=2, max_context=256, max_horizon=256)
        self.assertTrue(np.isfinite(result["context"]).all())
        self.assertEqual(result["context"][-1], 1)
        self.assertEqual(result["diagnostics"]["max_observation_gap_hours"], 8)
        self.assertGreater(result["diagnostics"]["grid_points_in_large_gap"], 0)

    def test_future_observation_and_infinite_model_output_are_rejected(self):
        sample = prefix()
        sample["now"] = 3 * 3_600_000
        with self.assertRaises(ValueError):
            foundation.resample_prefix(sample, step_hours=2, max_context=256, max_horizon=256)
        valid = foundation.resample_prefix(prefix(), step_hours=2, max_context=256, max_horizon=256)
        with self.assertRaises(ValueError):
            foundation.decode_endpoint(valid, [np.nan, 2])

    def test_unusually_long_event_keeps_entire_observed_span_and_end(self):
        result = foundation.resample_prefix(prefix(times=(0, 400, 800), scores=(0, 20, 40), end=1600, cutoff=801),
                                           step_hours=2, max_context=256, max_horizon=256)
        self.assertLessEqual(len(result["context"]), 256)
        self.assertLessEqual(result["horizon"], 256)
        self.assertGreater(result["diagnostics"]["effective_step_hours"], 2)

    def test_feature_categories_and_missing_values_fit_on_training_only(self):
        frame = pd.DataFrame({"numeric": [1.0, np.nan, 3.0], "category": ["a", "b", "a"]})
        transform = foundation.fit_preprocessor(frame)
        before = copy.deepcopy(transform)
        result = foundation.transform_features(transform, pd.DataFrame({"numeric": [np.nan, 999.], "category": ["new", "a"]}))
        self.assertEqual(transform, before)
        np.testing.assert_array_equal(result[:, 0], [2., 999.])
        self.assertEqual(result[0, 1], -1)
        self.assertTrue(np.isfinite(result).all())

    def test_weight_fingerprint_rejects_changed_local_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights"
            path.write_bytes(b"original")
            identity = foundation.file_identity(path)
            path.write_bytes(b"modified")
            with self.assertRaises(ValueError):
                foundation.verify_file(identity)

    def test_context_vocabulary_does_not_expand_for_unknown_or_missing_query_groups(self):
        arrays = {"cases": [{"case_id": "a"}, {"case_id": "b"}], "base": pd.DataFrame({"x": [1., 2.]}),
                  "event_type": np.array(["normal", "challenge"]), "band_shares": [{"1": 1.}, {"2": 1.}],
                  "progress": np.array([.5, .5]), "remaining_h": np.array([12., 12.]), "rank": np.array([0, 1])}
        frame, design = foundation.build_feature_frame(arrays, "static_context")
        self.assertEqual(frame.shape, (2, 5))
        query = arrays | {"event_type": np.array(["future_type", None]), "band_shares": [{"future_band": 1.}, {}]}
        q, same = foundation.build_feature_frame(query, "static_context", design)
        np.testing.assert_array_equal(q.iloc[:, 1:], np.zeros((2, 4)))
        self.assertEqual(same.types, ("challenge", "normal"))
        self.assertEqual(same.bands, ("1", "2"))

    def test_grid_diagnostics_are_json_serializable_and_cover_queries(self):
        bundle = {"config": foundation.recipes()[-1], "training_grid_diagnostics": []}
        arrays = {"samples": [prefix(), prefix(end=4.5)], "cases": [{"case_id": "a"}, {"case_id": "b"}]}
        report = foundation.diagnostics(bundle, arrays)
        self.assertEqual(len(report["query_grid_diagnostics"]), 2)
        json.dumps(report, allow_nan=False)


class FoundationOfflineIntegrationTests(unittest.TestCase):
    def test_public_local_models_predict_and_roundtrip_without_network(self):
        if not all(row["available"] for row in foundation.availability().values()):
            self.skipTest("Optional local foundation dependencies/weights unavailable")
        # Real networks are expensive and PyTorch/OpenMP must be process isolated.
        script = r'''
import socket, tempfile, joblib, numpy as np, pandas as pd, torch
from types import SimpleNamespace
from analysis import v2_next_foundation as f
def no_network(*args, **kwargs):
    raise AssertionError("Offline inference attempted a network connection")
socket.socket.connect = no_network
torch.use_deterministic_algorithms(True)
samples=[]
for i in range(8):
    c=SimpleNamespace(event=i//2,rank=1000,kind='normal',start=0,end=24*3600000,
        times=np.arange(1,13)*3600000,scores=np.arange(1,13,dtype=float)*(i+1)*100)
    samples.append({'curve':c,'now':12*3600000,'cutoff':12*3600000,'score':float(c.scores[-1]),'case_id':str(i)})
a={'samples':samples,'cases':[{'case_id':str(i),'event':i//2} for i in range(8)],
   'base':pd.DataFrame({'x':np.arange(8,dtype=float),'sparse':[np.nan,1]*4}),
   'current':np.array([s['score'] for s in samples]),'history':[],
   'event_type':np.array(['normal','challenge']*4),'band_shares':[{'1':1.},{'2':1.}]*4,
   'progress':np.full(8,.5),'remaining_h':np.full(8,12.),'rank':np.zeros(8)}
labels={str(i):float(a['current'][i]*(1.5+i*.07)) for i in range(8)}
for config in f.recipes():
    b=f.fit_recipe(config,a,labels)
    members=f.predict_members(b,a);p=np.array(list(members.values()))
    assert p.shape==(3,8) and np.isfinite(p).all() and np.all(p>=a['current'])
    with tempfile.TemporaryDirectory() as d:
        path=d+'/model.joblib';joblib.dump(b,path)
        assert __import__('os').path.getsize(path)<1000000
        loaded=joblib.load(path)
        np.testing.assert_allclose(np.array(list(f.predict_members(loaded,a).values())),p,rtol=1e-6,atol=1e-6)
    if config['zero_shot']:
        np.testing.assert_array_equal(p[0],p[1]);np.testing.assert_array_equal(p[0],p[2])
print('offline-real-model-roundtrip-ok')
'''
        env = os.environ | {"DYLD_LIBRARY_PATH": "/opt/homebrew/opt/libomp/lib",
                           "TABPFN_DISABLE_TELEMETRY": "1", "HF_HUB_OFFLINE": "1",
                           "HF_HUB_DISABLE_TELEMETRY": "1", "TRANSFORMERS_OFFLINE": "1"}
        result = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True,
                                timeout=180, env=env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("offline-real-model-roundtrip-ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
