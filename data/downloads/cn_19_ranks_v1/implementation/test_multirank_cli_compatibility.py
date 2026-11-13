"""Expanding the database must not silently expand legacy two-rank training."""
import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest import mock


class MultiRankCLICompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.curves=[SimpleNamespace(event=e,rank=r) for e,r in
                     [(1,10),(1,1000),(1,2000),(2,20),(2,1000),(2,2000)]]
        self.target=self.curves[2]

    def test_baseline_cli_preserves_its_two_rank_training_scope(self):
        import predict
        def forecast(target,history,cutoff,**kwargs):
            self.assertEqual({c.rank for c in history},{1000,2000})
            return {'checked':True}
        for offline in [False,True]:
            with self.subTest(offline=offline),mock.patch.object(predict,'fetch_current',return_value=self.target), \
                mock.patch.object(predict,'load_local',return_value=(self.target,self.curves)), \
                mock.patch.object(predict,'load_curves',return_value=(self.curves,[])), \
                mock.patch.object(predict,'load_calibration',return_value=(None,None)), \
                mock.patch.object(predict,'predict_cutoff',side_effect=forecast),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(predict.main(['--event','1','--rank','2000','--json']+(['--offline'] if offline else [])),0)

    def test_context_cli_uses_exact_partner_and_preserves_training_scope(self):
        import predict_context as p
        def forecast(target,curves,cutoff,peer):
            self.assertEqual(peer.rank,1000)
            self.assertEqual({c.rank for c in curves},{1000,2000})
            return {'checked':True}
        for offline in [False,True]:
            with self.subTest(offline=offline),mock.patch.object(p,'fetch_current',side_effect=lambda event,rank:next(c for c in self.curves if c.event==event and c.rank==rank)), \
                mock.patch.object(p,'load_local',return_value=(self.target,self.curves)), \
                mock.patch.object(p,'load_curves',return_value=(self.curves,[])), \
                mock.patch.object(p,'forecast',side_effect=forecast),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(p.main(['--event','1','--rank','2000','--json']+(['--offline'] if offline else [])),0)
