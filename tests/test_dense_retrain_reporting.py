import unittest
import pandas as pd
from analysis import dense_retrain_comparison as comparison


class DenseReportingTests(unittest.TestCase):
    def test_bootstrap_orders_distinct_events_by_actual_time_not_rank_or_id(self):
        parts=pd.DataFrame([
            {'split':'test','event':313,'rank':2000,'start_ms':10},
            {'split':'test','event':308,'rank':1000,'start_ms':20},
            {'split':'test','event':313,'rank':1000,'start_ms':10},
            {'split':'test','event':308,'rank':2000,'start_ms':20},
            {'split':'train','event':226,'rank':1000,'start_ms':1},
        ])
        self.assertEqual(comparison.ordered_test_events(parts,{308,313}),[313,308])

    def test_bootstrap_refuses_unknown_activity_membership(self):
        parts=pd.DataFrame([{'split':'test','event':308,'rank':1000,'start_ms':20}])
        with self.assertRaisesRegex(ValueError,'membership'):
            comparison.ordered_test_events(parts,{308,999})

    def test_evaluation_binding_rejects_other_valid_run(self):
        from analysis import dense_retrain_report as report
        with self.assertRaisesRegex(ValueError,'evaluation'):
            report.require_evaluation_binding({'evaluation_id':'e','identity':{'blind_manifest_sha256':'other'}},{'evaluation_id':'e','dataset_id':'d'},{'dataset_id':'d'},'expected')

    def test_training_binding_rejects_other_valid_run(self):
        from analysis import dense_retrain_report as report
        from pathlib import Path
        with self.assertRaisesRegex(ValueError,'training'):
            report.require_training_binding({'dataset_id':'d','run_id':'other','blind_seal_sha256':'h'},{'dataset_id':'d','run_id':'expected'},'h',Path('/tmp/run'))

    def test_exact_inventory_rejects_equal_count_substitution(self):
        from analysis import dense_retrain_report as report
        with self.assertRaisesRegex(ValueError,'inventory'):
            report.require_inventory({'a','wrong'},{'a','b'})
