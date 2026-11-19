"""The current entry point rejects old populations before any training."""
import importlib.util
import unittest

import test_dense_dataset as fixtures


class CurrentBenchmarkTests(unittest.TestCase):
    def test_old_dataset_is_rejected_and_dense_dataset_is_accepted(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.current_benchmark'),
                             'Current benchmark entry point is missing')
        from analysis.current_benchmark import load_dataset
        from analysis.dense_dataset import build_dense_v1
        helper=fixtures.DenseDatasetTests();self.addCleanup(helper.doCleanups)
        root,old,_=helper.fixture()
        with self.assertRaisesRegex(ValueError,'dense|policy|population'):
            load_dataset('v1',old.root)
        build_dense_v1(old,root/'dense')
        data=load_dataset('v1',root/'dense')
        self.assertEqual(data.manifest['splits']['train']['event_ids'],list(range(995,985,-1)))
        with self.assertRaisesRegex(ValueError,'protocol|V2|version'):
            load_dataset('v2',data.root)


if __name__=='__main__':unittest.main()
