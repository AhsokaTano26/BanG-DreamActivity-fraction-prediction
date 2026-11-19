"""Scoring must be paired and activity-weighted, without silently losing cases."""
import importlib.util
import unittest

if importlib.util.find_spec('analysis.holdout_evaluate'):
    from analysis.holdout_evaluate import score_predictions
    from analysis import holdout_evaluate as evaluator
else:
    score_predictions = None


class HoldoutEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(score_predictions, 'fixed holdout scorer not implemented')
        self.cases = [{'case_id':'a','event':1,'rank':1000,'stage':'25%','event_type':'live'},
                      {'case_id':'b','event':1,'rank':2000,'stage':'50%','event_type':'live'},
                      {'case_id':'c','event':2,'rank':1000,'stage':'25%','event_type':'challenge'}]
        self.labels = {'a':100.,'b':100.,'c':100.}
        self.predictions = [{'case_id':c['case_id'],'predicted':p} for c,p in zip(self.cases,[110.,110.,130.])]

    def test_primary_metric_weights_events_equally(self):
        result = score_predictions(self.cases,self.labels,self.predictions)
        self.assertAlmostEqual(result['overall']['event_macro_mape_pct'],20.)
        self.assertAlmostEqual(result['overall']['case_mape_pct'],50/3)
        self.assertEqual(result['overall']['events'],2)
        self.assertEqual(result['overall']['cases'],3)

    def test_missing_or_duplicate_predictions_are_rejected(self):
        for bad in [self.predictions[:-1],self.predictions+[self.predictions[0]]]:
            with self.assertRaisesRegex(ValueError,'case|Case'):score_predictions(self.cases,self.labels,bad)

    def test_extra_or_nonfinite_predictions_are_rejected(self):
        for bad in [self.predictions+[{'case_id':'x','predicted':100.}],
                    self.predictions[:-1]+[{'case_id':'c','predicted':float('nan')}]]:
            with self.assertRaises(ValueError):score_predictions(self.cases,self.labels,bad)

    def test_paired_errors_use_frozen_event_identity(self):
        self.assertTrue(hasattr(evaluator,'paired_event_errors'),'paired scorer must use frozen event mapping')
        altered = [r | {'event':999} for r in self.predictions]
        errors = evaluator.paired_event_errors(self.cases,self.labels,altered)
        self.assertEqual(list(errors.index),[1,2])
        self.assertAlmostEqual(errors.loc[1],10.)
        self.assertAlmostEqual(errors.loc[2],30.)


if __name__ == '__main__':unittest.main()
