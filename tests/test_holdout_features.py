import importlib.util
import unittest
import numpy as np

from analysis.backtest import Curve, snapshot
from analysis.event_features import EventFeatures

if importlib.util.find_spec('analysis.holdout_features'):
    from analysis.holdout_features import HoldoutFeatureBuilder
else:
    HoldoutFeatureBuilder = None


class HoldoutFeatureTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(HoldoutFeatureBuilder)
        self.metadata = EventFeatures({str(e):{'characters':[{'characterId':1}]} for e in [1,2,3]},
                                      {'1':{'bandId':1}}, {'1':{'bandName':['A']*5}}, {})
        hour = 3600000
        def curve(event,offset,rank=1000):
            return Curve(event,rank,'live',offset*hour,(offset+100)*hour,
                         np.arange(offset,offset+101,dtype=np.int64)*hour,np.arange(1,102,dtype=float)*100)
        self.past = curve(1,0)
        self.target = curve(3,400)
        self.sample = snapshot(self.target,450*hour)
        self.sample.update({'case_id':'test:3:1000:50%','peer_curve':None})

    def test_history_only_uses_training_pool(self):
        builder = HoldoutFeatureBuilder([self.past],self.metadata)
        features = builder.frame([self.sample],'history').iloc[0]
        self.assertEqual(features['history_rank_events'],1)
        # There is no API accepting validation/test histories during evaluation.
        self.assertEqual([c.event for c in builder.history],[1])

    def test_target_and_later_training_finals_are_excluded(self):
        builder = HoldoutFeatureBuilder([self.target,self.past],self.metadata)
        self.assertEqual(builder.frame([self.sample],'history').iloc[0]['history_rank_events'],1)
        early = snapshot(self.past,50*3600000)
        early.update({'case_id':'train:1:1000:50%','peer_curve':None})
        self.assertEqual(builder.frame([early],'history').iloc[0]['history_rank_events'],0)

    def test_missing_explicit_peer_does_not_search_full_curves(self):
        peer = Curve(3,2000,'live',self.target.start,self.target.end,self.target.times,self.target.scores*2)
        builder = HoldoutFeatureBuilder([self.past,peer],self.metadata)
        self.assertEqual(builder.frame([self.sample],'peer').iloc[0]['peer_available'],0)
        self.sample['peer_curve'] = peer
        self.assertEqual(builder.frame([self.sample],'peer').iloc[0]['peer_available'],1)


if __name__ == '__main__':unittest.main()
