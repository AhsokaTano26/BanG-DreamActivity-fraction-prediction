"""Check metadata definitions and strictly historical feature statistics."""
from dataclasses import replace
import importlib.util
import unittest

import numpy as np
from analysis.backtest import Curve, snapshot

if importlib.util.find_spec('analysis.event_features'):
    from analysis.event_features import EventFeatures
else:
    EventFeatures = None

HOUR=3600000


class EventFeatureTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(EventFeatures, 'event features are not implemented')
        self.events={'1':{'eventType':'live_try','characters':[{'characterId':1},{'characterId':2}],
                          'attributes':[{'attribute':'cool'}],
                          'members':[{'situationId':10,'seq':1},{'situationId':11,'seq':2}]},
                     '2':{'eventType':'live_try','characters':[{'characterId':1},{'characterId':3}],
                          'attributes':[{'attribute':'pure'}],'members':[]}}
        self.characters={'1':{'bandId':1},'2':{'bandId':1},'3':{'bandId':2}}
        self.bands={'1':{'bandName':['A']*5},'2':{'bandName':['B']*5}}
        self.cards={'10':{'characterId':1,'rarity':5,'type':'limited','releasedAt':[None,None,None,str(50*HOUR),None]},
                    '11':{'characterId':2,'rarity':5,'type':'permanent','releasedAt':[None,None,None,str(150*HOUR),None]}}
        self.features=EventFeatures(self.events,self.characters,self.bands,self.cards)

    def curve(self,event,start=0,rank=2000):
        ts=start+np.arange(201)*HOUR
        return Curve(event,rank,'live_try',start,start+200*HOUR,ts,np.arange(201)*100.)

    def test_single_band_and_mixed_participants_are_distinct(self):
        single=self.features.static(self.curve(1),100*HOUR)
        mixed=self.features.static(self.curve(2),100*HOUR)
        self.assertEqual(single['band_group'],'1')
        self.assertEqual(mixed['band_group'],'mixed')
        self.assertEqual(single['band_share_1'],1.)
        self.assertEqual(mixed['band_share_1'],.5)
        self.assertEqual(mixed['band_share_2'],.5)

    def test_unreleased_cn_cards_do_not_enter_a_past_snapshot(self):
        result=self.features.static(self.curve(1),100*HOUR)
        self.assertEqual(result['available_member_cards'],1)
        self.assertEqual(result['limited_member_cards'],1)
        self.assertEqual(result['featured_character'],'1')
        self.assertEqual(result['featured_character_2'],0.)

    def test_future_event_final_cannot_change_historical_heat(self):
        target=self.curve(2,1000*HOUR);past=self.curve(1);future=self.curve(1,2000*HOUR)
        sample=snapshot(target,target.start+100*HOUR)
        first=self.features.historical(sample,[past,future])
        altered=replace(future,scores=future.scores*10000)
        second=self.features.historical(sample,[past,altered])
        self.assertEqual(first,second)
        self.assertEqual(first['history_rank_events'],1)

    def test_target_event_other_rank_is_excluded_from_history(self):
        target=self.curve(2,1000*HOUR);same=self.curve(2,0,rank=1000)
        result=self.features.historical(snapshot(target,target.start+100*HOUR),[same])
        self.assertEqual(result['history_rank_events'],0)

    def test_peer_features_use_only_the_observed_other_rank_prefix(self):
        target=self.curve(2,1000*HOUR);other=self.curve(2,1000*HOUR,rank=1000)
        cutoff=target.start+100*HOUR
        values=other.scores.copy();values[101:]*=10000
        first=self.features.peer(snapshot(target,cutoff),[target,other])
        second=self.features.peer(snapshot(target,cutoff),[target,replace(other,scores=values)])
        self.assertEqual(first,second)
        self.assertEqual(first['peer_available'],1.)
        self.assertEqual(first['log_peer_current_ratio'],0.)


if __name__=='__main__':
    unittest.main()
