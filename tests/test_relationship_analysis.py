"""Scientific invariants for prefix features, clustered uncertainty and endpoint model."""
import unittest
import numpy as np
import pandas as pd
from analysis.backtest import Curve,snapshot
from analysis.event_features import EventFeatures
from analysis.relationship_features import relationship_frame
from analysis.relationship_statistics import clustered_ols
from analysis import relationship_statistics as relation_stats
from analysis.relationship_benchmark import decode,target_values,weights
from unittest.mock import patch
from analysis.relationship_benchmark import fit_recipe
from analysis import relationship_benchmark as relation_benchmark
from analysis.relationship_features import select_features


class RelationshipTests(unittest.TestCase):
    def test_changed_dependency_versions_are_rejected(self):
        self.assertTrue(hasattr(relation_benchmark,'verify_software'))
        with patch('analysis.relationship_benchmark.importlib.metadata.version',return_value='new'):
            with self.assertRaisesRegex(ValueError,'version'):
                relation_benchmark.verify_software({'numpy':'old'})
    def test_missing_peer_interactions_use_imputed_main_effect(self):
        self.assertTrue(hasattr(relation_stats,'peer_interactions'))
        frame=pd.DataFrame({'rank1000':[1.,0.,1.,0.],
                            'gap_log_1000_2000':[.2,.4,np.nan,np.nan],
                            'gap_change_24h':[.05,.1,np.nan,np.nan]})
        out=relation_stats.peer_interactions(frame)
        np.testing.assert_allclose(out.rank_gap,out.gap_log_1000_2000*out.rank1000)
        np.testing.assert_allclose(out.rank_gapchange,out.gap_change_24h*out.rank1000)
        self.assertAlmostEqual(out.rank_gap.iloc[2],.3)

    def test_invariant_dynamics_survive_score_unit_change(self):
        h=3600000;t=np.arange(51)*h
        c=Curve(1,1000,'live_try',0,100*h,t,np.r_[np.arange(45.)+1,np.full(6,45.)])
        meta=EventFeatures({'1':{'characters':[{'characterId':1}]}},{'1':{'bandId':1}},{'1':{'bandName':['A']*5}},{})
        before=select_features(relationship_frame([snapshot(c,50*h)],meta),'invariant')
        c.scores*=100
        after=select_features(relationship_frame([snapshot(c,50*h)],meta),'invariant')
        pd.testing.assert_frame_equal(before,after,atol=1e-10,rtol=1e-10)

    def test_rank_experts_slice_global_activity_weights(self):
        from types import SimpleNamespace
        samples=[{'curve':SimpleNamespace(event=e,rank=r),'case_id':str(i),'score':100.,'progress':.5}
                 for i,(e,r) in enumerate([(1,1000),(1,1000),(1,2000),(2,1000)])]
        frame=pd.DataFrame({'progress':[.5]*4,'rank':[str(s['curve'].rank) for s in samples]})
        recipe={'backend':'catboost','variant':'invariant_rank','target':'residual','loss':'RMSE'}
        models=[]
        class Learner:
            def fit(self,x,y,sample_weight):self.w=sample_weight;models.append(self)
        with patch('analysis.relationship_benchmark.CatBoostRegressor',side_effect=lambda **kw:Learner()):
            fit_recipe(recipe,frame,samples,{str(i):200. for i in range(4)})
        self.assertAlmostEqual(models[0].w[:2].sum()+models[1].w.sum(),models[0].w[2])

    def test_feature_values_ignore_target_and_peer_future(self):
        h=3600000;t=np.arange(101)*h
        c=Curve(1,1000,'live_try',0,100*h,t,np.arange(1,102.)*100)
        peer=Curve(1,2000,'live_try',0,100*h,t,np.arange(1,102.)*75)
        s=snapshot(c,50*h);s['peer_curve']=peer
        meta=EventFeatures({'1':{'characters':[{'characterId':1}]}},{'1':{'bandId':1}},{'1':{'bandName':['A']*5}},{})
        before=relationship_frame([s],meta)
        c.scores[51:]*=1000;peer.scores[51:]*=1000
        pd.testing.assert_frame_equal(before,relationship_frame([s],meta))

    def test_remaining_rate_prediction_converges_to_current_at_end(self):
        s=[{'progress':.5,'score':100.},{'progress':.999999,'score':100.}]
        p=decode(s,np.array([0.,0.]),'remaining')
        self.assertAlmostEqual(p[0],200.)
        self.assertLess(abs(p[1]-100),.001)
        train=[{'progress':.5,'score':100.,'case_id':'a'}]
        target=target_values(train,{'a':200.},'remaining')
        self.assertAlmostEqual(decode(train,target,'remaining')[0],200.)

    def test_duplicate_snapshots_do_not_create_independent_evidence(self):
        rng=np.random.default_rng(42);g=np.repeat(np.arange(40),4);x=rng.normal(size=len(g))
        y=2*x+rng.normal(size=40)[g]+rng.normal(size=len(g))*.1
        frame=pd.DataFrame({'intercept':1.,'x':x})
        one,_,_=clustered_ols(frame,y,g,np.ones(len(g)))
        two,_,_=clustered_ols(pd.concat([frame]*10,ignore_index=True),np.tile(y,10),np.tile(g,10),np.ones(len(g)*10)/10)
        np.testing.assert_allclose(one.beta,two.beta,atol=1e-10)
        self.assertGreater(two.cluster_se.iloc[1]/one.cluster_se.iloc[1],.95)
        self.assertLess(two.cluster_se.iloc[1]/one.cluster_se.iloc[1],1.05)

    def test_training_activity_weights_remain_equal(self):
        from types import SimpleNamespace
        samples=[{'curve':SimpleNamespace(event=e)} for e in [1,1,1,2]]
        w=weights(samples);self.assertAlmostEqual(w[:3].sum(),w[3])


if __name__=='__main__':unittest.main()
