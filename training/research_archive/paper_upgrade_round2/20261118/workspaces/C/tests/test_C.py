"""C fixture tests. Every test has a causal/fairness failure it should catch."""
from pathlib import Path
import sys
import unittest
import tempfile
import json
import gzip
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import c_fairness as c

MIN = 1706504400000

def frame(event, start, end, error=None, ranks=(1000,2000)):
    rows=[]
    for rank in ranks:
        for j, progress in enumerate((.1,.5,.95)):
            row=dict(case_id=f'train:{event}:{rank}:{j}',event=event,rank=rank,phase=j,
                     progress=progress,start_ms=start,end_ms=end,cutoff=start+int((end-start)*progress),
                     observed_at=start+int((end-start)*progress),current=100.,event_type='normal',
                     band_group='a',split='train',duration_h=100.,remaining_h=50.,
                     rate_6h_relative=np.nan,rate_24h_relative=np.nan,base_prediction=200.)
            if error is not None:row['log_error']=error[rank]
            rows.append(row)
    return pd.DataFrame(rows)

class Contract(unittest.TestCase):
    def function(self, name):
        value=getattr(c,name,None)
        if not callable(value):self.fail('C implementation missing: '+name)
        return value

    def test_strict_release_equality_cannot_influence_prediction(self):
        predict=self.function('c1_predict')
        past=frame(1,MIN,MIN+1000,{1000:.2,2000:-.1})
        current=frame(2,MIN+1000,MIN+2000)
        with self.assertRaisesRegex(ValueError,'earlier completed'):
            predict(past,current,query_start_ms=MIN+1000)

    def test_current_outcome_fields_fail_closed(self):
        predict=self.function('c1_predict')
        past=frame(1,MIN,MIN+1000,{1000:.2,2000:-.1})
        current=frame(2,MIN+2000,MIN+3000)
        for name in ['actual','log_error','expert__independent']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                predict(past,current.assign(**{name:1.}),query_start_ms=MIN+2000)

    def test_single_line_replacement_retains_opposite_independent_exactly(self):
        predict=self.function('c1_predict')
        past=frame(1,MIN,MIN+1000,{1000:.2,2000:-.1})
        current=frame(2,MIN+2000,MIN+3000)
        values,_=predict(past,current,query_start_ms=MIN+2000)
        for method,keep in [('legacy_only_T1000',2000),('legacy_only_T2000',1000),('swapped_only_T2000',1000)]:
            mask=current['rank'].to_numpy()==keep
            np.testing.assert_array_equal(values[method][mask],values['independent'][mask])
        expected=200.*np.exp(-.2/11.)
        self.assertAlmostEqual(values['independent'][0],expected)

    def test_same_filter_identity_on_shared_support_with_missing_anchor(self):
        predict=self.function('c1_predict')
        h1=frame(1,MIN,MIN+1000,{1000:.2,2000:-.1})
        h2=frame(2,MIN+2000,MIN+3000,{1000:.4},ranks=(1000,))
        current=frame(3,MIN+4000,MIN+5000)
        values,_=predict(pd.concat([h1,h2]),current,query_start_ms=MIN+4000)
        np.testing.assert_allclose(values['same_filter_pair'],values['same_panel_independent'],rtol=1e-14)
        self.assertEqual(len(values),18)
        self.assertTrue(all(np.isfinite(x).all() and len(x)==6 for x in values.values()))

    def test_history_group_quantity_span_and_strict_freeze(self):
        groups=self.function('training_groups')
        clock=pd.DataFrame([dict(event=i+1,start_ms=MIN+i*100,end_ms=MIN+i*100+80) for i in range(60)])
        result=groups(clock,endpoints=[32,44,56],query_count=4)
        first=result[0]
        self.assertEqual(first['query_events'],[33,34,35,36])
        self.assertEqual(first['groups']['N32full'],list(range(1,33)))
        self.assertEqual(first['groups']['N16span'],[1,3,5,7,9,11,13,15,18,20,22,24,26,28,30,32])
        self.assertEqual(first['groups']['N16recent'],list(range(17,33)))
        changed=clock.copy();changed.loc[31,'end_ms']=MIN+3200
        with self.assertRaisesRegex(ValueError,'completed'):
            groups(changed,endpoints=[32],query_count=4)

    def test_weights_equalize_events_then_ranks_even_with_prefix_imbalance(self):
        weights=self.function('event_rank_weights')
        events=np.array([1,1,1,2,2]);ranks=np.array([1000,1000,2000,1000,2000])
        w=weights(events,ranks);w=w/w.sum()
        np.testing.assert_allclose(w,[.125,.125,.25,.25,.25])

    def test_exact_parameter_count_and_rank_routing_without_training(self):
        network=self.function('network')
        import torch
        for d in [1,7,59]:
            shared=network(d,'shared');ind=network(d,'independent')
            self.assertEqual(sum(p.numel() for p in shared.parameters()),32*d+66)
            self.assertEqual(sum(p.numel() for p in ind.parameters()),32*d+66)
            x=torch.zeros((4,d));rank=torch.tensor([0,1,0,1])
            self.assertEqual(tuple(shared(x,rank).shape),(4,))
            self.assertEqual(tuple(ind(x,rank).shape),(4,))
            with self.assertRaises(ValueError):shared(x,rank+2)

    def test_actual_compute_manifest_does_not_mislabel_optimizer_matching(self):
        budget=self.function('compute_budget')
        shared=budget(7,10,30,'shared');ind=budget(7,10,60,'independent',rank_rows=[4,6])
        self.assertEqual(shared['forward_linear_MACs'],86400)
        self.assertEqual(ind['forward_linear_MACs'],76800)
        self.assertEqual(shared['optimizer_steps'],30)
        self.assertEqual(ind['optimizer_steps'],120)
        self.assertEqual(shared['parameter_updates'],8700)
        self.assertEqual(ind['parameter_updates'],17400)

    def test_prepare_only_does_not_load_answers_or_fit_and_never_overwrites(self):
        prepare=self.function('prepare_c2')
        with tempfile.TemporaryDirectory() as tmp:
            dataset=Path(tmp)/'data';dataset.mkdir()
            pd.DataFrame([dict(case_id=f'train:{i+1}:1000:1',event=i+1,rank=1000,start_ms=MIN+i*100,end_ms=MIN+i*100+80) for i in range(67)]).to_csv(dataset/'train_cases.csv',index=False)
            # Invalid label bytes would fail any attempt to open/parse answers.
            (dataset/'train_labels.csv').write_bytes(b'not-a-csv')
            run=Path(tmp)/'run';prepare(run,dataset=dataset,verify_frozen=False)
            self.assertTrue((run/'configuration_seal.json').exists())
            self.assertFalse((run/'models').exists())
            with self.assertRaises(FileExistsError):prepare(run,dataset=dataset,verify_frozen=False)

    def test_only_requested_past_train_label_values_are_materialized(self):
        load=self.function('allowed_train_labels')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'train_labels.csv'
            path.write_text('case_id,actual\npast,300\nfuture,not-numeric-answer\n')
            self.assertEqual(load(path,{'past'}),{'past':300.})
            with self.assertRaises(ValueError):load(path,{'absent'})

    def test_preregistration_input_digests_match_actual_bytes(self):
        hashes=self.function('input_hashes')
        registration=json.loads((HERE/'preregistration.json').read_text())
        actual=hashes()
        self.assertTrue(all(len(x)==64 for x in actual.values()))
        self.assertEqual(registration['C1']['source_sha256'],actual)

    def test_future_answer_perturbation_leaves_earlier_release_and_forecast_identical(self):
        release=self.function('release_query');predict=self.function('c1_predict')
        records=pd.concat([frame(1,MIN,MIN+1000,{1000:.2,2000:-.1}),frame(2,MIN+2000,MIN+3000,{1000:50.,2000:-50.})])
        records['actual']=999.;records['expert__independent']=999.
        historical,current=release(records,event=2,start_ms=MIN+2000)
        before,_=predict(historical,current,query_start_ms=MIN+2000)
        changed=records.copy();changed.loc[changed.event==2,['actual','log_error','expert__independent']]=-1e8
        h2,t2=release(changed,event=2,start_ms=MIN+2000)
        after,_=predict(h2,t2,query_start_ms=MIN+2000)
        self.assertFalse({'actual','log_error','expert__independent'}&set(t2))
        self.assertFalse({'actual','expert__independent'}&set(h2))
        for name in before:np.testing.assert_array_equal(before[name],after[name])

    def test_missing_case_cannot_be_silently_dropped_from_score(self):
        metrics=self.function('metric_summary')
        cases=pd.DataFrame([dict(case_id='a',event=1,rank=1000,predicted=110.,actual=100.),dict(case_id='b',event=2,rank=2000,predicted=50.,actual=100.)])
        result=metrics(cases)
        self.assertAlmostEqual(result['event_macro_mape_pct'],30.)
        self.assertAlmostEqual(result['event_rank_macro_mape_pct'],30.)
        self.assertAlmostEqual(result['mae'],30.)
        panel=self.function('check_panel')
        with self.assertRaises(ValueError):panel(cases.iloc[:1],{'a','b'})

    def test_training_prefix_loader_keeps_query_outcomes_out_and_rejects_future_points(self):
        loader=self.function('load_train_prefixes')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);rec=dict(event=1,rank=1000,kind='normal',start=MIN,end=MIN+100000,
                times=[MIN+1000,MIN+2000,MIN+3000,MIN+4000],scores=[10,20,30,40],
                case_id='train:1:1000:4000',stage='test',cutoff=MIN+4000,peer=None)
            pd.DataFrame([dict(case_id=rec['case_id'],event=1,rank=1000,start_ms=MIN,end_ms=MIN+100000,
                               current=40,observed_at=MIN+4000,cutoff=MIN+4000)]).to_csv(path/'train_cases.csv',index=False)
            with gzip.open(path/'train_inputs.jsonl.gz','wt') as f:f.write(json.dumps(rec)+'\n')
            cases,x=loader(path)
            self.assertEqual(len(cases),1);self.assertEqual(len(x),1)
            self.assertEqual(cases.progress.iloc[0],.04)
            self.assertFalse({'actual','target','log_error'}&set(cases))
            rec['times'][-1]=MIN+5000
            with gzip.open(path/'train_inputs.jsonl.gz','wt') as f:f.write(json.dumps(rec)+'\n')
            with self.assertRaises(ValueError):loader(path)

    def test_empty_history_fallback_is_bounded_base_for_all_methods(self):
        predict=self.function('c1_predict')
        empty=frame(1,MIN,MIN+1000,{1000:.2,2000:-.1}).iloc[:0]
        target=frame(2,MIN+2000,MIN+3000)
        values,_=predict(empty,target,query_start_ms=MIN+2000)
        for name,value in values.items():
            with self.subTest(name=name):np.testing.assert_array_equal(value,[200.]*6)

    def test_score_retains_raw_residuals_and_separate_answer_seal(self):
        scorer=self.function('score');seal=self.function('seal')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);run=path/'run';run.mkdir();inputs=path/'inputs';inputs.mkdir()
            frame1=frame(1,MIN,MIN+1000)
            frame1['predicted']=180.;frame1['method']='independent';frame1['period']='temporal_diagnostic'
            frame1.to_csv(run/'predictions.csv.gz',index=False)
            labels=frame1[['case_id']].assign(actual=100.);labels.to_csv(inputs/'train_residuals.csv.gz',index=False)
            (run/'configuration.json').write_text(json.dumps({'experiment':'C1','inputs_directory':str(inputs),
                'data_hashes':{str(inputs/'train_residuals.csv.gz'):c.sha(inputs/'train_residuals.csv.gz')}}))
            seal(run,'blind_prediction_seal.json');scorer(run,path/'scored','C1')
            result=pd.read_csv(path/'scored'/'event_rank_phase_residuals.csv')
            self.assertTrue({'raw_mean','corrected_mean'}<=set(result))
            np.testing.assert_allclose(result.raw_mean,np.log(2.))
            np.testing.assert_allclose(result.corrected_mean,np.log(1.8))
            self.assertTrue((path/'scored'/'evaluation_seal.json').exists())

    def test_c1_context_fields_retain_text_codes_in_mixed_csv(self):
        loader=self.function('load_c1_records')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            pd.DataFrame([{'event_type':'normal','band_group':'005'},{'event_type':'mission_live','band_group':'21+22'}]).to_csv(path/'train_residuals.csv.gz',index=False)
            data=loader(path)
            self.assertEqual(data.band_group.tolist(),['005','21+22'])
            self.assertTrue(all(isinstance(x,str) for x in data.event_type))

    def test_query_seal_records_clock_case_sets_and_prediction_sha_before_reuse(self):
        sealing=self.function('seal_query')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);(path/'predictions').mkdir()
            h=frame(1,MIN,MIN+1000,{1000:.2,2000:-.1});q=frame(2,MIN+2000,MIN+3000)
            output=path/'predictions'/'q.csv';q.assign(predicted=200.).to_csv(output,index=False)
            result=sealing(path,output,h,q,position=2)
            self.assertEqual(result['query_start_ms'],MIN+2000)
            self.assertEqual(result['max_history_end_ms'],MIN+1000)
            self.assertEqual(result['prediction_sha256'],c.sha(output))
            self.assertEqual(result['target_cases'],6)
            self.assertEqual(len(result['history_case_ids_sha256']),64)
            with self.assertRaises(FileExistsError):sealing(path,output,h,q,position=2)

    def test_score_rejects_changed_answer_source_before_parsing_answers(self):
        scorer=self.function('score')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);run=path/'run';run.mkdir();inputs=path/'inputs';inputs.mkdir()
            answers=inputs/'train_residuals.csv.gz';answers.write_bytes(b'bad-answer-bytes')
            (run/'configuration.json').write_text(json.dumps({'experiment':'C1','inputs_directory':str(inputs),
                'data_hashes':{str(answers):'0'*64}}))
            frame(1,MIN,MIN+1000).assign(predicted=180.,method='independent',period='warmup').to_csv(run/'predictions.csv.gz',index=False)
            c.seal(run,'blind_prediction_seal.json')
            try:
                scorer(run,path/'scored','C1')
            except Exception as error:
                self.assertIsInstance(error,ValueError)
                self.assertIn('Frozen data changed',str(error))
            else:self.fail('Changed answer source accepted')
            self.assertFalse((path/'scored').exists())

if __name__=='__main__':unittest.main()
