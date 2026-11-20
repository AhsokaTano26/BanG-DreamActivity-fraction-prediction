"""Read-only web routes and explicitly owned forecast process lifecycle."""
import json
from pathlib import Path
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from production.state import Ledger
from analysis.data_policy import MIN_EVENT_START_MS


class ForecastWebTests(unittest.TestCase):
    def setUp(self):
        from production import web
        self.web = web
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = dict(root=str(self.root), ranks=[1000, 2000], interval_seconds=900)
        self.now = int(time.time()*1000)
        self.row = dict(event=329, rank=1000, start_ms=self.now-3600000,
            end_ms=self.now+3600000, observed_at=self.now-1000, current_ep=50.,
            predicted_final_ep=105., available=True, progress=.5)
        self.payload = dict(status='ok', as_of_ms=self.now, active_events=[329],
            predictions=[self.row], model_path='/private/weights', errors=[])
        (self.root/'latest.json').write_text(json.dumps(self.payload))

    def server(self):
        server = self.web.make_server(self.config, host='127.0.0.1', port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.shutdown)
        return 'http://127.0.0.1:' + str(server.server_port)

    def test_api_reports_live_result_without_exposing_model_paths(self):
        base = self.server()
        with urlopen(base+'/api/predictions') as response:
            data = json.load(response)
            self.assertEqual(response.headers['Cache-Control'], 'no-store')
        self.assertEqual(data['predictions'][0]['predicted_final_ep'], 105.)
        self.assertFalse(data['stale'])
        self.assertNotIn('/private/weights', json.dumps(data))

    def test_expired_activity_and_old_observation_are_marked_unavailable(self):
        self.payload['as_of_ms'] -= 4*3600000
        self.payload['predictions'][0]['end_ms'] = self.now-1
        (self.root/'latest.json').write_text(json.dumps(self.payload))
        data = self.web.public_snapshot(self.config, now_ms=self.now)
        self.assertTrue(data['stale'])
        self.assertFalse(data['predictions'][0]['available'])
        self.assertIn('结束', data['predictions'][0]['reason'])

    def test_unlisted_files_and_write_requests_are_rejected(self):
        base = self.server()
        for path in ['/state.sqlite3', '/api/../../config.json', '/runtime/live/config.json']:
            with self.assertRaises(HTTPError) as error:
                urlopen(base+path)
            self.assertEqual(error.exception.code, 404)
            error.exception.close()
        with self.assertRaises(HTTPError) as error:
            urlopen(Request(base+'/api/predictions', method='POST', data=b'{}'))
        self.assertEqual(error.exception.code, 405)
        error.exception.close()

    def test_history_deduplicates_revisions_and_filters_event_and_rank(self):
        ledger = Ledger(self.root)
        event = dict(event=329, start_ms=self.now-3600000, end_ms=self.now+3600000)
        row = self.row | dict(raw_predicted_final_ep=110., phase=1)
        ledger.record(event, [row], cutoff_ms=self.now, issued_ms=self.now, model_id='a')
        ledger.record(event, [row | dict(predicted_final_ep=106.)],
            cutoff_ms=self.now, issued_ms=self.now+1, model_id='a')
        with urlopen(self.server()+'/api/history?event=329&rank=1000') as response:
            history = json.load(response)['points']
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['predicted_final_ep'], 106.)
        self.assertEqual(self.web.forecast_history(self.root, 329, 2000), [])

    def test_invalid_history_query_is_400(self):
        base = self.server()
        for query in ['event=-1&rank=1000', 'event=329&rank=999', 'rank=1000']:
            with self.assertRaises(HTTPError) as error:
                urlopen(base+'/api/history?'+query)
            self.assertEqual(error.exception.code, 400)
            error.exception.close()

    def test_history_checkpoints_preserve_early_issued_prediction_before_late_revision(self):
        self.history_store();ledger=Ledger(self.root)
        start=self.now-5*3600000;end=self.now-3*3600000
        event=dict(event=328,start_ms=start,end_ms=end);observed=start+1800000
        row=dict(rank=1000,phase=1,observed_at=observed,current_ep=25.,
                 raw_predicted_final_ep=110.,predicted_final_ep=95.,available=True)
        ledger.record(event,[row],cutoff_ms=observed,issued_ms=observed,model_id='past')
        ledger.record(event,[row|dict(predicted_final_ep=106.)],cutoff_ms=observed,
                      issued_ms=start+5400000,model_id='past')
        data=self.web.history_payload(self.root,328,1000,now_ms=self.now)
        checkpoints={row['key']:row for row in data['checkpoints']}
        self.assertEqual(data['points'][0]['predicted_final_ep'],106.)
        self.assertEqual(checkpoints['progress_50']['predicted_final_ep'],95.)
        self.assertEqual(checkpoints['progress_75']['predicted_final_ep'],106.)
        early=self.web.history_payload(self.root,328,1000,now_ms=start+3600000)
        self.assertEqual(early['points'][0]['predicted_final_ep'],95.)
        with urlopen(self.server()+'/api/history?event=328&rank=1000') as response:
            self.assertEqual(len(json.load(response)['checkpoints']),8)

    def test_service_without_results_has_meaningful_waiting_response(self):
        (self.root/'latest.json').unlink()
        data = self.web.public_snapshot(self.config, now_ms=self.now)
        self.assertEqual(data['status'], 'waiting_for_data')
        self.assertEqual(data['predictions'], [])
        self.assertEqual(data['ranks'], [1000, 2000])

    def test_event_card_uses_cn_name_dates_type_and_real_banner_asset(self):
        metadata = self.root/'metadata'
        metadata.mkdir()
        (metadata/'events.json').write_text(json.dumps({'329':{
            'eventName':['日文名',None,None,'Beyond my Imagination'],
            'eventType':'challenge','bannerAssetBundleName':'banner_event329',
            'startAt':[1000,None,None,str(self.row['start_ms'])],
            'endAt':[2000,None,None,str(self.row['end_ms'])]}}))
        data = self.web.public_snapshot(self.config, now_ms=self.now)
        event = data['events'][0]
        self.assertEqual(event['name'], 'Beyond my Imagination')
        self.assertEqual(event['type_label'], '挑战LIVE')
        self.assertEqual(event['start_ms'], self.row['start_ms'])
        self.assertEqual(event['jp_start_ms'], 1000)
        self.assertEqual(event['banner_url'], 'https://bestdori.com/assets/cn/homebanner_rip/banner_event329.png')

    def test_actual_curve_comes_from_early_observations_and_excludes_future(self):
        start = MIN_EVENT_START_MS
        database = self.root/'data.sqlite3'
        with sqlite3.connect(database) as con:
            con.executescript('CREATE TABLE events(event INTEGER,start_ms INTEGER,end_ms INTEGER); CREATE TABLE curves(event INTEGER,rank INTEGER,points_json TEXT);')
            con.execute('INSERT INTO events VALUES(?,?,?)', (329,start,start+7200000))
            con.execute('INSERT INTO curves VALUES(?,?,?)', (329,1000,json.dumps([
                {'time':start-1,'ep':1},{'time':start+1000,'ep':10},
                {'time':start+3600000,'ep':30},{'time':start+3600001,'ep':40}])))
        curve = self.web.observed_curve(self.root,329,1000,now_ms=start+3600000)
        self.assertEqual(curve['points'], [{'time':start+1000,'ep':10.}, {'time':start+3600000,'ep':30.}])
        self.assertEqual(curve['start_ms'], start)
        self.assertEqual(self.web.observed_curve(self.root,329,2000,now_ms=start+3600000)['points'], [])
        with urlopen(self.server()+'/api/curve?event=329&rank=1000') as response:
            self.assertEqual(len(json.load(response)['points']),3)

    def test_stop_reaps_owned_child_and_does_not_start_another_cycle(self):
        worker = self.web.CycleWorker(self.config, self.root/'config.json', interval=60)
        with patch.object(worker, 'cycle_command', return_value=[sys.executable, '-c', 'import time; time.sleep(60)']):
            worker.start()
            self.addCleanup(worker.stop)
            deadline = time.monotonic()+4
            while not worker.snapshot().get('pid') and time.monotonic() < deadline:
                time.sleep(.02)
            pid = worker.snapshot().get('pid')
            self.assertIsNotNone(pid)
            worker.stop()
        self.assertFalse(worker.thread.is_alive())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
        self.assertEqual(worker.snapshot()['state'], 'stopped')

    def history_store(self):
        with sqlite3.connect(self.root/'data.sqlite3') as con:
            con.executescript('''CREATE TABLE events(event INTEGER PRIMARY KEY,start_ms INTEGER,end_ms INTEGER,kind TEXT,name TEXT);
                CREATE TABLE curves(event INTEGER,rank INTEGER,points_json TEXT,verified_closed INTEGER,updated_ms INTEGER);''')
            for event,start,end in [(328,self.now-5*3600000,self.now-3*3600000),
                                     (329,self.row['start_ms'],self.row['end_ms']),
                                     (9999,MIN_EVENT_START_MS-7200000,MIN_EVENT_START_MS-3600000),
                                     (1,self.now+3600000,self.now+7200000)]:
                con.execute('INSERT INTO events VALUES(?,?,?,?,?)',(event,start,end,'challenge','activity '+str(event)))
                times=[start+1,start+(end-start)//4,start+(end-start)//2,end-60000,end+1000]
                con.execute('INSERT INTO curves VALUES(?,?,?,?,?)',(event,1000,json.dumps([
                    {'time':t,'ep':v} for t,v in zip(times,[1,25,50,90,100])]),1,self.now))

    def test_catalog_contains_history_but_excludes_predense_and_future_by_time(self):
        self.history_store()
        data=self.web.public_snapshot(self.config,now_ms=self.now)
        self.assertEqual([e['event'] for e in data['events']],[329,328])
        self.assertEqual(data['selected_event'],329)
        self.assertEqual(data['events'][1]['status'],'finished')
        self.assertEqual(data['predictions'][0]['predicted_final_ep'],105.)

    def test_history_uses_verified_final_and_marks_absent_live_predictions(self):
        self.history_store()
        before=(self.root/'latest.json').read_bytes()
        with urlopen(self.server()+'/api/predictions?event=328') as response:
            data=json.load(response)
        self.assertEqual(data['scope'],'historical')
        self.assertEqual(data['selected_event'],328)
        row=next(r for r in data['predictions'] if r['rank']==1000)
        self.assertTrue(row['observation_available'])
        self.assertEqual(row['actual_final_ep'],100.)
        self.assertIsNone(row['predicted_final_ep'])
        self.assertIsNone(row['relative_error_pct'])
        self.assertFalse(row['available'])
        self.assertIn('没有上线预测',row['reason'])
        self.assertFalse(data['stale'])
        self.assertEqual((self.root/'latest.json').read_bytes(),before)

    def test_history_reads_last_issued_revision_for_the_selected_event(self):
        self.history_store();ledger=Ledger(self.root)
        event=dict(event=328,start_ms=self.now-5*3600000,end_ms=self.now-3*3600000)
        observed=self.now-4*3600000
        row=dict(rank=1000,phase=1,observed_at=observed,current_ep=50.,raw_predicted_final_ep=110.,predicted_final_ep=105.,available=True)
        ledger.record(event,[row],cutoff_ms=observed,issued_ms=observed,model_id='past')
        ledger.record(event,[row|dict(predicted_final_ep=106.)],cutoff_ms=observed,issued_ms=observed+1,model_id='past')
        data=self.web.public_snapshot(self.config,now_ms=self.now,event_id=328)
        self.assertEqual(data['predictions'][0]['predicted_final_ep'],106.)
        self.assertEqual(ledger.summary()['forecasts'],1)
        self.assertEqual(ledger.summary()['forecast_revisions'],1)

    def test_verified_final_compares_last_issued_revision_without_changing_forecasts(self):
        self.history_store();ledger=Ledger(self.root)
        event=dict(event=328,start_ms=self.now-5*3600000,end_ms=self.now-3*3600000)
        observed=self.now-4*3600000
        row=dict(rank=1000,phase=1,observed_at=observed,current_ep=50.,
                 raw_predicted_final_ep=110.,predicted_final_ep=95.,available=True)
        ledger.record(event,[row],cutoff_ms=observed,issued_ms=observed,model_id='past')
        low=self.web.public_snapshot(self.config,now_ms=self.now,event_id=328)['predictions'][0]
        self.assertEqual(low['relative_error_pct'],-5.)
        self.assertEqual(low['absolute_error_pct'],5.)
        ledger.record(event,[row|dict(predicted_final_ep=106.)],
                      cutoff_ms=observed,issued_ms=observed+1,model_id='past')
        before=ledger.summary()
        data=self.web.public_snapshot(self.config,now_ms=self.now,event_id=328)
        checked=data['predictions'][0]
        self.assertEqual(checked['actual_final_ep'],100.)
        self.assertEqual(checked['relative_error_pct'],6.)
        self.assertEqual(checked['absolute_error_pct'],6.)
        self.assertIsNone(data['predictions'][1]['relative_error_pct'])
        self.assertEqual(ledger.summary(),before)
        with sqlite3.connect(self.root/'data.sqlite3') as con:
            values=json.loads(con.execute('SELECT points_json FROM curves WHERE event=328 AND rank=1000').fetchone()[0])
            values[-1]['ep']=200
            con.execute('UPDATE curves SET points_json=? WHERE event=328 AND rank=1000',(json.dumps(values),))
        corrected=self.web.public_snapshot(self.config,now_ms=self.now,event_id=328)['predictions'][0]
        self.assertEqual(corrected['relative_error_pct'],-47.)
        self.assertEqual(ledger.summary(),before)

    def test_forecast_comparison_waits_for_quality_verified_positive_final(self):
        self.history_store();ledger=Ledger(self.root)
        event=dict(event=328,start_ms=self.now-5*3600000,end_ms=self.now-3*3600000)
        observed=self.now-4*3600000
        ledger.record(event,[dict(rank=1000,phase=1,observed_at=observed,current_ep=50.,
            raw_predicted_final_ep=110.,predicted_final_ep=105.,available=True)],
            cutoff_ms=observed,issued_ms=observed,model_id='past')
        with sqlite3.connect(self.root/'data.sqlite3') as con:
            con.execute('UPDATE curves SET verified_closed=0 WHERE event=328')
        row=self.web.public_snapshot(self.config,now_ms=self.now,event_id=328)['predictions'][0]
        self.assertIsNone(row['relative_error_pct'])
        with sqlite3.connect(self.root/'data.sqlite3') as con:
            values=[{'time':event['start_ms']+i*3600000//4,'ep':0} for i in range(4)]
            values.append({'time':event['end_ms'],'ep':0})
            con.execute('UPDATE curves SET verified_closed=1,points_json=? WHERE event=328',(json.dumps(values),))
        row=self.web.public_snapshot(self.config,now_ms=self.now,event_id=328)['predictions'][0]
        self.assertIsNone(row['relative_error_pct'])

    def test_final_metadata_retains_real_reporting_clock_without_future_release(self):
        self.history_store();end=self.now-3*3600000
        early=self.web.observed_curve(self.root,328,1000,now_ms=end-1)
        self.assertNotIn('terminal_score',early)
        finished=self.web.observed_curve(self.root,328,1000,now_ms=end+1001)
        self.assertEqual(finished['terminal_score'],dict(ep=100.,reported_at=end+1000,event_end_ms=end))
        self.assertTrue(all(p['time']<=end for p in finished['points']))

    def test_invalid_or_disallowed_history_event_returns_bad_request(self):
        self.history_store();base=self.server()
        for event in [0,-1,1,9999,404040]:
            with self.assertRaises(HTTPError) as error:
                urlopen(base+'/api/predictions?event='+str(event))
            self.assertEqual(error.exception.code,400);error.exception.close()


if __name__ == '__main__':
    unittest.main()
