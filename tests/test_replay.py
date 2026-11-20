"""Historical replay publishes prefixes, never future observations or finals."""
import json
from pathlib import Path
import tempfile
import unittest

from production.data import Store
from production import web
from analysis.data_policy import MIN_EVENT_START_MS as S

H=3600000


class ReplayTests(unittest.TestCase):
    def test_source_releases_only_prefix_and_confirms_final_after_end(self):
        from production.replay import ReplaySource
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=Store(root/'source');destination=Store(root/'demo')
            source.metadata_directory.mkdir()
            for name in ('events','characters','bands','cards'):
                (source.metadata_directory/(name+'.json')).write_text('{}')
            points=[dict(time=S+i*H,ep=10+i*10) for i in range(5)]
            with source._connect() as con:
                con.execute('INSERT INTO events VALUES(?,?,?,?,?)',(328,S,S+4*H,'normal','Past event'))
                con.execute('INSERT INTO curves(event,rank,points_json,updated_ms) VALUES(?,?,?,?)',
                            (328,1000,json.dumps(points),S+5*H))
            replay=ReplaySource(source.root,328)
            replay.publish(destination,S+2*H)
            with destination._connect() as con:
                row=con.execute('SELECT * FROM curves').fetchone()
                self.assertEqual([p['time'] for p in json.loads(row['points_json'])],[S,S+H,S+2*H])
                self.assertFalse(row['verified_closed'])
            self.assertNotIn('terminal_score',web.observed_curve(destination.root,328,1000,now_ms=S+2*H))
            with self.assertRaises(ValueError):replay.publish(destination,S+H)
            replay.publish(destination,S+4*H+600001)
            self.assertEqual(web.observed_curve(destination.root,328,1000,now_ms=S+4*H+600001)['terminal_score']['ep'],50.)

    def test_http_clock_keeps_replay_in_progress_until_simulated_end(self):
        from urllib.request import urlopen
        import threading
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);store=Store(root)
            with store._connect() as con:
                con.execute('INSERT INTO events VALUES(?,?,?,?,?)',(328,S,S+4*H,'normal','Past event'))
            (root/'latest.json').write_text(json.dumps(dict(status='ok',as_of_ms=S+2*H,
                active_events=[328],predictions=[])))
            server=web.make_server(dict(root=str(root),ranks=[1000],interval_seconds=900),
                                   port=0,clock=lambda:S+2*H)
            thread=threading.Thread(target=server.serve_forever);thread.start()
            try:
                base='http://127.0.0.1:'+str(server.server_port)
                with urlopen(base+'/api/predictions') as response: data=json.load(response)
                self.assertEqual(data['server_time_ms'],S+2*H)
                self.assertEqual(data['events'][0]['status'],'active')
                with self.assertRaises(Exception):urlopen(base+'/api/curve?event=328&rank=1000&as_of='+str(S+3*H))
            finally:server.shutdown();server.server_close();thread.join()
