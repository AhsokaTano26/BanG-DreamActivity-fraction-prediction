import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest


class RankDownloadTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('download_ranks'))
        import download_ranks as d
        self.d=d

    def test_top10_tracks_the_boundary_not_the_final_tenth_player(self):
        points=[{'time':1000,'uid':i+1,'value':100-i} for i in range(10)]
        points += [{'time':2000,'uid':i+1,'value':200-i} for i in range(1,11)]
        points += [{'time':3000,'uid':i+1,'value':300-i} for i in range(9)]
        curve,quality=self.d.normalize_top({'points':points,'users':[]})
        self.assertEqual(curve,[{'time':1000,'ep':91},{'time':2000,'ep':190}])
        self.assertEqual(quality['incomplete_top_snapshots'],1)

    def test_duplicate_user_does_not_make_an_incomplete_top10_complete(self):
        points=[{'time':1000,'uid':i+1,'value':100-i} for i in range(9)]
        points.append(points[0].copy())
        curve,quality=self.d.normalize_top({'points':points,'users':[]})
        self.assertEqual(curve,[]);self.assertEqual(quality['incomplete_top_snapshots'],1)

    def test_cutoffs_sort_deduplicate_and_flag_decreases_without_rewriting(self):
        curve,q=self.d.normalize_cutoffs({'result':True,'cutoffs':[{'time':2000,'ep':20},
            {'time':1000,'ep':30},{'time':2000,'ep':20}]})
        self.assertEqual(curve,[{'time':1000,'ep':30},{'time':2000,'ep':20}])
        self.assertEqual(q['duplicate_points'],1);self.assertEqual(q['score_decreases'],1)
        for payload in [{'result':False,'cutoffs':[]},{'result':True},
            {'result':True,'cutoffs':[{'time':1,'ep':-1}]},
            {'result':True,'cutoffs':[{'time':1,'ep':2},{'time':1,'ep':3}]}]:
            with self.assertRaises(ValueError):self.d.normalize_cutoffs(payload)

    def test_metadata_uses_real_cn_start_and_all_nineteen_ranks(self):
        def event(start):return {'startAt':[None,None,None,start,None],
                                'endAt':[None,None,None,start+1000 if start is not None else None,None]}
        ids=self.d.started_events({'5':event(2000),'100':event(1000),'1':event(5000),'2':event(None)},3000)
        self.assertEqual(ids,[100,5])
        self.assertEqual(self.d.RANKS,[10,20,30,40,50,100,200,300,400,500,1000,1500,2000,3000,4000,5000,10000,20000,30000])

    def test_merge_preserves_old_only_points_and_accepts_source_corrections(self):
        old=[{'time':1,'ep':10},{'time':2,'ep':20}]
        merged,q=self.d.merge_points(old,[{'time':2,'ep':19},{'time':3,'ep':30}])
        self.assertEqual(merged,[{'time':1,'ep':10},{'time':2,'ep':19},{'time':3,'ep':30}])
        self.assertEqual(q,{'added_points':1,'corrected_points':1,'retained_old_points':1})

    def test_offline_download_resume_and_backup_import(self):
        import asyncio,httpx
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);db=root/'db.sqlite3';run=root/'run'
            with sqlite3.connect(db) as con:
                con.execute('CREATE TABLE event(EventID INTEGER,EventBand TEXT,EventName TEXT,EventType TEXT,StartAt INTEGER,EndAt INTEGER,PointRank TEXT,ID TEXT PRIMARY KEY,Rank INTEGER,Country TEXT)')
                con.execute('INSERT INTO event VALUES(1,NULL,"old","live_try",1,3000,?,"old-id",20,"中国大陆")',
                            (json.dumps([{'time':1000,'ep':8}]),))
            calls=[]
            metadata={'1':{'startAt':[None,None,None,'1',None],'endAt':[None,None,None,'3000',None],
                           'eventName':[None,None,None,'活动',None],'eventType':'live_try'}}
            attempts={}
            def handler(request):
                calls.append(str(request.url))
                if 'all.5.json' in request.url.path:return httpx.Response(200,json=metadata)
                rank=int(request.url.params.get('tier','10'));attempts[rank]=attempts.get(rank,0)+1
                if rank==20 and attempts[rank]==1:return httpx.Response(503,text='temporary')
                if rank==10:return httpx.Response(200,json={'points':[{'time':2000,'uid':i+1,'value':100-i} for i in range(10)],'users':[]})
                return httpx.Response(200,json={'result':True,'cutoffs':[] if rank==30 else [{'time':2000,'ep':20}]})
            asyncio.run(self.d.download(run,db,ranks=[10,20,30],rate=10000,workers=2,transport=httpx.MockTransport(handler)))
            count=len(calls)
            asyncio.run(self.d.download(run,db,ranks=[10,20,30],rate=10000,workers=2,resume=True,transport=httpx.MockTransport(handler)))
            self.assertEqual(len(calls),count)
            # A WAL-only edit must prevent a stale import even when main-file
            # bytes do not change; restoring logical content makes it eligible.
            external=sqlite3.connect(db)
            try:
                external.execute('PRAGMA journal_mode=WAL');external.execute('PRAGMA wal_autocheckpoint=0')
                external.execute('PRAGMA wal_checkpoint(TRUNCATE)');physical=self.d.sha256(db)
                external.execute('UPDATE event SET EventName="changed"');external.commit()
                self.assertEqual(physical,self.d.sha256(db))
                with self.assertRaises(ValueError):self.d.apply_download(run,db)
                external.execute('UPDATE event SET EventName="old"');external.commit()
            finally:external.close()
            # Simulate the disk failing after SQLite commits but before the
            # final receipt is written. The durable intent permits completion.
            from unittest import mock
            original_write=self.d.write_json
            def fail_receipt(path,value):
                if Path(path).name=='database_receipt.json':raise OSError('simulated receipt failure')
                return original_write(path,value)
            with mock.patch.object(self.d,'write_json',side_effect=fail_receipt):
                with self.assertRaises(OSError):self.d.apply_download(run,db)
            receipt=self.d.apply_download(run,db)
            self.assertTrue(receipt['recovered_after_commit'])
            self.assertTrue(Path(receipt['backup']).exists())
            with sqlite3.connect(db) as con:
                rows=con.execute('SELECT Rank,PointRank FROM event ORDER BY Rank').fetchall()
                self.assertEqual([r[0] for r in rows],[10,20])
                self.assertEqual(json.loads(rows[1][1]),[{'time':1000,'ep':8},{'time':2000,'ep':20}])
            self.assertEqual(self.d.apply_download(run,db),receipt)
            # Independently recompute the top10 boundary from raw JSON and
            # verify the database union against its pre-import backup.
            import audit_rank_download as audit_module
            from types import SimpleNamespace
            with mock.patch.object(audit_module,'HoldoutDataset',return_value=SimpleNamespace(
                    manifest={'dataset_id':audit_module.EXPECTED_HOLDOUT})):
                audit_module.audit(run,db,root/'audit')
            checked=json.loads((root/'audit/verification.json').read_text())
            self.assertTrue(checked['all_imported_curves_match_source_plus_backup'])
            self.assertEqual(checked['database_points'],3)
            # A completed download must fail verification if an artifact is tampered with.
            (run/'curves/1_20.json').write_text('[]')
            with self.assertRaises(ValueError):self.d.verify_download(run)

    def test_frozen_dataset_is_not_an_import_destination(self):
        with self.assertRaises(ValueError):self.d.check_database_path(Path('data/holdout_v1/snapshot.sqlite3'))

    def test_frozen_run_paths_and_symlinks_are_rejected_before_network(self):
        import asyncio,httpx
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);frozen=root/'holdout_v1';frozen.mkdir();link=root/'alias';link.symlink_to(frozen)
            db=root/'db.sqlite3'
            with sqlite3.connect(db) as con:con.execute('CREATE TABLE event(x)')
            def handler(request):raise AssertionError('Network reached for a frozen destination')
            for run in [frozen/'new-run',link/'new-run']:
                with self.assertRaises(ValueError):
                    asyncio.run(self.d.download(run,db,transport=httpx.MockTransport(handler)))
                self.assertFalse(run.exists())

    def test_database_identity_includes_committed_wal_content(self):
        with tempfile.TemporaryDirectory() as temp:
            db=Path(temp)/'db.sqlite3'
            con=sqlite3.connect(db);self.addCleanup(con.close)
            con.execute('PRAGMA journal_mode=WAL');con.execute('PRAGMA wal_autocheckpoint=0')
            con.execute('CREATE TABLE event(x)');con.execute('INSERT INTO event VALUES(1)');con.commit()
            con.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            before=self.d.database_identity(db);physical=self.d.sha256(db)
            con.execute('UPDATE event SET x=2');con.commit()
            self.assertEqual(physical,self.d.sha256(db))
            self.assertNotEqual(before,self.d.database_identity(db))


if __name__=='__main__':unittest.main()
