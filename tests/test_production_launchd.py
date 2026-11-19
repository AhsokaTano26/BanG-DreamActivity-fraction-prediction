import importlib
from pathlib import Path
import tempfile
import unittest

class ScheduleTests(unittest.TestCase):
    def test_schedule_preserves_venv_path_and_uses_bounded_cycle(self):
        module=importlib.import_module('production.launchd')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'live';cfg={'root':str(root),'interval_seconds':900}
            spec=module.plist_spec(cfg,root/'config.json')
            self.assertEqual(spec['StartInterval'],900)
            self.assertTrue(spec['RunAtLoad'])
            self.assertFalse(spec.get('KeepAlive',False))
            self.assertEqual(spec['ProgramArguments'][0],str(module.ROOT/'.venv/bin/python'))
            self.assertIn('cycle',spec['ProgramArguments'])
            self.assertEqual(spec['WorkingDirectory'],str(module.ROOT))

    def test_existing_other_config_is_not_replaced(self):
        module=importlib.import_module('production.launchd')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'launch.plist'
            import plistlib
            path.write_bytes(plistlib.dumps({'Label':module.LABEL,'ProgramArguments':['unrelated']}))
            with self.assertRaises(FileExistsError):module.validate_existing(path,{'Label':module.LABEL,'ProgramArguments':['new']})
