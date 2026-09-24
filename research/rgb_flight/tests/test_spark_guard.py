"""Admission tests for the unified-memory migration; no model or GPU allocation."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import Mock
from types import SimpleNamespace

if os.name!='nt':
    spec=importlib.util.spec_from_file_location('spark_guard',Path(__file__).resolve().parents[1]/'spark_guard.py')
    guard=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)


@unittest.skipIf(os.name=='nt','Spark uses native Linux fcntl and process groups')
class SparkGuardTests(unittest.TestCase):
    def rejected(self,stop_marker,used_fraction):
        with tempfile.TemporaryDirectory() as folder:
            home=Path(folder)
            root=home/'uav-rgb-flight'
            (root/'source').mkdir(parents=True)
            if stop_marker:
                (root/'STOP').write_text('operator cancelled')
            sentinel=home/'child_started'
            command=[sys.executable,'-c',f'from pathlib import Path; Path({str(sentinel)!r}).touch()']
            sample=dict(total_bytes=128*1024**3,available_bytes=int((1-used_fraction)*128*1024**3),used_fraction=used_fraction,swap_used_bytes=0)
            with patch.object(Path,'home',return_value=home),patch.object(guard,'memory',return_value=sample),patch.object(guard.shutil,'disk_usage',return_value=SimpleNamespace(free=1024**4)),patch.object(sys,'argv',['guard','--name','test','--seconds','10','--',*command]):
                self.assertEqual(guard.main(),2)
            self.assertFalse(sentinel.exists(),'Rejected work must never execute')
            result=json.loads(next((root/'runs').glob('*/result.json')).read_text())
            return result['reason']

    def test_eighty_percent_shared_memory_blocks_execution(self):
        self.assertIn('Unified-memory',self.rejected(False,.8))

    def test_operator_stop_blocks_execution(self):
        self.assertIn('stop marker',self.rejected(True,.1))

    def test_checkpoint_requested_before_ceiling_and_lock_held_through_cleanup(self):
        with tempfile.TemporaryDirectory() as folder:
            home=Path(folder)
            root=home/'uav-rgb-flight'
            (root/'source').mkdir(parents=True)
            low=dict(total_bytes=128*1024**3,available_bytes=100*1024**3,used_fraction=.2,swap_used_bytes=0)
            high=dict(low,available_bytes=30*1024**3,used_fraction=.76)
            child=Mock(pid=999999,returncode=0)
            child.poll.side_effect=[None,0,0]
            cleanup_checked=[]
            def killpg(pid,sig):
                if sig==0:
                    raise ProcessLookupError()
                with (root/'job.lock').open('a') as competitor:
                    with self.assertRaises(BlockingIOError):
                        guard.fcntl.flock(competitor,guard.fcntl.LOCK_EX|guard.fcntl.LOCK_NB)
                    cleanup_checked.append(True)
            with patch.object(Path,'home',return_value=home),patch.object(guard,'memory',side_effect=[low,high]), \
                 patch.object(guard.shutil,'disk_usage',return_value=SimpleNamespace(free=1024**4)), \
                 patch.object(guard.shutil,'which',return_value=None),patch.object(guard.subprocess,'Popen',return_value=child), \
                 patch.object(guard.os,'killpg',side_effect=killpg),patch.object(guard.time,'sleep'), \
                 patch.object(sys,'argv',['guard','--name','checkpoint','--seconds','600','--','unused']):
                self.assertEqual(guard.main(),0)
            run=next((root/'runs').iterdir())
            self.assertTrue((run/'CHECKPOINT_REQUEST').is_file())
            self.assertTrue(cleanup_checked)
            self.assertTrue(json.loads((run/'result.json').read_text())['checkpoint_requested'])

    def test_failed_owned_container_cleanup_cannot_report_success(self):
        with tempfile.TemporaryDirectory() as folder:
            home=Path(folder)
            root=home/'uav-rgb-flight'
            (root/'source').mkdir(parents=True)
            child=Mock(pid=999999,returncode=0)
            child.poll.return_value=0
            sample=dict(total_bytes=128*1024**3,available_bytes=100*1024**3,used_fraction=.2,swap_used_bytes=0)
            with patch.object(Path,'home',return_value=home),patch.object(guard,'memory',return_value=sample), \
                 patch.object(guard.shutil,'disk_usage',return_value=SimpleNamespace(free=1024**4)), \
                 patch.object(guard.shutil,'which',return_value='/usr/bin/docker'), \
                 patch.object(guard.subprocess,'Popen',return_value=child), \
                 patch.object(guard.os,'killpg',side_effect=ProcessLookupError()), \
                 patch.object(guard.subprocess,'check_output',return_value='owned-container\n'), \
                 patch.object(guard.subprocess,'run',side_effect=guard.subprocess.CalledProcessError(1,'docker stop')), \
                 patch.object(sys,'argv',['guard','--name','cleanup','--seconds','600','--','unused']):
                self.assertEqual(guard.main(),2)
            result=json.loads(next((root/'runs').glob('*/result.json')).read_text())
            self.assertEqual(result['status'],'cleanup_failed')


if __name__=='__main__':
    unittest.main()
