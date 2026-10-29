"""A native GUI cleanup must not accept stale success or stop running physics."""
import json
from pathlib import Path
import tempfile
import unittest
from scripts.run_live import monitor


class LiveLauncherTests(unittest.TestCase):
    def run_monitor(self,terminal):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'run_status.json';path.write_text('{"status":"passed"}')
            previous=path.stat().st_mtime_ns
            now=[0.]
            class Process:
                terminated=False
                def poll(self):return -15 if self.terminated else None
                def terminate(self):self.terminated=True
                def wait(self,timeout=None):return -15
            process=Process()
            def sleep(dt):
                now[0]+=dt
                if now[0]>=.2 and path.read_text()=='{"status":"passed"}':
                    path.write_text('{"status":"running"}')
                if now[0]>=.5:path.write_text(json.dumps({'status':terminal}))
                if now[0]<.5:self.assertFalse(process.terminated)
            result,cleaned=monitor(process,path,previous,grace=.3,clock=lambda:now[0],sleep=sleep)
            self.assertTrue(cleaned);self.assertGreaterEqual(now[0],.8)
            return result

    def test_fresh_success_only_and_owned_idle_child_cleanup(self):
        self.assertEqual(self.run_monitor('passed'),0)

    def test_cancelled_run_is_not_success(self):
        self.assertEqual(self.run_monitor('cancelled'),130)

    def test_failed_run_is_not_success(self):
        self.assertEqual(self.run_monitor('failed'),1)


if __name__=='__main__':unittest.main()
