import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from local_llm.cli import main


class AcceleratorCLITests(unittest.TestCase):
    def test_calibration_exports_report_and_always_closes_owned_runtime(self):
        runtime = Mock(); runtime.job = {'state': 'complete'}; runtime.profile = {'winner': 'standard'}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'report.json'
            with patch('local_llm.accelerator.Accelerator', return_value=runtime), patch('local_llm.discovery.inspect_model'), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['calibrate', 'installed.gguf', '--output', str(output)]), 0)
            self.assertEqual(json.loads(output.read_text()), runtime.profile)
        runtime.close.assert_called_once()
        runtime.optimize.assert_called_once_with(None)

    def test_failed_calibration_and_ctrl_c_release_owned_runtime(self):
        for error, status in [(ValueError('bad model'), 2), (KeyboardInterrupt(), 130)]:
            runtime = Mock(); runtime.load.side_effect = error
            with patch('local_llm.accelerator.Accelerator', return_value=runtime), patch('local_llm.discovery.inspect_model'), contextlib.redirect_stderr(io.StringIO()):
                if status == 2:
                    with self.assertRaises(SystemExit) as result: main(['calibrate', 'installed.gguf'])
                    self.assertEqual(result.exception.code, 2)
                else: self.assertEqual(main(['calibrate', 'installed.gguf']), 130)
            runtime.close.assert_called_once()

    def test_serve_default_auto_and_explicit_native_are_forwarded_without_starting_server(self):
        for args, engine in [(['serve'], 'auto'), (['serve', '--engine', 'native'], 'native')]:
            with patch('local_llm.cli.serve_http') as serve:
                self.assertEqual(main(args), 0)
            self.assertEqual(serve.call_args.args[-1], engine)


if __name__ == '__main__': unittest.main()
