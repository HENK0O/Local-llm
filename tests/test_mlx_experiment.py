import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from local_llm.cli import main
from local_llm.mlx_experiment import run_experiment, validate_model


class MLXExperimentTests(unittest.TestCase):
    def model(self, root):
        for name in ('config.json','tokenizer.json','tokenizer_config.json'):
            (root/name).write_text('{}')
        (root/'model.safetensors').write_bytes(b'fixture')

    def test_missing_local_weights_or_custom_code_cannot_trigger_download_or_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with self.assertRaises(ValueError): validate_model(root/'not-installed')
            self.model(root)
            self.assertEqual(validate_model(root),root.resolve())
            (root/'config.json').write_text(json.dumps({'auto_map':{'AutoModel':'custom.Model'}}))
            with self.assertRaises(ValueError): validate_model(root)

    def test_worker_is_offline_and_uses_an_explicit_isolated_interpreter(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.model(root)
            with patch('local_llm.mlx_experiment.platform.system', return_value='Darwin'), patch('local_llm.mlx_experiment.platform.machine', return_value='arm64'), patch('local_llm.mlx_experiment.subprocess.run', return_value=SimpleNamespace(returncode=0,stdout='{"experimental":true}',stderr='')) as run:
                self.assertTrue(run_experiment(root,'/tmp/isolated/bin/python')['experimental'])
                self.assertEqual(run.call_args.args[0][:3], ['/tmp/isolated/bin/python','-m','local_llm.mlx_experiment'])
                env=run.call_args.kwargs['env']
                self.assertEqual(env['HF_HUB_OFFLINE'],'1')
                self.assertEqual(env['TRANSFORMERS_OFFLINE'],'1')
                self.assertNotIn('shell',run.call_args.kwargs)
                run.return_value=SimpleNamespace(returncode=1,stdout='',stderr='Install mlx-lm separately')
                with self.assertRaisesRegex(ValueError,'Install mlx-lm separately'): run_experiment(root)

    def test_cli_exports_experimental_report_without_starting_app_server(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'report.json'
            with patch('local_llm.mlx_experiment.run_experiment',return_value={'experimental':True,'comparison_with_llama_cpp':None}) as run, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['mlx-experiment',folder,'--python','/tmp/mlx/bin/python','--output',str(output)]),0)
            self.assertTrue(json.loads(output.read_text())['experimental'])
            self.assertEqual(run.call_args.args[1],Path('/tmp/mlx/bin/python'))


if __name__ == '__main__': unittest.main()
