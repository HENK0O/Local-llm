import json
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from local_llm.discovery import inspect_model, mtp_head_count, file_provenance
from local_llm.model_advisor import (ModelAdvisor, NoRedirect, assess_variants, hub_repository,
                                     variant_files, variant_identity, MAX_RESPONSE)
from local_llm.accelerator import Accelerator, ExecutionConfig
from local_llm.server import ChatService
from test_discovery import chat_toy

GIB = 1024 ** 3
SHA = 'a' * 40


def item(name='FineTune-8B-Q8_0.gguf', size=8*GIB, repository='owner/FineTune-GGUF'):
    return SimpleNamespace(id='model', path='/existing/' + name, name=name, size_bytes=size,
                           repository=repository, quantization='Q8_0', mtp_heads=0)


def data(files):
    return {'id': 'owner/FineTune-GGUF', 'sha': SHA, 'private': False, 'gated': False,
            'siblings': [{'rfilename': filename, 'size': size} for filename, size in files]}


class AdvisorTests(unittest.TestCase):
    def test_exact_finetune_and_complete_gguf_only_with_revision_pinned_links(self):
        response = data([('FineTune-8B-Q4_K_M.gguf',4*GIB), ('Base-8B-Q4_K_M.gguf', 4*GIB),
                         ('FineTune-27B-Q4_K_M.gguf', 4*GIB), ('FineTune-8B-Q8_0.gguf',8*GIB),
                         ('FineTune-8B-Q4_K_M-00001-of-00002.gguf',2*GIB),
                         ('MTP/mtp-FineTune-8B-Q4_0.gguf',GIB), ('mmproj-F16.gguf',GIB),
                         ('../FineTune-8B-Q4_K_M.gguf',4*GIB), ('FineTune-8B-Q5_K_M.gguf',None)])
        rows = variant_files(item(),response)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['url'], 'https://huggingface.co/owner/FineTune-GGUF/blob/' + SHA + '/FineTune-8B-Q4_K_M.gguf')
        self.assertEqual(variant_identity('Qwen3.8-27B-UD-IQ3_S.gguf'), ('qwen3-8-27b','IQ3_S'))
        self.assertEqual(variant_identity('SmolLM2-360M-Instruct.official.Q8_0.gguf')[0], 'smollm2-360m-instruct')

    def test_memory_fit_depends_on_context_free_memory_and_total_not_speed_predictions(self):
        metadata = {'general.architecture':'llama', 'llama.block_count':32,
                    'llama.embedding_length':4096,'llama.attention.head_count':32,
                    'llama.attention.head_count_kv':8,'llama.context_length':8192}
        rows=variant_files(item(),data([('FineTune-8B-Q4_K_M.gguf',4*GIB),('FineTune-8B-F16.gguf',16*GIB)]))
        current, variants=assess_variants(item(),rows,metadata,12*GIB,2*GIB,4096)
        self.assertFalse(current['now']['fits'])
        self.assertEqual(len(variants),1)
        self.assertTrue(variants[0]['machine']['fits'])
        self.assertFalse(variants[0]['now']['fits'])
        self.assertIsNone(variants[0]['measured_speed_gain'])
        self.assertEqual(variants[0]['saved_bytes'],4*GIB)
        _, unknown=assess_variants(item(),rows,metadata,None,None,4096)
        self.assertIsNone(unknown[0]['machine']['fits'])

    def test_public_api_is_bounded_and_rejects_private_gated_redirected_or_unverified_data(self):
        response = Mock()
        response.__enter__ = Mock(return_value=response); response.__exit__ = Mock(return_value=False)
        opener = Mock(); opener.open.return_value=response
        for change in ({'private':True},{'gated':'auto'},{'sha':'main'},{'id':'other/repo'}, {'siblings':None}):
            response.read.return_value=json.dumps(dict(data([]),**change)).encode()
            with patch('local_llm.model_advisor.build_opener',return_value=opener):
                with self.assertRaises(ValueError): hub_repository('owner/FineTune-GGUF')
        response.read.return_value=b'x'*(MAX_RESPONSE+1)
        with patch('local_llm.model_advisor.build_opener',return_value=opener):
            with self.assertRaises(ValueError): hub_repository('owner/FineTune-GGUF')
        response.read.return_value=json.dumps(data([])).encode()
        with patch('local_llm.model_advisor.build_opener',return_value=opener):
            hub_repository('owner/FineTune-GGUF')
        request=opener.open.call_args[0][0]
        self.assertEqual(request.full_url,'https://huggingface.co/api/models/owner/FineTune-GGUF?blobs=true')
        self.assertNotIn('Authorization',request.headers)
        for bad in ('https://evil.test/a','../owner/repo','owner/repo?token=secret','owner/repo/extra'):
            with self.assertRaises(ValueError): hub_repository(bad)
        with self.assertRaises(ValueError): NoRedirect().redirect_request(None,None,302,'',{},'https://evil.test')

    def test_background_cache_is_bounded_and_unknown_origin_never_guesses(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'FineTune-8B-Q8_0.gguf'; path.write_bytes(b'fixture')
            selected=item(); selected.path=str(path)
            event=threading.Event(); calls=[]
            def fetch(repo):
                calls.append(repo); event.wait(2); return data([('FineTune-8B-Q4_K_M.gguf',4*GIB)])
            advisor=ModelAdvisor(fetch)
            reader=SimpleNamespace(metadata={'general.architecture':'llama','tokenizer.ggml.tokens':['PRIVATE']})
            with patch('local_llm.model_advisor.GGUFReader',return_value=reader):
                self.assertEqual(advisor.get(selected)['state'],'pending')
                selected.id='other'; self.assertEqual(advisor.get(selected)['state'],'pending')
                selected.id='third'; self.assertEqual(advisor.get(selected)['state'],'busy')
                event.set()
                deadline=time.monotonic()+3
                selected.id='model'
                while advisor.get(selected)['state']=='pending' and time.monotonic()<deadline: time.sleep(.01)
                result=advisor.get(selected)
            self.assertEqual(result['state'],'ready')
            self.assertEqual(len(calls),2)
            self.assertNotIn('PRIVATE',json.dumps(result))
            self.assertNotIn(folder,json.dumps(result))
            selected.id='unknown'; selected.repository=None
            with patch('local_llm.model_advisor.GGUFReader',return_value=reader):
                advisor.get(selected)
                deadline=time.monotonic()+3
                while advisor.get(selected)['state']=='pending' and time.monotonic()<deadline: time.sleep(.01)
            self.assertIn('Origine',advisor.get(selected)['message'])
            self.assertEqual(len(calls),2)

    def test_network_failure_is_cached_briefly_and_can_be_retried(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'FineTune-8B-Q8_0.gguf'; path.write_bytes(b'fixture')
            selected=item(); selected.path=str(path)
            fetch=Mock(side_effect=[OSError('offline'),data([])])
            advisor=ModelAdvisor(fetch)
            with patch('local_llm.model_advisor.GGUFReader',return_value=SimpleNamespace(metadata={})):
                advisor.get(selected)
                deadline=time.monotonic()+3
                while advisor.get(selected)['state']=='pending' and time.monotonic()<deadline: time.sleep(.01)
                self.assertEqual(advisor.get(selected)['state'],'unavailable')
                self.assertEqual(fetch.call_count,1)
                with patch('local_llm.model_advisor.time.monotonic',return_value=time.monotonic()+31):
                    self.assertEqual(advisor.get(selected)['state'],'pending')
                    deadline=time.time()+3
                    while advisor.get(selected)['state']=='pending' and time.time()<deadline: time.sleep(.01)
                    self.assertEqual(advisor.get(selected)['state'],'ready')
                self.assertEqual(fetch.call_count,2)

    def test_offline_repository_error_does_not_block_local_mtp_advice(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'FineTune-8B-Q8_0.gguf'; path.write_bytes(b'fixture')
            selected=item(); selected.path=str(path); selected.mtp_heads=1
            advisor=ModelAdvisor(Mock(side_effect=OSError('offline')))
            with patch('local_llm.model_advisor.GGUFReader',return_value=SimpleNamespace(metadata={})):
                advisor.get(selected)
                deadline=time.monotonic()+3
                while advisor.get(selected)['state']=='pending' and time.monotonic()<deadline: time.sleep(.01)
            result=advisor.get(selected,mtp_supported=True)
            self.assertEqual(result['state'],'unavailable')
            self.assertTrue(result['mtp']['supported'])


class DirectCheckpointTests(unittest.TestCase):
    def test_huggingface_symlink_keeps_gguf_name_and_read_in_place(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); blob=root/'blobs'/'hash'; blob.parent.mkdir(); blob.write_bytes(b'fixture')
            snapshot=root/'models--owner--FineTune-GGUF'/'snapshots'/SHA; snapshot.mkdir(parents=True)
            link=snapshot/'FineTune-8B-Q8_0.gguf'; link.symlink_to(blob)
            reader=SimpleNamespace(metadata={'general.architecture':'qwen35'},tensors={})
            with patch('local_llm.discovery.GGUFReader',return_value=reader): selected=inspect_model(link)
            self.assertEqual(selected.path,str(link))
            self.assertEqual(selected.repository,'owner/FineTune-GGUF')
            self.assertEqual(Path(selected.path).stat().st_ino,blob.stat().st_ino)

    def test_library_import_is_persisted_without_copy_and_refresh_retains_it(self):
        with tempfile.TemporaryDirectory() as folder, patch('local_llm.server.default_model_roots',return_value=[]):
            root=Path(folder); model=chat_toy(root/'existing'); state=root/'state'
            with patch.dict('os.environ',{'LOCAL_LLM_STATE_DIR':str(state)}): service=ChatService()
            try:
                inode=(model/'weights.npz').stat().st_ino
                service.import_model({'path':str(model)})
                self.assertEqual((model/'weights.npz').stat().st_ino,inode)
                self.assertEqual(len(service.available_models(refresh=True)['models']),1)
                self.assertEqual(json.loads((state/'library.json').read_text()),[str(model)])
                with patch.dict('os.environ',{'LOCAL_LLM_STATE_DIR':str(state)}): restored=ChatService()
                self.assertEqual(len(restored.catalog),1)
                restored.telemetry.close()
                for bad in ('relative.gguf',str(root/'missing.gguf'),str(state/'library.json')):
                    with self.assertRaises(ValueError): service.import_model({'path':bad})
            finally: service.telemetry.close()

    def test_mtp_requires_real_embedded_head_and_runtime_method_no_draft_file(self):
        reader=SimpleNamespace(metadata={'general.architecture':'qwen35','qwen35.nextn_predict_layers':1},
                               tensors={'blk.64.nextn.eh_proj.weight':None})
        self.assertEqual(mtp_head_count(reader),1)
        self.assertEqual(mtp_head_count(SimpleNamespace(metadata=reader.metadata,tensors={})),0)
        runtime=Accelerator(executable='llama-server');runtime.path=Path('/existing/model.gguf')
        with patch.object(runtime,'available',return_value={'specialized_methods':['draft-mtp']}), patch('local_llm.accelerator.GGUFReader',return_value=reader):
            configs=runtime._candidate_configs(ExecutionConfig(),[])
        mtp=[c for c in configs.values() if c.speculative=='draft-mtp']
        self.assertEqual([c.draft_tokens for c in mtp],[2,4,8,16])
        self.assertTrue(all(c.draft_path is None for c in mtp))
        with self.assertRaises(ValueError): ExecutionConfig(speculative='draft-mtp',draft_path='/extra.gguf')


class MTPMemoryTests(unittest.TestCase):
    def test_insufficient_mtp_memory_keeps_running_target_untouched(self):
        runtime=Accelerator(executable='llama-server')
        runtime.path=Mock(); runtime.path.stat.return_value.st_size=1
        runtime.memory_probe=Mock(return_value=2*GIB)
        plan={'estimated_bytes':int(1.3*GIB),'reserve_bytes':GIB//2,'kv_bytes_per_token':256*1024}
        with patch('local_llm.accelerator.GGUFReader',return_value=SimpleNamespace(metadata={})), patch('local_llm.accelerator.memory_plan',return_value=plan), patch.object(runtime,'_stop') as stop:
            with self.assertRaisesRegex(ValueError,'contexte MTP'): runtime._start(ExecutionConfig(speculative='draft-mtp'))
        stop.assert_not_called()

    def test_mtp_command_uses_main_checkpoint_and_no_second_weight_path(self):
        runtime=Accelerator(executable='llama-server')
        runtime.path=Path('/existing/model.gguf');runtime.model_id='target'
        process=Mock();process.poll.return_value=None
        client=Mock();client._request.side_effect=[{'status':'ok'}, {'default_generation_settings':{'n_ctx':4096},'total_slots':1}]
        with patch('local_llm.accelerator.socket.socket') as socket, patch('local_llm.accelerator.subprocess.Popen',return_value=process) as start, patch('local_llm.accelerator.LMStudioClient',return_value=client):
            socket.return_value.__enter__.return_value.getsockname.return_value=('127.0.0.1',54321)
            try:
                runtime._start(ExecutionConfig(speculative='draft-mtp',draft_tokens=8,slots=1))
                command=start.call_args[0][0]
                self.assertEqual(command[command.index('--model')+1],'/existing/model.gguf')
                self.assertEqual(command[command.index('--spec-draft-n-max')+1],'8')
                self.assertNotIn('--spec-draft-model',command)
            finally: runtime.close()
