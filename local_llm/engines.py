"""Choose an installed execution backend without substituting model weights."""
from pathlib import Path

from .accelerator import Accelerator
from .mac_runtime import MacRuntime, model_engines


class EngineManager:
    def __init__(self, executable=None, state_dir=None):
        self.llama = Accelerator(executable, state_dir)
        self.mac = MacRuntime(self.llama.state_dir)
        self.active = self.llama

    def __getattr__(self, name):
        return getattr(self.active, name)

    @property
    def memory_probe(self):
        return self.active.memory_probe

    @memory_probe.setter
    def memory_probe(self, value):
        self.llama.memory_probe = self.mac.memory_probe = value

    def available(self):
        llama, mac = self.llama.available(), self.mac.available()
        return dict(llama, available=llama['available'] or mac['available'],
                    engines={'llamacpp': llama, 'mlx': mac, 'mtplx': dict(mac, available=mac['available'] and 'mtplx' in mac.get('packages', {}))})

    def model_engines(self, item):
        if Path(item.path).suffix.lower() == '.gguf':
            return ['llamacpp'] if self.llama.available()['available'] and item.architecture not in {None, 'dflash', 'dspark', 'bert', 'nomic-bert'} else []
        return model_engines(item.path, self.mac.available())

    def describe(self):
        data = self.active.describe()
        return dict(data, available=self.available()['available'], engines=self.available()['engines'],
                    engine=data.get('engine', 'llamacpp'),
                    attribution=data.get('attribution', 'llama.cpp · GGML'),
                    scope=data.get('scope', 'llama.cpp · Metal' if data.get('gpu') else 'llama.cpp'))

    def load(self, item, available=None, engine='auto'):
        candidates = self.model_engines(item)
        if not candidates or engine != 'auto' and engine not in candidates:
            raise ValueError('Ce moteur n’est pas disponible pour ce checkpoint. Aucun autre modèle n’a été substitué.')
        target = self.llama if 'llamacpp' in candidates else self.mac
        # Only one owned model is resident. Do not stop external engines.
        if target is not self.active:
            self.active.unload()
        self.active = target
        if target is self.llama:
            return dict(target.load(item, available), **{k: v for k, v in self.describe().items() if k in ('engine', 'engines', 'scope', 'attribution')})
        target.load(item, available, engine)
        return self.describe()

    def unload(self):
        self.active.unload()
        return self.describe()

    def close(self):
        self.llama.close()
        self.mac.close()
