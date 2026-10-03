"""Bounded local slot snapshots. These files contain private conversation state.

No prompt text in metadata, no downloads, and no arbitrary runtime filenames.
Only completed streams are saved. Incompatibility or corruption falls back cold.
"""
import hashlib
import json
import os
import re
import secrets
import time
import threading
from functools import wraps
from pathlib import Path

MIB = 1024 ** 2
NAME = re.compile(r'^[a-f0-9]{64}$')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(MIB), b''):
            h.update(chunk)
    return h.hexdigest()


def synchronized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)
    return call


class KVStore:
    def __init__(self, state_dir):
        self.lock = threading.RLock()
        self.root = Path(state_dir) / 'kv-cache'
        self.enabled = True
        self.budget = 512 * MIB
        self.last = None
        self.error = None
        try:
            settings = self.root / 'settings.json'
            if settings.is_file() and not settings.is_symlink() and settings.stat().st_size < 4096:
                value = json.loads(settings.read_text())
                if not isinstance(value, dict):
                    raise ValueError('invalid settings')
                if type(value.get('enabled')) is bool:
                    self.enabled = value['enabled']
        except (OSError, ValueError):
            pass

    @synchronized
    def prepare(self):
        if self.root.is_symlink():
            raise ValueError('Le dossier du cache ne peut pas être un lien symbolique.')
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        self.prune()
        return self.root

    def _write(self, path, value):
        temp = self.root / (secrets.token_hex(32) + '.tmp')
        try:
            with temp.open('x') as handle:
                temp.chmod(0o600)
                json.dump(value, handle, ensure_ascii=False)
            os.replace(temp, path)
        finally:
            self._discard(temp)

    def _discard(self, path):
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            self.error = 'Nettoyage du cache impossible : ' + str(exc)[:200]

    def key(self, conversation, binding):
        return digest({'conversation': conversation, 'binding': binding})

    def _records(self):
        records = []
        if not self.root.is_dir() or self.root.is_symlink():
            return records
        for path in self.root.glob('*.json'):
            if not NAME.fullmatch(path.stem) or path.is_symlink():
                continue
            try:
                if path.stat().st_size > 65536:
                    raise ValueError('oversized metadata')
                value = json.loads(path.read_text())
                binary = self.root / (path.stem + '.bin')
                if binary.is_symlink() or not binary.is_file() or binary.stat().st_size != value['bytes']:
                    raise ValueError('incomplete snapshot')
                used = value.get('used', 0)
                if type(used) not in (float, int):
                    raise ValueError('invalid timestamp')
                records.append((used, path.stem, binary.stat().st_size))
            except (OSError, ValueError, TypeError, KeyError):
                self.remove_key(path.stem)
        return records

    def remove_key(self, key):
        if NAME.fullmatch(key):
            for suffix in ('.json', '.bin'):
                self._discard(self.root / (key + suffix))

    @synchronized
    def remove(self, conversation, binding):
        self.remove_key(self.key(conversation, binding))

    def prune(self):
        records = sorted(self._records())
        total = sum(size for _, _, size in records)
        for _, key, size in records:
            if total <= self.budget:
                break
            self.remove_key(key)
            total -= size
        valid = {key for _, key, _ in self._records()}
        # Orphans arise if the app exits between saving a binary and metadata.
        if self.root.is_dir() and not self.root.is_symlink():
            for path in self.root.iterdir():
                if NAME.fullmatch(path.stem) and (path.suffix == '.tmp' or path.suffix == '.bin' and path.stem not in valid):
                    self._discard(path)

    @synchronized
    def describe(self):
        records = self._records()
        return {'enabled': self.enabled, 'budget_bytes': self.budget,
                'bytes': sum(size for _, _, size in records), 'entries': len(records),
                'last': self.last, 'error': self.error}

    @synchronized
    def configure(self, enabled=None, clear=False):
        self.prepare()
        if enabled is not None:
            if type(enabled) is not bool:
                raise ValueError('enabled doit être un booléen.')
            self.enabled = enabled
            self._write(self.root / 'settings.json', {'enabled': enabled})
        if clear or not self.enabled:
            for _, key, _ in self._records():
                self.remove_key(key)
            self.last = None
        return self.describe()

    @synchronized
    def save(self, client, slot, conversation, binding, messages, timings):
        if not self.enabled or not binding:
            return
        temporary = secrets.token_hex(32) + '.tmp'
        path = self.root / temporary
        key = self.key(conversation, binding)
        try:
            started = time.perf_counter()
            result = client._request('/slots/%d?action=save' % slot, {'filename': temporary}, timeout=30)
            if (path.is_symlink() or not path.is_file() or path.stat().st_size > self.budget or
                    type(result.get('n_saved')) is not int or result['n_saved'] <= 0):
                return
            path.chmod(0o600)
            size = path.stat().st_size
            previous = self.root / (key + '.json')
            cold_ms = timings.get('prompt_ms')
            old = {}
            if previous.is_file() and not previous.is_symlink() and previous.stat().st_size <= 65536:
                old = json.loads(previous.read_text())
                if not isinstance(old, dict):
                    raise ValueError('invalid snapshot metadata')
                history = [digest(m) for m in messages]
                if timings.get('cache_n', 0) and history[:len(old.get('history', []))] == old.get('history'):
                    if old.get('history') != history and isinstance(cold_ms, (int, float)):
                        cold_ms += old.get('prefill_ms') or 0
                    else:
                        cold_ms = old.get('prefill_ms', cold_ms)
            sha256 = file_digest(path)
            estimate = max(.001, (time.perf_counter() - started) * 1.5)
            if old.get('restore_measured') and old.get('bytes', 0) > 0:
                estimate = old['restore_seconds'] * size / old['bytes']
            metadata = {'binding': binding, 'bytes': size, 'sha256': sha256,
                        'history': [digest(m) for m in messages], 'tokens': result['n_saved'],
                        'prompt_tokens': timings.get('prompt_n'), 'prefill_ms': cold_ms,
                        'used': time.time(), 'save_seconds': time.perf_counter() - started,
                        # Initial estimate from this snapshot’s measured write + hash cost.
                        # Actual restore cost replaces it after the first restoration.
                        'restore_seconds': estimate, 'restore_measured': bool(old.get('restore_measured'))}
            os.replace(path, self.root / (key + '.bin'))
            self._write(self.root / (key + '.json'), metadata)
            self.prune()
            self.error = None
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.error = 'Cache disque indisponible : ' + str(exc)[:200]
        finally:
            self._discard(path)

    @synchronized
    def restore(self, client, slot, conversation, binding, messages):
        self.last = {'restored': False, 'restore_seconds': None}
        if not self.enabled or not binding:
            return False
        key = self.key(conversation, binding)
        meta, path = self.root / (key + '.json'), self.root / (key + '.bin')
        if not meta.is_file() or meta.is_symlink() or meta.stat().st_size > 65536:
            return False
        try:
            value = json.loads(meta.read_text())
            history = [digest(m) for m in messages]
            if value['binding'] != binding or history[:len(value['history'])] != value['history']:
                return False
            # A warm save lacks a useful cold cost; keep the previous cold bound.
            cold = value.get('prefill_ms')
            if not isinstance(cold, (float, int)) or cold <= 0 or value['restore_seconds'] >= cold / 1000:
                self.last['reason'] = 'recalcul estimé moins coûteux'
                return False
            started = time.perf_counter()
            if path.is_symlink() or path.stat().st_size != value['bytes'] or file_digest(path) != value['sha256']:
                raise ValueError('snapshot corrompu')
            result = client._request('/slots/%d?action=restore' % slot, {'filename': key + '.bin'}, timeout=30)
            elapsed = time.perf_counter() - started
            if result.get('n_restored') != value['tokens']:
                raise ValueError('restauration incomplète')
            value.update(used=time.time(), restore_seconds=elapsed, restore_measured=True)
            self._write(meta, value)
            self.last = {'restored': True, 'restore_seconds': elapsed, 'tokens': result['n_restored'],
                         'bytes': value['bytes'], 'save_seconds': value['save_seconds']}
            return True
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.remove_key(key)
            # A failed restore may have partially filled the slot: erase before cold fallback.
            client._request('/slots/%d?action=erase' % slot, {}, timeout=10)
            self.error = 'Cache ignoré : ' + str(exc)[:200]
            return False
