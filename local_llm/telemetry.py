"""Small cached, local-only system readings, independent from the inference lock."""
from __future__ import annotations

import ctypes
import math
import os
import platform
import re
import subprocess
import threading
import time
from pathlib import Path

from .recommendations import detect_hardware


def parse_vm_stat(text, total):
    page = re.search(r'page size of (\d+) bytes', text)
    if not page:
        raise ValueError('Taille des pages inconnue')
    values = {key: int(value) for key, value in re.findall(r'^([^:\n]+):\s+(\d+)\.?$', text, re.M)}
    required = ['Pages active', 'Pages inactive', 'Pages wired down', 'Pages speculative',
                'Pages occupied by compressor', 'Pages purgeable', 'File-backed pages']
    if any(k not in values for k in required):
        raise ValueError('Statistiques mémoire incomplètes')
    pages = sum(values[k] for k in required[:5]) - values['Pages purgeable'] - values['File-backed pages']
    return max(0, min(total, pages * int(page.group(1))))


def linux_temperature(root=Path('/sys/class/hwmon')):
    values = []
    for directory in sorted(root.glob('hwmon*'))[:32]:
        try:
            name = (directory / 'name').read_text().strip()
            if name not in {'coretemp', 'k10temp', 'zenpower', 'cpu_thermal', 'soc_thermal'}:
                continue
            for sensor in sorted(directory.glob('temp*_input'))[:128]:
                try:
                    value = float(sensor.read_text()) / 1000
                    if math.isfinite(value) and 0 < value <= 150:
                        values.append(value)
                except (OSError, ValueError):
                    continue
        except OSError:
            continue
    return max(values) if values else None


class SystemTelemetry:
    def __init__(self):
        self._lock = threading.Lock()
        self._last = -float('inf')
        self._snapshot = None
        self._total = None
        self._initialized = False
        self._sensor = None

    def snapshot(self):
        with self._lock:
            if self._snapshot is not None and time.monotonic() - self._last < 3:
                return dict(self._snapshot)
            system = platform.system()
            if not self._initialized:
                self._total = detect_hardware()['memory_bytes']
                if system == 'Darwin':
                    try:
                        from .sensors_macos import MacSensors
                        self._sensor = MacSensors()
                    except (OSError, AttributeError, ValueError):
                        pass
                self._initialized = True
            used = rss = temperature = None
            memory_method = None
            try:
                if system == 'Darwin':
                    text = subprocess.check_output(['vm_stat'], text=True, timeout=2)
                    used = parse_vm_stat(text, self._total) if self._total else None
                    rss = int(subprocess.check_output(['ps', '-o', 'rss=', '-p', str(os.getpid())], text=True, timeout=2).strip()) * 1024
                    memory_method = 'Pages actives + inactives + câblées + spéculatives + compressées, hors pages de fichiers et purgeables.'
                elif system == 'Windows':
                    class Memory(ctypes.Structure):
                        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [(k, ctypes.c_ulonglong) for k in ('total', 'available', 'page_total', 'page_available', 'virtual_total', 'virtual_available', 'extended')]
                    status = Memory()
                    status.length = ctypes.sizeof(status)
                    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                        self._total = status.total
                        used = status.total - status.available
                    class ProcessMemory(ctypes.Structure):
                        _fields_ = [('size', ctypes.c_ulong), ('faults', ctypes.c_ulong)] + [(k, ctypes.c_size_t) for k in ('peak', 'rss', 'paged_peak', 'paged', 'nonpaged_peak', 'nonpaged', 'pagefile', 'pagefile_peak')]
                    process = ProcessMemory()
                    process.size = ctypes.sizeof(process)
                    ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
                    fn = ctypes.windll.psapi.GetProcessMemoryInfo
                    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
                    if fn(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(process), process.size):
                        rss = process.rss
                    memory_method = 'RAM physique totale moins RAM disponible.'
                else:
                    values = {k: int(v.split()[0]) * 1024 for k, v in (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
                    self._total = values['MemTotal']
                    used = self._total - values['MemAvailable']
                    for line in Path('/proc/self/status').read_text().splitlines():
                        if line.startswith('VmRSS:'):
                            rss = int(line.split()[1]) * 1024
                    memory_method = 'MemTotal moins MemAvailable ; RSS du processus de l’application.'
            except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError):
                pass
            try:
                if system == 'Darwin' and self._sensor:
                    temperature = self._sensor.temperature()
                elif system == 'Linux':
                    temperature = linux_temperature()
            except (OSError, ValueError):
                pass
            temperature_method = {
                'Darwin': 'Moyenne des capteurs SMC CPU identifiés (lecture seule).',
                'Linux': 'Maximum des capteurs CPU hwmon.',
            }.get(system)
            self._snapshot = dict(
                sampled_at=time.time(), interval_seconds=3,
                memory_total_bytes=self._total, memory_used_bytes=used,
                process_rss_bytes=rss, memory_method=memory_method,
                cpu_temperature_celsius=temperature,
                temperature_method=temperature_method,
                temperature_note=None if temperature is not None else 'Aucun capteur CPU accessible. Une valeur absente n’est jamais estimée.',
                process_note='RAM résidente du serveur local-llm uniquement. LM Studio est un autre processus ; sa RAM est incluse dans le total système.',
            )
            self._last = time.monotonic()
            return dict(self._snapshot)

    def close(self):
        with self._lock:
            if self._sensor is not None:
                self._sensor.close()
                self._sensor = None
