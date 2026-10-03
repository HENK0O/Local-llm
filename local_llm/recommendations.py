"""Offline hardware-aware suggestions. Memory estimates are not throughput claims."""
from __future__ import annotations

import ctypes
import os
import platform
import subprocess
from pathlib import Path

GIB = 1024 ** 3
# Curated open-weight Apache-2.0 models, checked against publisher cards 2026-10-02.
MODELS = [
    dict(name="SmolLM2 360M Instruct", family="SmolLM2-360M", memory_gib=1.2,
         runtime="local-llm", format="GGUF Q8_0", purpose="Réponses rapides, essais du moteur",
         download_url="https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/blob/main/smollm2-360m-instruct-q8_0.gguf",
         url="https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct"),
    dict(name="SmolLM2 1.7B Instruct", family="SmolLM2-1.7B", memory_gib=4.0,
         runtime="local-llm", format="GGUF Q8_0", purpose="Plus de capacité pour rédiger et résumer",
         download_url="https://huggingface.co/bartowski/SmolLM2-1.7B-Instruct-GGUF/blob/main/SmolLM2-1.7B-Instruct-Q8_0.gguf",
         url="https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct"),
    dict(name="Qwen3 8B", family="Qwen3-8B", memory_gib=8.0,
         runtime="local-llm", format="GGUF Q4_K_M", purpose="Modèle polyvalent plus exigeant",
         download_url="https://huggingface.co/Qwen/Qwen3-8B-GGUF/blob/main/Qwen3-8B-Q4_K_M.gguf",
         url="https://huggingface.co/Qwen/Qwen3-8B-GGUF"),
]


def detect_hardware():
    total = available = None
    cpu = platform.processor() or platform.machine()
    try:
        if platform.system() == "Darwin":
            def sysctl(key):
                return subprocess.check_output(["sysctl", "-n", key], text=True, timeout=2).strip()
            total = int(sysctl("hw.memsize"))
            cpu = sysctl("machdep.cpu.brand_string")
        elif platform.system() == "Windows":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (k, ctypes.c_ulonglong) for k in ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended")]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                total, available = status.total, status.available
        else:
            values = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, value = line.split(":", 1)
                values[key] = int(value.strip().split()[0]) * 1024
            total, available = values.get("MemTotal"), values.get("MemAvailable")
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return dict(cpu=cpu, logical_cores=os.cpu_count(), system=platform.system(),
                architecture=platform.machine(), memory_bytes=total,
                available_memory_bytes=available,
                gpu_note="Le moteur direct utilise les accélérateurs disponibles via llama.cpp. Les variantes doivent être mesurées sur cette machine.")


def recommend_models(hardware=None, installed=()):
    hardware = hardware if hardware is not None else detect_hardware()
    total = hardware.get("memory_bytes")
    # Reserve OS memory and leave headroom for short contexts / other applications.
    budget = max(0, min(total * .6, total - 3 * GIB)) if total else None
    items = []
    for model in MODELS:
        item = dict(model)
        item["fits"] = None if budget is None else item["memory_gib"] * GIB <= budget
        item["installed_id"] = next((m.id for m in installed if
            item["family"].lower() in m.name.lower() and m.compatible and
            (item["format"].split()[-1] in m.quantization)), None)
        if item["fits"] is not False:
            items.append(item)
    local = [m for m in items if m["runtime"] == "local-llm"]
    preferred = (local[-1] if (hardware.get("logical_cores") or 1) >= 4 else local[0]) if local and budget is not None else None
    for item in items:
        item["recommended"] = item is preferred
    return dict(hardware=hardware, budget_bytes=budget, models=items,
                method="Estimations prudentes de mémoire pour un contexte court (environ 2 048 tokens), poids et cache inclus. 40 % de la RAM, avec au moins 3 Gio, restent réservés. La RAM disponible et le GPU ne sont pas utilisés pour prédire la vitesse. Aucun tok/s n’est promis ; validez avec une génération sur votre machine.",
                catalog_checked="2026-10-02")
