"""Verified GGUF alternatives, without downloading weights or executing Hub code.

Only the selected model's repository identifier is sent to the public Hub API.
Conversation text, filesystem paths, tokens and hardware details stay local.
"""
from __future__ import annotations

import copy
import json
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from urllib.request import Request, build_opener, HTTPRedirectHandler, ProxyHandler

from .calibration import memory_plan
from .discovery import valid_repository
from .gguf import GGUFReader

GIB = 1024 ** 3
MAX_RESPONSE = 4 * 1024 * 1024
QUANT = re.compile(r"(?:[.-](?:UD|IQ))?[-._](IQ[1-4]_[A-Z0-9_]+|Q[2-8]_[A-Z0-9_]+|BF16|F16|F32)$", re.I)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Redirection refusée : seuls les liens directs Hugging Face sont utilisés.")


def hub_repository(repository):
    if not valid_repository(repository):
        raise ValueError("Identifiant Hugging Face invalide")
    url = "https://huggingface.co/api/models/" + quote(repository, safe="/") + "?blobs=true"
    # No ambient credentials/proxy, no redirects to another host, bounded JSON.
    opener = build_opener(ProxyHandler({}), NoRedirect())
    with opener.open(Request(url, headers={"Accept": "application/json", "User-Agent": "local-llm-model-advisor"}), timeout=5) as response:
        raw = response.read(MAX_RESPONSE + 1)
    if len(raw) > MAX_RESPONSE:
        raise ValueError("Catalogue Hugging Face trop volumineux")
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("id") != repository or data.get("private") or data.get("gated"):
        raise ValueError("Dépôt public sans restriction requis")
    if not re.fullmatch(r"[a-f0-9]{40}", str(data.get("sha", ""))):
        raise ValueError("Révision Hugging Face non vérifiable")
    if not isinstance(data.get("siblings"), list) or len(data["siblings"]) > 1024:
        raise ValueError("Liste de fichiers invalide")
    return data


def variant_identity(filename):
    name = PurePosixPath(filename).name
    if not name.lower().endswith(".gguf") or re.search(r"(?:^|[-_.])(mmproj|mtp|dflash|dspark|eagle)(?:[-_.]|$)", filename, re.I):
        return None
    if re.search(r"-\d{5}-of-\d{5}\.gguf$", name, re.I):
        return None  # Do not link one shard as if it were a complete checkpoint.
    stem = name[:-5]
    match = QUANT.search(stem)
    if not match:
        return None
    family = re.sub(r"[._-]+", "-", stem[:match.start()].replace(".official", "")).strip("-").lower()
    return family, match.group(1).upper()


def quant_bits(quant):
    match = re.match(r"I?Q([1-8])_", quant)
    return int(match.group(1)) if match else {"F16": 16, "BF16": 16, "F32": 32}.get(quant)


def variant_files(item, data):
    identity = variant_identity(Path(item.path).name)
    if not identity:
        return []
    found = []
    for sibling in data["siblings"]:
        if not isinstance(sibling, dict):
            continue
        filename = sibling.get("rfilename")
        if not isinstance(filename, str) or len(filename) > 512:
            continue
        parts = PurePosixPath(filename).parts
        if filename.startswith("/") or "\\" in filename or any(p in {"..", "."} for p in parts) or any(ord(c) < 32 for c in filename):
            continue
        variant = variant_identity(filename)
        # Exact fine-tune + parameter count + version; base_model is insufficient.
        if not variant or variant[0] != identity[0] or variant[1] == identity[1]:
            continue
        lfs = sibling.get("lfs")
        size = sibling.get("size") or (lfs.get("size") if isinstance(lfs, dict) else None)
        if type(size) is not int or not 0 < size <= 1024 * GIB:
            continue
        found.append({"quantization": variant[1], "size_bytes": size,
                      "file": filename, "repository": data["id"], "revision": data["sha"],
                      "url": "https://huggingface.co/" + data["id"] + "/blob/" + data["sha"] + "/" + quote(filename, safe="/")})
    return found


def fit(metadata, size, budget, context):
    if budget is None:
        return {"fits": None, "estimated_bytes": None, "context": context}
    try:
        plan = memory_plan(metadata, size, budget, required=context)
        return {"fits": True, "estimated_bytes": plan["estimated_bytes"], "context": plan["context"]}
    except ValueError:
        return {"fits": False, "estimated_bytes": None, "context": context}


def assess_variants(item, variants, metadata, total, available, context):
    identity = variant_identity(Path(item.path).name)
    current_bits = quant_bits(identity[1]) if identity else None
    machine_budget = max(0, total - 3 * GIB) if total is not None else None
    current = {"quantization": identity[1] if identity else item.quantization,
               "size_bytes": item.size_bytes, "now": fit(metadata, item.size_bytes, available, context),
               "machine": fit(metadata, item.size_bytes, machine_budget, context)}
    alternatives = []
    for value in variants:
        item_fit = fit(metadata, value["size_bytes"], machine_budget, context)
        if item_fit["fits"] is False:
            continue
        bits = quant_bits(value["quantization"])
        smaller = value["size_bytes"] < item.size_bytes * .95
        higher = current_bits is not None and bits is not None and bits > current_bits
        if not smaller and not higher:
            continue
        row = dict(value, now=fit(metadata, value["size_bytes"], available, context), machine=item_fit,
                   saved_bytes=max(0, item.size_bytes - value["size_bytes"]),
                   benefit="RAM réduite" if smaller else "Précision à comparer",
                   tradeoff="La quantification change les poids et peut changer les réponses. La vitesse reste à mesurer sur cette machine.",
                   verified_file=True, measured_speed_gain=None)
        alternatives.append(row)
    # Prefer viable moderate quants rather than the smallest (and lowest quality) file.
    preferred = {"Q4_K_M": 0, "Q4_K_XL": 0, "Q5_K_M": 1, "Q6_K": 2, "Q8_0": 3}
    alternatives.sort(key=lambda row: (row["now"]["fits"] is False,
                                       0 if current["machine"]["fits"] and row["benefit"] == "Précision à comparer" and (quant_bits(row["quantization"]) or 32) <= 6 else 1,
                                       preferred.get(row["quantization"], 5), abs(row["size_bytes"] - item.size_bytes)))
    return current, alternatives[:6]


class ModelAdvisor:
    """At most two background lookups; six-hour bounded cache, no chat blocking."""
    def __init__(self, fetch=hub_repository):
        self.fetch = fetch
        self.lock = threading.Lock()
        self.entries = OrderedDict()

    def get(self, item, total=None, available=None, context=4096, mtp_supported=False):
        path = Path(item.path)
        if path.suffix.lower() != ".gguf" or not path.is_file():
            return {"state": "unavailable", "model_id": item.id, "message": "Les suggestions de variantes concernent les checkpoints GGUF."}
        stat = path.stat()
        key = (item.id, stat.st_size, stat.st_mtime_ns)
        with self.lock:
            entry = self.entries.get(key)
            lifetime = 30 if entry is not None and entry["state"] == "unavailable" else 6 * 3600
            if entry is not None and time.monotonic() - entry["created"] > lifetime:
                self.entries.pop(key)
                entry = None
            if entry is None:
                if sum(e["state"] == "pending" for e in self.entries.values()) >= 2:
                    return {"state": "busy", "model_id": item.id, "message": "Recherche en cours pour un autre modèle…"}
                entry = {"state": "pending", "created": time.monotonic(), "variants": [], "message": "Vérification des fichiers sur Hugging Face…"}
                self.entries[key] = entry
                while len(self.entries) > 24:
                    removable = next(k for k, e in self.entries.items() if e["state"] != "pending")
                    self.entries.pop(removable)
                threading.Thread(target=self._lookup, args=(key, item), daemon=True).start()
            snapshot = copy.deepcopy(entry)
        metadata = snapshot.get("metadata", {})
        current, variants = assess_variants(item, snapshot["variants"], metadata, total, available, context)
        return {"state": snapshot["state"], "model_id": item.id, "model_name": item.name,
                "message": snapshot["message"], "variants": variants, "current": current,
                "repository": item.repository, "checked_at": snapshot.get("checked_at"),
                "context": context, "available_bytes": available, "machine_budget_bytes": max(0, total - 3 * GIB) if total else None,
                "mtp": {"embedded": item.mtp_heads > 0, "heads": item.mtp_heads,
                        "supported": item.mtp_heads > 0 and mtp_supported,
                        "message": ("Têtes MTP déjà présentes dans ce fichier : aucun téléchargement ni copie des poids. La calibration teste leur gain avant de les activer." if item.mtp_heads else "Aucune tête MTP intégrée détectée. Un modèle auxiliaire exige une compatibilité vérifiée avec ces poids.")},
                "method": "Liens vérifiés via l’API publique Hugging Face et fixés à une révision. Estimation prudente avec le contexte choisi, les buffers et une réserve mémoire. Aucun débit prédit. Seul l’identifiant du dépôt est envoyé, jamais les messages ni les chemins locaux."}

    def _lookup(self, key, item):
        result = {"state": "ready", "variants": [], "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        try:
            reader = GGUFReader(Path(item.path))
            # Memory geometry only; tokenizer and chat templates never leave this process.
            architecture = reader.metadata.get("general.architecture", "")
            result["metadata"] = {k: v for k, v in reader.metadata.items() if k == "general.architecture" or k.startswith(str(architecture) + ".")}
            if not item.repository:
                result["message"] = "Origine du checkpoint inconnue : aucune variante attribuée au hasard. Les optimisations du fichier installé restent disponibles."
            else:
                data = self.fetch(item.repository)
                result["variants"] = variant_files(item, data)
                result["message"] = "Variantes du même checkpoint vérifiées sur Hugging Face." if result["variants"] else "Aucune autre quantification complète du même checkpoint n’a été vérifiée dans ce dépôt."
        except (OSError, ValueError, TypeError, KeyError):
            result.update(state="unavailable", message="Vérification Hugging Face indisponible. Le modèle local fonctionne sans cette recherche.", variants=[])
        finally:
            with self.lock:
                if key in self.entries:
                    self.entries[key].update(result)
