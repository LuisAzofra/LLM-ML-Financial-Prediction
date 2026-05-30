"""
trial_registry.py — contador honesto de pruebas para multiple-testing.

Registra cada configuración distinta probada (config_hash, universe) en un JSON
persistente bajo cache/. El recuento de pruebas distintas alimenta el Deflated
Sharpe Ratio: cuantas más configuraciones se prueban, mayor el haircut por
data-mining. Solo stdlib.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from typing import Any, Dict, List

_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache')


def registry_path() -> str:
    """Ruta del registro de pruebas: cache/trial_registry.json."""
    return os.path.join(_CACHE_DIR, 'trial_registry.json')


def _load() -> List[Dict[str, Any]]:
    path = registry_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _distinct_count(entries: List[Dict[str, Any]]) -> int:
    return len({(e.get('config_hash'), e.get('universe')) for e in entries})


def register_trial(variant_label: str, config: Dict[str, Any], universe: str) -> int:
    """Registra (config_hash, universe) si es nuevo y devuelve el nº de pruebas distintas."""
    entries = _load()
    config_hash = hashlib.sha1(json.dumps(config, sort_keys=True, default=str).encode('utf-8')).hexdigest()
    seen = {(e.get('config_hash'), e.get('universe')) for e in entries}
    if (config_hash, universe) not in seen:
        entries.append({
            'ts': datetime.now().isoformat(timespec='seconds'),
            'variant_label': variant_label,
            'config_hash': config_hash,
            'universe': universe,
        })
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(registry_path(), 'w', encoding='utf-8') as f:
            json.dump(entries, f, indent=2)
    return _distinct_count(entries)


def count_trials() -> int:
    """Nº de pares (config_hash, universe) distintos en el registro (0 si no existe)."""
    return _distinct_count(_load())
