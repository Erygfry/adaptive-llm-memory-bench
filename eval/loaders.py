"""Загрузка корпуса (gold) и captures (фактический прогон Kotlin bench)."""
from __future__ import annotations

import json
from pathlib import Path


def load_scenarios(corpus_dir: Path) -> dict[str, dict]:
    """Все *.json сценарии из corpus_dir, ключ — scenario id."""
    scenarios: dict[str, dict] = {}
    for path in sorted(corpus_dir.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        sid = data["id"]
        if sid in scenarios:
            raise ValueError(f"Дубликат scenario id '{sid}' ({path})")
        scenarios[sid] = data
    return scenarios


def load_captures(captures_dir: Path) -> list[dict]:
    """Все *-capture.json из captures_dir. Список, а не dict — на один
    сценарий несколько captures (по одному на RunMode, ablation)."""
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(captures_dir.glob("*-capture.json"))
    ]
