"""Валидатор полного контекста для FULL_DIALOGUE-прогона.

Проверяет, что в каждом capture весь диалог реально влез в KV без overflow'а
(shift_context не сработал → ранние сессии не выпали). Опирается на поля
kvPosAtProbe / nCtxUsed, которые BenchmarkRunner пишет в caption начиная с
2026-06-06.

Критерий overflow'а (native_engine.cpp): shift_context срабатывает когда
позиция достигает nCtx - OVERFLOW_HEADROOM (=4). Если kvPosAtProbe оказался
≥ nCtx - HEADROOM_SAFETY — диалог был обрезан, контекст НЕ полный.

Запуск:  python eval/validate_fullctx.py captures-lme-fullctx
Код возврата != 0, если хоть один capture не прошёл валидацию.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

# Запас от потолка: если pos подошёл ближе этого к nCtx — считаем что был
# (или почти был) overflow. native OVERFLOW_HEADROOM=4, берём с щедрым буфером.
HEADROOM_SAFETY = 64


def main(argv: list[str]) -> int:
    cap_dir = Path(argv[1]) if len(argv) > 1 else Path("captures-lme-fullctx")
    if not cap_dir.is_dir():
        print(f"[validate] нет каталога {cap_dir}")
        return 2

    files = sorted(cap_dir.glob("*-FULL_DIALOGUE-capture.json"))
    if not files:
        print(f"[validate] нет FULL_DIALOGUE-captures в {cap_dir}")
        return 2

    print(f"[validate] проверяю {len(files)} FULL_DIALOGUE-captures в {cap_dir}\n")
    ok = 0
    overflow = 0
    no_field = 0
    rows = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        sid = d.get("scenarioId", f.stem)
        for qa in d.get("qaCaptures", []):
            pos = qa.get("kvPosAtProbe", -1)
            nctx = qa.get("nCtxUsed", -1)
            if pos < 0 or nctx < 0:
                no_field += 1
                status = "NO_FIELD (старый capture без kvPosAtProbe)"
            elif pos >= nctx - HEADROOM_SAFETY:
                overflow += 1
                status = f"⚠ OVERFLOW pos={pos} ~ nCtx={nctx}"
            else:
                ok += 1
                fill = pos / nctx * 100
                status = f"OK pos={pos}/{nctx} ({fill:.0f}% заполнения)"
            rows.append((sid, qa.get("turn", "?"), status))

    for sid, turn, status in rows:
        print(f"  {sid:42} turn={turn:>3}  {status}")

    print(f"\n── итог ── OK={ok}  overflow={overflow}  no_field={no_field}")
    if overflow:
        print("[validate] ✗ часть диалогов обрезана — контекст НЕ полный, "
              "подними -Dbench.nctx и перепрогони эти сценарии.")
        return 1
    if no_field:
        print("[validate] ✗ часть captures без kvPosAtProbe — перегенери их "
              "свежим бенчем (поле добавлено 2026-06-06).")
        return 1
    print("[validate] ✓ все диалоги влезли целиком — контекст действительно полный.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
