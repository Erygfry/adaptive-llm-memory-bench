"""Линтер корпуса сценариев memory-bench.

Свип по всем сценариям (ru-*.json, lme-*.json и пр.): ловит авторские огрехи
ДО прогона судьи, чтобы не жечь компьют/API на битых проверках.

Проверки:
  ERROR  — ломает пайплайн (невалидный JSON, неизвестный qa.type, нет gold,
           кривой force_after_turns, дыры в нумерации turn'ов).
  WARN   — подозрительно, стоит глянуть (must_mention-токен отсутствует в gold —
           вероятно слишком строгий гейт; must_not_mention-токен ЕСТЬ в gold;
           CJK-символы в тексте; нет instruction-probe/abstention и т.п.).

Запуск:  python eval/lint_corpus.py            (из memory-bench/)
         python memory-bench/eval/lint_corpus.py [путь_к_scenarios]
Код возврата != 0, если есть хотя бы один ERROR.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

# Windows-консоль часто в cp1251 — принудительно UTF-8, иначе кириллица/символы рушат вывод
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

ALLOWED_QA_TYPES = {
    "recall", "multi_fact", "abstention", "update",
    "temporal", "gist", "synthesis",
}
# CJK / fullwidth — частый артефакт автогенерации в русском тексте
CJK_RE = re.compile(r"[　-鿿＀-￯]")


class Report:
    def __init__(self, path: Path):
        self.path = path
        self.errors: list[str] = []
        self.warns: list[str] = []
        self.info: dict = {}

    def err(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warns.append(msg)


def _norm(s: str) -> str:
    return s.casefold()


def lint_file(path: Path) -> Report:
    r = Report(path)
    raw = path.read_text(encoding="utf-8")

    # 1. CJK-загрязнение (по сырому тексту — ловит и в content, и в gold)
    for m in CJK_RE.finditer(raw):
        line = raw.count("\n", 0, m.start()) + 1
        r.warn(f"CJK-символ '{m.group()}' (U+{ord(m.group()):04X}) ~строка {line}")

    # 2. Валидность JSON
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        r.err(f"невалидный JSON: {e}")
        return r

    # 3. Топ-уровень
    for field in ("id", "dialog", "eviction_policy"):
        if field not in data:
            r.err(f"нет обязательного поля '{field}'")
    for field in ("ram_tier", "starting_date", "language"):
        if field not in data:
            r.warn(f"нет поля '{field}'")

    dialog = data.get("dialog", [])
    if not isinstance(dialog, list) or not dialog:
        r.err("dialog пустой или не список")
        return r

    # 4. Нумерация и роли turn'ов
    turns = [t.get("turn") for t in dialog]
    expected = list(range(1, len(dialog) + 1))
    if turns != expected:
        r.err(f"нумерация turn'ов не 1..{len(dialog)} (есть дыры/повторы)")
    max_turn = len(dialog)
    for t in dialog:
        if t.get("role") not in ("user", "assistant"):
            r.err(f"turn {t.get('turn')}: role='{t.get('role')}' (ожидался user/assistant)")

    # 5. force_after_turns
    ev = data.get("eviction_policy", {})
    faf = ev.get("force_after_turns", [])
    if faf != sorted(faf):
        r.err(f"force_after_turns не по возрастанию: {faf}")
    if len(set(faf)) != len(faf):
        r.err(f"force_after_turns содержит повторы: {faf}")
    for x in faf:
        if not (1 <= x <= max_turn):
            r.err(f"force_after_turns: {x} вне диапазона 1..{max_turn}")

    # 6. QA-probe'ы
    qa_types: Counter = Counter()
    has_instruction = False
    n_probes = 0
    for t in dialog:
        checks = t.get("checks") or {}
        qa = checks.get("qa")
        if not qa:
            continue
        n_probes += 1
        turn_id = t.get("turn")
        qtype = qa.get("type")
        qa_types[qtype] += 1

        if qtype not in ALLOWED_QA_TYPES:
            r.err(f"turn {turn_id}: неизвестный qa.type='{qtype}'")
        if t.get("role") != "user":
            r.err(f"turn {turn_id}: qa-probe на роли '{t.get('role')}' (должен быть user)")

        gold = qa.get("gold")
        if not gold or not str(gold).strip():
            r.err(f"turn {turn_id}: пустой gold")
            gold = ""
        gold_n = _norm(str(gold))

        must = qa.get("must_mention", []) or []
        must_not = qa.get("must_not_mention", []) or []
        if not isinstance(must, list) or not isinstance(must_not, list):
            r.err(f"turn {turn_id}: must_mention/must_not_mention должны быть списками")
            must, must_not = [], []

        # must_mention — жёсткий гейт (metrics.py): токен ОБЯЗАН быть в ответе.
        # Если его нет даже в gold — почти наверняка гейт слишком строгий.
        if qtype != "abstention":
            for tok in must:
                if not str(tok).strip():
                    r.err(f"turn {turn_id}: пустой токен в must_mention")
                elif _norm(str(tok)) not in gold_n:
                    r.warn(f"turn {turn_id} ({qtype}): must_mention '{tok}' "
                           f"отсутствует в gold — возможно слишком строгий гейт")

        # must_not_mention — токен НЕ должен быть в ответе. Если он есть в gold,
        # значит gold сам себе противоречит (подозрительно).
        for tok in must_not:
            if str(tok).strip() and _norm(str(tok)) in gold_n:
                r.warn(f"turn {turn_id} ({qtype}): must_not_mention '{tok}' "
                       f"присутствует в gold — проверь")

        if qa.get("instruction_compliance"):
            has_instruction = True
            if qtype != "synthesis":
                r.warn(f"turn {turn_id}: instruction_compliance на типе '{qtype}' "
                       f"(обычно на synthesis)")

    # 7. Сводка по файлу
    dates = {t.get("date") for t in dialog if t.get("date")}
    r.info = {
        "turns": len(dialog),
        "sessions": len(dates),
        "probes": n_probes,
        "by_type": dict(qa_types),
        "has_instruction": has_instruction,
        "has_abstention": qa_types.get("abstention", 0) > 0,
    }
    if not has_instruction:
        r.warn("нет instruction_compliance probe (нет проверки соблюдения инструкции)")
    if not r.info["has_abstention"]:
        r.warn("нет abstention probe")
    missing = ALLOWED_QA_TYPES - set(qa_types)
    if missing:
        r.warn(f"не покрыты типы QA: {sorted(missing)}")

    return r


def main(argv: list[str]) -> int:
    # argv[1] — каталог (опц.), argv[2] — glob-шаблон (опц., по умолчанию *.json)
    scen_dir = Path(argv[1]) if len(argv) > 1 else Path("corpus/scenarios")
    if not scen_dir.exists():
        # запуск из корня репо
        alt = Path("memory-bench/corpus/scenarios")
        scen_dir = alt if alt.exists() else scen_dir
    pattern = argv[2] if len(argv) > 2 else "*.json"
    files = sorted(scen_dir.glob(pattern))
    if not files:
        print(f"[lint] не найдено сценариев в {scen_dir}")
        return 1

    total_err = total_warn = 0
    type_totals: Counter = Counter()
    print(f"[lint] {len(files)} сценариев в {scen_dir}\n")

    for path in files:
        rep = lint_file(path)
        total_err += len(rep.errors)
        total_warn += len(rep.warns)
        for k, v in rep.info.get("by_type", {}).items():
            type_totals[k] += v

        status = "ERROR" if rep.errors else ("WARN" if rep.warns else "OK")
        info = rep.info
        summary = ""
        if info:
            summary = (f"  turns={info['turns']} sess={info['sessions']} "
                       f"probes={info['probes']}")
        print(f"[{status:5}] {path.name}{summary}")
        for e in rep.errors:
            print(f"        [x] {e}")
        for w in rep.warns:
            print(f"        [!] {w}")

    print("\n── сводка ──")
    print(f"файлов: {len(files)}   ошибок: {total_err}   предупреждений: {total_warn}")
    print("probe'ы по типам: " + ", ".join(f"{k}={v}" for k, v in sorted(type_totals.items())))
    return 1 if total_err else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
