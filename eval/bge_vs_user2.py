"""Офлайн-сравнение retrieval-качества нескольких эмбеддинг-моделей по одним и
тем же захваченным капчам. Для каждого QA-probe берём:
  • `question` (текст запроса),
  • `candidatePool` (полный пул фактов на момент probe — то, что extraction
    реально написал),
  • `retrievedTopK` (что система с USER2-small и гибридным retrieval’ом
    фактически достала на устройстве).

Дальше каждый «ranker» отдаёт свой top-K тех же фактов, а DeepSeek-судья по
каждому набору говорит «нужный для ответа факт присутствует? YES/NO». Recall =
попаданий / число probe’ов соответствующего типа. Это позволяет сравнить:

  ◦ USER2-system     — реальный on-device retrieval (гибрид FTS+вектор,
                        фактический `retrievedTopK` из капчи);
  ◦ USER2-prefix     — USER2-small В Python с правильными префиксами
                        `search_query:` / `search_document:` (тестируем
                        гипотезу префикс-фикса: USER2 обучалась Nomic-style,
                        в проде префиксы НЕ проставлены — это, возможно, и есть
                        основная причина 34% zero-retrievals);
  ◦ USER2-bare       — USER2-small в Python БЕЗ префиксов (контроль: должен
                        приблизительно совпасть с USER2-system, выявит разницу
                        гибрид-vs-чистый-вектор);
  ◦ BGE-M3           — тяжёлый мультиязычный референс (2.3 ГБ);
  ◦ (далее можно добавить rubert-mini-frida, BERTA, EmbeddingGemma).

Зачем офлайн: модели грузятся только в Python-скрипт, на устройство не
тащим; и сравнение не зависит от LLM-генерации — хватает капч ОДНОГО режима
(FACTS_ONLY). Расход ресурсов на CPU небольшой (короткие тексты, ~6 фактов в
пуле), можно гонять параллельно с Kotlin-бенчем без заметного замедления.

Запуск:
  DEEPSEEK_API_KEY=sk-... python eval/bge_vs_user2.py
  python eval/bge_vs_user2.py --mock                            # без API
  python eval/bge_vs_user2.py --rankers system,user2-prefix     # быстрее, без BGE
  python eval/bge_vs_user2.py --include-diffuse                 # +gist/synthesis
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from judge import Judge, MockJudge  # noqa: E402
from loaders import load_captures   # noqa: E402

# Типы probe’ов с конкретным целевым фактом в памяти. abstention — нечего
# доставать; gist/synthesis — диффузные (нужен весь диалог, не топ-K).
RETRIEVAL_TYPES = {"recall", "temporal", "update", "multi_fact"}


def emb_text(fact: dict) -> str:
    """Текст для эмбеддинга — как embText в проде (content + keywords + context)."""
    parts = [fact.get("content", "")]
    kw = fact.get("keywords") or []
    if kw:
        parts.append(" ".join(kw))
    ctx = fact.get("context")
    if ctx:
        parts.append(ctx)
    return " ".join(p for p in parts if p).strip()


def content_of(fact: dict) -> str:
    return fact.get("content", "")


# ─── Ranker’ы ─────────────────────────────────────────────────────────────

class SystemRanker:
    """USER2-small как она реально работает в проде — гибридный retrieval
    (FTS+вектор) с порогом и tier-default K. Берём прямо из капчи, ничего
    не пересчитываем (это honest baseline системы)."""
    name = "USER2-system"

    def top_k(self, question, captured_top, pool, k):
        return [content_of(f) for f in captured_top]

    def k_for(self, captured_top, k):
        return len(captured_top)


class HFRanker:
    """transformers напрямую с ЯВНЫМ pooling — чтобы точно совпадать с проdом.
    USER2-small использует CLS pooling (EmbeddingModel.kt). sentence-transformers
    автодетектит pooling из конфига и может выдать другие вектора → не годится
    для честного сравнения. Здесь pooling задан явно."""

    def __init__(self, model_name: str, label: str,
                 pooling: str = "cls",
                 query_prefix: str = "", doc_prefix: str = "",
                 max_length: int = 512,
                 device: str | None = None):
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch  # noqa: F401
        except ImportError:
            sys.exit("[ranker] pip install transformers torch")
        t0 = time.time()
        print(f"[ranker] загружаю {label} ← {model_name} (pooling={pooling})…")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        if device:
            self.model = self.model.to(device)
        self.model.eval()
        self.pooling = pooling
        self.qp = query_prefix
        self.dp = doc_prefix
        self.max_length = max_length
        self.name = label
        print(f"[ranker]   готово за {time.time()-t0:.1f}s")

    def _encode(self, texts):
        import torch
        import torch.nn.functional as F
        with torch.no_grad():
            inp = self.tok(texts, return_tensors="pt",
                           truncation=True, padding=True,
                           max_length=self.max_length)
            inp = {k: v.to(self.model.device) for k, v in inp.items()}
            out = self.model(**inp)
            h = out.last_hidden_state  # [B, L, D]
            if self.pooling == "cls":
                v = h[:, 0, :]
            else:  # "mean" с учётом attention_mask
                mask = inp["attention_mask"].unsqueeze(-1).float()
                v = (h * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
            v = F.normalize(v, p=2, dim=-1)
        return v.cpu().numpy()

    def top_k(self, question, captured_top, pool, k):
        import numpy as np
        if not pool or k <= 0:
            return []
        q_text = self.qp + question
        d_texts = [self.dp + emb_text(f) for f in pool]
        embs = self._encode([q_text] + d_texts)
        q, docs = embs[0], embs[1:]
        scores = docs @ q
        order = np.argsort(-scores)[:k]
        return [content_of(pool[i]) for i in order]

    def k_for(self, captured_top, k):
        return k


def build_rankers(names: list[str], device: str | None, top_k: int) -> list:
    """Имена → готовые объекты-ranker’ы. Грузим только запрошенные модели."""
    builders = {
        "system": lambda: SystemRanker(),
        # USER2-small: CLS pooling (как в проде, EmbeddingModel.kt).
        # Без префиксов — повторяем прод-поведение через transformers как sanity-check.
        "user2-bare": lambda: HFRanker("deepvk/USER2-small", "USER2-bare",
                                       pooling="cls"),
        # USER2-small с префиксами по Nomic-схеме, для которой её обучали.
        "user2-prefix": lambda: HFRanker("deepvk/USER2-small", "USER2-prefix",
                                         pooling="cls",
                                         query_prefix="search_query: ",
                                         doc_prefix="search_document: "),
        # BGE-M3: тоже CLS pooling (per FlagEmbedding), длинный контекст.
        "bge-m3": lambda: HFRanker("BAAI/bge-m3", "BGE-M3",
                                    pooling="cls", max_length=8192,
                                    device=device),
    }
    out = []
    for n in names:
        n = n.strip()
        if n not in builders:
            sys.exit(f"[ranker] неизвестный ranker '{n}' (есть: {list(builders)})")
        out.append(builders[n]())
    return out


# ─── main ─────────────────────────────────────────────────────────────────

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures-dir", default="captures")
    ap.add_argument("--mode", default="FACTS_ONLY",
                    help="режим капч (FACTS_ONLY/FACTS_PLUS_SUMMARY)")
    ap.add_argument("--rankers", default="system,user2-prefix,bge-m3",
                    help="CSV: system,user2-bare,user2-prefix,bge-m3")
    ap.add_argument("--top-k", type=int, default=6,
                    help="бюджет K для оффлайн-ранкеров (system берёт свой "
                         "фактический len(retrievedTopK))")
    ap.add_argument("--device", default=None, help="cpu / cuda (для BGE)")
    ap.add_argument("--mock", action="store_true", help="MockJudge без API")
    ap.add_argument("--include-diffuse", action="store_true",
                    help="включить gist/synthesis")
    ap.add_argument("--out", default="results/retrieval_recall.md")
    args = ap.parse_args(argv[1:])

    cap_dir = Path(args.captures_dir)
    if not cap_dir.exists():
        alt = Path("memory-bench/captures")
        cap_dir = alt if alt.exists() else cap_dir
    captures = [c for c in load_captures(cap_dir) if c.get("mode") == args.mode]
    if not captures:
        sys.exit(f"[bge] нет капч {args.mode} в {cap_dir}")

    types = set(RETRIEVAL_TYPES)
    if args.include_diffuse:
        types |= {"gist", "synthesis"}

    judge = MockJudge() if args.mock else Judge()
    ranker_names = [r.strip() for r in args.rankers.split(",") if r.strip()]
    rankers = build_rankers(ranker_names, args.device, args.top_k)
    print(f"[bge] судья={judge.name}  капч={len(captures)} ({args.mode})  "
          f"rankers={[r.name for r in rankers]}\n")

    # счётчики [hits,total] по (ranker, type)
    stats = {r.name: defaultdict(lambda: [0, 0]) for r in rankers}
    n_eval = 0
    skipped_no_pool = 0

    for cap in captures:
        sid = cap.get("scenarioId", "?")
        for qa in cap.get("qaCaptures", []):
            qtype = qa["check"]["type"]
            if qtype not in types:
                continue
            pool = qa.get("candidatePool", [])
            top = qa.get("retrievedTopK", [])
            if not pool:
                skipped_no_pool += 1
                continue
            gold = qa["check"]["gold"]
            n_eval += 1

            # Каждый ranker отдаёт свой top-K из пула; судья — присутствует ли gold
            marks = []
            for r in rankers:
                k = r.k_for(top, args.top_k)
                texts = r.top_k(qa["question"], top, pool, k)
                hit = judge.retrieval_contains(gold, texts)
                stats[r.name][qtype][0] += int(hit)
                stats[r.name][qtype][1] += 1
                marks.append(f"{r.name}={'+' if hit else '-'}")
            print(f"[{sid} t{qa['turn']} {qtype}] " + "  ".join(marks)
                  + f"  (pool={len(pool)} k_sys={len(top)})")

    # ─── сводка ─────────────────────────────────────────────────────────
    all_types = sorted({t for s in stats.values() for t in s})
    headers = ["qa_type"] + [r.name for r in rankers]
    lines = ["# Retrieval recall: сравнение эмбеддинг-моделей", "",
             f"- режим капч: {args.mode}", f"- судья: {judge.name}",
             f"- probe’ов оценено: {n_eval}",
             f"- ranker’ы: {', '.join(r.name for r in rankers)}", ""]

    def rate(d, t):
        h, n = d[t]
        return f"{h}/{n} ({h/n:.0%})" if n else "—"

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "---|" * len(headers))
    for t in all_types:
        row = [t] + [rate(stats[r.name], t) for r in rankers]
        lines.append("| " + " | ".join(row) + " |")

    # итого
    totals = []
    for r in rankers:
        h = sum(v[0] for v in stats[r.name].values())
        n = sum(v[1] for v in stats[r.name].values())
        totals.append(f"**{h}/{n} ({h/n:.0%})**" if n else "—")
    lines.append("| **итого** | " + " | ".join(totals) + " |")

    print("\n" + "\n".join(lines))
    if skipped_no_pool:
        print(f"\n[bge] пропущено без candidatePool: {skipped_no_pool}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[bge] → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
