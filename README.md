# memory-bench

Бенчмарк системы долговременной памяти для on-device LLM ассистента
(`adaptive-llm`). Корпус диалоговых сценариев + Python-харнес оценки.

## Архитектура

Двухкомпонентная, потому что прогон системы и судейство — разные задачи:

```
adaptive-llm-bench/  (Kotlin, отдельный репо)
  прогоняет сценарий через настоящую систему памяти (llama.cpp + sqlite-vec +
  USER2-small embedding через JNI), захватывает:
    - QA-ответы модели
    - DB snapshots (факты + summary на checkpoint'ах)
    - retrieved top-K для каждого QA
  → пишет captures/<scenario>-capture.json

memory-bench/  (этот репо)
  corpus/  — 50 сценариев (gold/expected) + SCHEMA.md
  eval/    — Python: читает scenario gold + capture.json → DeepSeek judge →
             метрики → CSV + графики
  captures/ — сюда Kotlin bench кладёт сырые прогоны (gitignored)
  results/  — метрики и графики (gitignored)
```

Почему так: корпус — переиспользуемая спецификация (любая memory-система может
прогнаться). Judge на Python — удобнее API-клиенты, pandas, matplotlib для
таблиц/графиков диплома. Kotlin делает «дорогую» часть (модель через JNI).

## Метрики (см. SCHEMA.md для деталей сценариев)

**Primary — end-to-end QA** (LoCoMo/LongMemEval-style):
- `recall` — одиночный факт извлечён и применён в ответе
- `multi_fact` — несколько фактов скомбинированы в ответе
- `abstention` — модель не выдумывает то, чего не было
- `update` — отвечает актуальным значением, не stale
- `temporal` — корректно понимает абсолютные даты

**Secondary — диагностика:**
- retrieval top-K hit (всплыл ли нужный факт)
- DB snapshot (extraction recall, conflict resolution, summary fidelity)

## Запуск

```bash
pip install -r requirements.txt
export DEEPSEEK_API_KEY=sk-...        # judge
python eval/run_eval.py --corpus corpus/scenarios --captures captures --out results
```

## Judge

DeepSeek V4-Pro (open weights, дёшево, воспроизводимо) как primary judge,
Claude Sonnet 4.5 как cross-check на подмножестве. См. `eval/judge.py`.
