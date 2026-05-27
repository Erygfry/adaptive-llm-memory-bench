# Corpus scenario schema

Спецификация формата dialog-сценариев для `adaptive-llm-bench`. Один сценарий = один JSON-файл в этой папке. Runner читает все `*.json`, прогоняет систему по каждому, собирает метрики.

См. [project_benchmark_design.md](../../../.claude/projects/c--Users-Petr-degree/memory/project_benchmark_design.md) для общей методологии.

---

## 1. Структура файла

```
corpus/
├── SCHEMA.md                    ← этот документ
├── examples/
│   └── 001-basic-extraction.json
├── extraction/                  ← сценарии фокусом на extraction quality
├── conflict/                    ← на conflict resolution
├── retrieval/                   ← на retrieval recall@K
├── multi-hop/                   ← на multi-hop precision
├── summary/                     ← на summary fidelity
└── memory-growth/               ← на динамику eviction'ов
```

Подпапки — для удобства разметки. Runner это игнорирует, ему всё равно где лежит файл — он смотрит на поле `category` внутри.

---

## 2. Top-level поля

| Поле | Тип | Обяз. | Описание |
|---|---|---|---|
| `id` | string | да | Уникальный slug в kebab-case, обычно совпадает с именем файла |
| `description` | string | да | Краткое описание что тестируется |
| `category` | enum | да | `extraction` / `conflict` / `retrieval` / `multi_hop` / `summary` / `memory_growth` / `mixed` |
| `language` | enum | да | `ru` / `en` / `mixed` |
| `starting_date` | string ISO date | да | Дата первого turn'а по умолчанию. Если turn не указывает свою `date` — наследует от предыдущего, а первый — от этого поля. См. §4 про advancing clock |
| `ram_tier` | enum | нет (default `MID`) | `HIGH` / `UPPER_MID` / `MID` / `LOW`. Для `mode: "force"` влияет только на summary cap / lastN cap / topK |
| `eviction_policy` | object | да | См. §3 |
| `dialog` | array of turns | да | См. §4 |
| `db_checkpoints` | array | нет (но обычно есть) | См. §5 |

---

## 3. eviction_policy

```json
"eviction_policy": {
  "mode": "force",
  "force_after_turns": [12, 17, 22, 27]
}
```

| Поле | Описание |
|---|---|
| `mode` | `force` — точное управление, `auto` — естественные триггеры по T_max, `hybrid` — force до точки потом auto (опционально, MVP не реализуется) |
| `force_after_turns` | Только для `mode: "force"`. Список **абсолютных номеров turn'ов** (любой роли). Eviction вызывается ПОСЛЕ обработки этого turn'а. Обычно указывают assistant turn — после генерации ответа, когда KV максимально заполнен |

Force eviction вызывает `EvictionEngine.runEviction(...)` напрямую — bypass'ит проверку T_max. Если на момент force истории мало (`allInKv.size <= effectiveLastN`), eviction корректно скипнется без побочных эффектов.

---

## 4. Dialog

Массив turn'ов. Каждый turn — JSON-объект.

```json
{
  "turn": 1,
  "role": "user",
  "content": "Привет, я Алексей."
}
```

| Поле | Тип | Описание |
|---|---|---|
| `turn` | int | 1-based порядковый номер |
| `role` | enum | `user` / `assistant` |
| `content` | string | Текст сообщения |
| `date` | string ISO date | Опционально. Дата на которой происходит этот turn. См. ниже про advancing clock |
| `checks` | object | Опционально, только для user turn'ов. См. §4.1 |

### Advancing clock между turn'ами

Дата advance'ится по сценарию чтобы симулировать диалог растянутый на дни/недели/месяцы. Это критично потому что:
- `LocalDate.now()` в extraction prompt управляет absolute-date math («послезавтра» → ISO)
- `nowSec()` в FactsRepository проставляет created_at / valid_from / last_access — без advance'а все факты по timestamps одного часа, periodic cleanup и last_access ordering невозможно тестировать

Правила:
- Если turn имеет `date` поле — runner вызывает `BenchClock.setScenarioTime(date)` перед обработкой
- Иначе — наследует дату от предыдущего turn'а
- Первый turn без `date` использует top-level `starting_date`
- Внутри одного turn'а время фиксировано — все операции (eviction, insert, retrieval, last_access update) видят одно и то же «now»

Пример:
```json
{"turn": 1,  "role": "user", "content": "...", "date": "2026-05-25"},
{"turn": 2,  "role": "assistant", "content": "..."},
{"turn": 12, "role": "assistant", "content": "..."},
{"turn": 13, "role": "user", "content": "...", "date": "2026-05-26"},   // ← next day
...
{"turn": 19, "role": "user", "content": "...", "date": "2026-06-15"}    // ← 3 weeks later
```

### 4.1 Inline checks на user turn

Каждое user-сообщение в проде триггерит retrieval перед composing prompt'а. Аннотация `checks` говорит runner'у: «после этого user message захвати retrieval результат и сравни с expected».

```json
{
  "turn": 17,
  "role": "user",
  "content": "Где я живу?",
  "checks": {
    "retrieval_top_k_includes": ["fact-location"],
    "retrieval_k": 6
  }
}
```

| Поле | Описание |
|---|---|
| `retrieval_top_k_includes` | Массив `id_alias`'ов фактов которые **обязательно** должны быть в top-K |
| `retrieval_top_k_includes_all` | Multi-hop: ВСЕ из массива должны быть в top-K (строгий AND) |
| `retrieval_k` | Опционально, override K (иначе используется `topKFactsFor(ram_tier)`) |

`id_alias` — string-ссылка на факт описанный в `db_checkpoints[*].expected_active_facts[*].id_alias`. Runner мэтчит реальные `fact.id` из БД к этим алиасам через LLM-judge (по `content_paraphrases`).

---

## 5. db_checkpoints

После каждого forced eviction'а (или в указанные моменты для auto-mode), runner снимает состояние БД и сравнивает с ожиданием.

```json
"db_checkpoints": [
  {
    "after_turn": 12,
    "expected_active_facts": [...],
    "expected_summary_includes": [...]
  }
]
```

| Поле | Описание |
|---|---|
| `after_turn` | Номер user turn'а после которого проверять. Обычно совпадает с `force_after_turns` |
| `expected_active_facts` | Список фактов которые должны быть `valid_to IS NULL` в БД на момент checkpoint'а. См. §5.1 |
| `expected_invalidated_aliases` | Опционально. Список `id_alias`'ов из ПРЕДЫДУЩИХ checkpoint'ов которые на этом checkpoint'е должны стать `valid_to != NULL` (т.е. UPDATE-superseded или удалены через cleanup) |
| `expected_summary_includes` | Список строк/концептов — rubric для LLM-judge. Каждый item: «упоминает ли summary этот аспект?». Score = fraction matched |

### 5.1 expected_active_facts entry

```json
{
  "id_alias": "fact-location",
  "content_paraphrases": [
    "User lives in Moscow",
    "User's residence: Moscow",
    "Пользователь живёт в Москве"
  ],
  "category": "personal_info",
  "importance_range": [5, 8],
  "event_date": null
}
```

| Поле | Описание |
|---|---|
| `id_alias` | Локальный к сценарию string-id для cross-reference из `checks` и `expected_invalidated_aliases` |
| `content_paraphrases` | Любая из этих формулировок считается матчем. LLM-judge берёт extracted fact и спрашивает: «эквивалентна ли любому из этих?» |
| `category` | Exact match (`personal_info` / `preference` / `goal` / `instruction` / `event` / `relationship`) |
| `importance_range` | `[min, max]`, оба inclusive. Модель может присвоить разную важность, разумный диапазон снижает false negatives |
| `event_date` | Только для `category: "event"`. ISO `YYYY-MM-DD`. Проверяется через date math на event_date поле в БД |

---

## 6. Полный пример

См. [examples/001-basic-extraction.json](examples/001-basic-extraction.json).

Покрывает:
- Извлечение personal_info (имя, город, работа)
- Absolute date extraction (событие «послезавтра» в `event_date`)
- Multi-hop retrieval (запрос требующий двух фактов)
- Conflict resolution (смена работы → UPDATE)
- Summary fidelity check

---

## 7. Что НЕ в схеме (пока)

- **End-to-end QA probes** — для Слоя 2 (LongMemEval-style). Будет отдельная схема `qa_probes` или отдельные сценарии.
- **Cross-chat scenarios** — пока один чат на сценарий.
- **`hybrid` eviction mode** — MVP только `force` и `auto`.
- **`expected_decisions_since_last_checkpoint`** — явное отслеживание ADD/UPDATE/NOOP. Сейчас выводится из diff'а (новые active + invalidated). Если понадобится точное логирование — добавим инструментирование EvictionEngine.
