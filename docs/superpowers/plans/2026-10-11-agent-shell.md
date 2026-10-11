# Агент: оболочка диалога — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Любой свободный текст игрока с `is_dev` уходит агенту: разговор живёт до конца вечера, за один ответ можно вызвать до восьми расчётов `/ask`, каждое число ответа сверено кодом со сводкой разбора и выходами расчётов, игрок получает одно из трёх состояний с подписью «Посчитано: …» и кнопкой «не то посчитано».

**Architecture:** Бот маршрутизирует текст (меню → начатый ввод → ручной ответ по эскалации → агент) и ставит задачу `agent`. Воркер собирает контекст (сводка последнего разбора через `NumberBook`), читает историю вечера из `sessions.agent_history`, зовёт фасад `llm(purpose="agent", tools, message_history, instructions)`, проверяет ответ чистыми функциями `explanation.agent_checks` (одна реализация для прода и evals), при провале — одна попытка переписать, затем пишет историю, журнал (`traces.spans`) и отдаёт сообщение, собранное `presentation`. Фасад начинает считать лимитер, строку `llm_calls` и таймаут на КАЖДЫЙ запрос к провайдеру (обёртка над моделью pydantic-ai), а не на прогон агента. Новый пакет `harness.agent` не конвейерный (ему нужна база, как `calcs`); чистые части — в `explanation`.

**Tech Stack:** Python 3.12, pydantic 2, pydantic-ai 2.38.0 (`WrapperModel`, `ModelMessagesTypeAdapter`, `ToolReturn(metadata=…)`, `FunctionModel` в тестах), SQLAlchemy 2 async + Postgres + alembic, aiogram (только роутер), httpx (`TelegramSender`), pytest.

**Spec:** [docs/superpowers/specs/2026-10-11-agent-shell-design.md](../specs/2026-10-11-agent-shell-design.md)

**Имплементатору.** Каждое утверждение плана о коде сверено с рабочим деревом на 2026-10-11 (коммит `7897bc7`), но дерево живое: **открой файл и сверь строку до правки**. Расхождение плана с кодом — вопрос оркестратору, а не повод «подогнать».

## Отклонения от спеки — на решение владельца

1. **`jobs.type` и `llm_calls.purpose` не переименовываются, а дополняются** (спека §8: «`question` → `agent`», «`question_answer` → `agent`»). Причина в самой спеке, §10: у игроков без `is_dev` `/ask` остаётся до открытия агента, а `_run_question` пишет задачи `question` и строки `question_answer` (`worker/pipeline.py:1234`, `calcs/routing.py:388-394`). Переименование уронило бы `/ask` CHECK-ограничением в день миграции. Миграция 0014 ДОБАВЛЯЕТ `agent` в оба набора; удаление `question`/`question_answer` — миграцией того же шага, что снимает `/ask` (Task 6, список «при открытии»).
2. **HH-скан не ставит указатель разбора** (спека §4: «последний разбор (рука или HH)»). Код: скан пишет `tournaments.scan_summary`, строки `analyses` не создаёт (`pipeline.py:610-731`, `AnalysesRepo.save` зовётся только в `:1114` и `:1190`). FK `sessions.last_analysis_id → analyses` указать на HH не может. Решение плана: разбор руки (`deep_dive`, скриншот) ставит указатель, законченный скан HH **сбрасывает** его в NULL — чтобы после нового файла агент не отвечал про старую руку. Про сам файл агент отвечает инструментами статистики (окно `evening`). Указатель на турнир — спека №3, если понадобится.
3. **«Ответ без вызова — отказ» читается по всему вечеру, а не по ходу.** Спека §5: «Ответ без вызова и без разбора — отказ». Но §6 разрешает числа «выходов всех расчётов, которые остались в истории вечера» — уточняющий вопрос («а почему так?») по прошлому расчёту без нового вызова иначе получал бы отказ. План: отказ, если нет сводки разбора И в оставшейся истории вечера (включая этот ход) нет ни одного расчёта.
4. **Сверка терминов — по меткам хода, а не «у того действия, о котором речь»** (спека §6). Какое действие имеет в виду фраза, код механически не знает. План: термин словаря допустим, если его метка есть среди меток контекста (метки классификатора всех точек разбора) или среди величин расчётов, оставшихся в истории (`FrequencyStat` выхода). Граница записывается в EVALS.md вместе с границами допущения и лучшей линии.
5. **Сверка чисел не разрешает числа из вопроса игрока.** Спека: «каждое число — из сводки разбора или выхода расчёта». «У меня 12 бб, что открывать?» → ответ с «12 бб» не пройдёт и уйдёт в состояние 2. План держит спеку буквально; если на стенде это главный источник отказов — решение владельца (вопрос 3 ниже).
6. **«Тихого закрытия» сессии в коде нет** (спека §8: «`/new` или тихое закрытие»). Сессию закрывает только `SessionsRepo.close_active` (`memory/repos.py:372-397`, единственная запись `closed_at` в `src/`). План стирает историю там.
7. **Лимит длины текста агенту** — бриф молчит; план берёт существующий `MAX_QUESTION_CHARS = 300` (`bot/handlers.py:177`): вопрос едет в промпт целиком, то же обоснование.

## Global Constraints

Действуют в КАЖДОЙ задаче.

- **Правило зависимостей (CLAUDE.md).** `contracts`, `parsers`, `normalizer`, `engine`, `analysis`, `explanation`, `presentation` не импортируют Телеграм, БД и `harness.platform`. Про инфраструктуру знают только `bot`, `worker`, `memory`, `platform` — и новые `calcs`, `agent` (не конвейерные, им нужна база). `explanation` не импортирует `presentation` (обратная зависимость уже есть: `presentation/messages.py:163` импортирует `explanation.hand_replay`).
- **Бот не импортирует `harness.agent` и `harness.calcs`** — иначе в процесс бота едут `pokerkit`/`eval7` (`test_the_bot_does_not_load_the_dictionary`, `tests/test_calcs.py:551`; `test_bot_image_does_not_import_calculation_stack`, `tests/test_entrypoints.py:173`).
- **Агент маршрутизирует и излагает, считает код.** Каждое число ответа — из сводки разбора (`NumberBook`) или выхода расчёта, оставшегося в истории вечера. Не прошло дважды — числа без прозы. Текст модели не правится «чтобы сошлось».
- **Никогда не выдумывать числа о деньгах (CLAUDE.md).** Расхождение с hand history — отказ с логом.
- **Судить решение против диапазона, а не против вскрытой карты (CLAUDE.md).** «Ошибка/неверно/неправильно» в ответе — нарушение (`error_words_in`).
- **Лучшая линия на флопе и тёрне не называется; «лик» — только из выхода `leaks`; выборка ниже 20 возможностей стата — упомянуть, не опираться** (спека §2). Две «20» разные: `MIN_OTHER_HANDS_FOR_FREQUENCIES` (`explanation/hand_replay.py:117`) считает руки, правило агента — возможности стата.
- **Потолок — 8 вызовов расчёта на ход.** Девятый получает отказ текстом.
- **Доступ v1 — только `players.is_dev`.** Остальным свободный текст отвечает как сегодня (`unknown_text_msg`), и у них остаётся `/ask`.
- **Промпты не коммитятся и не пушатся.** `src/harness/[^/]+/prompts/` закрыт `.githooks/pre-push:21`; тесты, читающие промпт, гейтятся маркером из `tests/conftest.py`. Текст системного промпта агента в этом плане НЕ приводится (план публичный).
- **Репозиторий публичный.** В тестах, докстрингах и коммитах — только синтетика (метки P1…P7, номера рук `SYN…`, `TM99` как уже принятый синтетический номер `tests/test_verdict_text.py:110`). Никаких реальных `TM…`.
- **Докстринг утверждает только закреплённое тестом.** Докстринги, которые задача делает ложью, правятся той же задачей — список в каждой задаче.
- **Тексты игроку — только в `presentation/messages.py`** (кнопки — `presentation/keyboards.py`), на «вы», единица «BB» латиницей вплотную к числу (SESSIONS_UX, «Единица»), знаменатель всегда, «наблюдений нет» вместо 0%.
- **Бюджет сообщения — 4096 символов** (`_TELEGRAM_TEXT_LIMIT`, `presentation/messages.py:1967`): режется проза модели (`_fitted`, `:2124`), не числа и не подпись.
- **Команды проверки:** `uv run pytest -q -ra` (смотреть строку `skipped`: без фикстур и промптов пропуски есть, их число не должно расти сверх новых гейтированных тестов этого плана), затем `uv run ruff check . && uv run pyright`.
- **Коммиты локальные, без пуша.** Сообщение по-русски, последняя строка `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. `git add` — явный список файлов, никогда `-A`.
- **Кто пишет:** задачи 1–4 — `implementer-core` (Opus), 5–6 — `implementer` (Sonnet), 7–8 — оркестратор. После каждой задачи с логикой — `reviewer` (Opus); бриф ревьюеру включает раздел «Review Focus» дословно.

## Review Focus

Пять классов входа, на которых спека кусается, а тесты легко промахиваются. Каждый закреплён тестом в задаче-владельце.

1. **Подпись меню против текста вопроса.** «Мои лики» — кнопка меню, «мои лики за вечер» — вопрос агенту; совпадение только точное (`menus._SCREENS.get(text)`). Тест: `test_a_menu_label_opens_its_screen_and_a_question_about_it_goes_to_the_agent` (Task 5).
2. **Обозначения карт против чисел.** `A5s`, `T9o`, `KQ`, `J♥️9♥️`, `K-7-2` — не числа; голая цифровая пара (`99`, `22`) — всегда число (решение R10, 2026-10-11: модель пишет пары мастями или словами); «9 BB», «22 раздачи», «9.5» — числа всегда. Тесты: `test_card_tokens_are_not_numbers`, `test_a_pocket_pair_is_a_card_only_when_the_hero_holds_it`, `test_board_ranks_are_cards_only_when_they_are_the_board` (Task 3).
3. **Обрезка истории уносит разрешённые числа — и не оставляет сироту.** Число из выброшенного хода перестаёт быть разрешённым; первый оставшийся запрос не начинается с ответа на вызов инструмента, которого уже нет. Тесты: `test_trimming_drops_whole_turns_and_their_numbers`, `test_trimming_leaves_no_orphan_tool_return` (Task 4).
4. **`/new` посреди хода и ожидание эскалации.** Ход, закончившийся после `/new`, не воскрешает историю закрытой сессии; текст во время ожидания ручного ответа по эскалации остаётся эскалации, не агенту. Тесты: `test_history_is_not_written_into_a_closed_session` (Task 2), `test_text_while_an_escalation_waits_for_a_number_stays_with_the_escalation` (Task 5).
5. **Ход агента, упёршийся в сроки.** Зависший запрос к провайдеру снимается таймаутом запроса (строка `llm_calls` закрыта `error`), а не дедлайном задачи; ход с тремя запросами пишет три строки. Тесты: `test_a_hung_provider_request_is_cut_by_the_request_timeout`, `test_a_turn_with_two_tool_calls_logs_three_requests` (Task 1). И не-`is_dev`: `test_a_player_without_access_gets_the_old_answer_and_no_job` (Task 5).

---

### Task 1: Фасад `llm()` — история, назначение `agent`, таймаут и лимитер на каждый запрос

**Агент:** `implementer-core`, затем `reviewer` (правка платформы: лимитер, учёт денег провайдера).

Сегодня `_attempt` держит слот лимитера, строку `llm_calls` и HTTP-ретрай вокруг ЦЕЛОГО `agent.run()` (`platform/llm.py:419-480`), поэтому ход с вызовами инструментов пишет одну строку (`test_a_tool_call_round_trip_logs_one_row`, `tests/test_question_routing.py:276-307`, его докстринг прямо называет это занижением). Правка переносит слот, строку, HTTP-ретрай и новый таймаут ВНУТРЬ одного запроса к провайдеру — через обёртку модели (`pydantic_ai.models.wrapper.WrapperModel`, сверено в установленной 2.38.0: `request(messages, model_settings, model_request_parameters) -> ModelResponse`). Схема-ретрай остаётся в `__call__` и по-прежнему перезапускает прогон целиком.

**Files:**
- Modify: `src/harness/platform/llm.py` — модульный докстринг (пункт 2, строки 12-17), `CallMeta` (126-142), `LLM.__call__` (239-330), `_resolve_model` (332-355), `_attempt` (357-487); новый класс `_MeteredModel`.
- Modify: `src/harness/platform/config.py` — `Config` (128-170): три поля с умолчаниями в КОНЦЕ (тесты строят `Config(...)` семью именованными аргументами — `tests/test_llm_facade.py:49`, `tests/test_question_routing.py:41`, `tests/test_worker_pipeline.py:103`; поля без умолчаний их уронили бы).
- Modify: `.env.example` — раздел «Провайдер LLM»: три закомментированные строки (правило шапки файла: необязательные — закомментированы).
- Modify: `tests/test_llm_facade.py` (новые тесты), `tests/test_question_routing.py:276-307` (переписать тест одной строки).

**Interfaces:**
- Produces:
  - `Config.llm_agent_model: str = "anthropic:claude-sonnet-5-5"`, `Config.llm_request_timeout_s: int = 60`, `Config.agent_history_max_tokens: int = 30_000`; `from_env` читает `LLM_AGENT_MODEL`, `LLM_REQUEST_TIMEOUT_S`, `AGENT_HISTORY_MAX_TOKENS` через `optional_env`/`optional_int`.
  - `LLM.__call__(purpose: Literal["vision_extract", "vision_extract_fallback", "verdict_text", "tournament_text", "question_answer", "agent"], schema: type[T], *, prompt: str, images: Sequence[bytes] = (), tools: Sequence[Tool[None]] = (), message_history: Sequence[ModelMessage] = (), instructions: str | None = None, trace_id: int) -> tuple[T, CallMeta]`.
  - `CallMeta.messages: tuple[ModelMessage, ...] = ()` — `result.all_messages()` удачного прогона (история + новые сообщения).
- Consumes: Task 2 (CHECK `llm_calls.purpose` с `agent`) — **до миграции тест с `purpose="agent"` упадёт на CHECK**. Поэтому тесты этой задачи на запись строк зовут существующие назначения (`verdict_text`, `question_answer`); строку `agent` закрепляет Task 4.

- [ ] **Step 1: Падающие тесты**

В `tests/test_llm_facade.py` (импорты добавить: `from pydantic_ai.messages import ModelRequest, ToolReturnPart`; `from pydantic_ai.tools import Tool`; `from pydantic_ai.models.test import TestModel` уже есть):

```python
async def test_a_turn_with_two_tool_calls_logs_three_requests(db_factory):
    """Ход с двумя вызовами инструмента — три обращения к провайдеру и три строки.

    Окно темпа `PgLimiter` считает по `llm_calls`; одна строка на прогон
    занижала нагрузку ровно там, где она растёт, — у агента с инструментами.
    """
    trace_id = await _make_trace_scope(db_factory, tg_user_id=21)
    calls = {"n": 0}

    async def echo(value: int) -> str:
        """Вернуть число строкой."""
        return str(value)

    def scripted(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls["n"] += 1
        if calls["n"] <= 2:
            return ModelResponse(parts=[ToolCallPart(tool_name="echo", args={"value": calls["n"]})])
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"text": "ok"})])

    llm = LLM(cfg, db_factory, model_override=FunctionModel(scripted))
    await llm("verdict_text", Out, prompt="п", tools=[Tool(echo, name="echo")], trace_id=trace_id)

    rows = await fetch_all(db_factory, "select purpose, status from llm_calls order by id")
    assert calls["n"] == 3
    assert rows == [("verdict_text", "ok")] * 3


async def test_a_hung_provider_request_is_cut_by_the_request_timeout(db_factory):
    """Зависший запрос снимается своим таймаутом, строка закрыта `error`.

    Прежде запрос ничем не ограничивался (тех-спека §7, «Известный дефект»), и
    бюджет задачи снимал попытку целиком вместе с уже посчитанным.
    """
    trace_id = await _make_trace_scope(db_factory, tg_user_id=22)

    async def hung(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        await asyncio.sleep(10)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"text": "ok"})])

    short = replace_cfg(llm_request_timeout_s=1)
    llm = LLM(short, db_factory, model_override=FunctionModel(hung))
    started = time.monotonic()
    with pytest.raises(LLMProviderError):
        await llm("verdict_text", Out, prompt="п", trace_id=trace_id)

    assert time.monotonic() - started < 5
    rows = await fetch_all(db_factory, "select status from llm_calls")
    assert rows == [("error",)]


async def test_the_history_reaches_the_model_and_comes_back_with_the_new_turn(db_factory):
    """`message_history` уходит модели, `CallMeta.messages` — история плюс ход."""
    trace_id = await _make_trace_scope(db_factory, tg_user_id=23)
    llm = LLM(cfg, db_factory, model_override=TestModel())
    _out, first = await llm("verdict_text", Out, prompt="первый", trace_id=trace_id)
    seen: list[int] = []

    def counting(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(len(messages))
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"text": "ok"})])

    llm2 = LLM(cfg, db_factory, model_override=FunctionModel(counting))
    _out2, second = await llm2(
        "verdict_text", Out, prompt="второй", message_history=first.messages, trace_id=trace_id
    )

    assert seen[0] > 1, "история не дошла до модели"
    assert len(second.messages) > len(first.messages)
    assert second.messages[: len(first.messages)] == first.messages


async def test_instructions_reach_the_model(db_factory):
    trace_id = await _make_trace_scope(db_factory, tg_user_id=24)
    seen: list[str | None] = []

    def peeking(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info.instructions)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"text": "ok"})])

    llm = LLM(cfg, db_factory, model_override=FunctionModel(peeking))
    await llm("verdict_text", Out, prompt="п", instructions="СВОДКА-СИНТ", trace_id=trace_id)

    assert seen and "СВОДКА-СИНТ" in (seen[0] or "")


def test_the_agent_purpose_takes_its_own_model():
    llm = LLM(replace_cfg(llm_agent_model="anthropic:agent-test"), None)  # type: ignore[arg-type]
    assert llm._resolve_model("agent") == "anthropic:agent-test"


def test_config_reads_the_agent_variables_with_defaults(monkeypatch):
    for name, value in (
        ("LLM_VERDICT_MODEL", "anthropic:v"),
        ("LLM_MAX_CONCURRENCY", "4"),
        ("LLM_MAX_PER_MINUTE", "20"),
        ("DATABASE_URL", "postgresql://x"),
        ("TELEGRAM_TOKEN", "t"),
    ):
        monkeypatch.setenv(name, value)
    for name in ("LLM_AGENT_MODEL", "LLM_REQUEST_TIMEOUT_S", "AGENT_HISTORY_MAX_TOKENS"):
        monkeypatch.delenv(name, raising=False)
    loaded = Config.from_env()
    assert loaded.llm_agent_model == "anthropic:claude-sonnet-5-5"
    assert loaded.llm_request_timeout_s == 60
    assert loaded.agent_history_max_tokens == 30_000
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT_S", "5")
    assert Config.from_env().llm_request_timeout_s == 5
```

Помощник рядом с `cfg` (строка ~57):

```python
def replace_cfg(**changes: object) -> Config:
    """Тот же тестовый `Config` с заменёнными полями — `Config` заморожен."""
    from dataclasses import replace

    return replace(cfg, **changes)  # type: ignore[arg-type]
```

`LLM(cfg, None)` в `test_the_agent_purpose_takes_its_own_model`: конструктор строит `PgLimiter(session_factory, ...)` — сверить, что `PgLimiter.__init__` (`platform/limiter.py`) фабрику только сохраняет и не трогает; если трогает — передать `db_factory` и сделать тест `async`.

В `tests/test_question_routing.py` тест `test_a_tool_call_round_trip_logs_one_row` (276-307) переименовать в `test_a_tool_call_round_trip_logs_a_row_per_request`; ожидание `[("question_answer", "ok")]` → `[("question_answer", "ok"), ("question_answer", "ok")]`; докстринг: «Вызов с инструментами — строка `llm_calls` на каждое обращение к провайдеру: выбор инструмента и ответ по его выходу — два запроса, и окно темпа `PgLimiter` считает оба».

- [ ] **Step 2: Запустить и убедиться, что падают**

Run: `uv run pytest tests/test_llm_facade.py tests/test_question_routing.py -q -ra` → FAIL (нет `message_history`, `instructions`, полей конфига; одна строка вместо трёх).

- [ ] **Step 3: Конфиг**

```python
    llm_verdict_model: str
    llm_max_concurrency: int
    llm_max_per_minute: int
    database_url: str
    telegram_token: str
    # Модель агента (спека агента №1, §9). Необязательна: умолчание — Sonnet 5.5.
    # Строка — тот же формат `"<provider>:<model>"`, что у остальных.
    llm_agent_model: str = "anthropic:claude-sonnet-5-5"
    # Таймаут ОДНОГО запроса к провайдеру, в секундах. До него запрос ничем не
    # ограничивался (тех-спека §7, «Известный дефект»).
    llm_request_timeout_s: int = 60
    # Потолок истории вечера агента в оценочных токенах (`agent.history.estimate_tokens`).
    agent_history_max_tokens: int = 30_000
```

В `from_env`:

```python
            llm_agent_model=optional_env("LLM_AGENT_MODEL", "anthropic:claude-sonnet-5-5"),
            llm_request_timeout_s=optional_int("LLM_REQUEST_TIMEOUT_S", 60),
            agent_history_max_tokens=optional_int("AGENT_HISTORY_MAX_TOKENS", 30_000),
```

**Идентификатор модели проверить до коммита.** В установленной pydantic-ai список известных имён кончается на `anthropic:claude-sonnet-5` (`models/_known_model_names.py`); `claude-sonnet-5-5` там нет. Строка не обязана быть в списке (это тип-подсказка), но сам идентификатор у провайдера — вопрос владельцу (открытый вопрос 1); умолчание правится одной строкой.

`.env.example`, после `LLM_VERDICT_MODEL=`:

```
# Модель агента (свободный текст после разбора). НЕОБЯЗАТЕЛЬНА: умолчание в коде —
# Sonnet 5.5. Раскомментировал — значит задал.
#LLM_AGENT_MODEL=

# Таймаут одного запроса к провайдеру, секунды. НЕОБЯЗАТЕЛЬНА, умолчание 60.
#LLM_REQUEST_TIMEOUT_S=

# Потолок истории вечера агента в оценочных токенах. НЕОБЯЗАТЕЛЬНА, умолчание 30000.
# При превышении выбрасываются самые старые ходы целиком, и числа из них перестают
# быть разрешёнными в ответе.
#AGENT_HISTORY_MAX_TOKENS=
```

Строку про `LLM_VERDICT_MODEL` («на неё падают все назначения, кроме зрения, в том числе `question_answer` — единственный вызов…») поправить: «кроме зрения и агента».

- [ ] **Step 4: Обёртка модели и новый `_attempt`**

```python
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings


class _MeteredModel(WrapperModel):
    """Модель, у которой КАЖДЫЙ запрос к провайдеру — свой слот лимитера, своя
    строка `llm_calls`, свой таймаут и свои HTTP-ретраи.

    Прогон агента с инструментами — несколько запросов; считать его одним значило
    занижать окно темпа `PgLimiter` ровно там, где нагрузка растёт
    (`test_a_turn_with_two_tool_calls_logs_three_requests`). Таймаут обёрнут
    вокруг одного запроса, а не прогона: зависший запрос снимается сам и не
    съедает бюджет задачи (`test_a_hung_provider_request_is_cut_by_the_request_timeout`).

    `last_call_id` — строка последнего запроса: `_attempt` переписывает ей статус
    на `schema_error`, если ответ этого запроса не прошёл схему уже после того,
    как сам запрос удался.
    """

    def __init__(
        self,
        wrapped: Model | str,
        *,
        llm: LLM,
        trace_id: int,
        purpose: str,
        provider: str,
        model_name: str,
        deadline: float,
    ) -> None:
        super().__init__(wrapped)
        self._llm = llm
        self._trace_id = trace_id
        self._purpose = purpose
        self._provider = provider
        self._model_name = model_name
        self._deadline = deadline
        self.last_call_id: int | None = None

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        llm = self._llm
        last_http_error: ModelHTTPError | None = None
        for http_attempt in range(_MAX_HTTP_ATTEMPTS):
            async with llm._limiter.slot(deadline=self._deadline):
                call_id = await llm._log_started(
                    trace_id=self._trace_id,
                    provider=self._provider,
                    model=self._model_name,
                    purpose=self._purpose,
                )
                self.last_call_id = call_id
                started = time.monotonic()
                try:
                    async with asyncio.timeout(llm._cfg.llm_request_timeout_s):
                        response = await self.wrapped.request(
                            messages, model_settings, model_request_parameters
                        )
                except asyncio.CancelledError:
                    await asyncio.shield(llm._log_finished(call_id, status="error"))
                    raise
                except TimeoutError as exc:
                    await llm._log_finished(call_id, status="error")
                    raise LLMProviderError(
                        f"{self._purpose}: запрос к провайдеру не уложился в "
                        f"{llm._cfg.llm_request_timeout_s} с"
                    ) from exc
                except ModelHTTPError as exc:
                    await llm._log_finished(call_id, status="error")
                    last_http_error = exc
                    if http_attempt + 1 >= _MAX_HTTP_ATTEMPTS or not _is_retryable_http_error(exc):
                        raise LLMProviderError(
                            f"{self._purpose}: провайдер вернул {exc.status_code} после "
                            f"{http_attempt + 1} попыт(ок)"
                        ) from exc
                except Exception:
                    await llm._log_finished(call_id, status="error")
                    raise
                else:
                    usage = response.usage
                    await llm._log_finished(
                        call_id,
                        status="ok",
                        tokens_in=usage.input_tokens,
                        tokens_out=usage.output_tokens,
                        cost=usage.cost,
                        latency_ms=round((time.monotonic() - started) * 1000),
                        served_by=_served_by(response),
                    )
                    return response
            await _sleep(_backoff_delay(http_attempt))
        raise LLMProviderError(
            f"{self._purpose}: провайдер недоступен после {_MAX_HTTP_ATTEMPTS} попыт(ок)"
        ) from last_http_error
```

Порядок `except` важен: `TimeoutError` из `asyncio.timeout` — до общего `Exception`; `CancelledError` — первым (тот же довод, что сегодня в `_attempt:428-438`).

`usage.cost` у `RequestUsage` есть (`pydantic_ai/usage.py:131`), но заполняет ли его библиотека на этом месте или позже, в графе агента, — сверить на живом прогоне `TestModel`; если `None` — колонка `llm_calls.cost` у строки пуста (она nullable), а `CallMeta.cost_usd` по-прежнему берётся из `result.usage.cost` прогона. Записать в отчёт, что увидел.

`__call__`: в сигнатуру — `message_history`, `instructions` (см. Interfaces), в `Literal` — `"agent"`. Агент строится на обёртке, дедлайн по-прежнему считается один раз (под локальным семафором) и уходит в обёртку:

```python
        model: Model | str = (
            self._model_override if self._model_override is not None else self._resolve_model(purpose)
        )
        provider, model_name = _describe_model(model)
        user_prompt = [...]  # как сейчас
        async with self._local_semaphore:
            deadline = time.monotonic() + _limiter_module._ACQUIRE_TIMEOUT_S
            metered = _MeteredModel(
                model, llm=self, trace_id=trace_id, purpose=purpose,
                provider=provider, model_name=model_name, deadline=deadline,
            )
            agent: Agent[None, T] = Agent(
                metered, output_type=schema, retries=0, tools=tools,
                deps_type=type(None), instructions=instructions,
            )
            last_schema_error: UnexpectedModelBehavior | None = None
            for _schema_attempt in range(_MAX_SCHEMA_ATTEMPTS):
                try:
                    return await self._attempt(agent, metered, user_prompt, message_history, model_name)
                except UnexpectedModelBehavior as exc:
                    last_schema_error = exc
            raise LLMSchemaError(...)  # как сейчас
```

`_attempt` сжимается до одного прогона:

```python
    async def _attempt(
        self,
        agent: Agent[None, T],
        metered: _MeteredModel,
        user_prompt: list[str | BinaryContent],
        message_history: Sequence[ModelMessage],
        model_name: str,
    ) -> tuple[T, CallMeta]:
        """Один прогон агента. Слоты, строки `llm_calls`, таймаут и HTTP-ретраи —
        внутри `_MeteredModel.request`, на каждый запрос. Здесь — только схема:
        ответ, не прошедший `schema`, переписывает статус строки ПОСЛЕДНЕГО
        запроса на `schema_error` (запрос удался, ответ — нет) и уходит в
        `__call__`, который решает про схема-ретрай."""
        started = time.monotonic()
        try:
            result = await agent.run(user_prompt, message_history=list(message_history) or None)
        except UnexpectedModelBehavior:
            if metered.last_call_id is not None:
                await self._log_status(metered.last_call_id, "schema_error")
            raise
        usage = result.usage
        return result.output, CallMeta(
            model=result.response.model_name or model_name,
            tokens_in=usage.input_tokens or 0,
            tokens_out=usage.output_tokens or 0,
            cost_usd=float(usage.cost) if usage.cost is not None else None,
            latency_ms=round((time.monotonic() - started) * 1000),
            messages=tuple(result.all_messages()),
        )

    async def _log_status(self, call_id: int, status: Literal["schema_error"]) -> None:
        """Переписать только статус — токены и цена удачного запроса остаются."""
        async with self._session_factory() as session:
            await session.execute(update(LlmCall).where(LlmCall.id == call_id).values(status=status))
            await session.commit()
```

`CallMeta`: добавить последним полем `messages: tuple[ModelMessage, ...] = ()` (frozen + slots + умолчание — допустимо, т.к. поле последнее). `CallMeta.latency_ms` теперь — длительность прогона целиком (в докстринге `CallMeta` сказать это).

`_resolve_model`: перед `return self._cfg.llm_verdict_model` — `if purpose == "agent": return self._cfg.llm_agent_model`.

`instructions=instructions` в `Agent(...)`: проверить в установленной библиотеке, что `Agent.__init__` принимает `instructions: str | None` (сверено: `agent/__init__.py:519`, `instructions: AgentInstructions[AgentDepsT] = None`). Инструкции не переигрываются из истории: модель берёт инструкции текущего запроса (`models/__init__.py:952-985`), поэтому сводка разбора в инструкциях каждого хода не копится в истории.

- [ ] **Step 5: Запустить файл и весь набор фасада**

Run: `uv run pytest tests/test_llm_facade.py tests/test_question_routing.py -q -ra` → PASS. Особо смотреть, что старые тесты фасада не поменяли ожиданий: `test_retry_on_schema_error_then_fail` (две `schema_error`), `test_backoff_*` (порядок `enter/exit/sleep`), `test_call_shares_one_deadline_across_*` (один дедлайн), `test_cancellation_finalizes_row_and_repropagates`, `test_unexpected_exception_finalizes_row_as_error`. Падение любого из них — не повод править тест: это значит, что обёртка поменяла наблюдаемое поведение; разобраться и доложить.

Run: `uv run pytest -q -ra` → PASS; `uv run ruff check . && uv run pyright` → чисто.

- [ ] **Step 6: Коммит**

```bash
git add src/harness/platform/llm.py src/harness/platform/config.py .env.example tests/test_llm_facade.py tests/test_question_routing.py
git commit -m "$(cat <<'EOF'
Фасад модели: история, назначение agent, таймаут и учёт на каждый запрос к провайдеру

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

**Докстринги, которые становятся ложью:**
- `platform/llm.py:289-294` (докстринг `__call__`, абзац про `tools`): «строка `llm_calls` на них по-прежнему одна, закреплено `test_a_tool_call_round_trip_logs_one_row`» → «на каждый запрос своя строка (`test_a_turn_with_two_tool_calls_logs_three_requests`, `test_a_tool_call_round_trip_logs_a_row_per_request`)».
- `platform/llm.py:12-17` (модульный, пункт 2): «"Попытка" здесь — именно одно HTTP-обращение» — остаётся верным, дописать «включая каждый запрос внутри прогона с инструментами».
- `platform/llm.py:405-417` (докстринг `_attempt`): «`agent.run()` им НЕ оборачивается… таймаут самого запроса к провайдеру не входит в периметр» — удаляется вместе с телом; эквивалент про дедлайн переносится в докстринг `_MeteredModel`.
- `platform/llm.py:341-343` (`_resolve_model`): «на неё падает `question_answer` — единственный вызов модели, который конвейер сегодня делает» → «…кроме агента, у которого своя модель».
- `worker/pipeline.py:22-25` (модульный, «Дедлайн задачи»): «отдельного таймаута на LLM-запрос по-прежнему нет» → «у каждого запроса к провайдеру свой таймаут (`LLM_REQUEST_TIMEOUT_S`), бюджет задачи остаётся сверху».
- `worker/pipeline.py:138-141` (комментарий к `"question"` в `_JOB_DEADLINE_S`): ссылка на `test_a_tool_call_round_trip_logs_one_row` → новое имя.
- `.env.example`, абзац про `LLM_VERDICT_MODEL` (см. Step 3).

---

### Task 2: Миграция 0014, модели, репозитории, квота

**Агент:** `implementer-core`, затем `reviewer` (схема и данные).

**Files:**
- Create: `migrations/versions/0014_agent_shell.py`
- Modify: `src/harness/memory/models.py` — `Session` (141-155), `EvalCase` (496-520, CHECK 500-508, `hand_id` 513), `Job` CHECK `type_allowed` (539-542), `LlmCall` CHECK `purpose_allowed` (609-619).
- Modify: `src/harness/memory/repos.py` — `SessionsRepo` (229-397, `close_active` 372-397), `AnalysesRepo` (724-853), `JobsRepo` (927-992), `QUOTA_INTERACTIVE_JOB_TYPES` и комментарий (1071-1078), `EvalCasesRepo.add` (1192-1206).
- Modify: `src/harness/platform/queue.py` — `_DEFAULT_PRIORITY_BY_TYPE` (68-74) и модульный докстринг (28-38).
- Test: `tests/test_memory.py`, `tests/test_queue.py`.

**Interfaces:**
- Produces:
  - колонки `sessions.agent_history JSONB NULL`, `sessions.last_analysis_id BIGINT NULL FK → analyses.id` (имя FK по соглашению `_NAMING_CONVENTION`: `fk_sessions_last_analysis_id_analyses`);
  - CHECK: `jobs.type` += `'agent'`; `llm_calls.purpose` += `'agent'`; `eval_cases.kind` += `'agent_routing'`; `eval_cases.source` += `'agent_miss_button'`; `eval_cases.hand_id` → NULL;
  - `@dataclass(frozen=True, slots=True) class AgentState: history: list[dict[str, Any]] | None; last_analysis_id: int | None`
  - `SessionsRepo.agent_state(session_id: int) -> AgentState`
  - `SessionsRepo.save_agent_history(session_id: int, history: list[dict[str, Any]]) -> bool` — `False`, если сессия уже закрыта (ничего не записано)
  - `SessionsRepo.set_last_analysis(session_id: int, analysis_id: int | None) -> None`
  - `SessionsRepo.close_active` дополнительно ставит `agent_history = NULL`
  - `AnalysesRepo.get(analysis_id: int) -> AnalysisRecord | None`
  - `@dataclass(frozen=True, slots=True) class AgentTurnRecord: job_id: int; trace_id: int; question: str; turn: dict[str, Any]`
  - `JobsRepo.agent_turn(job_id: int, player_id: int) -> AgentTurnRecord | None`
  - `EvalCasesRepo.add(*, kind: str, hand_id: int | None, ground_truth: dict, source: str, field: str | None = None) -> int`
  - `QUOTA_INTERACTIVE_JOB_TYPES = ("deep_dive", "screenshot_analyze", "agent")`
  - `_DEFAULT_PRIORITY_BY_TYPE["agent"] = 100`

- [ ] **Step 1: Падающие тесты** (`tests/test_memory.py`; импорты — `SessionsRepo`, `AnalysesRepo`, `JobsRepo`, `EvalCasesRepo`, `QuotaRepo` из `harness.memory.repos`, `Job`, `Trace` из `harness.memory.models`; помощники `_make_session`, `make_min_raw` уже в файле — сверить их сигнатуры)

```python
async def test_the_evening_history_is_written_read_and_erased_by_new(db):
    session_id = await _make_session(db)
    repo = SessionsRepo(db)
    assert (await repo.agent_state(session_id)).history is None

    assert await repo.save_agent_history(session_id, [{"kind": "request", "parts": []}]) is True
    assert (await repo.agent_state(session_id)).history == [{"kind": "request", "parts": []}]

    row = await db.get(SessionRow, session_id)
    await repo.close_active(row.player_id)
    assert (await repo.agent_state(session_id)).history is None


async def test_history_is_not_written_into_a_closed_session(db):
    """Ход, закончившийся после `/new`, не воскрешает историю закрытого вечера."""
    session_id = await _make_session(db)
    repo = SessionsRepo(db)
    row = await db.get(SessionRow, session_id)
    await repo.close_active(row.player_id)

    assert await repo.save_agent_history(session_id, [{"kind": "request", "parts": []}]) is False
    assert (await repo.agent_state(session_id)).history is None


async def test_the_last_analysis_pointer_is_set_and_reset(db):
    session_id = await _make_session(db)
    raw = RawHand.model_validate(make_min_raw())
    hand_id = await HandsRepo(db).save_raw(session_id=session_id, raw=raw)
    analysis_id = await AnalysesRepo(db).save(
        hand_id=hand_id, result=AnalysisResult(hand_no="SYN1", points=[]), decision_points=[]
    )
    repo = SessionsRepo(db)
    await repo.set_last_analysis(session_id, analysis_id)
    assert (await repo.agent_state(session_id)).last_analysis_id == analysis_id
    assert (await AnalysesRepo(db).get(analysis_id)).hand_id == hand_id
    await repo.set_last_analysis(session_id, None)
    assert (await repo.agent_state(session_id)).last_analysis_id is None


async def test_an_agent_miss_is_stored_without_a_hand(db):
    case_id = await EvalCasesRepo(db).add(
        kind="agent_routing",
        hand_id=None,
        ground_truth={"question": "как часто я вхожу в банк", "calls": []},
        source="agent_miss_button",
    )
    row = await db.get(EvalCase, case_id)
    assert row.hand_id is None and row.kind == "agent_routing"


async def test_an_agent_turn_is_found_only_for_its_player(db):
    session_id = await _make_session(db)
    row = await db.get(SessionRow, session_id)
    job = Job(type="agent", player_id=row.player_id, session_id=session_id, payload={"text": "вопрос"})
    db.add(job)
    await db.flush()
    db.add(Trace(job_id=job.id, spans=[{"name": "agent_turn", "state": "prose", "calls": []}]))
    await db.flush()

    found = await JobsRepo(db).agent_turn(job.id, row.player_id)
    assert found is not None and found.question == "вопрос" and found.turn["state"] == "prose"
    assert await JobsRepo(db).agent_turn(job.id, row.player_id + 1) is None


async def test_an_agent_message_spends_the_daily_quota_and_a_failed_one_does_not(db):
    session_id = await _make_session(db)
    row = await db.get(SessionRow, session_id)
    player = await db.get(Player, row.player_id)
    player.quota_daily = 2
    db.add(Job(type="agent", player_id=player.id, session_id=session_id, payload={}, status="done"))
    db.add(Job(type="agent", player_id=player.id, session_id=session_id, payload={}, status="failed"))
    await db.flush()

    quota = await QuotaRepo(db).check(player.id)
    assert (quota.left, quota.total) == (1, 2)
```

`SessionRow` в тесте — `from harness.memory.models import Session as SessionRow`. Тест compare-metadata (`test_the_models_say_the_same_as_the_migrations`, `tests/test_memory.py:2135`) уже есть и закрепит совпадение моделей с миграцией — новых тестов на это не нужно.

`tests/test_queue.py` — `test_an_agent_job_is_as_urgent_as_a_hand_analysis`: `enqueue(type="agent", …)` → `priority == 100` (помощники scope — сверить, какие уже есть в файле).

Миграция вперёд-назад на базе с данными — по образцу `test_migration_0013_turns_colour_keys_into_colours_of_their_owners` (`tests/test_memory.py:1973`), свой контейнер, ревизии номерами:

```python
def test_migration_0014_adds_the_agent_and_rolls_back_without_its_rows():
    """Вперёд: колонки и новые значения CHECK. Назад: строки агента удалены,
    иначе старые CHECK их не примут; `eval_cases` без руки удалены по той же причине."""
    from alembic import command
    from sqlalchemy import create_engine
    from testcontainers.community.postgres import PostgresContainer

    from tests.conftest import alembic_config

    with PostgresContainer("postgres:16-alpine") as container:
        dsn = container.get_connection_url(driver="psycopg")
        config = alembic_config(dsn)
        command.upgrade(config, "0014")
        engine = create_engine(dsn)
        try:
            with engine.begin() as conn:
                player = conn.execute(
                    text("insert into players (tg_user_id) values (9201) returning id")
                ).scalar_one()
                session_id = conn.execute(
                    text(
                        "insert into sessions (player_id, started_at, title, agent_history) "
                        "values (:p, now(), 'вечер', '[]'::jsonb) returning id"
                    ),
                    {"p": player},
                ).scalar_one()
                job = conn.execute(
                    text(
                        "insert into jobs (type, payload, session_id, player_id) "
                        "values ('agent', '{}'::jsonb, :s, :p) returning id"
                    ),
                    {"s": session_id, "p": player},
                ).scalar_one()
                trace = conn.execute(
                    text("insert into traces (job_id) values (:j) returning id"), {"j": job}
                ).scalar_one()
                conn.execute(
                    text(
                        "insert into llm_calls (trace_id, provider, model, purpose) "
                        "values (:t, 'anthropic', 'm', 'agent')"
                    ),
                    {"t": trace},
                )
                conn.execute(
                    text(
                        "insert into eval_cases (kind, hand_id, ground_truth, source) "
                        "values ('agent_routing', null, '{}'::jsonb, 'agent_miss_button')"
                    )
                )
            command.downgrade(config, "0013")
            with engine.begin() as conn:
                assert conn.execute(text("select count(*) from jobs")).scalar_one() == 0
                assert conn.execute(text("select count(*) from llm_calls")).scalar_one() == 0
                assert conn.execute(text("select count(*) from eval_cases")).scalar_one() == 0
                columns = {
                    row[0]
                    for row in conn.execute(
                        text(
                            "select column_name from information_schema.columns "
                            "where table_name = 'sessions'"
                        )
                    )
                }
                assert "agent_history" not in columns and "last_analysis_id" not in columns
        finally:
            engine.dispose()
```

- [ ] **Step 2: Запустить** → FAIL (нет колонок, значений CHECK, методов).

- [ ] **Step 3: Миграция**

```python
"""агент: история вечера, указатель разбора, тип задачи и назначение вызова

Сессия получает историю диалога с агентом (`agent_history`, сериализованные
сообщения pydantic-ai) и указатель на последний разбор вечера
(`last_analysis_id`). `jobs.type` и `llm_calls.purpose` получают `agent` —
ДОБАВЛЕНИЕМ: `question`/`question_answer` остаются, пока `/ask` жив у игроков
без доступа к агенту (спека агента №1, §10). `eval_cases` принимает строку без
руки — «не то посчитано» на вопрос без разбора.

Откат удаляет строки, которых старые CHECK не примут (тот же приём, что 0009).

Revision ID: 0014
Revises: 0013
Create Date: 2026-10-11 12:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = '0014'
down_revision: str | Sequence[str] | None = '0013'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_JOB_TYPES_OLD = "type IN ('screenshot_analyze', 'hh_scan', 'deep_dive', 'eval_run', 'question')"
_JOB_TYPES_NEW = (
    "type IN ('screenshot_analyze', 'hh_scan', 'deep_dive', 'eval_run', 'question', 'agent')"
)
_PURPOSE_OLD = (
    "purpose IN ('vision_extract', 'vision_extract_fallback', 'verdict_text', "
    "'tournament_text', 'question_answer')"
)
_PURPOSE_NEW = (
    "purpose IN ('vision_extract', 'vision_extract_fallback', 'verdict_text', "
    "'tournament_text', 'question_answer', 'agent')"
)
_KIND_OLD = "kind IN ('vision_field', 'verdict_confirm', 'verdict_dispute')"
_KIND_NEW = "kind IN ('vision_field', 'verdict_confirm', 'verdict_dispute', 'agent_routing')"
_SOURCE_OLD = "source IN ('escalation', 'disagree_button', 'manual')"
_SOURCE_NEW = "source IN ('escalation', 'disagree_button', 'manual', 'agent_miss_button')"


def upgrade() -> None:
    op.add_column('sessions', sa.Column('agent_history', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column('sessions', sa.Column('last_analysis_id', sa.BigInteger(), nullable=True))
    op.create_foreign_key(
        op.f('fk_sessions_last_analysis_id_analyses'), 'sessions', 'analyses', ['last_analysis_id'], ['id']
    )
    op.drop_constraint('type_allowed', 'jobs', type_='check')
    op.create_check_constraint('type_allowed', 'jobs', _JOB_TYPES_NEW)
    op.drop_constraint('purpose_allowed', 'llm_calls', type_='check')
    op.create_check_constraint('purpose_allowed', 'llm_calls', _PURPOSE_NEW)
    op.drop_constraint('kind_allowed', 'eval_cases', type_='check')
    op.create_check_constraint('kind_allowed', 'eval_cases', _KIND_NEW)
    op.drop_constraint('source_allowed', 'eval_cases', type_='check')
    op.create_check_constraint('source_allowed', 'eval_cases', _SOURCE_NEW)
    op.alter_column('eval_cases', 'hand_id', existing_type=sa.BigInteger(), nullable=True)


def downgrade() -> None:
    op.execute("DELETE FROM eval_cases WHERE hand_id IS NULL OR kind = 'agent_routing' "
               "OR source = 'agent_miss_button'")
    op.alter_column('eval_cases', 'hand_id', existing_type=sa.BigInteger(), nullable=False)
    op.drop_constraint('source_allowed', 'eval_cases', type_='check')
    op.create_check_constraint('source_allowed', 'eval_cases', _SOURCE_OLD)
    op.drop_constraint('kind_allowed', 'eval_cases', type_='check')
    op.create_check_constraint('kind_allowed', 'eval_cases', _KIND_OLD)
    op.execute(
        "DELETE FROM llm_calls WHERE purpose = 'agent' OR trace_id IN "
        "(SELECT id FROM traces WHERE job_id IN (SELECT id FROM jobs WHERE type = 'agent'))"
    )
    op.execute("DELETE FROM traces WHERE job_id IN (SELECT id FROM jobs WHERE type = 'agent')")
    op.execute("DELETE FROM jobs WHERE type = 'agent'")
    op.drop_constraint('purpose_allowed', 'llm_calls', type_='check')
    op.create_check_constraint('purpose_allowed', 'llm_calls', _PURPOSE_OLD)
    op.drop_constraint('type_allowed', 'jobs', type_='check')
    op.create_check_constraint('type_allowed', 'jobs', _JOB_TYPES_OLD)
    op.drop_constraint(op.f('fk_sessions_last_analysis_id_analyses'), 'sessions', type_='foreignkey')
    op.drop_column('sessions', 'last_analysis_id')
    op.drop_column('sessions', 'agent_history')
```

Имена CHECK в `op.drop_constraint` — короткие (`'type_allowed'`), как в 0009 (соглашение имён дособерёт `ck_jobs_type_allowed`); сверить по 0009, что так и работает, а не по памяти. Старое `_PURPOSE_OLD` взять из модели (`models.py:617-618` — там уже `tournament_text`, миграция 0010), не из 0009.

- [ ] **Step 4: Модели**

`Session` — после `title`:

```python
    # История диалога с агентом за вечер: сообщения pydantic-ai,
    # `ModelMessagesTypeAdapter` (спека агента №1, §8). Уходит с закрытием
    # сессии (`SessionsRepo.close_active`).
    agent_history: Mapped[Any | None] = mapped_column(JSONB)
    # Последний разбор руки этого вечера — о нём отвечает агент. NULL — разбора
    # нет либо после него пришёл файл HH (`worker.pipeline`, план агента, отклонение 2).
    last_analysis_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("analyses.id"))
```

`EvalCase.hand_id` → `Mapped[int | None] = mapped_column(BigInteger, ForeignKey("hands.id"))`; CHECK-строки — как `_KIND_NEW`/`_SOURCE_NEW`. `Job` CHECK — `_JOB_TYPES_NEW`; `LlmCall` CHECK — `_PURPOSE_NEW`, комментарий над ним дополнить строкой про `agent`.

- [ ] **Step 5: Репозитории и очередь**

```python
@dataclass(frozen=True, slots=True)
class AgentState:
    """История вечера и указатель разбора — то, с чем воркер начинает ход агента."""

    history: list[dict[str, Any]] | None
    last_analysis_id: int | None
```

В `SessionsRepo`:

```python
    async def agent_state(self, session_id: int) -> AgentState:
        row = (
            await self.db.execute(
                select(SessionRow.agent_history, SessionRow.last_analysis_id).where(
                    SessionRow.id == session_id
                )
            )
        ).one()
        return AgentState(history=row.agent_history, last_analysis_id=row.last_analysis_id)

    async def save_agent_history(self, session_id: int, history: list[dict[str, Any]]) -> bool:
        """Записать историю, только пока сессия открыта.

        Ход агента может закончиться после `/new`: тогда сессия уже закрыта, её
        история стёрта, и запись вернула бы её обратно
        (`test_history_is_not_written_into_a_closed_session`).
        """
        result = await self.db.execute(
            update(SessionRow)
            .where(SessionRow.id == session_id, SessionRow.closed_at.is_(None))
            .values(agent_history=history)
            .execution_options(synchronize_session=False)
            .returning(SessionRow.id)
        )
        return result.first() is not None

    async def set_last_analysis(self, session_id: int, analysis_id: int | None) -> None:
        await self.db.execute(
            update(SessionRow)
            .where(SessionRow.id == session_id)
            .values(last_analysis_id=analysis_id)
            .execution_options(synchronize_session=False)
        )
```

`close_active`: `.values(closed_at=datetime.now(UTC), agent_history=None)`; докстринг дополнить «история диалога с агентом уходит вместе с вечером (спека агента №1, §8)».

`AnalysesRepo.get`:

```python
    async def get(self, analysis_id: int) -> AnalysisRecord | None:
        """Разбор по его номеру — через `get_by_hand`, чтобы точки собирались одним путём."""
        hand_id = await self.db.scalar(select(Analysis.hand_id).where(Analysis.id == analysis_id))
        return None if hand_id is None else await self.get_by_hand(hand_id)
```

`JobsRepo.agent_turn` (импорт `Trace` из `harness.memory.models` в repos — сверить, что его там нет, и добавить):

```python
    async def agent_turn(self, job_id: int, player_id: int) -> AgentTurnRecord | None:
        """Ход агента по номеру задачи из кнопки — только этого игрока.

        Ход берётся из последнего трейса задачи (у повторной попытки трейс свой):
        запись `agent_turn` в `spans` кладёт воркер (`pipeline._run_agent`).
        """
        job = await self.db.scalar(
            select(Job).where(Job.id == job_id, Job.player_id == player_id, Job.type == "agent")
        )
        if job is None:
            return None
        trace = await self.db.scalar(
            select(Trace).where(Trace.job_id == job_id).order_by(Trace.id.desc()).limit(1)
        )
        if trace is None:
            return None
        turn = next((s for s in trace.spans or [] if s.get("name") == "agent_turn"), None)
        if turn is None:
            return None
        return AgentTurnRecord(
            job_id=job.id, trace_id=trace.id, question=str((job.payload or {}).get("text", "")), turn=turn
        )
```

`EvalCasesRepo.add`: `hand_id: int | None`.

Квота: `QUOTA_INTERACTIVE_JOB_TYPES = ("deep_dive", "screenshot_analyze", "agent")`. Комментарий 1071-1077 переписать: `agent` входит — одно сообщение агенту, одна единица (спека агента №1, §9), подпись счётчика становится «запросов» (Task 5); `question` по-прежнему не входит — `/ask` живёт до открытия агента у игроков без доступа и подписи «запросов» у них не меняет смысла, но решение «вопрос бесплатен» принято раньше и этим планом не пересматривается.

Очередь: `"agent": 100,` в `_DEFAULT_PRIORITY_BY_TYPE`; модульный докстринг (28-38) — абзац про `question` дополнить `agent` («сообщение агенту — тоже 100»).

- [ ] **Step 6: Запустить**

Run: `uv run pytest tests/test_memory.py tests/test_queue.py tests/test_bot_handlers.py -q -ra` → PASS. `test_quota_counts_only_interactive_jobs` (`tests/test_bot_handlers.py:849`) — сверить, что он не перечисляет типы буквально; если перечисляет — дополнить `agent`.
Run: `uv run pytest -q -ra`, `uv run ruff check . && uv run pyright` → чисто.

- [ ] **Step 7: Коммит**

```bash
git add migrations/versions/0014_agent_shell.py src/harness/memory/models.py src/harness/memory/repos.py src/harness/platform/queue.py tests/test_memory.py tests/test_queue.py
git commit -m "$(cat <<'EOF'
Схема агента: история вечера и указатель разбора в сессии, тип задачи agent, «не то посчитано» без руки

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

**Докстринги, которые становятся ложью:**
- `memory/repos.py:1071-1077` (комментарий квоты) — см. Step 5.
- `memory/repos.py:1187` (`EvalCasesRepo`): «подтверждения/возражения игрока, эскалации vision» → дописать «и „не то посчитано“ агента».
- `memory/models.py:497` (`EvalCase`): «Копится сам — этажи 2 и 4» → «этажи 2, 4 и 5».
- `platform/queue.py:28-38` — см. Step 5.
- `bot/handlers.py:701-704` (докстринг `handle_question_command`, «Дневной лимит разборов вопрос не тратит… счётчик подписан «разборов X/Y»») — правится в Task 5 вместе с подписью.

---

### Task 3: Чистые проверки ответа агента и сводка разбора (`explanation`)

**Агент:** `implementer-core`, затем `reviewer` (верность чисел — граница системы).

Одна реализация для прода и evals (EVALS.md: «вторая реализация "для evals" разошлась бы с первой»). Модуль не знает ни pydantic-ai, ни базы: всё приходит аргументами.

**Files:**
- Create: `src/harness/explanation/agent_checks.py`
- Modify: `src/harness/explanation/digest.py` — метод `NumberBook.prose`
- Modify: `src/harness/explanation/__init__.py` — реэкспорт
- Test: `tests/test_agent_checks.py` (новый), `tests/test_faithfulness.py` не трогать

**Interfaces:**
- Consumes: `numbers_in`, `unsupported_numbers`, `error_words_in` (`explanation/faithfulness.py:112, 121, 171`); `verdict_digest` (`explanation/verdict_text.py:275`); `hand_replay(en, *, stats=None)` (`explanation/hand_replay.py:575` — **аргумента `ev_loss_bb` у неё нет**, не передавать); `postflop_line_detail` (`contracts/analysis.py:329`); `class_of` (`contracts/ranges.py:21`).
- Produces:
  - `NumberBook.prose(text: str) -> str` — регистрирует каждое число `numbers_in(text)`
  - `@dataclass(frozen=True, slots=True) class AgentContext: text: str; allowed: frozenset[float]; labels: frozenset[str]; hero_class: str | None; board_ranks: tuple[str, ...]`
  - `agent_context(result: AnalysisResult, enriched: EnrichedHand) -> AgentContext`
  - `strip_card_tokens(text: str, *, hero_class: str | None = None, board_ranks: Sequence[str] = ()) -> str`
  - `check_agent_answer(answer: str, *, allowed: frozenset[float], labels: frozenset[str], leaks_called: bool, hero_class: str | None = None, board_ranks: Sequence[str] = ()) -> list[str]` — пустой список = прошло; иначе нарушения словами (едут в трейс и в запрос «переписать»)
  - константы `ASSUMPTION_MARK = "(допущение)"`, `TERM_LABELS: dict[str, tuple[re.Pattern[str], frozenset[str]]]`

- [ ] **Step 1: Падающие тесты** (`tests/test_agent_checks.py`)

```python
"""Проверки ответа агента и сводка разбора — без базы и без сети.

Проверяется КОД вокруг модели: что разрешено и что отвергается. Механическая
проверка допущений, терминов и лучшей линии частичная по построению (спека
агента №1, §6): явные обороты ловятся здесь, остальное — судья (спека №2).
"""

from __future__ import annotations

import pytest

from harness.contracts import (
    POSTFLOP_LINE_DETAIL,
    ActionTag,
    Combination,
    HandCategory,
    HandStrength,
    Line,
    PostflopLineDetail,
    StrengthClass,
)
from harness.explanation.agent_checks import (
    ASSUMPTION_MARK,
    agent_context,
    check_agent_answer,
    strip_card_tokens,
)
from harness.explanation.digest import NumberBook
from harness.explanation.faithfulness import numbers_in, unsupported_numbers
from tests.test_hand_replay import _postflop_hand
from tests.test_verdict_text import _point, _result

_NONE: frozenset[float] = frozenset()


def _check(answer: str, **kw) -> list[str]:
    kw.setdefault("allowed", _NONE)
    kw.setdefault("labels", frozenset())
    kw.setdefault("leaks_called", False)
    return check_agent_answer(answer, **kw)


# --- числа и обозначения карт ---------------------------------------------------


def test_a_number_from_the_allowed_set_passes_and_an_invented_one_does_not():
    assert _check("Вы входите в банк в 100.0% раздач (2 из 2).", allowed=frozenset({100.0, 2.0})) == []
    assert any("числа" in v for v in _check("Вы входите в 43.0% раздач.", allowed=frozenset({100.0})))


@pytest.mark.parametrize("text", ["A5s", "T9o", "KQ и AA", "J♥️9♥️", "10♠", "борд K-7-2"])
def test_card_tokens_are_not_numbers(text: str):
    assert numbers_in(strip_card_tokens(text)) == []


@pytest.mark.parametrize("text", ["9 BB", "22 раздачи", "9.5", "99%", "стек 22.5"])
def test_real_numbers_stay_numbers(text: str):
    assert numbers_in(strip_card_tokens(text, hero_class="99", board_ranks=("Q", "J", "6"))) != []


def test_a_pocket_pair_is_a_card_only_when_the_hero_holds_it():
    """Чисто цифровую пару (`22`…`99`) от суммы шаблоном не отличить (тех-спека
    §5.7). Решение: пара — карта, только если это класс руки героя в разборе."""
    assert numbers_in(strip_card_tokens("у вас 99", hero_class="99")) == []
    assert numbers_in(strip_card_tokens("у вас 99")) == [99.0]
    assert numbers_in(strip_card_tokens("у вас 99", hero_class="J9s")) == [99.0]


def test_board_ranks_are_cards_only_when_they_are_the_board():
    """Без разбора — ранги борда с буквой (`K-7-2`); с разбором — только то,
    что лежит на борде: «7-2» на борде Q-J-6 — числа, и они не разрешены."""
    assert numbers_in(strip_card_tokens("на борде Q-J-6", board_ranks=("Q", "J", "6"))) == []
    assert numbers_in(strip_card_tokens("на борде K-7-2", board_ranks=("Q", "J", "6"))) == [7.0, 2.0]
    assert numbers_in(strip_card_tokens("на борде K-7-2")) == []


# --- допущение ------------------------------------------------------------------


def test_a_range_claim_without_the_mark_is_refused_and_with_it_passes():
    assert any(ASSUMPTION_MARK in v for v in _check("Его диапазон тут широкий."))
    assert _check(f"Его диапазон тут широкий {ASSUMPTION_MARK}.") == []


def test_a_range_sentence_citing_only_allowed_numbers_needs_no_mark():
    """Фраза о диапазоне, все числа которой из сводки (зона «предполагая»
    разбора), — цитата расчёта, а не допущение агента."""
    assert _check("В допущении о диапазоне 1.5% всех рук.", allowed=frozenset({1.5})) == []


def test_a_decimal_point_does_not_split_a_sentence():
    assert _check(f"Диапазон на 6.6 банка {ASSUMPTION_MARK}.", allowed=frozenset({6.6})) == []


# --- лучшая линия на флопе и тёрне ---------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["На флопе лучше было чекнуть.", "На тёрне вам стоило поставить.", "На терне правильно было бы сдаться."],
)
def test_the_best_line_on_flop_or_turn_is_refused(text: str):
    assert any("лучшая линия" in v for v in _check(text))


@pytest.mark.parametrize(
    "text", ["Это ничего не стоило на флопе.", "На ривере лучше было коллировать."]
)
def test_what_is_not_a_best_line_verdict_on_flop_or_turn_passes(text: str):
    assert not any("лучшая линия" in v for v in _check(text))


# --- термины --------------------------------------------------------------------


def test_a_term_passes_only_with_its_classifier_label():
    assert any("c-бет" in v for v in _check("Ваш c-бет на флопе."))
    assert _check("Ваш c-бет на флопе.", labels=frozenset({"cbet"})) == []
    assert _check("Ваш с-бет на флопе.", labels=frozenset({"cbet_flop"})) == []


def test_a_word_that_merely_contains_a_term_is_not_the_term():
    assert _check("Сетка вечера и пробег.") == []


# --- «лик» и «ошибка» -----------------------------------------------------------


def test_the_word_leak_needs_the_leaks_calculation():
    assert any("лик" in v for v in _check("Это ваш лик."))
    assert _check("Это ваш лик.", leaks_called=True) == []
    assert _check("Клик и ликвидность.") == []


def test_calling_a_decision_a_mistake_is_refused():
    assert any("ошибкой" in v for v in _check("Здесь вы ошиблись."))


def test_an_empty_answer_is_a_violation():
    assert _check("   ") == ["пустой ответ"]


# --- сводка разбора -------------------------------------------------------------


def _with_line() -> dict[str, object]:
    detail = PostflopLineDetail(
        hand=HandStrength(
            category=HandCategory.TOP_PAIR_WEAK_KICKER,
            strength=StrengthClass.MEDIUM,
            combination=Combination.PAIR,
            ranks=["J"],
            plays="hand",
            kicker=None,
        ),
        draw=None,
        draw_missed=False,
        backdoors=[],
        overcards=None,
        line=Line(action=ActionTag.CHECK_FOLD, barrel=None, size_pct=None, size_tag=None, purpose=None),
        fold_threshold=None,
        draw_call=None,
        showdown=None,
    )
    return {POSTFLOP_LINE_DETAIL: detail.model_dump(mode="json")}


def test_the_context_registers_every_number_it_prints():
    """Тот же инвариант, что у выжимок (`digest.py`): число в сводке — разрешено."""
    ctx = agent_context(_result([_point(dp_index=0, ev_diff_bb=-1.4)]), _postflop_hand())
    assert unsupported_numbers(ctx.text, ctx.allowed) == []


def test_the_context_carries_the_hand_board_and_classifier_labels():
    ctx = agent_context(_result([_point(dp_index=0, ev_diff_bb=-1.4, detail=_with_line())]), _postflop_hand())
    assert ctx.hero_class == "J9s"
    assert ctx.board_ranks == ("6", "J", "Q")
    assert {"top_pair_weak_kicker", "check_fold"} <= ctx.labels


def test_prose_registers_every_number_of_a_code_built_text():
    book = NumberBook()
    book.prose("Флоп: 6♠ J♦ Q♦ (банк 6.6): вы чек → CO бет 3.0, итог −2.6BB")
    assert {6.6, 3.0, -2.6} <= book.allowed
```

`_point(detail=…)` у `test_verdict_text._point` создаёт префлоп-точку (`street=Street.PREFLOP`) — метки классификатора от улицы не зависят, тесту этого достаточно. Сверить, что `PointVerdict` с `spot=PUSHFOLD_UNOPENED` и `postflop_line` в `detail` проходит валидаторы (`contracts/analysis.py:283-326`); если нет — собрать `PointVerdict` в тесте явно со `street=Street.FLOP` и подходящим `spot`.

- [ ] **Step 2: Запустить** → FAIL (модуля нет).

- [ ] **Step 3: `NumberBook.prose`** (`digest.py`, импорт `from harness.explanation.faithfulness import numbers_in` — циклов нет: `faithfulness` импортирует только `contracts`)

```python
    def prose(self, text: str) -> str:
        """Текст, собранный КОДОМ из фактов руки (реплей): каждое его число разрешено.

        Для готовых строк, а не для величин: у величины свой формат (`bb`, `pct`,
        `count`). Модельный текст сюда не попадает никогда — регистрация сделала
        бы разрешённым выдуманное (`test_prose_registers_every_number_of_a_code_built_text`).
        """
        self._allowed.update(numbers_in(text))
        return text
```

Модульный докстринг `digest.py` дополнить: «`prose` — исключение для текста, который собрал код, а не модель».

- [ ] **Step 4: `agent_checks.py`**

Код ниже прогнан на примерах из тестов в черновике (scratchpad оркестратора) — регулярки сверены, а не написаны по памяти.

```python
"""Проверки ответа агента и сводка разбора для его контекста (спека агента №1, §4, §6).

Одна реализация на прод (`harness.agent`) и evals (спека №2): вторая разошлась
бы с первой на первой же правке. Модуль чистый — ни pydantic-ai, ни базы:
разрешённые числа, метки и признак вызова `leaks` приходят аргументами.

**Граница, а не гарантия.** Допущение, термины и лучшая линия проверяются по
явным оборотам; что сказано другими словами, ловит судья (спека №2). Это
записано в EVALS.md как граница.

**Обозначения карт — не числа.** `A5s`, `T9o`, `KQ`, карта с мастью, ранги борда
вырезаются ДО сверки чисел. Чисто цифровая пара (`99`) от суммы шаблоном не
отличима (тех-спека §5.7); решение — она карта, только если это класс руки
героя в сводке (`test_a_pocket_pair_is_a_card_only_when_the_hero_holds_it`).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from harness.contracts import AnalysisResult, EnrichedHand, class_of, postflop_line_detail
from harness.explanation.digest import NumberBook
from harness.explanation.faithfulness import error_words_in, numbers_in, unsupported_numbers
from harness.explanation.hand_replay import hand_replay
from harness.explanation.verdict_text import verdict_digest

__all__ = [
    "ASSUMPTION_MARK",
    "TERM_LABELS",
    "AgentContext",
    "agent_context",
    "check_agent_answer",
    "strip_card_tokens",
]

ASSUMPTION_MARK = "(допущение)"

_RANK = r"(?:10|[2-9TJQKA])"
_EDGE_L = r"(?<!\w)(?<!\d[.,])"
_EDGE_R = r"(?![\w%]|[.,]\d)"
_SUITED_CARD_RE = re.compile(_RANK + r"[♠♥♦♣]️?")
_CLASS_RE = re.compile(_EDGE_L + r"([2-9TJQKA])([2-9TJQKA])([so])?" + _EDGE_R)
_BOARD_RE = re.compile(_EDGE_L + _RANK + r"(?:[-–]?" + _RANK + r"){2,4}" + _EDGE_R)
# Точка внутри числа («6.6») предложение не кончает.
_SENTENCE_RE = re.compile(r"(?:[^.!?\n]|\.(?=\d))+[.!?]?")
_RANGE_RE = re.compile(r"диапазон|рендж", re.IGNORECASE)
# «стоило» — только с инфинитивом или «бы»: «ничего не стоило» — не упрёк.
_BEST_LINE_RE = re.compile(
    r"лучше было|было лучше|лучше бы|правильн\w* было бы|следовало|надо было|нужно было"
    r"|стоило\s+(?:бы|[а-яё]+ть)\b",
    re.IGNORECASE,
)
_FLOP_TURN_RE = re.compile(r"флоп|т[её]рн", re.IGNORECASE)
_LEAK_RE = re.compile(r"(?<![а-яё])лик(?:и|ов|ам|ами|ах|а|ом|у|е)?(?![а-яё])", re.IGNORECASE)

# Термин → метки, при которых он допустим: значения `ActionTag`, `HandCategory`,
# `DrawKind` (контекст разбора) и `FrequencyStat` (выход расчёта).
TERM_LABELS: dict[str, tuple[re.Pattern[str], frozenset[str]]] = {
    "c-бет": (re.compile(r"[cс]-?бет|контбет", re.IGNORECASE),
              frozenset({"cbet", "delayed_cbet", "cbet_flop", "fold_to_cbet"})),
    "донк": (re.compile(r"(?<![а-яё])донк", re.IGNORECASE), frozenset({"donk"})),
    "проба": (re.compile(r"(?<![а-яё])проб(?:а|у|ой|ы|е)(?![а-яё])", re.IGNORECASE),
              frozenset({"probe"})),
    "баррель": (re.compile(r"баррел", re.IGNORECASE),
                frozenset({"barrel", "barrel_turn", "barrel_river"})),
    "чек-рейз": (re.compile(r"чек-?рейз", re.IGNORECASE), frozenset({"check_raise"})),
    "сет": (re.compile(r"(?<![а-яё])сет(?:а|ом|у|ы)?(?![а-яё])", re.IGNORECASE),
            frozenset({"set"})),
    "оверпара": (re.compile(r"оверпар", re.IGNORECASE), frozenset({"overpair"})),
    "топ-пара": (re.compile(r"топ-?пар", re.IGNORECASE),
                 frozenset({"top_pair_strong_kicker", "top_pair_weak_kicker"})),
    "гатшот": (re.compile(r"гатшот", re.IGNORECASE), frozenset({"gutshot", "double_gutshot"})),
    "флеш-дро": (re.compile(r"флеш-?дро", re.IGNORECASE), frozenset({"flush"})),
}


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Сводка последнего разбора для агента: текст, разрешённые числа, метки
    классификатора, класс руки героя и ранги борда (для вырезания карт)."""

    text: str
    allowed: frozenset[float]
    labels: frozenset[str]
    hero_class: str | None
    board_ranks: tuple[str, ...]


def agent_context(result: AnalysisResult, enriched: EnrichedHand) -> AgentContext:
    """Сводка разбора: ход руки (реплей, код), судимые точки (выжимка вердикта),
    метки классификатора всех точек. Каждое число — через `NumberBook`."""
    book = NumberBook()
    replay = book.prose(hand_replay(enriched).plain)
    verdict = verdict_digest(result)
    labels: set[str] = set()
    label_lines: list[str] = []
    for point in result.points:
        line = postflop_line_detail(point)
        if line is None:
            continue
        found = []
        if line.hand is not None:
            found.append(line.hand.category.value)
        if line.draw is not None:
            found.extend(kind.value for kind in line.draw.kinds)
        if line.line is not None:
            found.append(line.line.action.value)
        labels.update(found)
        label_lines.append(f"Точка {point.street.value}: метки классификатора — {', '.join(found)}.")
    hand = enriched.hand
    cards = hand.dealt.get(hand.hero_label, [])
    hero_class = class_of(cards[0], cards[1]) if len(cards) == 2 else None
    board = tuple(card[0] for street in sorted(hand.boards) for card in hand.boards[street])
    text = "\n\n".join(part for part in (replay, verdict.text, "\n".join(label_lines)) if part)
    return AgentContext(
        text=text,
        allowed=book.allowed | verdict.allowed,
        labels=frozenset(labels),
        hero_class=hero_class,
        board_ranks=board,
    )
```

`sorted(hand.boards)` — `Street` это `StrEnum`? сверить в `contracts/raw.py`: если сортировка по строке даст «flop, river, turn», собирать борд обходом `for street in Street` (порядок объявления), как делает `hand_replay`. Ранг «T» у десятки — сверить формат карт (`"Th"`), `card[0]` даёт `"T"`.

```python
def strip_card_tokens(
    text: str, *, hero_class: str | None = None, board_ranks: Sequence[str] = ()
) -> str:
    """Текст без обозначений карт — то, что идёт в сверку чисел."""
    text = _SUITED_CARD_RE.sub(" ", text)
    board = Counter(board_ranks)

    def board_sub(match: re.Match[str]) -> str:
        ranks = re.findall(_RANK, match.group(0))
        has_letter = any(rank in "TJQKA" for rank in ranks)
        if (board and not (Counter(ranks) - board)) or (has_letter and not board):
            return " "
        return match.group(0)

    def class_sub(match: re.Match[str]) -> str:
        high, low, suit = match.group(1), match.group(2), match.group(3)
        if high.isdigit() and low.isdigit() and suit is None:
            return " " if match.group(0) == hero_class else match.group(0)
        return " "

    return _CLASS_RE.sub(class_sub, _BOARD_RE.sub(board_sub, text))


def check_agent_answer(
    answer: str,
    *,
    allowed: frozenset[float],
    labels: frozenset[str],
    leaks_called: bool,
    hero_class: str | None = None,
    board_ranks: Sequence[str] = (),
) -> list[str]:
    """Нарушения ответа словами; пустой список — ответ прошёл (спека §6)."""
    text = answer.strip()
    if not text:
        return ["пустой ответ"]

    def cleaned(part: str) -> str:
        return strip_card_tokens(part, hero_class=hero_class, board_ranks=board_ranks)

    violations: list[str] = []
    invented = unsupported_numbers(cleaned(text), allowed)
    if invented:
        violations.append(f"числа не из расчёта и не из разбора — {invented}")
    for sentence in _SENTENCE_RE.findall(text):
        quoted = sentence.strip()
        if _RANGE_RE.search(sentence) and ASSUMPTION_MARK not in sentence:
            numbers = numbers_in(cleaned(sentence))
            if not numbers or unsupported_numbers(cleaned(sentence), allowed):
                violations.append(f"утверждение о диапазоне без метки {ASSUMPTION_MARK} — «{quoted}»")
        if _BEST_LINE_RE.search(sentence) and _FLOP_TURN_RE.search(sentence):
            violations.append(f"лучшая линия на флопе или тёрне — «{quoted}»")
    for term, (pattern, needed) in TERM_LABELS.items():
        if pattern.search(text) and not (needed & labels):
            violations.append(f"термин «{term}» без такой метки классификатора")
    if _LEAK_RE.search(text) and not leaks_called:
        violations.append("слово «лик» без расчёта leaks")
    errors = error_words_in(text)
    if errors:
        violations.append(f"решение названо ошибкой — {errors}")
    return violations
```

`__init__.py`: реэкспорт `AgentContext`, `agent_context`, `check_agent_answer`, `strip_card_tokens`; модульный докстринг пакета («Три выхода…») дополнить четвёртым: проверки и сводка для агента.

Если `import hand_replay` из `agent_checks` создаёт цикл через `explanation/__init__` — импортировать модулем (`from harness.explanation.hand_replay import hand_replay`), как написано выше; `__init__` импортирует `agent_checks` последним.

- [ ] **Step 5: Запустить**

Run: `uv run pytest tests/test_agent_checks.py tests/test_faithfulness.py tests/test_question_text.py tests/test_verdict_text.py -q -ra` → PASS.
Run: `uv run pytest -q -ra`, `uv run ruff check . && uv run pyright` → чисто.

- [ ] **Step 6: Коммит**

```bash
git add src/harness/explanation/agent_checks.py src/harness/explanation/digest.py src/harness/explanation/__init__.py tests/test_agent_checks.py
git commit -m "$(cat <<'EOF'
Проверки ответа агента и сводка разбора: числа, карты, допущение, термины, лик, лучшая линия

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

**Докстринги, которые становятся ложью:**
- `explanation/__init__.py:1-10` — «Три выхода» (см. Step 4).
- `explanation/digest.py:7-13` — «величина не печатается мимо книги» остаётся верным; дописать про `prose` (Step 3).
- `explanation/faithfulness.py:9-13` («Одни и те же функции работают в двух местах… `verdict_text`… `evals/verdict/checks.py`») — верно, но неполно: `agent_checks` зовёт `unsupported_numbers`/`error_words_in` третьим потребителем; дописать одной фразой.

---

### Task 4: Пакет `harness.agent`, задача `agent` в воркере, история, журнал, указатель разбора

**Агент:** `implementer-core`, затем `reviewer` (конвейер воркера, данные, граница «считает код»).

**Files:**
- Create: `src/harness/agent/__init__.py`, `src/harness/agent/history.py`, `src/harness/agent/prompt.py`, `src/harness/agent/run.py`
- Modify: `src/harness/calcs/routing.py` — `CalcRun` (131-141), `_perform` (144-163), модульный докстринг (пункт 3, строки 18-22)
- Modify: `src/harness/worker/pipeline.py` — `_JOB_DEADLINE_S` (137-146), `Sender` (220-233), `Deps` (236-260), `_dispatch` (1268-1282), новый `_run_agent`; указатель разбора в `_run_deep_dive` (1186-1195) и `_run_screenshot` (1104-1119), сброс в `_run_hh_scan` (после `save_scan_summary`, строка 722)
- Modify: `src/harness/worker/main.py` — `TelegramSender` (178-281): метод `typing`; `main` (443-480): `Deps(agent_history_max_tokens=cfg.agent_history_max_tokens)`
- Modify: `tests/conftest.py` — маркер `requires_agent_prompt`
- Test: `tests/test_agent.py` (новый), `tests/test_worker_pipeline.py` (`FakeSender.typing`, тесты задачи `agent`), `tests/test_entrypoints.py` (`typing` у `TelegramSender`)

**Interfaces:**
- Consumes: Task 1 (`LLM.__call__` с `message_history`/`instructions`, `CallMeta.messages`, назначение `agent`), Task 2 (`SessionsRepo.agent_state/save_agent_history/set_last_analysis`, `AnalysesRepo.get`, CHECK), Task 3 (`AgentContext`, `agent_context`, `check_agent_answer`).
- Produces:
  - `calcs.routing.CalcRun(max_calcs: int = 1)`; поля `results: list[CalcResult]`, `calls: list[dict[str, Any]]`; свойства `result -> CalcResult | None` (первый), `digest -> Digest | None` (первого) — `/ask` читает их как раньше (`routing.py:395-401`)
  - `_perform(...) -> str | ToolReturn` — удачный расчёт возвращает `ToolReturn(return_value=digest.text, metadata={"calc": …, "allowed": sorted(digest.allowed), "stat": …})`; модель видит тот же текст, метаданные модели не уходят (`pydantic_ai/messages.py:1019-1020`)
  - `agent.history`: `load_history(raw: list[dict[str, Any]] | None) -> list[ModelMessage]`; `dump_history(messages: Sequence[ModelMessage]) -> list[dict[str, Any]]`; `estimate_tokens(messages) -> int`; `turns(messages: list[ModelMessage]) -> list[list[ModelMessage]]`; `trim_history(messages, max_tokens: int) -> list[ModelMessage]`; `allowed_from_history(messages) -> frozenset[float]`; `calcs_in_history(messages) -> frozenset[str]`; `stats_in_history(messages) -> frozenset[str]`
  - `agent.prompt`: `SYSTEM_PROMPT_PATH: Path`; `load_system_prompt() -> str`; `instructions_for(system: str, context: AgentContext | None) -> str`
  - `agent.run`: `MAX_CALCS_PER_TURN = 8`; `class AgentDraft(ModelOutput, BaseModel): answer: str`; `@dataclass(frozen=True) class AgentTurn: state: Literal["prose", "numbers", "refusal"]; prose: str | None; results: tuple[CalcResult, ...]; from_context: bool; messages: tuple[ModelMessage, ...]; violations: tuple[tuple[str, ...], ...]; calls: tuple[dict[str, Any], ...]` и метод `journal() -> dict[str, Any]`; `async def run_agent_turn(db: AsyncSession, llm: LLM, *, player_id: int, session_id: int, text: str, trace_id: int, history: Sequence[ModelMessage], context: AgentContext | None, system_prompt: str | None = None) -> AgentTurn`
  - `Sender.typing(chat_id: int) -> None`; `Deps.agent_history_max_tokens: int = 30_000`
  - задача `agent`: `payload {"text": str}`, дедлайн 120 с, запись `trace.record("agent_turn", …)`

- [ ] **Step 1: Маркер промпта** (`tests/conftest.py`, рядом с `requires_prompts`, строки 48-57)

```python
PROMPT_AGENT = SRC / "agent" / "prompts" / "system.md"
# Отдельный маркер, а не пятое условие `PROMPTS_PRESENT`: у владельца промпты
# изложения есть, а промпт агента появится позже (Task 7 плана агента) — общее
# условие на это время выключило бы ВСЕ тесты с промптами.
requires_agent_prompt = pytest.mark.skipif(
    not PROMPT_AGENT.exists(),
    reason="нет системного промпта агента (agent/prompts) — закрыт политикой публикации",
)
```

Тесты этой задачи промпт НЕ читают: `run_agent_turn` принимает `system_prompt`, тесты передают синтетическую строку. Гейтится только тест загрузчика.

- [ ] **Step 2: Падающие тесты** (`tests/test_agent.py`)

```python
"""Ход агента: инструменты, потолок, проверки, попытка переписать, история.

В сеть не ходим: модель — `FunctionModel`, сценарий по прогонам задаёт тест.
База настоящая — инструменты делают выборку SQL (довод `test_question_routing.py`).
Выбор расчёта здесь не проверяется: это eval маршрутизации (спека №2).
"""

from __future__ import annotations

from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from sqlalchemy import text as sql_text

from harness.agent.history import (
    allowed_from_history,
    calcs_in_history,
    dump_history,
    load_history,
    trim_history,
)
from harness.agent.prompt import instructions_for, load_system_prompt
from harness.agent.run import MAX_CALCS_PER_TURN, run_agent_turn
from harness.contracts import CalcName
from harness.explanation.agent_checks import agent_context
from harness.platform.llm import LLM
from tests.conftest import requires_agent_prompt
from tests.test_hand_replay import _postflop_hand
from tests.test_question_routing import _TEST_CFG, _seeded
from tests.test_verdict_text import _point, _result

_SYSTEM = "СИСТЕМНЫЙ-СИНТ"


class _TurnScript:
    """Сценарий по прогонам: в каждом — вызовы инструментов, потом ответ.

    Шаг считается от ПОСЛЕДНЕГО сообщения игрока, а не от начала истории:
    иначе прошлые ходы вечера сдвигали бы сценарий (у `_Script` из
    `test_question_routing.py` истории нет, у агента — есть).
    """

    def __init__(self, runs: list[tuple[list[tuple[str, dict[str, Any]]], str]]) -> None:
        self.runs = runs
        self.run = -1
        self.requests = 0
        self.instructions: list[str] = []
        self.returns: list[str] = []

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.requests += 1
        self.instructions.append(info.instructions or "")
        self.returns = [
            part.content
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart) and isinstance(part.content, str)
        ]
        starts = [
            index
            for index, message in enumerate(messages)
            if isinstance(message, ModelRequest)
            and any(isinstance(part, UserPromptPart) for part in message.parts)
        ]
        step = sum(isinstance(message, ModelResponse) for message in messages[starts[-1]:])
        if step == 0:
            self.run += 1
        calls, answer = self.runs[self.run]
        if step < len(calls):
            name, args = calls[step]
            return ModelResponse(parts=[ToolCallPart(tool_name=name, args=args)])
        assert info.output_tools
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"answer": answer})])


_VPIP = ("hero_frequency", {"stat": "vpip"})
_GOOD = "Вы входите в банк в 100.0% раздач (2 из 2)."


async def _turn(db_factory, session, script, *, scope, text="вопрос", history=(), context=None):
    player_id, session_id, trace_id = scope
    llm = LLM(_TEST_CFG, db_factory, model_override=FunctionModel(script))
    return await run_agent_turn(
        session, llm, player_id=player_id, session_id=session_id, text=text,
        trace_id=trace_id, history=history, context=context, system_prompt=_SYSTEM,
    )


async def test_a_number_from_the_tool_reaches_the_player_with_its_calculation(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8101)
        turn = await _turn(db_factory, session, _TurnScript([([_VPIP], _GOOD)]), scope=scope)
    assert turn.state == "prose" and turn.prose == _GOOD
    assert [r.calc for r in turn.results] == [CalcName.HERO_FREQUENCY]


async def test_several_calculations_in_one_turn_are_all_listed(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8102)
        script = _TurnScript([([_VPIP, ("leaks", {})], _GOOD)])
        turn = await _turn(db_factory, session, script, scope=scope)
    assert [r.calc for r in turn.results] == [CalcName.HERO_FREQUENCY, CalcName.LEAKS]


async def test_the_ninth_calculation_is_refused(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8103)
        script = _TurnScript([([_VPIP] * (MAX_CALCS_PER_TURN + 1), _GOOD)])
        turn = await _turn(db_factory, session, script, scope=scope)
    assert len(turn.results) == MAX_CALCS_PER_TURN
    assert any("8" in text and "расчёт" in text.lower() for text in script.returns)


async def test_an_answer_from_the_context_alone_passes_without_a_call(db_factory):
    """Вопрос про руку часто отвечается сводкой: числа из неё разрешены."""
    ctx = agent_context(_result([_point(dp_index=0, ev_diff_bb=-1.4)]), _postflop_hand())
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8104)
        script = _TurnScript([([], "Банк на флопе был 6.6, вы сдались на ставку 3.0.")])
        turn = await _turn(db_factory, session, script, scope=scope, context=ctx)
    assert turn.state == "prose" and turn.results == () and turn.from_context
    assert _SYSTEM in script.instructions[0] and "6.6" in script.instructions[0]


async def test_an_answer_without_a_call_and_without_a_context_is_refused(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8105)
        script = _TurnScript([([], "На BB вы минусовой, это нормально.")])
        turn = await _turn(db_factory, session, script, scope=scope)
    assert turn.state == "refusal" and turn.prose is None


async def test_one_rewrite_then_numbers_without_prose(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8106)
        script = _TurnScript([
            ([_VPIP], "Вы входите в 43.0% раздач."),
            ([], "Вы входите в 44.0% раздач."),
        ])
        turn = await _turn(db_factory, session, script, scope=scope)
    assert turn.state == "numbers" and turn.prose is None
    assert len(turn.violations) == 2 and script.run == 1, "переписать — ровно один раз"


async def test_a_rewrite_that_passes_shows_the_prose(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8107)
        script = _TurnScript([([_VPIP], "Вы входите в 43.0% раздач."), ([], _GOOD)])
        turn = await _turn(db_factory, session, script, scope=scope)
    assert turn.state == "prose" and turn.prose == _GOOD


async def test_a_number_from_an_earlier_turn_of_the_evening_stays_allowed(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8108)
        first = await _turn(db_factory, session, _TurnScript([([_VPIP], _GOOD)]), scope=scope)
        second = await _turn(
            db_factory, session, _TurnScript([([], "Те же 100.0% (2 из 2).")]),
            scope=scope, history=first.messages,
        )
    assert second.state == "prose"


async def test_the_word_leak_without_the_leaks_calculation_is_rewritten(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8109)
        script = _TurnScript([([_VPIP], _GOOD + " Это ваш лик."), ([], _GOOD)])
        turn = await _turn(db_factory, session, script, scope=scope)
    assert turn.state == "prose" and any("лик" in v for v in turn.violations[0])


async def test_a_turn_with_three_model_requests_logs_three_agent_rows(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8110)
        await _turn(db_factory, session, _TurnScript([([_VPIP, ("leaks", {})], _GOOD)]), scope=scope)
    async with db_factory() as session:
        rows = (await session.execute(sql_text("select purpose, status from llm_calls order by id"))).all()
    assert [tuple(r) for r in rows] == [("agent", "ok")] * 3


async def test_history_survives_serialization(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8111)
        turn = await _turn(db_factory, session, _TurnScript([([_VPIP], _GOOD)]), scope=scope)
    restored = load_history(dump_history(turn.messages))
    assert calcs_in_history(restored) == {"hero_frequency"}
    assert allowed_from_history(restored) >= {100.0, 2.0}


async def test_trimming_drops_whole_turns_and_their_numbers(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8112)
        first = await _turn(db_factory, session, _TurnScript([([_VPIP], _GOOD)]), scope=scope)
        second = await _turn(
            db_factory, session, _TurnScript([([("leaks", {})], "Расхождений нет.")]),
            scope=scope, history=first.messages,
        )
    assert 100.0 in allowed_from_history(second.messages)
    last_turn_only = trim_history(second.messages, max_tokens=_tokens_of_last_turn(second.messages))
    assert calcs_in_history(last_turn_only) == {"leaks"}
    assert "hero_frequency" not in calcs_in_history(last_turn_only)
    first_numbers = allowed_from_history(first.messages) - allowed_from_history(last_turn_only)
    assert first_numbers, "выброшенный ход не унёс ни одного числа — проверять нечего"


async def test_trimming_leaves_no_orphan_tool_return(db_factory):
    """Первый оставшийся запрос не начинается с ответа на вызов, которого нет:
    провайдер такую историю отвергает."""
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8113)
        first = await _turn(db_factory, session, _TurnScript([([_VPIP], _GOOD)]), scope=scope)
        second = await _turn(
            db_factory, session, _TurnScript([([], "Те же 100.0% (2 из 2).")]),
            scope=scope, history=first.messages,
        )
    trimmed = trim_history(second.messages, max_tokens=_tokens_of_last_turn(second.messages))
    head = trimmed[0]
    assert isinstance(head, ModelRequest)
    assert all(isinstance(part, UserPromptPart) for part in head.parts)


def test_instructions_carry_the_system_prompt_and_the_context():
    ctx = agent_context(_result([_point(dp_index=0, ev_diff_bb=-1.4)]), _postflop_hand())
    assert instructions_for(_SYSTEM, ctx).startswith(_SYSTEM)
    assert ctx.text in instructions_for(_SYSTEM, ctx)
    assert ctx.text not in instructions_for(_SYSTEM, None)


@requires_agent_prompt
def test_the_system_prompt_loads():
    assert load_system_prompt().strip()


def _tokens_of_last_turn(messages: list[ModelMessage] | tuple[ModelMessage, ...]) -> int:
    from harness.agent.history import estimate_tokens, turns

    return estimate_tokens(turns(list(messages))[-1])
```

`_seeded` и `_TEST_CFG` импортируются из `tests/test_question_routing.py` (строки 41, 96) — тот же приём, что `tests/test_worker_pipeline.py:1993`.

- [ ] **Step 3: Запустить** → FAIL (пакета нет).

- [ ] **Step 4: `CalcRun` и `_perform`** (`calcs/routing.py`)

```python
from pydantic_ai import ToolReturn

_CALC_CAP = (
    "Потолок расчётов на один ответ — {cap}: ответьте по тому, что уже посчитано."
)


class CalcRun:
    """Расчёты одного хода и их выжимки — то, что копит прогон.

    `max_calcs=1` — `/ask`: второй успешный вызов получает отказ
    (`test_the_second_calculation_is_refused`). Агент передаёт 8 (спека агента
    №1, §5): девятый получает отказ (`test_the_ninth_calculation_is_refused`).
    `calls` — журнал каждого вызова, удачного или нет: имя, параметры, выход,
    зона, время (спека §8, «Журнал»).
    """

    def __init__(self, max_calcs: int = 1) -> None:
        self.max_calcs = max_calcs
        self.results: list[CalcResult] = []
        self.digests: list[Digest] = []
        self.calls: list[dict[str, Any]] = []

    @property
    def result(self) -> CalcResult | None:
        return self.results[0] if self.results else None

    @property
    def digest(self) -> Digest | None:
        return self.digests[0] if self.digests else None


# Зона доверия по инструменту (CLAUDE.md): все шесть расчётов — измерения и
# арифметика, ни один не опирается на угаданный диапазон.
_TOOL_ZONE: dict[CalcName, Zone] = {name: Zone.STRICT for name in CalcName}


async def _perform(
    db: AsyncSession, player_id: int, run_state: CalcRun, params: CalcParams
) -> str | ToolReturn:
    if len(run_state.results) >= run_state.max_calcs:
        refusal = _ONE_CALC_ONLY if run_state.max_calcs == 1 else _CALC_CAP.format(cap=run_state.max_calcs)
        run_state.calls.append({"calc": params.calc.value, "params": params.model_dump(mode="json"), "refused": True})
        return refusal
    started = time.monotonic()
    try:
        result = await run(db, player_id, params)
    except ValueError as exc:
        run_state.calls.append({
            "calc": params.calc.value, "params": params.model_dump(mode="json"),
            "error": str(exc), "duration_s": round(time.monotonic() - started, 3),
        })
        return f"Расчёт не выполнен: {exc}"
    digest = question_digest(result)
    run_state.results.append(result)
    run_state.digests.append(digest)
    run_state.calls.append({
        "calc": result.calc.value, "params": params.model_dump(mode="json"),
        "output": digest.text, "zone": _TOOL_ZONE[result.calc].value,
        "duration_s": round(time.monotonic() - started, 3),
    })
    stat = getattr(result, "stat", None)
    return ToolReturn(
        return_value=digest.text,
        metadata={
            "calc": result.calc.value,
            "allowed": sorted(digest.allowed),
            "stat": stat.value if stat is not None else None,
        },
    )
```

Аннотации шести функций-инструментов в `tools_for` (`-> str`) → `-> str | ToolReturn`. Сверить в установленной библиотеке, что `ToolReturn` импортируется из `pydantic_ai` верхнего уровня; если нет — из `pydantic_ai.messages`. Сверить, что `ToolReturnPart.content` у удачного вызова — строка `return_value` (на этом держатся `script.returns` в старых тестах `/ask`).

- [ ] **Step 5: `agent/history.py`**

```python
"""История вечера агента: сериализация, ходы, обрезка, разрешённые числа.

История — сообщения pydantic-ai (`ModelMessagesTypeAdapter`), живёт в
`sessions.agent_history` до конца вечера. Разрешённые числа и вызванные расчёты
читаются из метаданных ответов инструментов (`calcs.routing._perform` кладёт
туда выжимку), а не из текста: что ушло из истории обрезкой, перестаёт быть
разрешённым (`test_trimming_drops_whole_turns_and_their_numbers`).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ToolReturnPart,
    UserPromptPart,
)

# Оценка, а не счёт токенизатора: байты JSON / 3. Кириллица в UTF-8 — два байта
# на букву и около половины токена на букву, так что оценка завышает — потолок
# срабатывает раньше, а не позже.
_BYTES_PER_TOKEN = 3


def load_history(raw: list[dict[str, Any]] | None) -> list[ModelMessage]:
    return ModelMessagesTypeAdapter.validate_python(raw) if raw else []


def dump_history(messages: Sequence[ModelMessage]) -> list[dict[str, Any]]:
    """Сообщения в JSON без инструкций: системный промпт и сводка пересобираются
    на каждом ходе, хранить их копию в каждой строке незачем."""
    bare = [
        dataclasses.replace(m, instructions=None) if isinstance(m, ModelRequest) else m
        for m in messages
    ]
    return ModelMessagesTypeAdapter.dump_python(bare, mode="json")


def estimate_tokens(messages: Sequence[ModelMessage]) -> int:
    return len(ModelMessagesTypeAdapter.dump_json(list(messages))) // _BYTES_PER_TOKEN


def _opens_a_turn(message: ModelMessage) -> bool:
    return isinstance(message, ModelRequest) and any(
        isinstance(part, UserPromptPart) for part in message.parts
    )


def turns(messages: list[ModelMessage]) -> list[list[ModelMessage]]:
    """Ходы: каждый начинается запросом с репликой игрока."""
    result: list[list[ModelMessage]] = []
    for message in messages:
        if _opens_a_turn(message) or not result:
            result.append([message])
        else:
            result[-1].append(message)
    return result


def trim_history(messages: Sequence[ModelMessage], max_tokens: int) -> list[ModelMessage]:
    """Самые старые ходы — прочь целиком, пока история не влезет в потолок.

    Первый оставшийся запрос чистится от ответов инструментов: pydantic-ai
    склеивает соседние запросы, и ответ на вызов из выброшенного хода мог
    оказаться в одном запросе с новой репликой — провайдер такую историю
    отвергает (`test_trimming_leaves_no_orphan_tool_return`).
    """
    kept = turns(list(messages))
    while kept and estimate_tokens([m for turn in kept for m in turn]) > max_tokens:
        kept.pop(0)
    flat = [m for turn in kept for m in turn]
    if flat and isinstance(flat[0], ModelRequest):
        head = flat[0]
        parts = [p for p in head.parts if not isinstance(p, ToolReturnPart)]
        flat[0] = dataclasses.replace(head, parts=parts)
    return flat


def _metadata(messages: Sequence[ModelMessage]) -> list[dict[str, Any]]:
    return [
        part.metadata
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and isinstance(part.metadata, dict)
    ]


def allowed_from_history(messages: Sequence[ModelMessage]) -> frozenset[float]:
    return frozenset(float(n) for meta in _metadata(messages) for n in meta.get("allowed", []))


def calcs_in_history(messages: Sequence[ModelMessage]) -> frozenset[str]:
    return frozenset(str(meta["calc"]) for meta in _metadata(messages) if meta.get("calc"))


def stats_in_history(messages: Sequence[ModelMessage]) -> frozenset[str]:
    return frozenset(str(meta["stat"]) for meta in _metadata(messages) if meta.get("stat"))
```

Сверить в установленной 2.38.0: (а) `ToolReturnPart.metadata` переживает `dump_python`/`validate_python` (тест `test_history_survives_serialization`); (б) `ModelRequest` — dataclass, `dataclasses.replace` с `parts=` работает (`messages.py:1983-1990`).

- [ ] **Step 6: `agent/prompt.py`**

```python
"""Системный промпт агента и инструкции хода.

Файл промпта закрыт политикой публикации (`.githooks/pre-push:21`, путь
`src/harness/[^/]+/prompts/`), читается лениво (`read_prompt`): без файла падает
тот, кто зовёт модель, а не тот, кто импортировал пакет.
"""

from __future__ import annotations

from pathlib import Path

from harness.explanation.agent_checks import AgentContext
from harness.explanation.verdict_text import read_prompt

SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "system.md"


def load_system_prompt() -> str:
    return read_prompt(SYSTEM_PROMPT_PATH)


def instructions_for(system: str, context: AgentContext | None) -> str:
    """Системный промпт плюс сводка последнего разбора — в инструкциях, а не в
    реплике: инструкции текущего запроса не копятся в истории."""
    if context is None:
        return f"{system}\n\nРазбора руки в этом вечере нет: отвечайте только расчётами."
    return f"{system}\n\nСводка последнего разбора (числа — только отсюда или из расчётов):\n{context.text}"
```

Две фразы в `instructions_for` — служебные пометки кода, не промпт; если владелец захочет их формулировку — переносятся в `system.md` плейсхолдерами (тот же приём, что `{question}` в `question_prompt`, `explanation/question.py:250`).

- [ ] **Step 7: `agent/run.py`**

```python
"""Один ход агента: инструменты, проверки, одна попытка переписать.

Модель маршрутизирует и излагает, считает код (спека агента №1, §1). Проверки —
`explanation.agent_checks`, та же реализация, что в evals. Попытка переписать —
второй вызов модели с тем же контекстом и списком нарушений; это не схема-ретрай
фасада (`_MAX_SCHEMA_ATTEMPTS`) — два разных механизма (спека §6).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage
from sqlalchemy.ext.asyncio import AsyncSession

from harness.agent import prompt as _prompt
from harness.agent.history import allowed_from_history, calcs_in_history, stats_in_history
from harness.calcs.routing import CalcRun, tools_for
from harness.contracts import CalcResult, ModelOutput
from harness.explanation.agent_checks import AgentContext, check_agent_answer
from harness.platform.llm import LLM

__all__ = ["MAX_CALCS_PER_TURN", "AgentDraft", "AgentTurn", "run_agent_turn"]

MAX_CALCS_PER_TURN = 8


class AgentDraft(ModelOutput, BaseModel):
    """Что модели разрешено вернуть: слова. Подпись и числа собирает код."""

    answer: str


@dataclass(frozen=True)
class AgentTurn:
    state: Literal["prose", "numbers", "refusal"]
    prose: str | None
    results: tuple[CalcResult, ...]
    from_context: bool
    messages: tuple[ModelMessage, ...]
    violations: tuple[tuple[str, ...], ...]
    calls: tuple[dict[str, Any], ...]

    def journal(self) -> dict[str, Any]:
        """Запись `agent_turn` в `traces.spans` (спека §8, «Журнал»)."""
        return {
            "state": self.state,
            "answer": self.prose,
            "calls": list(self.calls),
            "checks": [list(v) for v in self.violations],
            "rewrite": len(self.violations) == 2,
            "from_context": self.from_context,
        }


def _violations(answer: str, messages: Sequence[ModelMessage], context: AgentContext | None) -> list[str]:
    return check_agent_answer(
        answer,
        allowed=(context.allowed if context else frozenset()) | allowed_from_history(messages),
        labels=(context.labels if context else frozenset()) | stats_in_history(messages),
        leaks_called="leaks" in calcs_in_history(messages),
        hero_class=context.hero_class if context else None,
        board_ranks=context.board_ranks if context else (),
    )


def _rewrite_request(violations: Sequence[str]) -> str:
    listed = "\n".join(f"- {v}" for v in violations)
    return f"Ответ не прошёл проверку:\n{listed}\nПерепишите ответ, исправив только это."


async def run_agent_turn(
    db: AsyncSession,
    llm: LLM,
    *,
    player_id: int,
    session_id: int,
    text: str,
    trace_id: int,
    history: Sequence[ModelMessage],
    context: AgentContext | None,
    system_prompt: str | None = None,
) -> AgentTurn:
    """Ход: вызов модели, проверки, не более одной попытки переписать.

    Отказ (состояние 3) — нет сводки разбора и ни одного расчёта в оставшейся
    истории вечера, включая этот ход (план, отклонение 3). Числа без прозы
    (состояние 2) — проза не прошла проверку дважды.
    """
    system = system_prompt if system_prompt is not None else _prompt.load_system_prompt()
    instructions = _prompt.instructions_for(system, context)
    run_state = CalcRun(max_calcs=MAX_CALCS_PER_TURN)
    tools = tools_for(db, player_id, session_id, run_state)

    draft, meta = await llm(
        "agent", AgentDraft, prompt=text, tools=tools,
        message_history=history, instructions=instructions, trace_id=trace_id,
    )
    messages = meta.messages
    grounded = context is not None or bool(calcs_in_history(messages))
    found = _violations(draft.answer, messages, context)
    violations: list[tuple[str, ...]] = [tuple(found)]
    prose: str | None = draft.answer.strip() if not found else None

    if grounded and found:
        draft, meta = await llm(
            "agent", AgentDraft, prompt=_rewrite_request(found), tools=tools,
            message_history=messages, instructions=instructions, trace_id=trace_id,
        )
        messages = meta.messages
        second = _violations(draft.answer, messages, context)
        violations.append(tuple(second))
        prose = draft.answer.strip() if not second else None

    state: Literal["prose", "numbers", "refusal"]
    if not grounded:
        state, prose = "refusal", None
    elif prose is not None:
        state = "prose"
    else:
        state = "numbers"
    return AgentTurn(
        state=state,
        prose=prose,
        results=tuple(run_state.results),
        from_context=context is not None and not run_state.results,
        messages=tuple(messages),
        violations=tuple(violations),
        calls=tuple(run_state.calls),
    )
```

`journal()["rewrite"]` — была ли вторая попытка (закреплено в тесте журнала, Step 9). `from_context` — «ответ стоит на сводке, расчётов в ходе нет»: им `presentation` выбирает подпись.

Импорт `from harness.agent import prompt as _prompt` — модулем, не функцией: тесты воркера подменяют `harness.agent.prompt.load_system_prompt`, а `from`-импорт связал бы имя один раз (урок, записанный в `worker/pipeline.py:633-638`).

`agent/__init__.py`: докстринг пакета (не конвейерный: нужна база, как `calcs`; бот его не импортирует) и реэкспорт `AgentTurn`, `run_agent_turn`.

- [ ] **Step 8: Воркер**

`Sender` — четвёртый метод:

```python
    async def typing(self, chat_id: int) -> None: ...
```

`Deps` — поле `agent_history_max_tokens: int = 30_000` (с комментарием: значение из `Config.agent_history_max_tokens`, `worker/main.py`).

`_JOB_DEADLINE_S`: `"agent": 120.0,` с комментарием «ход агента — до 8 расчётов и одна попытка переписать; цель задержки 30 с, бюджет как у вопроса (спека агента №1, §9)».

```python
# Телеграм гасит «печатает…» через ~5 с; обновляем чаще.
_TYPING_EVERY_S = 4.0


async def _keep_typing(sender: Sender, chat_id: int) -> None:
    """«Печатает…», пока идёт ход агента (спека §7). Сбой индикатора хода не роняет."""
    while True:
        try:
            await sender.typing(chat_id)
        except Exception:  # noqa: BLE001 — индикатор best-effort
            _log.warning("typing_indicator_failed", exc_info=True)
        await asyncio.sleep(_TYPING_EVERY_S)


async def _agent_context(session: AsyncSession, analysis_id: int | None) -> AgentContext | None:
    if analysis_id is None:
        return None
    analysis = await AnalysesRepo(session).get(analysis_id)
    if analysis is None:
        return None
    hand = await HandsRepo(session).get(analysis.hand_id)
    return None if hand.enriched is None else agent_context(analysis.result, hand.enriched)


async def _run_agent(job: JobModel, deps: Deps, trace: Trace) -> None:
    """Ход агента: контекст, история вечера, модель, проверки, журнал, ответ.

    Чекпоинта нет, как у `_run_question`: повтор зовёт модель заново, дубля
    сообщения не будет (`_send_idempotent`). История пишется только в открытую
    сессию (`SessionsRepo.save_agent_history`).
    """
    worker_id = job.locked_by
    async with deps.db_factory() as session:
        payload: dict[str, Any] = dict(job.payload)
        chat_id = await _chat_id(session, job.player_id)
        sessions = SessionsRepo(session)
        state = await sessions.agent_state(job.session_id)
        context = await _agent_context(session, state.last_analysis_id)
        history = trim_history(load_history(state.history), deps.agent_history_max_tokens)
        await session.commit()

        typing = asyncio.create_task(_keep_typing(deps.sender, chat_id))
        try:
            async with trace.span("agent"):
                turn = await run_agent_turn(
                    session, deps.llm,
                    player_id=job.player_id, session_id=job.session_id,
                    text=payload["text"], trace_id=trace.trace_id,
                    history=history, context=context,
                )
        finally:
            typing.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await typing

        trace.record("agent_turn", question=payload["text"], **turn.journal())
        await sessions.save_agent_history(
            job.session_id,
            dump_history(trim_history(turn.messages, deps.agent_history_max_tokens)),
        )
        await session.commit()
        msg = agent_turn_msg(turn.state, turn.prose, list(turn.results), from_context=turn.from_context, job_id=job.id)
        await _send_idempotent(deps, session, job.id, worker_id, "result_message_id", chat_id, msg)
        await session.commit()
```

Сообщение игроку собирает `presentation`. Воркеру оно нужно уже в этой задаче, поэтому **Task 4 кладёт в `presentation/messages.py` функцию `agent_turn_msg` с окончательной подписью и временным телом**; Task 5 переписывает тело (кнопка, подпись по разбору, бюджет 4096, тексты), подпись не меняется:

```python
def agent_turn_msg(
    state: Literal["prose", "numbers", "refusal"],
    prose: str | None,
    results: Sequence[CalcResult],
    *,
    from_context: bool,
    job_id: int,
) -> Msg:
    """Ответ агента. Временное тело (план агента, Task 4): кнопку, подпись по
    разбору и бюджет сообщения добавляет Task 5."""
    if state == "refusal":
        return question_refusal_msg()
    lines = [prose.strip(), ""] if state == "prose" and prose else []
    lines.append("Посчитано: " + ", ".join(_CALC_WORD[r.calc] for r in results) + ".")
    return Msg(text="\n".join(lines))
```

Реэкспорт в `presentation/__init__.py` (импорт и `__all__`). Тест воркера Step 9 опирается только на прозу и «Посчитано:», поэтому переживает Task 5 без правки.

`_dispatch`: перед `question` —

```python
    if job.type == "agent":
        await _run_agent(job, deps, trace)
        return None
```

Указатель разбора. `_run_deep_dive` (1186-1195): сохранить номер разбора и выставить указатель на ОБОИХ путях (новый разбор и чекпоинт):

```python
            if existing is not None:
                result = existing.result
                analysis_id = existing.id
            else:
                ...
                analysis_id = await analyses_repo.save(
                    hand_id=hand.id, result=result,
                    decision_points=hand.enriched.report.decision_points,
                )
                await session.commit()
            await SessionsRepo(session).set_last_analysis(job.session_id, analysis_id)
            await session.commit()
```

То же в `_run_screenshot` (1104-1119; путь мёртв с 2026-09-17, но держится в согласии). В `_run_hh_scan` после `save_scan_summary` (строка 722): `await SessionsRepo(session).set_last_analysis(job.session_id, None)` с комментарием «новый файл — агент больше не отвечает про старую руку (план агента, отклонение 2)».

Импорты в `pipeline.py`: `contextlib`; `SessionsRepo`; `from harness.agent.history import dump_history, load_history, trim_history`; `from harness.agent.run import run_agent_turn`; `from harness.explanation.agent_checks import AgentContext, agent_context`; `agent_turn_msg` из `harness.presentation`.

`worker/main.py`, `TelegramSender`:

```python
    async def typing(self, chat_id: int) -> None:
        """«Печатает…» в чате игрока (`sendChatAction`) — пока идёт ход агента."""
        await self._call("sendChatAction", {"chat_id": chat_id, "action": "typing"})
```

`main()`: `Deps(..., agent_history_max_tokens=cfg.agent_history_max_tokens)`.

- [ ] **Step 9: Тесты воркера** (`tests/test_worker_pipeline.py`)

`FakeSender` — счётчик и метод:

```python
        self.typings = 0

    async def typing(self, chat_id: int) -> None:
        self.typings += 1
```

```python
async def test_an_agent_job_answers_journals_and_keeps_the_evening_history(
    db_factory, queue, deps, fake_sender, monkeypatch
):
    """Задача `agent` целиком: ответ игроку, запись хода в журнал, история вечера."""
    from pydantic_ai.models.function import FunctionModel

    from tests.test_agent import _GOOD, _VPIP, _TurnScript
    from tests.test_calcs import _open_from, _store

    monkeypatch.setattr("harness.agent.prompt.load_system_prompt", lambda: "СИНТ")
    player_id, session_id = await _make_scope(db_factory)
    async with db_factory() as session:
        for index in (1, 2):
            await _store(session, session_id=session_id,
                         raw=_open_from("CO", tournament_id="T1", hand_no=f"A{index}"))
        await session.commit()

    jid = await queue.enqueue(type="agent", player_id=player_id, session_id=session_id,
                              payload={"text": "как часто я вхожу в банк"})
    job = await queue.claim("w1")
    assert job is not None
    script = _TurnScript([([_VPIP], _GOOD)])
    await run_job(job, replace(deps, llm=LLM(_TEST_CFG, db_factory, model_override=FunctionModel(script))))

    assert (await job_status(db_factory, jid)) == "done"
    assert any(_GOOD in text and "Посчитано:" in text for text in _all_texts(fake_sender))
    assert fake_sender.typings >= 1
    async with db_factory() as session:
        history = (await session.execute(
            text("select agent_history from sessions where id = :s"), {"s": session_id}
        )).scalar_one()
        spans = (await session.execute(
            text("select spans from traces where job_id = :j"), {"j": jid}
        )).scalar_one()
    assert history
    turn = next(s for s in spans if s["name"] == "agent_turn")
    assert turn["state"] == "prose" and turn["calls"][0]["calc"] == "hero_frequency"
    assert turn["calls"][0]["zone"] == "strict" and turn["rewrite"] is False


@requires_fixtures
async def test_a_hand_analysis_becomes_the_agent_context_and_a_new_file_resets_it(
    db_factory, queue, deps
):
    """Разбор руки ставит указатель вечера; новый файл HH его сбрасывает
    (план агента, отклонение 2): агент не отвечает про старую руку после нового файла."""
    player_id, session_id = await _make_scope(db_factory)
    jid, _raw = await _seed_one_hand_deep_dive_job(
        db_factory, queue, player_id=player_id, session_id=session_id
    )
    job = await queue.claim("w1")
    assert job is not None
    await run_job(job, deps)
    assert (await job_status(db_factory, jid)) == "done"

    async def pointer() -> int | None:
        async with db_factory() as session:
            return (await session.execute(
                text("select last_analysis_id from sessions where id = :s"), {"s": session_id}
            )).scalar_one()

    async with db_factory() as session:
        analysis_id = (await session.execute(text("select id from analyses"))).scalar_one()
    assert await pointer() == analysis_id

    scan = await enqueue_hh_scan(queue, FIXTURE_DAILY, player_id=player_id, session_id=session_id)
    job = await queue.claim("w1")
    assert job is not None
    await run_job(job, deps)
    assert (await job_status(db_factory, scan)) == "done"
    assert await pointer() is None
```

Тест гейтится фикстурами: все существующие тесты `deep_dive` этого файла стоят на `FIXTURE_DAILY` (`_seed_one_hand_deep_dive_job`, строка 476). Без фикстур он пропущен — назвать это в отчёте; сам репозиторный механизм закреплён без фикстур в Task 2 (`test_the_last_analysis_pointer_is_set_and_reset`). Сверить, что `enqueue_hh_scan` (строка 206) требует файл на диске по пути — `FIXTURE_DAILY` и есть путь.

`tests/test_entrypoints.py` (там тесты `TelegramSender` через `httpx.MockTransport` — сверить имя помощника): `test_typing_sends_a_chat_action` — тело запроса `{"chat_id": 7, "action": "typing"}` на `/sendChatAction`.

- [ ] **Step 10: Запустить**

Run: `uv run pytest tests/test_agent.py tests/test_worker_pipeline.py tests/test_question_routing.py tests/test_entrypoints.py tests/test_calcs.py -q -ra` → PASS. Старые тесты `/ask` (`test_the_second_calculation_is_refused` и соседи) обязаны пройти без правок — `CalcRun()` по умолчанию прежний.
Run: `uv run pytest -q -ra` (смотреть `skipped`), `uv run ruff check . && uv run pyright` → чисто.

- [ ] **Step 11: Коммит**

```bash
git add src/harness/agent/__init__.py src/harness/agent/history.py src/harness/agent/prompt.py src/harness/agent/run.py src/harness/calcs/routing.py src/harness/worker/pipeline.py src/harness/worker/main.py src/harness/presentation/messages.py src/harness/presentation/__init__.py tests/conftest.py tests/test_agent.py tests/test_worker_pipeline.py tests/test_entrypoints.py
git commit -m "$(cat <<'EOF'
Агент в воркере: ход с расчётами и проверками, одна попытка переписать, история вечера и журнал

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

**Докстринги, которые становятся ложью:**
- `calcs/routing.py:18-22` (модульный, пункт 3 «Один вопрос — один расчёт»): дописать «для `/ask`; у агента потолок 8 (`CalcRun(max_calcs=…)`)».
- `calcs/routing.py:131-137` (докстринг `CalcRun`) — заменяется целиком (Step 4).
- `calcs/routing.py:147-153` (докстринг `_perform`): «единственную попытку не тратит» → «попытку из потолка не тратит».
- `worker/pipeline.py:129-132` (комментарий `_Station`) — станция `ask` по-прежнему у `/ask`; агент станций прогресса не показывает (индикатор «печатает»), дописать.
- `worker/pipeline.py:1279-1281` (комментарий в `_dispatch`): «`screenshot_analyze`/`eval_run` существуют в CHECK… станций для них этот воркер ещё не знает» — `screenshot_analyze` станцию знает давно; поправить на «`eval_run`».
- `worker/main.py:178-181` (докстринг `TelegramSender`): «только эти два эндпоинта» — эндпоинтов уже четыре (`sendMessage`, `editMessageText`, `sendPhoto`, `sendChatAction`).

---

### Task 5: Бот — маршрутизация текста, доступ, квота, подача трёх состояний, «не то посчитано»

**Агент:** `implementer` (Sonnet; читает `.claude/SESSIONS_UX.md` до правки), затем `reviewer` (маршрутизация входа и данные `eval_cases`).

**Files:**
- Modify: `src/harness/bot/handlers.py` — `handle_text` (872-936), `UI_CALLBACK_PREFIXES` (187-201), `_dispatch_ui` (1084-1145), новые `agent_enabled`, `_agent_miss_reply`; импорт `AGENT_MISS_PREFIX`, новых сообщений
- Modify: `src/harness/bot/router.py` — `on_text` (284-297): «печатает» при `None`
- Modify: `src/harness/presentation/messages.py` — `_quota_line` (507-508), `quota_exceeded_msg` (1726-1734), тело `agent_turn_msg` (из Task 4), `agent_refusal_msg`, `agent_miss_saved_msg`; общий сборщик чисел расчёта из `question_msg` (2901-2930)
- Modify: `src/harness/presentation/keyboards.py` — `AGENT_MISS_PREFIX`, `agent_miss_button`
- Modify: `src/harness/presentation/__init__.py` — реэкспорт
- Test: `tests/test_bot_handlers.py`, `tests/test_presentation.py`, `tests/test_calcs.py` (бот не грузит агента)

**Interfaces:**
- Consumes: Task 2 (`JobsRepo.agent_turn`, `EvalCasesRepo.add(hand_id=None)`, квота с `agent`), Task 4 (`agent_turn_msg` — подпись, `trace.record("agent_turn", …)` — ключи `question`, `state`, `answer`, `calls`, `checks`, `rewrite`).
- Produces:
  - `agent_enabled(player: Player) -> bool` — **единственное место решения о доступе** (`return player.is_dev`); открытие агента игрокам = правка этой строки (Task 6)
  - `AGENT_MISS_PREFIX = "agentmiss:"`; `agent_miss_button(job_id: int) -> Btn`
  - `agent_turn_msg(...)` (подпись Task 4), `agent_refusal_msg() -> Msg`, `agent_miss_saved_msg() -> Msg`
  - `_quota_line` → `f"запросов {quota_left}/{quota_total} за 24 ч"`

- [ ] **Step 1: Падающие тесты** (`tests/test_bot_handlers.py`; `handle_text` импортируется локально, как в соседних тестах — `tests/test_bot_handlers.py:1320`)

```python
async def _make_dev(db_factory, player_id: int) -> None:
    async with db_factory() as session:
        await session.execute(text(f"update players set is_dev = true where id = {player_id}"))
        await session.commit()


async def test_free_text_of_a_dev_becomes_an_agent_job(db_factory, deps, invited):
    from harness.bot.handlers import handle_text

    await _make_dev(db_factory, invited)
    assert await handle_text(deps, _TG_USER_ID, "как часто я вхожу в банк") is None
    job = await fetch_one(db_factory, "select type, payload, priority from jobs")
    assert (job["type"], job["payload"], job["priority"]) == ("agent", {"text": "как часто я вхожу в банк"}, 100)


async def test_a_player_without_access_gets_the_old_answer_and_no_job(db_factory, deps, invited):
    from harness.bot.handlers import handle_text
    from harness.presentation import unknown_text_msg

    assert await handle_text(deps, _TG_USER_ID, "как часто я вхожу в банк") == unknown_text_msg()
    assert await fetch_all(db_factory, "select id from jobs") == []


async def test_a_menu_label_opens_its_screen_and_a_question_about_it_goes_to_the_agent(
    db_factory, deps, invited
):
    """Подпись кнопки — экран; вопрос со словами подписи — агенту. Совпадение точное."""
    from harness.bot.handlers import handle_text
    from harness.presentation import MENU_LEAKS

    await _make_dev(db_factory, invited)
    screen = await handle_text(deps, _TG_USER_ID, MENU_LEAKS)
    assert screen is not None
    assert await fetch_all(db_factory, "select id from jobs") == []
    assert await handle_text(deps, _TG_USER_ID, f"{MENU_LEAKS} за вечер — что там?") is None
    assert (await fetch_one(db_factory, "select type from jobs"))["type"] == "agent"


async def test_a_started_input_takes_the_text_before_the_agent(db_factory, deps, invited):
    from harness.bot.handlers import handle_text

    await _make_dev(db_factory, invited)
    async with db_factory() as session:
        await PlayersRepo(session).set_pending_input(invited, {"kind": "gg_nickname"})
        await session.commit()
    msg = await handle_text(deps, _TG_USER_ID, "SynthNick")
    assert msg is not None and "SynthNick" in msg.text
    assert await fetch_all(db_factory, "select id from jobs") == []


async def test_text_while_an_escalation_waits_for_a_number_stays_with_the_escalation(deps, db_factory):
    """Принятый край (спека §3): вопрос во время ручного ответа получает «не число»."""
    from harness.bot.handlers import handle_escalation_callback, handle_text
    from harness.presentation import vision_answer_not_a_number_msg

    job_id, _hand_id = await _awaiting_job(db_factory, deps)
    player = await fetch_one(db_factory, f"select player_id from jobs where id = {job_id}")
    await _make_dev(db_factory, player["player_id"])
    await handle_escalation_callback(deps, _TG_USER_ID, f"escalate:{job_id}:pot:manual")
    assert await handle_text(deps, _TG_USER_ID, "как часто я вхожу в банк") == vision_answer_not_a_number_msg()
    assert (await fetch_one(db_factory, "select count(*) as n from jobs where type = 'agent'"))["n"] == 0


async def test_an_agent_message_over_the_quota_is_refused(db_factory, deps, invited):
    from harness.bot.handlers import handle_text

    await _make_dev(db_factory, invited)
    async with db_factory() as session:
        await session.execute(text(f"update players set quota_daily = 0 where id = {invited}"))
        await session.commit()
    msg = await handle_text(deps, _TG_USER_ID, "как часто я вхожу в банк")
    assert msg is not None and "запросов" in msg.text.lower()
    assert await fetch_all(db_factory, "select id from jobs") == []


async def test_an_agent_text_longer_than_the_limit_is_refused(db_factory, deps, invited):
    from harness.bot.handlers import handle_text

    await _make_dev(db_factory, invited)
    msg = await handle_text(deps, _TG_USER_ID, "а" * (MAX_QUESTION_CHARS + 1))
    assert msg is not None and str(MAX_QUESTION_CHARS) in msg.text


async def test_the_wrong_calculation_button_writes_an_eval_case_without_a_hand(db_factory, deps, invited):
    from harness.bot.handlers import handle_ui_callback
    from harness.memory.models import Job, Trace
    from harness.presentation import AGENT_MISS_PREFIX, agent_miss_saved_msg

    player_id, session_id = await _seed_player(db_factory)
    async with db_factory() as session:
        job = Job(type="agent", player_id=player_id, session_id=session_id,
                  payload={"text": "как часто я вхожу в банк"}, status="done")
        session.add(job)
        await session.flush()
        session.add(Trace(job_id=job.id, spans=[{
            "name": "agent_turn", "question": "как часто я вхожу в банк", "state": "prose",
            "answer": "синт", "calls": [{"calc": "hero_frequency"}], "checks": [[]], "rewrite": False,
        }]))
        await session.commit()
        job_id = job.id

    assert await handle_ui_callback(deps, _TG_USER_ID, f"{AGENT_MISS_PREFIX}{job_id}") == agent_miss_saved_msg()
    case = await fetch_one(db_factory, "select kind, hand_id, source, ground_truth from eval_cases")
    assert (case["kind"], case["hand_id"], case["source"]) == ("agent_routing", None, "agent_miss_button")
    assert case["ground_truth"]["question"] == "как часто я вхожу в банк"
    assert case["ground_truth"]["calls"] == [{"calc": "hero_frequency"}]


async def test_the_wrong_calculation_button_of_another_player_writes_nothing(db_factory, deps, invited):
    from harness.bot.handlers import handle_ui_callback
    from harness.memory.models import Job
    from harness.presentation import AGENT_MISS_PREFIX

    other_id, other_session = await _seed_player(db_factory, tg_user_id=_TG_USER_ID + 1)
    async with db_factory() as session:
        job = Job(type="agent", player_id=other_id, session_id=other_session, payload={"text": "x"})
        session.add(job)
        await session.commit()
        job_id = job.id
    await handle_ui_callback(deps, _TG_USER_ID, f"{AGENT_MISS_PREFIX}{job_id}")
    assert await fetch_all(db_factory, "select id from eval_cases") == []
```

`invited` и `_seed_player` оба заводят игрока с `_TG_USER_ID` — сверить, что `get_or_create` возвращает того же игрока и тест не плодит двух.

Квота — правка существующих: `tests/test_bot_handlers.py:1534` (`разборов` → `запросов`), `tests/test_presentation.py:374, 821, 1645` (то же); `test_a_question_does_not_spend_the_daily_quota` (2257-2267) остаётся — `/ask` по-прежнему бесплатен; его докстринг («Счётчик подписан «разборов X/Y»») переписать: «вопрос `/ask` в счётчик не входит — решение принято до агента и планом агента не пересматривается».

`tests/test_presentation.py`:

```python
def test_the_agent_answer_names_every_calculation_and_carries_the_button():
    from harness.presentation import AGENT_MISS_PREFIX, agent_turn_msg
    from tests.test_question_text import _frequency, _leaks

    msg = agent_turn_msg("prose", "Синтетический ответ.", [_frequency(), _leaks()], from_context=False, job_id=7)
    assert msg.text.startswith("Синтетический ответ.")
    assert "Посчитано: ваша частота, типы расхождений." in msg.text
    assert [b.callback_data for row in msg.buttons for b in row] == [f"{AGENT_MISS_PREFIX}7"]


def test_the_agent_numbers_without_prose_show_the_calculation_and_the_button():
    from harness.presentation import agent_turn_msg
    from tests.test_question_text import _frequency

    msg = agent_turn_msg("numbers", None, [_frequency()], from_context=False, job_id=7)
    assert "5 из 7" in msg.text and msg.buttons


def test_the_agent_refusal_lists_what_can_be_counted_and_has_no_button():
    from harness.presentation import agent_turn_msg

    msg = agent_turn_msg("refusal", None, [], from_context=False, job_id=7)
    assert "Посчитать могу:" in msg.text and not msg.buttons


def test_an_answer_from_the_hand_says_so_in_its_signature():
    from harness.presentation import agent_turn_msg

    msg = agent_turn_msg("prose", "Синт.", [], from_context=True, job_id=7)
    assert "Посчитано: разбор раздачи." in msg.text


def test_a_long_agent_answer_fits_one_message_by_cutting_the_prose():
    from harness.presentation import agent_turn_msg
    from tests.test_question_text import _frequency

    msg = agent_turn_msg("prose", "слово " * 2000, [_frequency()], from_context=False, job_id=7)
    assert len(msg.text) <= 4096
    assert "Посчитано: ваша частота." in msg.text and "5 из 7" in msg.text
```

`_frequency()` (`tests/test_question_text.py:37`) — 5 из 7, `_leaks()` (`:68`). Сверить точный вывод `_measured` («71.4% (5 из 7)»).

`tests/test_calcs.py` — рядом с `test_the_bot_does_not_load_the_dictionary` (551): тот же subprocess, множество `{'harness.agent', 'harness.calcs', 'pokerkit', 'eval7'}` → `test_the_bot_does_not_load_the_agent` (или дописать `harness.agent` в существующий набор и обновить докстринг).

- [ ] **Step 2: Запустить** → FAIL.

- [ ] **Step 3: Presentation**

`keyboards.py`, рядом с `DISAGREE_PREFIX` (71):

```python
AGENT_MISS_PREFIX = "agentmiss:"


def agent_miss_button(job_id: int) -> Btn:
    """«Не то посчитано» под ответом агента — зеркало «Не согласен», но про маршрут
    (EVALS, этаж 5). Несёт номер задачи: ход лежит в её трейсе."""
    return Btn(text="Не то посчитано", callback_data=f"{AGENT_MISS_PREFIX}{job_id}")
```

`messages.py`:

```python
def _quota_line(quota_left: int, quota_total: int) -> str:
    return f"запросов {quota_left}/{quota_total} за 24 ч"
```

`quota_exceeded_msg`: «Дневной лимит разборов исчерпан. Следующий будет доступен через…» → «Дневной лимит запросов исчерпан. Следующий запрос — через {hours_to_free} ч, лимит считается за скользящие 24 ч.»

Числа расчёта — вынести из `question_msg` (2915-2924) в `_calc_numbers(result: CalcResult) -> list[str]` (ветки `isinstance` как есть); `question_msg` зовёт его — поведение `/ask` не меняется.

```python
def agent_turn_msg(
    state: Literal["prose", "numbers", "refusal"],
    prose: str | None,
    results: Sequence[CalcResult],
    *,
    from_context: bool,
    job_id: int,
) -> Msg:
    """Ответ агента — одно из трёх состояний (спека агента №1, §7).

    Подпись «Посчитано: …» собирает код по `CalcResult.calc` всех расчётов
    хода, модель в ней не участвует. В тесноте режется проза, не числа и не
    подпись (`test_a_long_agent_answer_fits_one_message_by_cutting_the_prose`).
    """
    if state == "refusal":
        return agent_refusal_msg()
    if results:
        names = ", ".join(_CALC_WORD[r.calc] for r in results)
        signature = f"Посчитано: {names}."
    elif from_context:
        signature = "Посчитано: разбор раздачи."
    else:
        signature = "Посчитано: расчёты выше в этом разговоре."
    numbers = [line for result in results for line in _calc_numbers(result) if line]
    tail = "\n".join([signature, *numbers])
    head = ""
    if state == "prose" and prose and prose.strip():
        budget = _TELEGRAM_TEXT_LIMIT - len(tail) - 2
        head = _fitted(prose.strip(), budget) + "\n\n"
    elif state == "numbers" and not results:
        head = "Ответ не сошёлся с числами разбора — спросите иначе.\n\n"
    return Msg(text=head + tail, buttons=[[agent_miss_button(job_id)]])


def agent_refusal_msg() -> Msg:
    """Ни расчёта, ни разбора — текста модели ни строки, и список того, что есть."""
    return Msg(text="На это у меня нет ни расчёта, ни разбора.\n\n" + _WHAT_CAN_BE_COUNTED)


def agent_miss_saved_msg() -> Msg:
    return Msg(text="Записал: посчитано не то. Это пойдёт в проверку маршрутизации.")
```

Тексты «Посчитано: разбор раздачи.», «расчёты выше в этом разговоре», «Ответ не сошёлся…», «Записал…» — решение имплементатора по брифу; назвать в отчёте как вопрос владельцу (единый голос). `_fitted` при отрицательном бюджете отдаёт только маркер — сверить, что при огромном списке чисел сообщение всё равно ≤ 4096; если нет, урезать и числа тем же `_fitted` с маркером, и закрепить тестом.

- [ ] **Step 4: Бот**

```python
def agent_enabled(player: Player) -> bool:
    """Достаётся ли свободный текст агенту. Единственное место решения о доступе:
    открытие агента игрокам (после спеки №2) — правка этой строки, и вместе с ней
    `/ask` у всех уходит в подсказку (Task 6 плана агента)."""
    return player.is_dev
```

`handle_text` — после ветки эскалации (строка ~932), вместо последних трёх строк:

```python
        if not agent_enabled(player) or not answer:
            reply = await _apply_pending_input(db, player, pending, answer)
            await db.commit()
            return reply
        if len(answer) > MAX_QUESTION_CHARS:
            await db.commit()
            return question_too_long_msg(MAX_QUESTION_CHARS)
        player_id = player.id
        await db.commit()

    quota = await check_quota(deps, player_id)
    if not quota.allowed:
        return quota_exceeded_msg(quota.hours_to_free)
    async with deps.db_factory() as db:
        session_id = (await SessionsRepo(db).active_or_create(player_id)).id
        await db.commit()
    await deps.queue.enqueue(
        type="agent", player_id=player_id, session_id=session_id, payload={"text": answer}
    )
    return None
```

Квота — до заведения сессии, тот же порядок и та же оговорка о мягком пределе, что у `handle_deep_dive_callback` (`handlers.py:657-670`).

`UI_CALLBACK_PREFIXES` += `AGENT_MISS_PREFIX`. `_dispatch_ui`, перед `return None`:

```python
    if data.startswith(AGENT_MISS_PREFIX):
        return await _agent_miss_reply(db, player, data.removeprefix(AGENT_MISS_PREFIX))
```

```python
async def _agent_miss_reply(db: AsyncSession, player: Player, raw_id: str) -> Msg:
    """«Не то посчитано»: ход агента — в `eval_cases` кандидатом в набор маршрутизации.

    Номер задачи приходит из кнопки (внешний мир) и сверяется с игроком в
    `JobsRepo.agent_turn` (`test_the_wrong_calculation_button_of_another_player_writes_nothing`).
    Руки у вопроса может не быть — `hand_id` пуст.
    """
    job_id = _button_number(raw_id)
    record = None if job_id is None else await JobsRepo(db).agent_turn(job_id, player.id)
    if record is None:
        return analysis_unavailable_msg()
    await EvalCasesRepo(db).add(
        kind="agent_routing",
        hand_id=None,
        ground_truth={
            "job_id": record.job_id,
            "trace_id": record.trace_id,
            "question": record.question,
            "calls": record.turn.get("calls", []),
            "answer": record.turn.get("answer"),
            "state": record.turn.get("state"),
        },
        source="agent_miss_button",
    )
    return agent_miss_saved_msg()
```

`analysis_unavailable_msg()` для чужой/пропавшей задачи — сверить его текст (он про разбор руки); если не подходит по смыслу — завести `agent_turn_unavailable_msg()` в `presentation` и назвать в отчёте.

`router.py`, `on_text`:

```python
        msg = await handle_text(deps, message.from_user.id, message.text)
        if msg is None:
            # Задача агента поставлена; дальше говорит воркер. «Печатает…» —
            # сразу, ещё до того, как воркер возьмёт задачу из очереди.
            await bot.send_chat_action(message.chat.id, "typing")
            return
```

Докстринг `on_text` («ник в руме либо число…») → «кнопка меню, начатый ввод, число по эскалации или вопрос агенту — решает `handle_text`».

- [ ] **Step 5: Запустить**

Run: `uv run pytest tests/test_bot_handlers.py tests/test_presentation.py tests/test_calcs.py -q -ra` → PASS.
Run: `uv run pytest -q -ra`, `uv run ruff check . && uv run pyright` → чисто.

- [ ] **Step 6: Коммит**

```bash
git add src/harness/bot/handlers.py src/harness/bot/router.py src/harness/presentation/messages.py src/harness/presentation/keyboards.py src/harness/presentation/__init__.py tests/test_bot_handlers.py tests/test_presentation.py tests/test_calcs.py
git commit -m "$(cat <<'EOF'
Бот: свободный текст is_dev — агенту, квота «запросов», три состояния ответа и кнопка «не то посчитано»

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

**Докстринги, которые становятся ложью:**
- `bot/handlers.py:873-901` (докстринг `handle_text`, первая строка «кнопка меню, начатый ввод, число по эскалации — в этом порядке») — дописать четвёртое звено «агент, если `agent_enabled`», и абзац «Ник больше не берётся из первого текста… текст без начатого ввода получает честное «не понял»» — «…у игрока без доступа к агенту».
- `bot/handlers.py:694-698` (докстринг `handle_question_command`, «Свободный текст в этом продукте уже занят…») — правится в Task 6.
- `bot/handlers.py:701-704` («Дневной лимит разборов вопрос не тратит… счётчик подписан «разборов X/Y»») → «…счётчик подписан «запросов X/Y»; `/ask` в него не входит».
- `presentation/messages.py:2611-2621` (`unknown_text_msg`, «Текст, который ничего не значит») — дописать «у игрока без доступа к агенту».
- `bot/router.py:286-291` (`on_text`) — Step 4.
- `presentation/messages.py:2901-2913` (`question_msg`) — числа теперь через `_calc_numbers`, смысл докстринга верен; проверить.

---

### Task 6: Снятие `/ask` — переключатель сейчас, удаление кода при открытии

**Агент:** `implementer`, затем `reviewer`.

Спека §10: «код удаления готовится и проверяется здесь, включается тем же переключателем доступа». План делает это так: **поведение «`/ask` снят» включается уже сейчас для игроков с доступом** (`agent_enabled`), а удаление мёртвого кода — отдельный коммит в момент открытия, по списку ниже. Один переключатель — одна функция `agent_enabled` (Task 5); открытие = `return True` в ней + коммит удаления.

**Files:**
- Modify: `src/harness/bot/handlers.py` — `handle_question_command` (691-730)
- Modify: `src/harness/presentation/messages.py` — `ask_moved_msg`
- Modify: `src/harness/presentation/__init__.py`
- Test: `tests/test_bot_handlers.py`

**Interfaces:**
- Consumes: `agent_enabled` (Task 5).
- Produces: `ask_moved_msg() -> Msg`.

- [ ] **Step 1: Падающие тесты**

```python
async def test_ask_of_a_player_with_the_agent_points_to_plain_text(db_factory, deps, invited):
    """У игрока с агентом `/ask` снят: вопрос пишется текстом, задача `question` не ставится."""
    from harness.presentation import ask_moved_msg

    await _make_dev(db_factory, invited)
    assert await handle_question_command(deps, _TG_USER_ID, "плюсовой ли я на BB") == ask_moved_msg()
    assert await fetch_all(db_factory, "select id from jobs") == []


async def test_ask_of_a_player_without_the_agent_still_works(db_factory, deps, invited):
    assert await handle_question_command(deps, _TG_USER_ID, "плюсовой ли я на BB") is None
    assert (await fetch_one(db_factory, "select type from jobs"))["type"] == "question"
```

Второй тест дублирует `test_a_question_becomes_a_job_for_the_worker` (2243) по смыслу — если так, не добавлять, а дописать в докстринг существующего «у игрока без доступа к агенту».

- [ ] **Step 2: Запустить** → FAIL.

- [ ] **Step 3: Реализация**

`handle_question_command`, сразу после проверки игрока:

```python
        if agent_enabled(player):
            await db.commit()
            return ask_moved_msg()
```

```python
def ask_moved_msg() -> Msg:
    """`/ask` у игрока с агентом: вопрос пишется просто текстом (спека агента №1, §10)."""
    return Msg(text="Команда больше не нужна — спрашивайте просто текстом.")
```

Докстринг `handle_question_command`: абзац «**Командой, а не свободным текстом.**» → «У игрока с доступом к агенту (`agent_enabled`) команда снята и отвечает подсказкой; у остальных живёт до открытия агента (спека агента №1, §10)».

- [ ] **Step 4: Запустить, коммит**

Run: `uv run pytest tests/test_bot_handlers.py -q -ra` → PASS; `uv run pytest -q -ra`, `uv run ruff check . && uv run pyright`.

```bash
git add src/harness/bot/handlers.py src/harness/presentation/messages.py src/harness/presentation/__init__.py tests/test_bot_handlers.py
git commit -m "$(cat <<'EOF'
/ask у игрока с агентом снят подсказкой; остальным живёт до открытия

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

**При открытии агента игрокам (после спеки №2) — один коммит, список сверен с деревом 2026-10-11:**
- `bot/handlers.py`: `agent_enabled` → `return True`; удалить `handle_question_command` (691-730), его строку в `__all__` (143), `MAX_QUESTION_CHARS` остаётся (им пользуется агент).
- `bot/router.py`: `on_ask` (177-185) и импорт `handle_question_command` (52).
- `worker/pipeline.py`: `_run_question` (1234-1265), ветка `question` в `_dispatch` (1269-1271), `"question"` в `_JOB_DEADLINE_S` (142), импорт `answer_question` (71), `question_msg`, `question_refusal_msg` из импорта `presentation` (116-117).
- `calcs/routing.py`: `answer_question`, `QuestionLLM`, `QuestionOutcome`, `_ONE_CALC_ONLY` и ветка `max_calcs == 1` в `_perform`; `__all__` (74).
- `explanation/question.py`: `question_prompt`, `check_answer`, `QuestionDraft`, `_PROMPT_PATH` (выжимка `question_digest` ОСТАЁТСЯ — её зовёт `_perform`); промпт `explanation/prompts/question.md` — на диске владельца и сервера, удалить там же; `tests/conftest.py:31, 51` (`PROMPT_QUESTION` в `PROMPTS_PRESENT`).
- `presentation/messages.py`: `question_msg`, `question_usage_msg`, `question_refusal_msg`, `question_too_long_msg` → переименовать в `agent_text_too_long_msg` (им пользуется агент), `ask_moved_msg`; реэкспорты `presentation/__init__.py`.
- `presentation/messages.py`: `help_msg` — строка `/ask ВОПРОС` → «спрашивайте просто текстом» (ревью задачи 6).
- `platform/queue.py`: `"question"` в `_DEFAULT_PRIORITY_BY_TYPE`; `platform/llm.py`: `"question_answer"` в `Literal`.
- Миграция 0015: `jobs.type` и `llm_calls.purpose` без `question`/`question_answer` (откат — по образцу 0009 наоборот; накопленные строки `question` переписать в `agent`? — решение владельца в тот день).
- Тесты: `tests/test_question_routing.py` (весь файл, кроме переезжающих в `test_agent.py` проверок инструментов: `test_the_tool_set_covers_every_calculation_name_exactly_once`, `test_every_tool_describes_its_parameters`, `test_an_unnamed_opponent_is_not_guessed`, `test_a_wrong_argument_value_is_handed_back_to_the_model`, `test_a_bad_parameter_does_not_spend_the_single_calculation` — перенести в агентские формулировки); `test_the_second_calculation_is_refused` уходит (спека §10), его смысл — в `test_the_ninth_calculation_is_refused`; `tests/test_question_text.py` — тесты `check_answer`; `tests/test_bot_handlers.py:2240-2305` (раздел `/ask`); `tests/test_worker_pipeline.py:1982-2060`.
- Документы: SESSIONS_UX раздел «Команда `/ask`», тех-спека таблица команд, ARCHITECTURE «Разговорный доступ» — финальная редакция без `/ask`.

---

### Task 7: Черновик системного промпта

**Кто:** оркестратор пишет черновик, владелец правит. Не `implementer`.

**Текст промпта в этом плане не приводится и в гит не попадает.**

**Files:**
- Create (не коммитится): `src/harness/agent/prompts/system.md`
- Modify: `.gitignore` — строка `src/harness/agent/prompts/` рядом со строками 40-41 (`parsers/prompts/`, `explanation/prompts/`). Pre-push хук путь уже закрывает (`.githooks/pre-push:21`), `.gitignore` — второй конец, чтобы файл не попал даже в локальный коммит.
- Modify: `docs/deploy-runbook.md` — раздел «приватные файлы» (около строк 140-170): каталогов промптов теперь три; `.dockerignore` (комментарий в шапке называет два каталога) — дописать третий; `artifacts` в `pyproject.toml:61` (`src/harness/**/prompts/*`) уже покрывает — не трогать.

- [ ] **Step 1:** `git status --porcelain | grep prompts` → пусто после создания файла (иначе `.gitignore` не сработал).
- [ ] **Step 2:** Черновик по разделам спеки §10, в этом порядке: роль (тренер MTT micro/low/middle, рум GG); правила утверждений — факт против допущения, буквальная метка `(допущение)`; процедура работы со статистикой (порог 20 возможностей конкретного стата, стат под спот, знаменатель всегда); процедура чтения диапазона по линии (стартовый — только из расчёта, сужение — допущение); запрет вердикта о лучшей линии на флопе и тёрне; суждение против диапазона, а не вскрытой карты; «лик» только из `leaks`; отказ от живых советов во время игры; нет PKO; словарь меток классификатора, которые встречаются в сводке (`check_fold`, `top_pair_weak_kicker`…), — русскими словами из `presentation/messages.py:1158-1194`; числа — только из сводки и расчётов, обозначения карт как есть.
- [ ] **Step 3:** `uv run pytest tests/test_agent.py -k system_prompt -q -ra` → PASS (тест перестал пропускаться).
- [ ] **Step 4:** Отдать владельцу на правку; правки — им в файл, пересборка образа (Task 9).
- [ ] **Step 5: Коммит** (только `.gitignore`, `.dockerignore`, runbook):

```bash
git add .gitignore .dockerignore docs/deploy-runbook.md
git commit -m "$(cat <<'EOF'
Промпт агента — третий закрытый каталог: .gitignore, .dockerignore, runbook

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Документы (спека §11)

**Кто:** оркестратор. Ревью — `reviewer` (докстринги и проза против кода).

**Files и правки** (строки — по дереву 2026-10-11):

- `CLAUDE.md:4-5` — «Место LLM в конвейере одно — ответ на вопрос игрока: там она ВЫБИРАЕТ расчёт из закрытого набора и излагает его выход» → «Место LLM в конвейере одно — агент над закрытым набором расчётов: на любой текст игрока после разбора он выбирает расчёты (до восьми на ответ) и излагает их выход и сводку разбора; каждое число сверяется кодом. В v1 — только `is_dev`».
- `.claude/ARCHITECTURE.md:359-391` («Разговорный доступ к своим данным»): вход — свободный текст (у `is_dev`; `/ask` у остальных до открытия); «третье — и последнее — место LLM» (369) → «место LLM — агент»; ограничение «один вопрос — один расчёт» (381-383) → «потолок 8 расчётов на ход; перебор срезов сдерживают правило „лик только из `leaks`“ и порог выборки 20 возможностей»; «нет подходящего расчёта — отказ» (384-385) → «ни расчёта в вечере, ни сводки разбора — отказ»; пятый пункт — история вечера и правило обрезки; шестой — проверки ответа (`explanation.agent_checks`) и одна попытка переписать.
- `.claude/SESSIONS_UX.md:475-500` (раздел «Команда `/ask`»): новый раздел «Свободный текст — агенту» перед ним: порядок (меню → начатый ввод → ручной ответ по эскалации → агент), принятый край с эскалацией, три состояния (с подписью «Посчитано: <список>», кнопка «не то посчитано»), «печатает…», квота «одно сообщение — одна единица, упавшее бесплатно»; раздел `/ask` сократить до «у игроков без доступа к агенту — до открытия». `:483-485` («Дневной лимит разборов вопрос не тратит…») — оставить для `/ask`, дописать про агента. `:363` (пример строки статуса «разборов 17/50») → «запросов 17/50». `:172` («ответы `/ask`») → «ответы агента и `/ask`». `:508-509` («Либо утверждение, либо недостающая выборка числом») — дописать правило агента: ниже 20 возможностей конкретного стата — упомянуть, не опираться; две «20» назвать явно (руки в реплее, возможности у агента).
- `.claude/EVALS.md:8-12` — «Место LLM… одно — выбор расчёта под вопрос игрока (`llm_calls.purpose = question_answer`)» → «агент (`purpose = agent`; `question_answer` — `/ask` до открытия)». Этаж 5 (`:99-128`): п.1 «Логировать выбранный расчёт» — сделано (`traces.spans`, запись `agent_turn`: вызовы, параметры, выход, зона, время, проверки, исход); п.2 «Кнопка „не то посчитано“» — сделана (`eval_cases.kind = agent_routing`, `source = agent_miss_button`, `hand_id` пуст у вопроса без руки); новый абзац «Граница механических проверок»: допущение, термины (по меткам хода, не по действию — отклонение 4) и лучшая линия ловятся по явным оборотам, остальное — судья (спека №2). «Порядок строительства», п.6 (`:139-141`) — первый шаг сделан.
- `docs/superpowers/specs/2026-08-28-poker-harness-tech-spec-design.md`:
  - `:502` (`sessions`) — колонки `agent_history jsonb?`, `last_analysis_id?` (FK `analyses`) и что их пишет;
  - `:511` (`eval_cases`) — `kind` += `agent_routing`, `source` += `agent_miss_button`, `hand_id?`;
  - `:517-523` (`jobs.type`) — `agent` (миграция 0014), `question` до снятия `/ask`;
  - `:532-541` (§7, фасад) — сигнатура с `message_history`, `instructions`; назначение `agent` (`LLM_AGENT_MODEL`);
  - `:555` («каждая попытка → строка в `llm_calls`») — «каждый запрос к провайдеру, включая запросы внутри прогона с инструментами»;
  - `:563-570` («Известный дефект: у запроса к модели нет собственного таймаута») — снять: таймаут на запрос (`LLM_REQUEST_TIMEOUT_S`, `_MeteredModel`), дедлайн лимитера и бюджет задачи — сверху;
  - `:653` (таблица команд, `/ask`) — «у игроков без доступа к агенту; у `is_dev` — подсказка»; `:658` (свободный текст) — «… → агент (`is_dev`), задача `agent`»; новая строка «„Не то посчитано“ (инлайн) | `_agent_miss_reply` | запись в `eval_cases` (kind=`agent_routing`)»;
  - `§5.7` (`:466-473`, «Проверка верности… Чисто цифровые пары… нужно решение, а не регэксп») — дописать: для агента решение принято (`agent_checks.strip_card_tokens`: вырезаются только однозначные карты — с мастями или буквами и борд через дефис; голая цифровая пара — всегда число); для текста вердикта §5.7 по-прежнему отложен.

- [ ] **Step 1:** Правки по списку; каждое утверждение о коде — сверить с кодом после задач 1-6 (а не с этим планом).
- [ ] **Step 2:** `grep -rn "один вопрос — один расчёт\|последнее — место LLM\|разборов [0-9]*/[0-9]*\|question_answer — единственный" CLAUDE.md .claude docs/superpowers/specs` → остались только сознательно оставленные (про `/ask`).
- [ ] **Step 3: Коммит**

```bash
git add CLAUDE.md .claude/ARCHITECTURE.md .claude/SESSIONS_UX.md .claude/EVALS.md docs/superpowers/specs/2026-08-28-poker-harness-tech-spec-design.md
git commit -m "$(cat <<'EOF'
Документы под агента: место LLM, свободный текст, квота «запросов», журнал и кнопка этажа 5, таймаут

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Приёмка — гейты, развёртывание на стенде, живая проверка владельцем

**Кто:** оркестратор; живую проверку делает владелец (`is_dev`).

- [ ] **Step 1: Гейты.** `uv run pytest -q -ra` — зелено; назвать строку `skipped` и сравнить с исходной (до плана): прирост допустим только на `requires_agent_prompt` (если промпта нет на машине прогона). **Зелёный прогон без фикстур не означает, что конвейер проверен** (CLAUDE.md). `uv run ruff check . && uv run pyright` — чисто.
- [ ] **Step 2: Промпт на месте.** `ls src/harness/agent/prompts/system.md` на машине сборки — ДО сборки образа (runbook: собранный без промпта образ доставкой файла не чинится).
- [ ] **Step 3: SHA в образе.** Сегодня образ коммита не несёт (в `Dockerfile` нет `LABEL`, сверено) — правило CLAUDE.md «сверить SHA коммита работающего контейнера» проверять нечем. Предложение владельцу (одна правка, отдельный коммит, ревью не нужно): в финальную стадию `Dockerfile` — `ARG GIT_SHA=unknown` и `LABEL org.opencontainers.image.revision=$GIT_SHA`. Если владелец против — проверка по косвенным признакам Step 5.
- [ ] **Step 4: Сборка и подъём.**

```bash
docker compose build --build-arg GIT_SHA=$(git rev-parse HEAD)
docker compose run --rm migrate
docker compose up -d
docker compose ps
```

- [ ] **Step 5: Сверка того, что работает новый код.**

```bash
git rev-parse HEAD
docker inspect --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}' $(docker compose ps -q worker)
docker inspect --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}' $(docker compose ps -q bot)
docker compose exec -T postgres psql -U harness -d harness -tAc 'select version_num from alembic_version'   # 0014
docker compose exec -T worker python -c "import harness.agent.run, pathlib, harness.agent.prompt as p; print(p.SYSTEM_PROMPT_PATH.exists())"   # True
```

Метки обоих контейнеров = `git rev-parse HEAD`; без метки — `docker inspect --format '{{.Created}}'` позже `git log -1 --format=%cI`. Только после этого докладывать.

- [ ] **Step 6: Живая проверка владельцем** (`is_dev`), в Телеграме:
  1. Разобрать руку («разобрать» под сводкой HH) → спросить текстом про эту руку («почему здесь фолд?») — ответ из сводки, подпись «Посчитано: разбор раздачи.», кнопка «не то посчитано».
  2. «Как часто я вхожу в банк за вечер?» → «Посчитано: ваша частота.», знаменатель есть.
  3. «Мои лики за вечер» текстом → вызван `leaks`; слово «лик» допустимо.
  4. Вопрос, на который нечего посчитать без разбора, сразу после `/new` → отказ со списком.
  5. Нажать «не то посчитано» → `select kind, hand_id, source from eval_cases order by id desc limit 1;` → `agent_routing | | agent_miss_button`.
  6. Кнопка нижнего меню «Мои лики» → экран, а не агент.
  7. `/ask …` у владельца → подсказка «спрашивайте текстом».
  8. Задержка: «печатает…» видно; ответ в пределах ~30 с — замерить по `llm_calls` (`select purpose, status, latency_ms from llm_calls where purpose='agent' order by id desc limit 10;`) и назвать числа, а не «быстро».
- [ ] **Step 7:** Итог владельцу: что прошло, что нет, числа задержки и доля ответов в состоянии 2 за сессию проверки (`traces.spans` → `agent_turn.state`).

---

## Открытые вопросы владельцу

1. **Идентификатор Sonnet 5.5 у провайдера.** Умолчание `anthropic:claude-sonnet-5-5` в списке известных имён установленной pydantic-ai отсутствует (там до `claude-sonnet-5`). Подтвердить строку или задать `LLM_AGENT_MODEL` в `.env`.
2. **Потолок истории** — умолчание 30 000 оценочных токенов (байты JSON / 3). Размер контекста Sonnet 5.5 и цену хода назвать, чтобы выбрать число.
3. **Числа из вопроса игрока** (отклонение 5): разрешать ли числа, которые игрок сам написал в этом ходе.
4. **Повтор упавшего хода.** Упавшая задача `agent` ждёт `reap()` до 10 минут и повторяется до трёх раз (`run_job`, `max_attempts` по умолчанию 3) — для разговора это долго. Оставить общее правило или ставить `agent` с одной попыткой (нужен параметр `max_attempts` у `enqueue`).
5. **Тексты, решённые имплементатором** (Task 5–6): подписи «Посчитано: разбор раздачи.» / «расчёты выше в этом разговоре», текст отказа, «Записал: посчитано не то…», «Команда больше не нужна…», подпись кнопки «Не то посчитано».
6. **`LABEL` с SHA в `Dockerfile`** (Task 9, Step 3).
