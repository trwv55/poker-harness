# Агент: диапазон соперника по чарту и эквити против него — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** На вопрос «стоило ли ставить?» агент пишет третий абзац ответа по расчётам, а не по своим знаниям: стартовый диапазон соперника берёт код из чарта (опен или защита BB, на столе 4–7 мест — 8-max по позиции от конца), агент сужает его целыми классами с пометкой «(допущение)», эквити героя против суженного считает ядро; назначение ставки и порог фолдов уже лежат в сводке разбора; первый эталон `evals/agent/` записан и закрыт хуком.

**Architecture:** Справочник `analysis/charts` получает второй вид книги — чарты защиты (`DefendBook`) с той же схемой файла и общей функцией ближайшей глубины; файла владельца ещё нет, и отсутствие файла — «чартов нет», а не падение. Чистый модуль `analysis/opponent_range.py` читает из руки линию соперника до флопа, глубину, раскладку и анте, берёт стартовый диапазон из чарта, проверяет сужение (подмножество, веса чарта) и мёртвые карты. Два новых имени в закрытом наборе `CalcName` — `opponent_range` (`strict`) и `equity_vs_range` (`assuming`) — исполняются в `calcs` (руку разбора читает `AnalysesRepo` по указателю вечера с проверкой владельца, эквити — Монте-Карло `analysis.tools.equity` под потолком итераций и с `calc_cache`). Инструментами они подключены только агенту: `/ask` остаётся с шестью. Выходы идут модели через `NumberBook`; проверка ответа различает пересказ чарта (факт) и эквити против суженного (допущение) по метаданным выходов. Сводка разбора (`agent_context`) печатает назначение и порог фолдов каждой постфлоп-точки героя; числа порога служат опорой арифметики порога.

**Tech Stack:** Python 3.12, pydantic 2, pydantic-ai 2.38.0 (`Tool`, `ToolReturn(metadata=…)`, `FunctionModel` в тестах), SQLAlchemy 2 async + Postgres (`calc_cache`, без миграций), eval7 (через `analysis.tools.equity`), pytest.

**Spec:** [docs/superpowers/specs/2026-10-14-agent-range-slice-design.md](../specs/2026-10-14-agent-range-slice-design.md) (контекст — [оболочка агента](../specs/2026-10-11-agent-shell-design.md))

**Имплементатору.** Каждое утверждение плана о коде сверено с рабочим деревом на 2026-10-14 (коммит `e57a324`), но дерево живое: **открой файл и сверь строку до правки**. Расхождение плана с кодом — вопрос оркестратору, а не повод «подогнать».

## Отклонения от спеки — на решение владельца

1. **Неполный стол для опенов уже сделан — §5 «это меняет поведение вердикта» неверно.** Коммит `d245843` (решение владельца 2026-10-10): `open_chart.chart_table` (`src/harness/analysis/open_chart.py:115-133`) судит стол на 4–7 мест 8-max чартом по числу игроков позади, ровно по таблице §5; модульный докстринг `open_chart.py:20-25` и `charts/lookup.py:22-25` это уже говорят. Значит, сетка 318 рук новых вердиктов открытия первым **не покажет**: код вердикта в этом срезе не меняется. План делает другое (Task 2): тестом закрепляет таблицу §5 **целиком** (сегодня в параметризации `tests/test_open_chart.py:470-492` нет ни одной строки 5 мест, нет SB и BB для 6 и BB для 7), подключает то же правило к защите (`defend_table`) и прогоном сетки доказывает, что вердикты не сдвинулись; владельцу — счётчик точек 4–7 мест, судимых 8-max с `d245843`, для сведения.
2. **Стартовый диапазон по чарту иначе не пройдёт проверку ответа.** Сегодня предложение со словом «диапазон», «рендж», «держ…» требует «(допущение)» по умолчанию (`agent_checks._opponent_claim`, `:416-435`; модульный докстринг `:7-8` прямо говорит «в v1 диапазоны не считает ни один расчёт»). Спека §6.1 ставит чарту зону `strict` — пересказ стартового диапазона допущением не является. План (Task 7): слово диапазона не требует метки, если в предложении есть слово «чарт» и число из выхода `opponent_range` этого вечера; прогноз о сопернике в том же предложении метку требует по-прежнему. Спека молчит — нужно решение.
3. **Число эквити против суженного — только с меткой.** Спека §6.2 требует строку о допущении в выходе, но в проверке ответа говорит лишь «числа этих выходов — числа расчёта». Без механики эквити против суженного диапазона прошло бы без метки во фразе без слова «диапазон» («против этих рук у вас 31%»). План (Task 7): доля из выхода `equity_vs_range` в предложении без «(допущение)» — нарушение.
4. **Порог из сводки — опора арифметики порога.** §7: «вывод "ставка за фолд-эквити" становится фактом». Но «окупается, если сбрасывает чаще N%» проверка пропускает, только когда N — число **расчёта** (`calc_numbers`, `agent/run.py:115`), а числа сводки туда не входят. План (Task 3): `AgentContext.threshold_numbers` — числа порога из сводки — добавляются к `calc_numbers` хода.
5. **Где спека молчит — решение плана:**
   - **Глубина соперника** (§6.1 «глубину соперника в BB»): у открывшего — мера `open_depth_bb` от его места (его стек до анте, но не больше самого глубокого из не сбросивших до опена); у BB — меньший из стеков BB и открывшего до анте. Обе в BB руки; ниже 13 BB — отказ «зона пуш-фолда».
   - **Линии:** открывший — один рейз до флопа (рейз → `raise`, олл-ин → `allin` опен-чарта); BB — колл или 3-бет единственного опена без входа между ними. Лимп SB в опен-чарте есть (`limp`), но §6.1 называет лимп отказом — отказ. Колд-колл между опеном и BB, 3-бет для открывшего, 4-бет, колл не из BB — отказ.
   - **Улица точки решения** нужна обоим инструментам: стартовый диапазон от неё не зависит, а «комбо с учётом карт героя и борда» — зависит. На префлопе живы все за столом, поэтому эквити на префлопе на полном столе — отказ «мультивей» (это и есть правило §3 «не входит: мультивей»).
   - **«Компактная запись»** (§6.1): `to_notation` компактную форму не восстанавливает намеренно (`notation.py:59-74`); выход печатает классы по одному, смешанные — с долей чарта.
   - **Голые пары в выходе** (`99`, `22:0.5`) регистрируются числами: иначе выжимка печатала бы незарегистрированное число. Следствие: после `opponent_range` число «99» в ответе разрешено весь вечер. Решение R10 («пары модель пишет мастями или словами») не меняется.
   - **Указатель разбора** (`sessions.last_analysis_id`) подставляет код, как `session_id` в окне; модель передаёт только позицию соперника, улицу и классы. Нет разбора в вечере — инструмент отвечает отказом.
   - **Файл защиты отсутствует** — все лукапы защиты `ChartMissing` → «чарта для этого спота нет». Отсутствие опен-файла остаётся ошибкой (`test_absent_file_is_a_clear_error`, не трогается).
6. **Первый эталон хранит выгрузку руки, а не ссылку на разбор в базе** (§9 разрешает оба): номера `analyses.id` стенда не переносятся между базами, а прогонщик спеки №2 должен собрать сводку без стенда. Каталог случая — `case.json` + `hand.json` (`EnrichedHand`) + `analysis.json` (`AnalysisResult`).

## Global Constraints

Действуют в КАЖДОЙ задаче.

- **Правило зависимостей (CLAUDE.md).** `contracts`, `parsers`, `normalizer`, `engine`, `analysis`, `explanation`, `presentation` не импортируют Телеграм, БД и `harness.platform`. Про инфраструктуру знают `bot`, `worker`, `memory`, `platform` и не конвейерные `calcs`, `agent` (им нужна база). Новый `analysis/opponent_range.py` — чистый: рука и пути к чартам приходят аргументами. `explanation` не импортирует `analysis` и `presentation` (сегодня ни одного такого импорта, сверено `grep`).
- **Бот не импортирует `harness.agent` и `harness.calcs`** (`test_the_bot_does_not_load_the_dictionary`, `tests/test_calcs.py`; `test_bot_image_does_not_import_calculation_stack`, `tests/test_entrypoints.py`).
- **Числа — только из инструментов и сводки.** Каждое число ответа — из `NumberBook` сводки, выхода расчёта в истории вечера или реплики игрока; выжимка не печатает число мимо книги. Текст модели не правится «чтобы сошлось».
- **Стартовый диапазон — только из чарта.** Ни модель, ни код не подставляют «похожий» чарт: соседняя позиция, раскладка, анте — отказ (`ChartMissing`); исключение одно — правило неполного стола §5 (решённое правило, не допущение).
- **Сужение — подмножество стартового, целыми классами.** Класс вне стартового диапазона или с нулевой долей в чарте — отказ; веса задаёт чарт, модель — нет.
- **Зона по инструменту:** `opponent_range` — `strict`, `equity_vs_range` — `assuming` (`_TOOL_ZONE`, журнал хода).
- **Суммы шансов не пишутся** (решение владельца 2026-10-14): эквити — только против диапазона и только из ядра. Код это не проверяет — промпт (Task 8) и эталон (Task 9).
- **Судить решение против диапазона, а не против вскрытой карты (CLAUDE.md).** Карты соперника не читаются ни одной новой функцией.
- **Репозиторий публичный.** В тестах, докстрингах и коммитах — только синтетика: метки `P1…P7`, номера рук `SYN…`, `TM99` (уже принятый синтетический номер, `tests/test_verdict_text.py:110`). Никаких реальных `TM…`. Карты руки-образца из спеки в тестах не повторять — свои синтетические.
- **Промпты не коммитятся.** `src/harness/[^/]+/prompts/` закрыт `.githooks/pre-push:21` и `.gitignore:40-42`. Текст промпта в этом плане не приводится — только названия разделов (Task 8).
- **`evals/agent/` — приватный датасет.** Добавляется в `FORBIDDEN_PATHS` хука и в `.gitignore` (Task 9) **до** того, как в каталоге появится первый файл.
- **Докстринг утверждает только закреплённое тестом.** Докстринги, которые задача делает ложью, правятся той же задачей — список в каждой задаче.
- **Тексты игроку — только в `presentation/messages.py`**, на «вы», «BB» латиницей вплотную к числу, знаменатель всегда (SESSIONS_UX).
- **Команды проверки:** `uv run pytest -q -ra` (смотреть строку `skipped`: без фикстур пропусков 5, без промптов — плюс гейтированные; число не должно расти сверх тех, что добавляет план — план не добавляет ни одного гейтированного теста), затем `uv run ruff check . && uv run pyright`.
- **Коммиты локальные, без пуша.** Сообщение по-русски, последняя строка `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. `git add` — явный список файлов, никогда `-A`.
- **Кто пишет:** задачи 1–7 — `implementer-core` (Opus; Task 5 трогает и печать игроку — тексты дословно из плана, потому что типы, выходы и печать связаны pyright'ом и разнести их по двум агентам нельзя), 8–11 — оркестратор. После каждой задачи с логикой — `reviewer` (Opus); бриф ревьюеру включает раздел «Review Focus» дословно.

## Review Focus

Пять классов входа, на которых спека кусается, а тесты легко промахиваются. Каждый закреплён тестом в задаче-владельце.

1. **Неполный стол — таблица §5 целиком, и для защиты тоже.** 5 мест, SB и BB, позиция открывшего в чарте защиты по тому же правилу; 3 и 9 мест — только своя запись. Тесты: `test_the_short_table_rule_covers_the_whole_table_of_the_spec`, `test_the_defence_maps_the_opener_by_the_same_rule`, `test_three_and_nine_handed_defences_keep_their_own_table` (Task 2), `test_a_three_handed_defence_has_no_chart`, `test_a_short_table_defence_reads_the_eight_max_chart_by_position_from_the_end` (Task 4).
2. **Линия соперника вне чартов — отказ словами, а не «похожий» диапазон.** Лимп-банк, лимп до опена, колд-колл между опеном и BB, 3-бет после опена (для открывшего), 4-бет, колл не из BB, позиция героя, позиции нет за столом, стек мельче 13 BB. Тест: `test_lines_outside_the_charts_are_refused` (Task 4), `test_a_short_stack_is_push_fold_not_a_chart` (Task 4).
3. **Сужение — подмножество и веса чарта.** Класс с долей 0 в чарте — вне стартового; отрезок, частично выходящий за старт, — отказ с именами классов; веса от модели — отказ; оставленные классы сохраняют долю чарта. Тесты: `test_a_class_with_zero_weight_in_the_chart_is_outside_the_start`, `test_a_class_outside_the_start_is_refused_by_name`, `test_weights_from_the_agent_are_refused`, `test_narrowing_keeps_the_chart_weights_of_the_kept_classes` (Task 4).
4. **Мёртвые карты, мультивей, детерминизм.** Комбо без карт героя и борда улицы; полностью заблокированный диапазон — отказ, а не эквити 0; три живых на улице — отказ; один и тот же вход — одно число, второй раз из `calc_cache`. Тесты: `test_dead_cards_leave_the_combos`, `test_a_fully_blocked_range_is_refused_not_zero`, `test_multiway_equity_is_refused` (Task 4), `test_equity_is_deterministic_and_cached` (Task 5).
5. **Факт против допущения в проверке ответа.** Пересказ стартового диапазона по чарту без метки — только со словом «чарт» и числом его выхода; прогноз в том же предложении метку требует; доля эквити против суженного — только с меткой, даже без слова «диапазон»; порог из сводки — опора арифметики порога. Тесты: `test_a_chart_range_sentence_needs_no_mark`, `test_a_chart_fact_does_not_cover_a_prediction_in_the_same_sentence`, `test_an_equity_against_a_narrowed_range_needs_the_mark`, `test_the_turn_checks_read_chart_and_assumed_numbers_from_history` (Task 7), `test_a_threshold_from_the_summary_backs_profit_arithmetic` (Task 3).

---

### Task 1: Чарты защиты — схема, загрузчик, лукап с общей ближайшей глубиной

**Агент:** `implementer-core`, затем `reviewer`.

Опен-чарты читает `ChartBook` (`src/harness/analysis/charts/lookup.py:215-292`); правило ближайшей глубины живёт внутри `ChartBook.nearest` (`:228-251`). Защите нужно то же правило — не копией: оно выносится в функцию `_nearest_depth`, которую зовут обе книги. Проверки файла (`_parse_file` `:351-376`, `_strategy_of` `:379-413`) тоже разбираются на общие куски. Файл владельца `defend_8max.json` ещё не существует — загрузчик отдаёт пустую книгу.

**Files:**
- Modify: `src/harness/analysis/charts/lookup.py` — модульный докстринг (`:1-43`), константы (после `:79`), `ChartKey` (`:112-116`, рядом `DefendKey`), `OpenStrategy` (`:119-146`, рядом `DefendStrategy`), `ChartEntry._charted_depth` (`:179-187`), `ChartBook.nearest` (`:228-251`), `_parse_file` (`:351-376`), `_strategy_of` (`:379-413`); новые `DefendEntry`, `DefendFileModel`, `DefendBook`, `load_defend_book`, `defend_strategy`, `_nearest_depth`, `_read_model`, `_check_shares`, `_check_declared`.
- Modify: `src/harness/analysis/charts/__init__.py` — докстринг (`:1-19`), реэкспорт (`:23-65`).
- Modify: `tests/test_charts.py` — новый раздел «чарты защиты» в конце файла (после `:358`).

**Interfaces:**
- Produces:
  - `DEFAULT_DEFEND_PATH: Path` = `charts/data/defend_8max.json`.
  - `class DefendKey(NamedTuple): seats: int; position: str; vs_position: str; depth_bb: float; ante_type: str`.
  - `@dataclass(frozen=True) class DefendStrategy: call_range: Range; raise_range: Range; allin_range: Range; def fold_weight(self, cls: str) -> float`.
  - `class DefendBook: path; all_keys() -> list[DefendKey]; nearest(seats: int, position: str, vs_position: str, eff_bb: float, ante_type: str) -> DefendKey; entry(key: DefendKey) -> DefendEntry; get(key: DefendKey) -> DefendStrategy`.
  - `load_defend_book(path: Path | None = None) -> DefendBook` — нет файла → пустая книга.
  - `defend_strategy(seats: int, position: str, vs_position: str, eff_bb: float, ante_type: str, *, path: Path | None = None) -> DefendStrategy`.
  - Отказы — те же классы: `ChartMissing`, `ChartPlaceholder`, `DepthNotCharted`, `ChartFileError`.
- Consumes: ничего нового.

- [ ] **Step 1: Падающие тесты** — в конец `tests/test_charts.py` (импорт расширить: `DefendKey`, `defend_strategy`, `load_defend_book`):

```python
# --- чарты защиты ---------------------------------------------------------------
#
# Схема — опен-чарта с двумя отличиями (спека 2026-10-14, §4): ключ несёт
# `vs_position`, действия — `call`, `raise` (3-бет), `allin`, сверка — `defend_pct`.
# Синтетика: 22+ (78) + A2s-AQs (44) + K9o+ (48) = 170 комбо колла, AKs наполовину
# 3-бетом (2) — 172 комбо = 12.97%.


def _defend(**over: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "seats": 8,
        "position": "BB",
        "vs_position": "BTN",
        "depth_bb": 25,
        "ante_type": "per_player",
        "source": "тестовый источник",
        "revised_at": "2026-10-14",
        "call": "22+, A2s-AQs, K9o+",
        "raise": "AKs:0.5",
    }
    entry.update(over)
    return entry


def _write_defend(tmp_path: Path, entries: list[dict[str, object]], **file_fields: object) -> Path:
    payload: dict[str, object] = {"schema_version": 3, "entries": entries}
    payload.update(file_fields)
    path = tmp_path / "defend.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_a_defence_entry_gives_three_actions_and_folds_the_rest(tmp_path: Path):
    path = _write_defend(tmp_path, [_defend(defend_pct="13.0%")])
    strategy = defend_strategy(8, "BB", "BTN", 25.0, "per_player", path=path)
    assert strategy.call_range.weight("A5s") == 1.0
    assert strategy.raise_range.weight("AKs") == 0.5
    assert strategy.allin_range.weights == {}
    assert strategy.fold_weight("AKs") == pytest.approx(0.5)
    assert strategy.fold_weight("72o") == 1.0


def test_the_defence_takes_the_nearest_depth_like_the_open(tmp_path: Path):
    """Одно правило на опен и защиту: ближайшая глубина, при равенстве — меньшая."""
    path = _write_defend(tmp_path, [_defend(depth_bb=20), _defend(depth_bb=30, call="22+")])
    book = load_defend_book(path)
    assert book.nearest(8, "BB", "BTN", 25.0, "per_player").depth_bb == 20.0
    assert book.nearest(8, "BB", "BTN", 26.0, "per_player").depth_bb == 30.0
    with pytest.raises(DepthNotCharted):
        book.nearest(8, "BB", "BTN", 12.0, "per_player")


def test_a_missing_defence_file_means_no_charts(tmp_path: Path):
    """Чарт защиты владелец кладёт позже: без файла лукап отказывает, а не падает."""
    absent = tmp_path / "нет-такого.json"
    assert load_defend_book(absent).all_keys() == []
    with pytest.raises(ChartMissing):
        defend_strategy(8, "BB", "BTN", 25.0, "per_player", path=absent)


def test_no_neighbour_defence_is_substituted(tmp_path: Path):
    path = _write_defend(tmp_path, [_defend()])
    with pytest.raises(ChartMissing):
        defend_strategy(8, "BB", "CO", 25.0, "per_player", path=path)
    with pytest.raises(ChartMissing):
        defend_strategy(8, "SB", "BTN", 25.0, "per_player", path=path)
    with pytest.raises(ChartMissing):
        defend_strategy(8, "BB", "BTN", 25.0, "bb_ante", path=path)


def test_a_defence_placeholder_is_refused(tmp_path: Path):
    path = _write_defend(tmp_path, [_defend(status="example")])
    assert DefendKey(8, "BB", "BTN", 25.0, "per_player") in load_defend_book(path).all_keys()
    with pytest.raises(ChartPlaceholder):
        defend_strategy(8, "BB", "BTN", 25.0, "per_player", path=path)


@pytest.mark.parametrize(
    "entries",
    [
        pytest.param([_defend(seats=4, vs_position="LJ")], id="открывшего нет за таким столом"),
        pytest.param([_defend(vs_position="BB")], id="защищающийся и открывший — одно место"),
        pytest.param([_defend(call="", **{"raise": ""})], id="защиты нет"),
        pytest.param([_defend(call="AKs", **{"raise": "AKs:0.5"})], id="доли руки больше 1"),
        pytest.param([_defend(defend_pct="15%")], id="defend_pct расходится с диапазонами"),
        pytest.param([_defend(limp="AA")], id="поле опен-чарта в чарте защиты"),
        pytest.param([_defend(depth_bb=10)], id="глубина ниже 15bb"),
        pytest.param([_defend(source=" ")], id="пустой source"),
        pytest.param([_defend(), _defend()], id="ключ дважды"),
        pytest.param([], id="ни одной записи"),
    ],
)
def test_a_malformed_defence_file_is_rejected(tmp_path: Path, entries: list[dict[str, object]]):
    with pytest.raises(ChartFileError):
        load_defend_book(_write_defend(tmp_path, entries))
```

- [ ] **Step 2:** `uv run pytest tests/test_charts.py -q -ra` → новые тесты FAIL (`ImportError: DefendKey`).

- [ ] **Step 3: Реализация.** В `lookup.py`:

После `DEFAULT_CHART_PATH` (`:79`):

```python
DEFAULT_DEFEND_PATH = Path(__file__).parent / "data" / "defend_8max.json"
```

После `ChartKey` (`:112-116`):

```python
class DefendKey(NamedTuple):
    seats: int
    position: str
    vs_position: str
    depth_bb: float
    ante_type: str


_K = TypeVar("_K", ChartKey, DefendKey)
```

После `OpenStrategy` (`:146`):

```python
@dataclass(frozen=True)
class DefendStrategy:
    """Эталонная защита против опена: колл, 3-бет рейзом, олл-ин; остальное — фолд."""

    call_range: Range
    raise_range: Range
    allin_range: Range

    def fold_weight(self, cls: str) -> float:
        """Доля фолда класса: всё, что не колл, не 3-бет и не олл-ин."""
        played = (
            self.call_range.weight(cls) + self.raise_range.weight(cls) + self.allin_range.weight(cls)
        )
        return max(0.0, 1.0 - played)
```

Проверку глубины вынести в функцию модуля (перед `ChartEntry`) и звать из обоих валидаторов:

```python
def _charted_depth(value: float) -> float:
    if not value >= MIN_CHART_DEPTH_BB:
        raise ValueError(
            f"глубина чарта {value}bb ниже {MIN_CHART_DEPTH_BB}bb — там эталон "
            f"считается равновесием пуш-фолда"
        )
    return value
```

`ChartEntry._charted_depth` (`:179-187`) → тело `return _charted_depth(value)` (имя метода сменить на `_depth`, чтобы не затенять функцию).

После `ChartFileModel` (`:207-212`):

```python
class DefendEntry(BaseModel):
    """Запись чарта защиты: ключ с позицией открывшего, провенанс и три действия.

    `extra="forbid"`: поле опен-чарта (`limp`, `open_pct`) здесь — ошибка загрузки
    (`test_a_malformed_defence_file_is_rejected`, «поле опен-чарта в чарте защиты»).
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    seats: int
    position: str
    vs_position: str
    depth_bb: float
    ante_type: str
    source: str
    revised_at: date
    status: Literal["chart", "example"] = "chart"
    call: str = ""
    raise_: str = Field(default="", alias="raise")
    allin: str = ""
    defend_pct: str | None = None

    @field_validator("ante_type", "source")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("поле обязательно и не может быть пустым")
        return value

    @field_validator("depth_bb")
    @classmethod
    def _depth(cls, value: float) -> float:
        return _charted_depth(value)

    @model_validator(mode="after")
    def _check_shape(self) -> DefendEntry:
        layout = POSITIONS_BY_COUNT.get(self.seats)
        if layout is None:
            raise ValueError(f"нет раскладки позиций для {self.seats} мест")
        for field, value in (("position", self.position), ("vs_position", self.vs_position)):
            if value not in layout:
                raise ValueError(
                    f"{field} {value!r} не встречается за столом на {self.seats} мест: {layout}"
                )
        if self.position == self.vs_position:
            raise ValueError("защищающийся и открывший — одна позиция")
        if not (self.call.strip() or self.raise_.strip() or self.allin.strip()):
            raise ValueError("защиты нет: пусты call, raise и allin")
        return self

    @property
    def key(self) -> DefendKey:
        return DefendKey(self.seats, self.position, self.vs_position, self.depth_bb, self.ante_type)


class DefendFileModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    readme: list[str] = []
    entries: list[DefendEntry]
```

Общая ближайшая глубина (перед `ChartBook`):

```python
def _nearest_depth(candidates: Sequence[_K], eff_bb: float, missing: str) -> _K:
    """Ключ с глубиной, ближайшей к стеку, среди ключей одной раскладки, позиций и
    анте; при равном расстоянии — меньшая глубина. Одно правило на опен и защиту
    (`test_the_nearest_charted_depth_is_taken`,
    `test_the_defence_takes_the_nearest_depth_like_the_open`)."""
    if not eff_bb >= MIN_LOOKUP_DEPTH_BB:
        raise DepthNotCharted(
            f"{eff_bb}bb ниже {MIN_LOOKUP_DEPTH_BB}bb — это зона пуш-фолда, "
            f"эталон там считается равновесием (analysis.tools.pushfold), а не чартом"
        )
    if not candidates:
        raise ChartMissing(missing)
    return min(candidates, key=lambda key: (abs(key.depth_bb - eff_bb), key.depth_bb))
```

`ChartBook.nearest` (`:228-251`) — тело заменить на:

```python
        return _nearest_depth(
            [
                key
                for key in self._entries
                if key.seats == seats and key.position == position and key.ante_type == ante_type
            ],
            eff_bb,
            f"нет ни одного чарта для (seats={seats}, position={position!r}, "
            f"ante_type={ante_type!r}) в {self.path}; соседняя позиция, раскладка "
            f"или тип анте не подставляются",
        )
```

После `ChartBook` (`:292`):

```python
class DefendBook:
    """Разобранный файл чартов защиты: точный ключ либо ближайшая снятая глубина."""

    def __init__(
        self, path: Path, entries: dict[DefendKey, tuple[DefendEntry, DefendStrategy]]
    ) -> None:
        self.path = path
        self._entries = entries

    def all_keys(self) -> list[DefendKey]:
        return list(self._entries)

    def nearest(
        self, seats: int, position: str, vs_position: str, eff_bb: float, ante_type: str
    ) -> DefendKey:
        return _nearest_depth(
            [
                key
                for key in self._entries
                if key.seats == seats
                and key.position == position
                and key.vs_position == vs_position
                and key.ante_type == ante_type
            ],
            eff_bb,
            f"нет чарта защиты {position} против {vs_position} для {seats} мест и анте "
            f"{ante_type!r} в {self.path}; соседний не подставляется",
        )

    def entry(self, key: DefendKey) -> DefendEntry:
        entry, _ = self._require(key)
        return entry

    def get(self, key: DefendKey) -> DefendStrategy:
        _, strategy = self._require(key)
        return strategy

    def _require(self, key: DefendKey) -> tuple[DefendEntry, DefendStrategy]:
        found = self._entries.get(key)
        if found is None:
            raise ChartMissing(f"нет чарта защиты по ключу {tuple(key)} в {self.path}")
        entry, strategy = found
        if entry.status == "example":
            raise ChartPlaceholder(
                f"запись {tuple(key)} помечена status='example' — образец формата, а не чарт"
            )
        return entry, strategy


_DEFEND_CACHE: dict[tuple[Path, int, int], DefendBook] = {}


def load_defend_book(path: Path | None = None) -> DefendBook:
    """Прочитать и проверить файл чартов защиты; кэш — как у `load_chart_book`.

    Файла нет — чартов защиты нет: пустая книга, и каждый лукап — `ChartMissing`
    (`test_a_missing_defence_file_means_no_charts`). Испорченный файл —
    `ChartFileError`, как у опенов.
    """
    target = (path or DEFAULT_DEFEND_PATH).resolve()
    try:
        stat = target.stat()
    except FileNotFoundError:
        return DefendBook(target, {})
    except OSError as exc:
        raise ChartFileError(f"файл чартов защиты не читается: {target} ({exc})") from exc
    cache_key = (target, stat.st_mtime_ns, stat.st_size)
    cached = _DEFEND_CACHE.get(cache_key)
    if cached is None:
        cached = _parse_defend_file(target)
        _DEFEND_CACHE[cache_key] = cached
    return cached


def defend_strategy(
    seats: int,
    position: str,
    vs_position: str,
    eff_bb: float,
    ante_type: str,
    *,
    path: Path | None = None,
) -> DefendStrategy:
    """Эталонная защита для стека: чарт ближайшей снятой глубины."""
    book = load_defend_book(path)
    return book.get(book.nearest(seats, position, vs_position, eff_bb, ante_type))
```

Разбор файла — общий кусок (заменяет `:352-367` в `_parse_file`):

```python
_M = TypeVar("_M", ChartFileModel, DefendFileModel)


def _read_model(path: Path, model: type[_M]) -> _M:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ChartFileError(f"файл чартов не читается: {path} ({exc})") from exc
    except json.JSONDecodeError as exc:
        raise ChartFileError(f"файл чартов не разобран как JSON: {path} ({exc})") from exc
    try:
        parsed = model.model_validate(payload)
    except ValidationError as exc:
        raise ChartFileError(f"файл чартов не прошёл проверку формы: {path}\n{exc}") from exc
    if parsed.schema_version != SCHEMA_VERSION:
        raise ChartFileError(
            f"версия схемы файла {parsed.schema_version}, код читает {SCHEMA_VERSION}: {path}"
        )
    return parsed


def _parse_defend_file(path: Path) -> DefendBook:
    model = _read_model(path, DefendFileModel)
    entries: dict[DefendKey, tuple[DefendEntry, DefendStrategy]] = {}
    for entry in model.entries:
        if entry.key in entries:
            raise ChartFileError(f"ключ {tuple(entry.key)} встречается дважды: {path}")
        where = f"запись {tuple(entry.key)} ({path})"
        strategy = DefendStrategy(
            call_range=_range_of(entry.call, "call", where),
            raise_range=_range_of(entry.raise_, "raise", where),
            allin_range=_range_of(entry.allin, "allin", where),
        )
        ranges = (strategy.call_range, strategy.raise_range, strategy.allin_range)
        _check_shares(ranges, where)
        _check_declared(
            entry.defend_pct, "defend_pct",
            100.0 * sum(r.fraction_of_hands() for r in ranges), where,
        )
        entries[entry.key] = (entry, strategy)
    if not entries:
        raise ChartFileError(f"в файле чартов защиты нет ни одной записи: {path}")
    return DefendBook(path, entries)
```

`_parse_file` начинается с `model = _read_model(path, ChartFileModel)`; остальное — как было (`:369-376`). `_strategy_of` (`:379-413`) — те же проверки через общие функции:

```python
def _check_shares(ranges: Sequence[Range], where: str) -> None:
    """Доли одной руки по действиям в сумме не больше 1."""
    classes = {cls for rng in ranges for cls in rng.weights}
    over = sorted(
        cls for cls in classes if sum(rng.weight(cls) for rng in ranges) > 1.0 + _SUM_TOLERANCE
    )
    if over:
        raise ChartFileError(f"{where}: доли действий в сумме больше 1 у {', '.join(over)}")


def _check_declared(declared: str | None, field: str, counted: float, where: str) -> None:
    """Заявленный процент совпадает с посчитанным до десятой — ловит опечатку переноса."""
    if declared is None:
        return
    match = _PERCENT.search(declared)
    if match is None:
        raise ChartFileError(f"{where}: {field} {declared!r} не содержит процента")
    if abs(float(match.group(1)) - counted) > _OPEN_PCT_TOLERANCE:
        raise ChartFileError(
            f"{where}: заявлено {match.group(1)}% ({field}), по диапазонам выходит "
            f"{counted:.2f}% — опечатка при переносе?"
        )
```

В `_strategy_of` после построения `strategy`: `_check_shares((strategy.raise_range, strategy.allin_range, strategy.limp_range), where)` и `_check_declared(entry.open_pct, "open_pct", 100.0 * (strategy.raise_range.fraction_of_hands() + strategy.allin_range.fraction_of_hands()), where)`. Импорты модуля: `from collections.abc import Sequence`, `from typing import Literal, NamedTuple, TypeVar`.

`charts/__init__.py`: в реэкспорт и `__all__` — `DEFAULT_DEFEND_PATH`, `DefendBook`, `DefendEntry`, `DefendKey`, `DefendStrategy`, `defend_strategy`, `load_defend_book`.

- [ ] **Step 4:** `uv run pytest tests/test_charts.py tests/test_open_chart.py -q -ra` → PASS (старые `test_malformed_file_is_rejected` и `test_the_nearest_charted_depth_is_taken` — без правок ожиданий; падение одного из них — поведение опенов сдвинулось, разобраться, а не править тест). Затем `uv run pytest -q -ra`, `uv run ruff check . && uv run pyright` → чисто.

- [ ] **Step 5: Коммит**

```bash
git add src/harness/analysis/charts/lookup.py src/harness/analysis/charts/__init__.py tests/test_charts.py
git commit -m "$(cat <<'EOF'
Справочник: чарты защиты — схема, загрузчик, лукап с общей ближайшей глубиной

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

**Докстринги, которые становятся ложью:**
- `charts/lookup.py:1` («Лукап опен-чартов: `(seats, position, ante_type)`…») → лукап опен-чартов и чартов защиты; абзац «Стратегия открытия — три действия» (`:32-36`) дополнить абзацем о защите (`call`/`raise`/`allin`, `defend_pct`, `vs_position` в ключе, нет файла — нет чартов, `test_a_missing_defence_file_means_no_charts`); список проверок формы (`:38-43`) — «пинит `test_malformed_file_is_rejected` и `test_a_malformed_defence_file_is_rejected`».
- `charts/__init__.py:1` («Справочник опен-диапазонов…») и `:7-9` (опен на глубоких стеках) → плюс чарты защиты BB, тот же каталог `charts/data/`.
- `ChartBook.nearest` (`:229-234`): правило ближайшей глубины описано в `_nearest_depth` — сослаться.

---

### Task 2: Неполный стол — таблица §5 целиком, то же правило для защиты, сетка

**Агент:** `implementer-core`, затем `reviewer` (объём ревью — только тесты и одна функция; массовый прогон рук — да, это гейт вердиктов).

Опенам правило уже работает (отклонение 1). Задача закрепляет таблицу §5 целиком и даёт защите ту же функцию, а не вторую копию.

**Files:**
- Modify: `src/harness/analysis/open_chart.py` — новая `defend_table` сразу после `chart_table` (`:115-133`); модульный докстринг `:20-25`.
- Modify: `tests/test_open_chart.py` — после `test_the_chart_table_counts_the_players_behind` (`:493-497`).

**Interfaces:**
- Produces: `defend_table(seats: int, position: str, vs_position: str) -> tuple[int, str, str]` — раскладка чарта, позиция защищающегося и открывшего в нём.
- Consumes: `chart_table(seats: int, position: str) -> tuple[int, str]` (`open_chart.py:115`).

- [ ] **Step 1: Падающие тесты**

```python
@pytest.mark.parametrize(
    ("seats", "position", "chart"),
    [
        (7, "UTG", "UTG+1"), (7, "UTG+1", "LJ"), (7, "HJ", "HJ"), (7, "CO", "CO"),
        (7, "BTN", "BTN"), (7, "SB", "SB"), (7, "BB", "BB"),
        (6, "UTG", "LJ"), (6, "HJ", "HJ"), (6, "CO", "CO"), (6, "BTN", "BTN"),
        (6, "SB", "SB"), (6, "BB", "BB"),
        (5, "UTG", "HJ"), (5, "CO", "CO"), (5, "BTN", "BTN"), (5, "SB", "SB"), (5, "BB", "BB"),
        (4, "UTG", "CO"), (4, "BTN", "BTN"), (4, "SB", "SB"), (4, "BB", "BB"),
    ],
)
def test_the_short_table_rule_covers_the_whole_table_of_the_spec(seats, position, chart):
    """Таблица §5 спеки 2026-10-14 строка в строку: стол на 4–7 мест — 8-max и
    позиция с тем же числом игроков позади до BB."""
    assert open_chart.chart_table(seats, position) == (8, chart)


@pytest.mark.parametrize(
    ("seats", "opener", "chart_opener"),
    [(7, "UTG", "UTG+1"), (6, "UTG", "LJ"), (5, "UTG", "HJ"), (4, "UTG", "CO"),
     (6, "BTN", "BTN"), (8, "LJ", "LJ")],
)
def test_the_defence_maps_the_opener_by_the_same_rule(seats, opener, chart_opener):
    assert open_chart.defend_table(seats, "BB", opener) == (8, "BB", chart_opener)


def test_three_and_nine_handed_defences_keep_their_own_table():
    """Меньше 4 и больше 8 мест — только своя запись (§5): позиции не сдвигаются."""
    assert open_chart.defend_table(3, "BB", "BTN") == (3, "BB", "BTN")
    assert open_chart.defend_table(9, "BB", "UTG+2") == (9, "BB", "UTG+2")
```

- [ ] **Step 2:** `uv run pytest tests/test_open_chart.py -q -ra` → `test_the_short_table_rule_…` PASS сразу (это закрепление, не новая логика — так и должно быть, отклонение 1), `defend_table` — FAIL (`AttributeError`).

- [ ] **Step 3: Реализация** (`open_chart.py`, после `chart_table`):

```python
def defend_table(seats: int, position: str, vs_position: str) -> tuple[int, str, str]:
    """Раскладка чарта защиты, позиция защищающегося и позиция открывшего в нём —
    то же правило неполного стола, что у опена (`chart_table`), к обоим местам
    (`test_the_defence_maps_the_opener_by_the_same_rule`,
    `test_three_and_nine_handed_defences_keep_their_own_table`)."""
    chart_seats, chart_position = chart_table(seats, position)
    _, chart_vs = chart_table(seats, vs_position)
    return chart_seats, chart_position, chart_vs
```

- [ ] **Step 4: Сетка — вердикты не сдвинулись.** На машине владельца с фикстурами (`fixtures/hh/`, вне репозитория). Скрипт — в scratchpad, не в репозиторий (печатает только счётчики; номера рук — только если что-то разошлось, и тогда в `reports/`, закрытый хуком):

```python
# <scratchpad>/open_chart_grid.py — запуск: uv run python <scratchpad>/open_chart_grid.py
from collections import Counter
from pathlib import Path

from harness.analysis import analyze_hand
from harness.engine import enrich
from harness.normalizer import normalize
from harness.parsers.hh_parser import parse_file

counts: Counter[tuple[int, bool, bool]] = Counter()
for path in sorted(Path("fixtures/hh").glob("*.txt")):
    for raw in parse_file(path.read_text(encoding="utf-8"), source_ref=path.name):
        en = enrich(normalize(raw))
        for point in analyze_hand(en).points:
            if point.spot == "open_chart":
                counts[(len(en.hand.players), "chart_seats" in point.detail, bool(point.best_action))] += 1
for key in sorted(counts):
    print(key, counts[key])
```

Прогнать на `e57a324` (`git stash` не нужен: до правки файла `open_chart.py` вывод тот же) и после коммита задачи; вывод обязан совпасть построчно. Затем `uv run pytest tests/test_regression_grid.py -q -ra` — строка `skipped` пуста (фикстуры на месте). Владельцу: «вердикты открытия первым не изменились; точек на столах 4–7 мест, судимых 8-max с `d245843`: <число из строк с `True` во втором поле>».

- [ ] **Step 5:** `uv run pytest -q -ra`, `uv run ruff check . && uv run pyright` → чисто.

- [ ] **Step 6: Коммит**

```bash
git add src/harness/analysis/open_chart.py tests/test_open_chart.py
git commit -m "$(cat <<'EOF'
Неполный стол: таблица §5 целиком в тестах и то же правило для защиты

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

**Докстринги, которые становятся ложью:**
- `open_chart.py:20-25` — дописать: «то же правило — у чарта защиты (`defend_table`)»; список тестов `chart_table` (`:118-124`) — добавить `test_the_short_table_rule_covers_the_whole_table_of_the_spec`.
- `charts/lookup.py:22-25` («вызывающая сторона явно спрашивает 8-max ключ… `open_chart.chart_table`») → «…`chart_table` у опена, `defend_table` у защиты».

---

### Task 3: Сводка разбора — назначение ставки и порог фолдов

**Агент:** `implementer-core`, затем `reviewer`.

Ядро уже кладёт назначение (`Line.purpose`) и порог (`PostflopLineDetail.fold_threshold`, только у блефа и полублефа — `postflop_line.py:163-165`, `_fold_threshold` `:390-410`) в `detail["postflop_line"]`. Сводка агента их не печатает (`agent_context`, `explanation/agent_checks.py:610-673`). Слова назначения — те же, что в разборе (`presentation/messages.py:1189-1197`, `_PURPOSE_WORD`), но `explanation` не импортирует `presentation` — копия словаря, равенство закреплено тестом.

**Files:**
- Modify: `src/harness/explanation/agent_checks.py` — импорт (`:96-101`, + `Purpose`), словарь после `_BACKDOOR_WORD` (`:551`), функция `_purpose_line` после `_hero_equity_line` (`:567-593`), `AgentContext` (`:598-607`), `agent_context` (`:610-673`).
- Modify: `src/harness/agent/run.py` — `_violations` (`:109-131`).
- Modify: `tests/test_agent_checks.py` (раздел «сводка разбора», после `:950`), `tests/test_agent.py` (нет — тест `_violations` чистый, кладётся в `tests/test_agent_checks.py`).

**Interfaces:**
- Produces: `AgentContext.threshold_numbers: frozenset[float] = frozenset()` (последнее поле, с умолчанием); строка сводки `Назначение, <улица>: <слово>[; …порог…].`
- Consumes: `PostflopLineDetail.line.purpose`, `PostflopLineDetail.fold_threshold` (`contracts/postflop.py:220-247`).

- [ ] **Step 1: Падающие тесты** (`tests/test_agent_checks.py`; импорты: `FoldThreshold`, `Purpose`, `SizeTag` из `harness.contracts`):

```python
def _bet_line(threshold: FoldThreshold | None, purpose: Purpose) -> dict[str, object]:
    detail = PostflopLineDetail(
        hand=None, draw=None, draw_missed=False, backdoors=[], overcards=None,
        line=Line(
            action=ActionTag.CBET, barrel=None, size_pct=0.32, size_tag=SizeTag.BLOCK,
            purpose=purpose,
        ),
        fold_threshold=threshold, draw_call=None, showdown=None,
    )
    return {POSTFLOP_LINE_DETAIL: detail.model_dump(mode="json")}


def _ctx(detail: dict[str, object]):
    return agent_context(
        _result([_point(dp_index=0, ev_diff_bb=-1.4, detail=detail)]), _postflop_hand()
    )


def test_the_context_names_the_purpose_and_the_fold_threshold_of_a_hero_bet():
    ctx = _ctx(_bet_line(FoldThreshold(bluff=0.24, semibluff=None, semibluff_free=False), Purpose.BLUFF))
    assert "Назначение, префлоп: блеф; ставка окупается, если соперник сбрасывает чаще 24%." in ctx.text
    assert 24.0 in ctx.allowed and ctx.threshold_numbers == frozenset({24.0})


def test_a_semibluff_names_both_thresholds_or_that_it_pays_without_folds():
    both = _ctx(_bet_line(FoldThreshold(bluff=0.24, semibluff=0.091, semibluff_free=False), Purpose.SEMIBLUFF))
    assert (
        "Назначение, префлоп: полублеф; порог фолдов как у чистого блефа 24%, "
        "с учётом аутов 9.1%." in both.text
    )
    assert both.threshold_numbers == frozenset({24.0, 9.1})
    free = _ctx(_bet_line(FoldThreshold(bluff=0.24, semibluff=0.0, semibluff_free=True), Purpose.SEMIBLUFF))
    assert "с учётом аутов ставка окупается и без фолдов." in free.text


def test_a_value_bet_names_its_purpose_without_a_threshold():
    ctx = _ctx(_bet_line(None, Purpose.VALUE))
    assert "Назначение, префлоп: вэлью." in ctx.text
    assert ctx.threshold_numbers == frozenset()


def test_the_purpose_words_are_the_words_of_the_hand_breakdown():
    """Копия словаря разбора (`explanation` не импортирует `presentation`)."""
    from harness.explanation.agent_checks import _PURPOSE_WORD as SUMMARY
    from harness.presentation.messages import _PURPOSE_WORD as BREAKDOWN

    assert SUMMARY == BREAKDOWN


def test_a_threshold_from_the_summary_backs_profit_arithmetic():
    """Порог в сводке — из ядра, не догадка: «окупается, если сбрасывает чаще 24%»
    с ним — арифметика порога (спека 2026-10-14, §7). Без `threshold_numbers` та
    же фраза — голый вывод без расчёта."""
    from harness.agent.run import _violations

    ctx = _ctx(_bet_line(FoldThreshold(bluff=0.24, semibluff=None, semibluff_free=False), Purpose.BLUFF))
    answer = "Ставка окупается, если соперник сбрасывает чаще 24%."
    assert _violations(answer, [], ctx, frozenset()) == []
    assert any(
        "в плюсе" in v
        for v in _check(answer, allowed=ctx.allowed, calc_numbers=frozenset(), labels=ctx.labels)
    )
```

- [ ] **Step 2:** `uv run pytest tests/test_agent_checks.py -q -ra` → новые FAIL.

- [ ] **Step 3: Реализация** (`agent_checks.py`):

```python
# Слова назначения — те же, что в разборе (`presentation.messages._PURPOSE_WORD`);
# копия, потому что `explanation` не импортирует `presentation`
# (`test_the_purpose_words_are_the_words_of_the_hand_breakdown`).
_PURPOSE_WORD: dict[Purpose, str] = {
    Purpose.VALUE: "вэлью",
    Purpose.MEDIUM_HAND: "ставка со средней рукой",
    Purpose.SEMIBLUFF: "полублеф",
    Purpose.BLUFF: "блеф",
    Purpose.CALL_STRONG: "колл с сильной рукой",
    Purpose.BLUFF_CATCH: "ловля блефа",
    Purpose.CALL_WITH_DRAW: "колл с дро",
}


def _purpose_line(
    line: PostflopLineDetail, street: Street, book: NumberBook, thresholds: NumberBook
) -> str | None:
    """Назначение и порог фолдов героя в точке — из ядра (`postflop_line`, §4.5–§4.6).
    Числа порога пишутся в обе книги: они числа сводки и опора арифметики порога
    (`AgentContext.threshold_numbers`;
    `test_the_context_names_the_purpose_and_the_fold_threshold_of_a_hero_bet`)."""
    played = line.line
    if played is None or played.purpose is None:
        return None
    text = f"Назначение, {_STREET_NAME[street]}: {_PURPOSE_WORD[played.purpose]}"
    fold = line.fold_threshold

    def pct(share: float) -> str:
        thresholds.pct(100.0 * share, trim=True)
        return book.pct(100.0 * share, trim=True)

    if fold is not None:
        if fold.semibluff_free:
            text += (
                f"; порог фолдов как у чистого блефа {pct(fold.bluff)}%, "
                "с учётом аутов ставка окупается и без фолдов"
            )
        elif fold.semibluff is not None:
            text += (
                f"; порог фолдов как у чистого блефа {pct(fold.bluff)}%, "
                f"с учётом аутов {pct(fold.semibluff)}%"
            )
        else:
            text += f"; ставка окупается, если соперник сбрасывает чаще {pct(fold.bluff)}%"
    return text + "."
```

`AgentContext` — последним полем `threshold_numbers: frozenset[float] = frozenset()`. В `agent_context`: перед циклом `thresholds = NumberBook()`, `purpose_lines: list[str] = []`; в цикле после `equity = …` — `purpose = _purpose_line(line, point.street, book, thresholds)`, `if purpose is not None: purpose_lines.append(purpose)`; в сборке текста (`:661-665`) части `(replay, verdict_text, "\n".join(label_lines), "\n".join(purpose_lines), equity_text)`; в `return` — `threshold_numbers=thresholds.allowed`.

`agent/run.py`, `_violations` — внутри `if context is not None:` добавить `calc_numbers |= context.threshold_numbers` (после строки `labels |= context.labels`, `:120`).

- [ ] **Step 4:** `uv run pytest tests/test_agent_checks.py tests/test_agent.py -q -ra` → PASS; особо: `test_the_context_registers_every_number_it_prints` (новая строка не печатает число мимо книги). `uv run pytest -q -ra`, `uv run ruff check . && uv run pyright` → чисто.

- [ ] **Step 5: Коммит**

```bash
git add src/harness/explanation/agent_checks.py src/harness/agent/run.py tests/test_agent_checks.py
git commit -m "$(cat <<'EOF'
Сводка агента: назначение ставки и порог фолдов из ядра; порог — опора арифметики

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

**Докстринги, которые становятся ложью:**
- `AgentContext` (`:599-601`): «текст, разрешённые числа, метки…» → плюс `threshold_numbers` — числа порога из ядра, опора арифметики порога.
- `agent_context` (`:611-618`): «метки классификатора всех точек и дро, бэкдоры, оверкарты героя» → плюс назначение и порог фолдов (`test_the_context_names_the_purpose_and_the_fold_threshold_of_a_hero_bet`).
- `check_agent_answer` (`:719-725`): «`calc_numbers` — числа из выходов расчётов разговора, без чисел сводки» → «…и числа порога сводки (`threshold_numbers`) — их добавляет вызывающий».

---

### Task 4: Ядро диапазона — линия соперника, стартовый диапазон, сужение, мёртвые карты

**Агент:** `implementer-core`, затем `reviewer` (мутации на правилах подмножества и линий).

Новый чистый модуль. Руку читает как есть (`CanonicalHand`): позиции — `PlayerState.position` (нормализатор), стеки — `PlayerState.stack` (стартовый, до постов: так его читает `classifier.table_state`, `:235`, `:268`), действия префлопа — `CanonicalAction` (посты блайндов в действия не входят, `contracts/canonical.py:75-79`). Карты соперника не читаются.

**Files:**
- Create: `src/harness/analysis/opponent_range.py`.
- Modify: `src/harness/contracts/calcs.py` — алиас `RangeLine` (после `POSITIONS`, `:74-76`) и `__all__` (`:40-65`); `src/harness/contracts/__init__.py` — реэкспорт (`:40-64`, `__all__` `:160`).
- Create: `tests/test_opponent_range.py`.

**Interfaces:**
- Produces (`harness.analysis.opponent_range`):
  - `class RangeRefused(ValueError)` — причина словами для модели.
  - `@dataclass(frozen=True) class RangeSpot: label: str; position: str; opener_position: str; line: RangeLine; seats: int; depth_bb: float; ante_type: str`.
  - `@dataclass(frozen=True) class StartRange: spot: RangeSpot; range: Range; chart: Literal["open", "defend"]; chart_seats: int; chart_position: str; vs_chart_position: str | None; chart_depth_bb: float`.
  - `@dataclass(frozen=True) class EquitySpot: hero: tuple[str, str]; board: list[str]; start: StartRange; narrowed: Range; live_combos: float; share_of_start: float` + свойство `cache_key: str`.
  - `range_spot(hand: CanonicalHand, position: str) -> RangeSpot`
  - `start_range(spot: RangeSpot, *, open_path: Path | None = None, defend_path: Path | None = None) -> StartRange`
  - `hero_cards(hand: CanonicalHand) -> list[str]`, `board_at(hand: CanonicalHand, street: Street) -> list[str]`, `live_at(hand: CanonicalHand, street: Street) -> list[str]`
  - `live_combos(rng: Range, dead: Collection[str]) -> float`
  - `narrow(start: Range, hands: str) -> Range`
  - `equity_spot(hand: CanonicalHand, position: str, street: Street, hands: str, *, open_path: Path | None = None, defend_path: Path | None = None) -> EquitySpot`
  - `contracts.RangeLine = Literal["open_raise", "open_allin", "call", "three_bet", "allin"]`.
- Consumes: `load_chart_book`, `load_defend_book`, `parse_range`, `to_notation`, `NotationError` (Task 1), `chart_table`, `defend_table` (Task 2), `combos_of_class` (`analysis/tools/equity.py:44`).

- [ ] **Step 1: Падающие тесты** — `tests/test_opponent_range.py`:

```python
"""Стартовый диапазон соперника по чарту и его сужение агентом — ядро без базы и сети.

Чарт защиты — синтетика во временном файле (файла владельца ещё нет); опен-чарт —
файл репозитория, ожидания сверяются с `open_strategy`, а не с числами чарта.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from harness.analysis.charts import open_strategy, parse_range
from harness.analysis.opponent_range import (
    RangeRefused,
    board_at,
    equity_spot,
    live_combos,
    narrow,
    range_spot,
    start_range,
)
from harness.contracts import Post, PostKind, Provenance, Range, RawHand, SeatInfo, Street
from tests.test_postflop_line import F, P, R, T, _bet, _call, _check, _fold, _played_out, _raise

_SB, _BB = 50, 100
# Кнопка — Hero: P5 — SB, P6 — BB, P1 — UTG, P2 — HJ, P3 — CO.
_SIX = ("P1", "P2", "P3", "Hero", "P5", "P6")
_BOARD = {F: ["Qc", "7d", "2s"], T: ["5h"], R: ["9s"]}
_DEFEND_CALL = "22+, A2s-AQs, KTs+, QTs+, JTs, ATo-AQo, KJo+"


def _raw(
    actions,
    *,
    labels: tuple[str, ...] = _SIX,
    button: str = "Hero",
    stacks: dict[str, int] | None = None,
    hero_cards: tuple[str, str] = ("Kd", "Tc"),
    boards: dict[Street, list[str]] | None = None,
) -> RawHand:
    stacks = stacks or {}
    at = labels.index(button)
    small, big = labels[(at + 1) % len(labels)], labels[(at + 2) % len(labels)]
    return RawHand(
        provenance=Provenance.HAND_HISTORY,
        source_ref="synthetic",
        hand_no="SYN-RANGE-1",
        tournament_id="TSYN",
        tournament_name="synthetic",
        level=1,
        sb=_SB,
        bb=_BB,
        ante=0,
        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        table_name="syn",
        max_seats=len(labels),
        button_seat=at + 1,
        seats=[SeatInfo(seat=n, label=label, stack=stacks.get(label, 10_000))
               for n, label in enumerate(labels, 1)],
        posts=[
            Post(label=small, kind=PostKind.SMALL_BLIND, amount=_SB),
            Post(label=big, kind=PostKind.BIG_BLIND, amount=_BB),
        ],
        dealt={"Hero": list(hero_cards)},
        actions=actions,
        boards=boards or _BOARD,
    )


def _hand(actions, **kw):
    return _played_out(_raw(actions, **kw)).hand


BTN_OPENS_BB_CALLS = [
    _fold(P, "P1"), _fold(P, "P2"), _fold(P, "P3"), _raise(P, "Hero", 250), _fold(P, "P5"),
    _call(P, "P6", 150), _check(F, "P6"), _bet(F, "Hero", 160), _call(F, "P6", 160),
]


def defend_file(tmp_path: Path, **over: object) -> Path:
    entry: dict[str, object] = {
        "seats": 8, "position": "BB", "vs_position": "BTN", "depth_bb": 25,
        "ante_type": "per_player", "source": "синтетика", "revised_at": "2026-10-14",
        "call": _DEFEND_CALL, "raise": "AKs, AKo:0.5",
    }
    entry.update(over)
    path = tmp_path / "defend.json"
    path.write_text(json.dumps({"schema_version": 3, "entries": [entry]}, ensure_ascii=False),
                    encoding="utf-8")
    return path


# --- линия соперника и стартовый диапазон ---------------------------------------


def test_the_bb_call_against_a_button_open_takes_the_defence_call_range(tmp_path: Path):
    spot = range_spot(_hand(BTN_OPENS_BB_CALLS), "BB")
    assert (spot.label, spot.line, spot.opener_position, spot.seats) == ("P6", "call", "BTN", 6)
    assert spot.depth_bb == 100.0
    start = start_range(spot, defend_path=defend_file(tmp_path))
    assert start.range == parse_range(_DEFEND_CALL)
    assert (start.chart, start.chart_seats, start.chart_position, start.vs_chart_position) == (
        "defend", 8, "BB", "BTN",
    )
    assert start.chart_depth_bb == 25.0


def test_a_bb_three_bet_takes_the_raise_range(tmp_path: Path):
    actions = [
        _fold(P, "P1"), _fold(P, "P2"), _fold(P, "P3"), _raise(P, "Hero", 250), _fold(P, "P5"),
        _raise(P, "P6", 800), _call(P, "Hero", 550),
    ]
    spot = range_spot(_hand(actions), "BB")
    assert spot.line == "three_bet"
    assert start_range(spot, defend_path=defend_file(tmp_path)).range == parse_range("AKs, AKo:0.5")


def test_the_opener_range_is_the_raise_range_of_the_open_chart():
    """Герой на BB защищается против опена BTN: диапазон BTN — рейз опен-чарта,
    стол на 6 мест — по 8-max BTN."""
    actions = [
        _fold(P, "P5"), _fold(P, "P6"), _fold(P, "P1"), _raise(P, "P2", 250), _fold(P, "P3"),
        _call(P, "Hero", 150),
    ]
    spot = range_spot(_hand(actions, button="P2"), "BTN")
    assert (spot.label, spot.line, spot.depth_bb) == ("P2", "open_raise", 100.0)
    start = start_range(spot)
    assert start.range == open_strategy(8, "BTN", 100.0, "per_player").raise_range
    assert (start.chart, start.chart_seats, start.chart_position) == ("open", 8, "BTN")


def test_a_short_table_defence_reads_the_eight_max_chart_by_position_from_the_end(tmp_path: Path):
    """6 мест: опен UTG — пятеро позади, как у LJ за 8-max (§5)."""
    actions = [
        _raise(P, "P1", 250), _fold(P, "P2"), _fold(P, "P3"), _fold(P, "Hero"), _fold(P, "P5"),
        _call(P, "P6", 150),
    ]
    start = start_range(
        range_spot(_hand(actions), "BB"), defend_path=defend_file(tmp_path, vs_position="LJ")
    )
    assert (start.chart_seats, start.vs_chart_position, start.spot.seats) == (8, "LJ", 6)


def test_a_three_handed_defence_has_no_chart(tmp_path: Path):
    actions = [_raise(P, "Hero", 250), _fold(P, "P2"), _call(P, "P3", 150)]
    spot = range_spot(_hand(actions, labels=("Hero", "P2", "P3")), "BB")
    with pytest.raises(RangeRefused, match="чарта для этого спота нет"):
        start_range(spot, defend_path=defend_file(tmp_path))


def test_without_the_defence_file_the_spot_has_no_chart(tmp_path: Path):
    with pytest.raises(RangeRefused, match="чарта для этого спота нет"):
        start_range(range_spot(_hand(BTN_OPENS_BB_CALLS), "BB"), defend_path=tmp_path / "нет.json")


def test_a_short_stack_is_push_fold_not_a_chart(tmp_path: Path):
    spot = range_spot(_hand(BTN_OPENS_BB_CALLS, stacks={"P6": 1_000}), "BB")
    assert spot.depth_bb == 10.0
    with pytest.raises(RangeRefused, match="пуш-фолд"):
        start_range(spot, defend_path=defend_file(tmp_path))


_LIMPED = [_call(P, "P1", 100), _fold(P, "P2"), _fold(P, "P3"), _fold(P, "Hero"),
           _call(P, "P5", 50), _check(P, "P6")]
_LIMP_THEN_RAISE = [_call(P, "P1", 100), _fold(P, "P2"), _fold(P, "P3"), _raise(P, "Hero", 400),
                    _fold(P, "P5"), _call(P, "P6", 300), _call(P, "P1", 300)]
_COLD_CALL = [_fold(P, "P1"), _fold(P, "P2"), _fold(P, "P3"), _raise(P, "Hero", 250),
              _call(P, "P5", 200), _call(P, "P6", 150)]
_FOUR_BET = [_fold(P, "P1"), _fold(P, "P2"), _fold(P, "P3"), _raise(P, "Hero", 250),
             _fold(P, "P5"), _raise(P, "P6", 800), _raise(P, "Hero", 2_000), _call(P, "P6", 1_200)]
_SB_CALLS = [_fold(P, "P1"), _fold(P, "P2"), _fold(P, "P3"), _raise(P, "Hero", 250),
             _call(P, "P5", 200), _fold(P, "P6")]
_HERO_BB_THREE_BETS = [_fold(P, "P5"), _fold(P, "P6"), _fold(P, "P1"), _raise(P, "P2", 250),
                       _fold(P, "P3"), _raise(P, "Hero", 800), _call(P, "P2", 550)]


@pytest.mark.parametrize(
    ("actions", "button", "position", "reason"),
    [
        pytest.param(_LIMPED, "Hero", "UTG", "никто не повышал", id="лимп-банк"),
        pytest.param(_LIMP_THEN_RAISE, "Hero", "BB", "лимп", id="лимп до опена"),
        pytest.param(_COLD_CALL, "Hero", "BB", "между опеном и BB", id="колд-колл перед BB"),
        pytest.param(_FOUR_BET, "Hero", "BB", "не колл и не 3-бет", id="4-бет"),
        pytest.param(_SB_CALLS, "Hero", "SB", "только для BB", id="колл не из BB"),
        pytest.param(_HERO_BB_THREE_BETS, "P2", "BTN", "3-бет", id="открывший после 3-бета"),
        pytest.param(BTN_OPENS_BB_CALLS, "Hero", "BTN", "героя", id="позиция героя"),
        pytest.param(BTN_OPENS_BB_CALLS, "Hero", "UTG+1", "нет позиции", id="позиции нет"),
    ],
)
def test_lines_outside_the_charts_are_refused(actions, button, position, reason):
    with pytest.raises(RangeRefused, match=reason):
        range_spot(_hand(actions, button=button), position)


# --- сужение и мёртвые карты -------------------------------------------------------

_START = Range(weights={"AA": 1.0, "KK": 1.0, "QQ": 0.0, "AKs": 0.5, "72o": 0.0})


def test_narrowing_keeps_the_chart_weights_of_the_kept_classes():
    assert narrow(_START, "AA, AKs") == Range(weights={"AA": 1.0, "AKs": 0.5})


def test_a_class_with_zero_weight_in_the_chart_is_outside_the_start():
    with pytest.raises(RangeRefused, match="72o"):
        narrow(_START, "AA, 72o")


def test_a_class_outside_the_start_is_refused_by_name():
    """«QQ+» — QQ, KK, AA: QQ в чарте с долей 0, сужение рук не добавляет."""
    with pytest.raises(RangeRefused, match="нет в стартовом диапазоне: QQ"):
        narrow(_START, "QQ+")


def test_weights_from_the_agent_are_refused():
    with pytest.raises(RangeRefused, match="веса"):
        narrow(_START, "AKs:1")


def test_dead_cards_leave_the_combos():
    rng = parse_range("AA, AKs")
    assert live_combos(rng, ()) == 10.0
    assert live_combos(rng, ("As",)) == 6.0  # AA: 3 без пиковой, AKs: 3
    assert live_combos(Range(weights={"AKs": 0.5}), ()) == 2.0


def test_the_board_is_the_board_of_the_street():
    hand = _hand(BTN_OPENS_BB_CALLS)
    assert board_at(hand, Street.PREFLOP) == []
    assert board_at(hand, Street.TURN) == ["Qc", "7d", "2s", "5h"]


def test_the_equity_spot_takes_the_street_board_and_the_share_of_the_start(tmp_path: Path):
    spot = equity_spot(
        _hand(BTN_OPENS_BB_CALLS), "BB", Street.FLOP, "QQ, 77, 22, AQs",
        defend_path=defend_file(tmp_path),
    )
    assert spot.hero == ("Kd", "Tc") and spot.board == ["Qc", "7d", "2s"]
    assert spot.live_combos == 12.0  # по 3 у каждой пары с картой на борде, AQs без Qc — 3
    dead = ("Kd", "Tc", "Qc", "7d", "2s")
    assert spot.share_of_start == pytest.approx(12.0 / live_combos(parse_range(_DEFEND_CALL), dead))


def test_a_fully_blocked_range_is_refused_not_zero(tmp_path: Path):
    hand = _hand(
        BTN_OPENS_BB_CALLS, hero_cards=("Kd", "Kc"),
        boards={F: ["Kh", "7d", "2s"], T: ["5h"], R: ["9s"]},
    )
    with pytest.raises(RangeRefused, match="заблокированы"):
        equity_spot(hand, "BB", Street.FLOP, "KK", defend_path=defend_file(tmp_path))


def test_multiway_equity_is_refused():
    """Три живых на флопе: мультивей — вне среза (§3)."""
    actions = [
        _fold(P, "P5"), _fold(P, "P6"), _fold(P, "P1"), _raise(P, "P2", 250), _call(P, "P3", 200),
        _call(P, "Hero", 150),
    ]
    with pytest.raises(RangeRefused, match="больше двух"):
        equity_spot(_hand(actions, button="P2"), "BTN", Street.FLOP, "AA")
```

- [ ] **Step 2:** `uv run pytest tests/test_opponent_range.py -q -ra` → FAIL (`ModuleNotFoundError`). Если какой-то `_hand(...)` падает в построении (движок не принял синтетику) — это ошибка фикстуры теста, поправить суммы в действиях, не код.

- [ ] **Step 3: Реализация.** `contracts/calcs.py` после `POSITIONS`:

```python
# Линия соперника до флопа, которую покрывают чарты (спека 2026-10-14, §6.1):
# опен рейзом или олл-ином — опен-чарт; колл, 3-бет, олл-ин BB против одного опена —
# чарт защиты. Остальное — отказ (`analysis.opponent_range.range_spot`).
RangeLine = Literal["open_raise", "open_allin", "call", "three_bet", "allin"]
```

`src/harness/analysis/opponent_range.py`:

```python
"""Стартовый диапазон соперника по чарту и его сужение агентом (спека 2026-10-14, §6).

**Стартовый — только из чарта.** Линию соперника до флопа, глубину, раскладку и
анте код читает из руки сам; модель называет только позицию соперника и улицу.
Опен-чарт даёт диапазон открывшего (рейз или олл-ин), чарт защиты — колл, 3-бет
или олл-ин BB против одного опена. Стол на 4–7 мест — 8-max по позиции от конца
(`open_chart.chart_table`, `defend_table`): это правило, а не допущение. Чего в
чартах нет — `RangeRefused` словами, и своего диапазона никто не подставляет
(`test_lines_outside_the_charts_are_refused`).

**Глубина** — в мерах чарта, стек до анте: у открывшего — его стек, но не больше
самого глубокого из не сбросивших до опена (та же мера, что `open_depth_bb`); у BB —
меньший из стеков BB и открывшего.

**Сужение — подмножество, целыми классами** (`narrow`): класс вне стартового или с
долей 0 — отказ; оставленные классы сохраняют долю чарта, модель весов не задаёт.

**Мёртвые карты** — карты героя и борд улицы — убираются из комбо (`live_combos`);
полностью заблокированный диапазон — отказ, а не эквити 0. Карты соперника и
вскрытие не читаются ни одной функцией.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from harness.analysis.charts import (
    ChartFileError,
    ChartMissing,
    ChartPlaceholder,
    DepthNotCharted,
    NotationError,
    load_chart_book,
    load_defend_book,
    parse_range,
    to_notation,
)
from harness.analysis.open_chart import chart_table, defend_table
from harness.analysis.tools.equity import combos_of_class
from harness.contracts import (
    ActionKind,
    CanonicalHand,
    Range,
    RangeLine,
    Street,
    all_classes,
)

__all__ = [
    "EquitySpot",
    "RangeRefused",
    "RangeSpot",
    "StartRange",
    "board_at",
    "equity_spot",
    "hero_cards",
    "live_at",
    "live_combos",
    "narrow",
    "range_spot",
    "start_range",
]

_STREETS: list[Street] = list(Street)
_HEADS_UP = 2


class RangeRefused(ValueError):
    """Диапазона по чарту нет — причина словами для модели. `ValueError`, чтобы
    `calcs.routing._perform` вернул её модели как «Расчёт не выполнен: …»."""


@dataclass(frozen=True)
class RangeSpot:
    label: str
    position: str
    opener_position: str
    line: RangeLine
    seats: int
    depth_bb: float
    ante_type: str


@dataclass(frozen=True)
class StartRange:
    spot: RangeSpot
    range: Range
    chart: Literal["open", "defend"]
    chart_seats: int
    chart_position: str
    vs_chart_position: str | None
    chart_depth_bb: float


@dataclass(frozen=True)
class EquitySpot:
    hero: tuple[str, str]
    board: list[str]
    start: StartRange
    narrowed: Range
    live_combos: float
    share_of_start: float

    @property
    def cache_key(self) -> str:
        """Ключ `calc_cache`: рука героя, борд, суженный диапазон по классам с долями."""
        return f"{''.join(self.hero)}|{''.join(self.board)}|{to_notation(self.narrowed)}"


def range_spot(hand: CanonicalHand, position: str) -> RangeSpot:
    """Линия соперника до флопа и ключ чарта — из руки; вне чартов — `RangeRefused`."""
    by_position = {p.position: p for p in hand.players}
    opponent = by_position.get(position)
    if opponent is None:
        raise RangeRefused(f"за этим столом нет позиции {position}")
    if opponent.label == hand.hero_label:
        raise RangeRefused(f"{position} — позиция героя, а не соперника")
    stack = {p.label: p.stack for p in hand.players}
    preflop = [a for a in hand.actions if a.street is Street.PREFLOP]
    raises = [a for a in preflop if a.kind is ActionKind.RAISE]
    if not raises:
        raise RangeRefused("до флопа никто не повышал — для лимп-банка чартов нет")
    opened_at = preflop.index(raises[0])
    if any(a.kind is ActionKind.CALL for a in preflop[:opened_at]):
        raise RangeRefused("до опена был лимп — такой линии в чартах нет")
    opener = raises[0].label
    opener_position = next(p.position for p in hand.players if p.label == opener)
    seats = len(hand.players)

    def spot(line: RangeLine, depth_chips: int) -> RangeSpot:
        return RangeSpot(
            label=opponent.label, position=position, opener_position=opener_position,
            line=line, seats=seats, depth_bb=depth_chips / hand.bb, ante_type=hand.ante_type,
        )

    if opponent.label == opener:
        if len(raises) > 1:
            raise RangeRefused("после опена был 3-бет — линии с 3-бетом для открывшего в чартах нет")
        folded = {a.label for a in preflop[:opened_at] if a.kind is ActionKind.FOLD}
        others = [s for label, s in stack.items() if label != opener and label not in folded]
        depth = min(stack[opener], max(others)) if others else stack[opener]
        return spot("open_allin" if raises[0].is_all_in else "open_raise", depth)
    if position != "BB":
        raise RangeRefused("чарты колла и 3-бета есть только для BB")
    own = [a for a in preflop if a.label == opponent.label]
    if not own or own[0].kind is ActionKind.FOLD:
        raise RangeRefused("BB сбросил карты до флопа")
    if any(a.kind is not ActionKind.FOLD for a in preflop[opened_at + 1 : preflop.index(own[0])]):
        raise RangeRefused("между опеном и BB кто-то вошёл — защиты против нескольких игроков в чартах нет")
    first = own[0]
    if first.kind is ActionKind.CALL and len(raises) == 1:
        line: RangeLine = "call"
    elif first.kind is ActionKind.RAISE and len(raises) == 2 and raises[1].label == opponent.label:
        line = "allin" if first.is_all_in else "three_bet"
    else:
        raise RangeRefused("линия BB до флопа — не колл и не 3-бет против одного опена: в чартах её нет")
    return spot(line, min(stack[opponent.label], stack[opener]))


def start_range(
    spot: RangeSpot, *, open_path: Path | None = None, defend_path: Path | None = None
) -> StartRange:
    """Стартовый диапазон по чарту ближайшей глубины. Отказы словами: зона пуш-фолда,
    чарта нет, образец формата, справочник не прочитался."""
    try:
        if spot.line in ("open_raise", "open_allin"):
            chart_seats, chart_position = chart_table(spot.seats, spot.position)
            book = load_chart_book(open_path)
            key = book.nearest(chart_seats, chart_position, spot.depth_bb, spot.ante_type)
            opened = book.get(key)
            rng = opened.raise_range if spot.line == "open_raise" else opened.allin_range
            return StartRange(spot, rng, "open", chart_seats, chart_position, None, key.depth_bb)
        chart_seats, chart_position, chart_vs = defend_table(
            spot.seats, spot.position, spot.opener_position
        )
        defence_book = load_defend_book(defend_path)
        defence_key = defence_book.nearest(
            chart_seats, chart_position, chart_vs, spot.depth_bb, spot.ante_type
        )
        defence = defence_book.get(defence_key)
        rng = {
            "call": defence.call_range,
            "three_bet": defence.raise_range,
            "allin": defence.allin_range,
        }[spot.line]
        return StartRange(
            spot, rng, "defend", chart_seats, chart_position, chart_vs, defence_key.depth_bb
        )
    except DepthNotCharted as exc:
        raise RangeRefused("стек в зоне пуш-фолда — чарта для этого спота нет") from exc
    except (ChartMissing, ChartPlaceholder) as exc:
        raise RangeRefused("чарта для этого спота нет") from exc
    except ChartFileError as exc:
        raise RangeRefused("справочник чартов не прочитался — чарта для этого спота нет") from exc


def hero_cards(hand: CanonicalHand) -> list[str]:
    return list(hand.dealt.get(hand.hero_label, []))


def board_at(hand: CanonicalHand, street: Street) -> list[str]:
    """Борд, открытый к улице точки решения; улицы, до которой раздача не дошла, —
    отказ."""
    if street is not Street.PREFLOP and not hand.boards.get(street):
        raise RangeRefused("до этой улицы раздача не дошла")
    upto = _STREETS[1 : _STREETS.index(street) + 1]
    return [card for s in upto for card in hand.boards.get(s, [])]


def live_at(hand: CanonicalHand, street: Street) -> list[str]:
    """Места, не сбросившие карты до начала улицы."""
    folded = {
        a.label
        for a in hand.actions
        if a.kind is ActionKind.FOLD and _STREETS.index(a.street) < _STREETS.index(street)
    }
    return [p.label for p in hand.players if p.label not in folded]


def live_combos(rng: Range, dead: Collection[str]) -> float:
    """Взвешенные комбо диапазона без мёртвых карт."""
    blocked = set(dead)
    return float(sum(
        weight * sum(1 for c1, c2 in combos_of_class(cls) if c1 not in blocked and c2 not in blocked)
        for cls, weight in rng.weights.items()
    ))


def narrow(start: Range, hands: str) -> Range:
    """Сужение агентом: классы без весов, только из стартового диапазона с ненулевой
    долей; доли — чарта."""
    if ":" in hands:
        raise RangeRefused("веса не задаются: перечислите классы рук, доли берутся из чарта")
    try:
        asked = parse_range(hands)
    except (NotationError, ValueError) as exc:
        raise RangeRefused(f"запись рук не разобрана: {exc}") from exc
    outside = [cls for cls in all_classes() if cls in asked.weights and start.weight(cls) <= 0.0]
    if outside:
        raise RangeRefused(
            "этих рук нет в стартовом диапазоне: " + ", ".join(outside)
            + " — сужение рук не добавляет"
        )
    return Range(weights={cls: start.weight(cls) for cls in asked.weights})


def equity_spot(
    hand: CanonicalHand,
    position: str,
    street: Street,
    hands: str,
    *,
    open_path: Path | None = None,
    defend_path: Path | None = None,
) -> EquitySpot:
    """Всё, что нужно расчёту эквити, кроме самого Монте-Карло: рука героя, борд
    улицы, стартовый и суженный диапазоны, их живые комбо."""
    cards = hero_cards(hand)
    if len(cards) != 2:
        raise RangeRefused("карты героя неизвестны — эквити считать не от чего")
    start = start_range(range_spot(hand, position), open_path=open_path, defend_path=defend_path)
    board = board_at(hand, street)
    live = live_at(hand, street)
    if start.spot.label not in live:
        raise RangeRefused("соперник сбросил карты до этой улицы")
    if len(live) > _HEADS_UP:
        raise RangeRefused("на этой улице в банке больше двух игроков — мультивей в этом срезе не считается")
    narrowed = narrow(start.range, hands)
    dead = [*cards, *board]
    combos = live_combos(narrowed, dead)
    if combos == 0.0:
        raise RangeRefused("все комбо этих рук заблокированы вашими картами и бордом")
    low, high = sorted(cards)
    return EquitySpot(
        hero=(low, high), board=board, start=start, narrowed=narrowed,
        live_combos=combos, share_of_start=combos / live_combos(start.range, dead),
    )
```

- [ ] **Step 4:** `uv run pytest tests/test_opponent_range.py -q -ra` → PASS. Мутации ревьюеру: убрать `start.weight(cls) <= 0.0` (останется проверка «класс в словаре») — `test_a_class_with_zero_weight_in_the_chart_is_outside_the_start` обязан упасть; убрать проверку колд-колла — `test_lines_outside_the_charts_are_refused[колд-колл перед BB]` обязан упасть. `uv run pytest -q -ra`, `uv run ruff check . && uv run pyright` → чисто.

- [ ] **Step 5: Коммит**

```bash
git add src/harness/analysis/opponent_range.py src/harness/contracts/calcs.py src/harness/contracts/__init__.py tests/test_opponent_range.py
git commit -m "$(cat <<'EOF'
Ядро: стартовый диапазон соперника по чарту, сужение классами, мёртвые карты

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

**Докстринги, которые становятся ложью:** нет (новый модуль). Сверить, что `contracts/calcs.py:21-25` («Оценка только там, где эталон вычислим… третьего источника в наборе нет») не противоречит — этот абзац о пороге частот; править в Task 5 вместе с новыми типами.

---

### Task 5: Два расчёта в словаре — контракты, исполнители, кэш эквити, выходы, печать

**Агент:** `implementer-core`, затем `reviewer` (тексты игроку — сверка с `.claude/SESSIONS_UX.md`, «Три состояния ответа»).

Новые имена входят в `CalcName`, реестр (`calcs/__init__.py:287-294`), выжимки (`explanation/question.py`) и печать (`presentation/messages.py`) одной задачей: без любого из четырёх pyright или `test_the_dictionary_covers_every_name_exactly_once` (`tests/test_frequency.py:280-289`) красные. Инструментами агенту они подключаются в Task 6; `/ask` их не получает никогда.

**Files:**
- Modify: `src/harness/contracts/calcs.py` — модульный докстринг (`:21-25`), `__all__` (`:40-65`), `CalcName` (`:79-87`), новые результаты после `ThresholdResult` (`:305`), параметры после `ThresholdParams` (`:381`), `CalcParams`/`CalcResult` (`:384-396`), `AGENT_ONLY_CALCS`; `src/harness/contracts/__init__.py` — реэкспорт.
- Modify: `src/harness/memory/repos.py` — `AnalysesRepo.enriched_hand` (после `get`, `:901-907`), `CalcCacheRepo.get` (после `get_all`, `:1110-1121`).
- Modify: `src/harness/calcs/__init__.py` — импорт (`:34-68`), исполнители, `REGISTRY` (`:287-294`), перегрузки `run` (`:297-316`), докстринг `run` (`:320-332`).
- Modify: `src/harness/explanation/question.py` — импорты (`:25-46`), `_CALC_BRIEF` (`:60-67`), `question_digest` (`:250-268`), `agent_digest` (`:366-389`).
- Modify: `src/harness/presentation/messages.py` — импорт результатов (`:124-131`), `_CALC_WORD` (`:2771-2778`), `_calc_answer` (`:2949-2959`), `agent_turn_msg` (`:2996`), `agent_refusal_msg` (`:3033`), новый `_AGENT_CAN_BE_COUNTED` рядом с `_WHAT_CAN_BE_COUNTED` (`:3078-3088`).
- Modify: `tests/test_question_text.py:144-146`, `tests/test_question_routing.py:122-129` и `:307-315`, `tests/test_agent_outputs.py`, `tests/test_presentation.py`, `tests/test_calcs.py`.

**Interfaces:**
- Produces:
  - `CalcName.OPPONENT_RANGE = "opponent_range"`, `CalcName.EQUITY_VS_RANGE = "equity_vs_range"`; `AGENT_ONLY_CALCS: frozenset[CalcName]`.
  - `OpponentRangeParams(calc, analysis_id: int, opponent_position: str, street: Street)`; `EquityVsRangeParams(calc, analysis_id: int, opponent_position: str, street: Street, hands: str)`.
  - `OpponentRangeResult(calc, opponent_position, street, line: RangeLine, opener_position, chart: Literal["open", "defend"], chart_seats, table_seats, chart_position, vs_chart_position: str | None = None, chart_depth_bb, range_text, hands_share, live_combos)`; `EquityVsRangeResult(calc, opponent_position, street, range_text, live_combos, share_of_start, equity)`.
  - `AnalysesRepo.enriched_hand(analysis_id: int, player_id: int) -> EnrichedHand | None`; `CalcCacheRepo.get(prefix: str, key: str) -> float | None`.
  - `calcs.AGENT_EQUITY_ITERATIONS: int = 50_000` (уточняется замером, Step 6), `calcs.AGENT_EQUITY_SEED: int = 42`.
- Consumes: `range_spot`, `start_range`, `equity_spot`, `board_at`, `hero_cards`, `live_combos` (Task 4); `equity_vs_range` (`analysis/tools/equity.py:155-164`); `strip_card_tokens` (`agent_checks.py:675`).

- [ ] **Step 1: Падающие тесты.**

`tests/test_agent_outputs.py` (импорты: `Street`, `OpponentRangeResult`, `EquityVsRangeResult` из `harness.contracts`; `strip_card_tokens` из `harness.explanation.agent_checks`):

```python
def range_result(**over) -> OpponentRangeResult:
    """Синтетика: 99 (6) + 66 наполовину (3) + A5s (4) + KQo (12) = 25 комбо из 1326."""
    base: dict[str, object] = dict(
        calc=CalcName.OPPONENT_RANGE, opponent_position="BB", street=Street.FLOP, line="call",
        opener_position="BTN", chart="defend", chart_seats=8, table_seats=6,
        chart_position="BB", vs_chart_position="BTN", chart_depth_bb=25.0,
        range_text="99, 66:0.5, A5s, KQo", hands_share=25 / 1326, live_combos=22.0,
    )
    base.update(over)
    return OpponentRangeResult.model_validate(base)


def equity_result(**over) -> EquityVsRangeResult:
    base: dict[str, object] = dict(
        calc=CalcName.EQUITY_VS_RANGE, opponent_position="BB", street=Street.FLOP,
        range_text="99, A5s", live_combos=7.0, share_of_start=0.318, equity=0.314,
    )
    base.update(over)
    return EquityVsRangeResult.model_validate(base)


@pytest.mark.parametrize("result", [range_result(), equity_result()], ids=["range", "equity"])
def test_the_range_outputs_register_every_number_that_stays_a_number(result: CalcResult):
    """Классы с буквой сверка вырезает (`strip_card_tokens`) — их цифры не
    регистрируются; голые пары и доли — числа, они разрешены."""
    digest = agent_digest(result)
    assert unsupported_numbers(strip_card_tokens(digest.text), digest.allowed) == []


def test_the_range_output_names_the_chart_and_the_short_table_rule():
    text = agent_digest(range_result()).text
    assert "Соперник: BB, линия до флопа — колл против опена BTN." in text
    assert (
        "Источник: чарт защиты BB против BTN, 8-max, 25 BB; за столом мест — 6: "
        "чарт 8-max по позиции от конца (правило, не допущение)." in text
    )
    assert "Доля рук: 1.9%; комбо без ваших карт и борда (флоп): 22." in text
    assert {99.0, 66.0, 0.5} <= agent_digest(range_result()).allowed


def test_an_eight_max_range_says_nothing_about_another_table():
    assert "за столом мест" not in agent_digest(range_result(table_seats=8)).text


def test_the_equity_output_says_the_narrowing_is_an_assumption():
    text = agent_digest(equity_result()).text
    assert "Диапазон сужен агентом — это допущение." in text
    assert "Комбо: 7 — 31.8% стартового диапазона." in text
    assert "Эквити вашей руки при вскрытии: 31.4%." in text


def test_ask_has_no_digest_for_agent_only_calculations():
    with pytest.raises(ValueError, match="расчёт агента"):
        question_digest(range_result())
```

`tests/test_question_text.py:144-146` → `assert {result.calc for result in ALL_RESULTS} == set(CalcName) - AGENT_ONLY_CALCS` (докстринг: «у каждого имени `/ask`…»). `tests/test_question_routing.py:122-129` → сравнение с `sorted(n.value for n in set(CalcName) - AGENT_ONLY_CALCS)`, имя теста — `test_the_ask_tool_set_covers_every_ask_calculation_exactly_once`; `:307` — `@pytest.mark.parametrize("name", sorted(n.value for n in set(CalcName) - AGENT_ONLY_CALCS))` (в Task 6 параметризация переходит на набор агента).

`tests/test_calcs.py` (импорты: `analyze_hand`, `EquityVsRangeParams`, `OpponentRangeParams`, `CalcCacheRepo`, `SessionsRepo`, `harness.calcs as calcs_module`, `harness.analysis.charts.lookup as lookup`, `from harness.analysis.tools.equity import equity_vs_range`; из `tests.test_opponent_range` — `BTN_OPENS_BB_CALLS`, `defend_file`, `_raw`; из `tests.test_postflop_line` — `_played_out`):

```python
async def stored_analysis(db, *, session_id: int, actions=None) -> int:
    """Синтетическая рука вечера с разбором: BTN открывает, BB коллирует, флоп."""
    raw = _raw(actions or BTN_OPENS_BB_CALLS)
    en = _played_out(raw)
    hand_id = await HandsRepo(db).save_raw(session_id=session_id, raw=raw)
    await HandsRepo(db).save_canonical(hand_id, en.hand)
    await HandsRepo(db).save_enriched(hand_id, en)
    return await AnalysesRepo(db).save(
        hand_id=hand_id, result=analyze_hand(en), decision_points=en.report.decision_points
    )


async def test_the_opponent_range_comes_from_the_chart_of_the_analysed_hand(db, tmp_path, monkeypatch):
    monkeypatch.setattr(lookup, "DEFAULT_DEFEND_PATH", defend_file(tmp_path))
    player_id, session_id = await _player_with_session(db, tg_user_id=9201)
    analysis_id = await stored_analysis(db, session_id=session_id)
    result = await run(db, player_id, OpponentRangeParams(
        analysis_id=analysis_id, opponent_position="BB", street=Street.FLOP,
    ))
    assert (result.calc, result.line, result.chart, result.vs_chart_position) == (
        CalcName.OPPONENT_RANGE, "call", "defend", "BTN",
    )
    assert (result.chart_seats, result.table_seats) == (8, 6)


async def test_another_players_analysis_is_not_read(db, tmp_path, monkeypatch):
    monkeypatch.setattr(lookup, "DEFAULT_DEFEND_PATH", defend_file(tmp_path))
    _owner, owner_session = await _player_with_session(db, tg_user_id=9202)
    stranger, _ = await _player_with_session(db, tg_user_id=9203)
    analysis_id = await stored_analysis(db, session_id=owner_session)
    with pytest.raises(ValueError, match="разбора этой руки нет"):
        await run(db, stranger, OpponentRangeParams(
            analysis_id=analysis_id, opponent_position="BB", street=Street.FLOP,
        ))


async def test_equity_is_deterministic_and_cached(db, tmp_path, monkeypatch):
    """Тот же вход — то же число, равное прямому расчёту с тем же сидом; второй раз
    — из `calc_cache`."""
    monkeypatch.setattr(lookup, "DEFAULT_DEFEND_PATH", defend_file(tmp_path))
    monkeypatch.setattr(calcs_module, "AGENT_EQUITY_ITERATIONS", 2_000)
    player_id, session_id = await _player_with_session(db, tg_user_id=9204)
    analysis_id = await stored_analysis(db, session_id=session_id)
    params = EquityVsRangeParams(
        analysis_id=analysis_id, opponent_position="BB", street=Street.FLOP,
        hands="QQ, 77, 22, AQs",
    )
    first = await run(db, player_id, params)
    prefix, key = calcs_module.equity_cache_prefix(), "KdTc|Qc7d2s|" + to_notation(
        parse_range("QQ, 77, 22, AQs")
    )
    assert await CalcCacheRepo(db).get(prefix, key) == first.equity
    assert first.equity == equity_vs_range(
        ("Kd", "Tc"), parse_range("QQ, 77, 22, AQs"), ["Qc", "7d", "2s"], iterations=2_000, seed=42,
    )
    assert (await run(db, player_id, params)).equity == first.equity
    assert first.live_combos == 12.0


async def test_a_narrowing_outside_the_start_is_refused_by_the_calculation(db, tmp_path, monkeypatch):
    monkeypatch.setattr(lookup, "DEFAULT_DEFEND_PATH", defend_file(tmp_path))
    player_id, session_id = await _player_with_session(db, tg_user_id=9205)
    analysis_id = await stored_analysis(db, session_id=session_id)
    with pytest.raises(ValueError, match="нет в стартовом диапазоне: AKo"):
        await run(db, player_id, EquityVsRangeParams(
            analysis_id=analysis_id, opponent_position="BB", street=Street.FLOP, hands="AKo",
        ))
```

Порядок `KdTc` в ключе — `sorted(["Kd", "Tc"])`; если имплементатор видит другой — ключ строится из `EquitySpot.cache_key`, тест взять оттуда же (`equity_spot(...).cache_key`), а не переписывать порядок.

`tests/test_presentation.py`:

```python
def test_agent_numbers_print_the_chart_range_and_the_assumed_equity():
    from harness.presentation import agent_turn_msg
    from tests.test_agent_outputs import equity_result, range_result

    msg = agent_turn_msg("numbers", None, [range_result(), equity_result()], from_context=True, job_id=7)
    assert "Посчитано: диапазон соперника по чарту, эквити против суженного диапазона." in msg.text
    assert (
        "Диапазон BB по чарту защиты BB против BTN (8-max, 25.0BB; мест за столом: 6): "
        "1.9% рук, 22 комбо." in msg.text
    )
    assert (
        "Эквити против диапазона BB, суженного по линии (допущение): 31.4% — 7 комбо, "
        "31.8% стартового." in msg.text
    )


def test_the_agent_list_names_the_range_calculations_and_ask_does_not():
    from harness.presentation import agent_refusal_msg, question_refusal_msg

    assert "стартовый диапазон соперника по чарту" in agent_refusal_msg().text
    assert "диапазон" not in question_refusal_msg().text
```

- [ ] **Step 2:** `uv run pytest tests/test_agent_outputs.py tests/test_calcs.py tests/test_presentation.py tests/test_question_text.py -q -ra` → новые FAIL.

- [ ] **Step 3: Контракты** (`contracts/calcs.py`):

```python
class CalcName(StrEnum):
    ...
    OPPONENT_RANGE = "opponent_range"
    EQUITY_VS_RANGE = "equity_vs_range"


# Расчёты только агента: им нужна рука разбора вечера и улица точки решения, у
# `/ask` нет ни того, ни другого (спека 2026-10-14, §6; `/ask` без изменений).
AGENT_ONLY_CALCS: frozenset[CalcName] = frozenset(
    {CalcName.OPPONENT_RANGE, CalcName.EQUITY_VS_RANGE}
)


class OpponentRangeResult(BaseModel):
    """Стартовый диапазон соперника по чарту — зона `strict`: диапазон не угадан, а
    взят из эталона владельца по линии, которую код прочёл в руке.

    `hands_share` — `Range.fraction_of_hands` (от 1326 комбо); `live_combos` —
    взвешенные комбо без карт героя и борда улицы `street`. `chart_seats` и
    `table_seats` расходятся на столе 4–7 мест (8-max по позиции от конца — правило).
    """

    calc: CalcName
    opponent_position: str
    street: Street
    line: RangeLine
    opener_position: str
    chart: Literal["open", "defend"]
    chart_seats: int
    table_seats: int
    chart_position: str
    vs_chart_position: str | None = None
    chart_depth_bb: float
    range_text: str
    hands_share: float
    live_combos: float


class EquityVsRangeResult(BaseModel):
    """Эквити героя при вскрытии против диапазона, суженного агентом, — зона
    `assuming`: сужение — допущение агента, классы — подмножество стартового с
    долями чарта. `share_of_start` — доля живых комбо суженного от стартового."""

    calc: CalcName
    opponent_position: str
    street: Street
    range_text: str
    live_combos: float
    share_of_start: float
    equity: float


class OpponentRangeParams(BaseModel):
    """`analysis_id` подставляет код (указатель разбора вечера), не модель."""

    calc: Literal[CalcName.OPPONENT_RANGE] = CalcName.OPPONENT_RANGE
    analysis_id: int
    opponent_position: str
    street: Street


class EquityVsRangeParams(BaseModel):
    """`hands` — классы без весов; `analysis_id` — от кода, как у `OpponentRangeParams`."""

    calc: Literal[CalcName.EQUITY_VS_RANGE] = CalcName.EQUITY_VS_RANGE
    analysis_id: int
    opponent_position: str
    street: Street
    hands: str
```

`CalcParams` и `CalcResult` дополнить обоими; `__all__` и `contracts/__init__.py` — `AGENT_ONLY_CALCS`, `EquityVsRangeParams`, `EquityVsRangeResult`, `OpponentRangeParams`, `OpponentRangeResult`, `RangeLine`.

- [ ] **Step 4: Память** (`memory/repos.py`):

```python
    async def enriched_hand(self, analysis_id: int, player_id: int) -> EnrichedHand | None:
        """Рука разбора — только если разбор этого игрока; чужой или пропавший — `None`
        (`test_another_players_analysis_is_not_read`)."""
        record = await self.get(analysis_id)
        if record is None:
            return None
        owner = await self.db.scalar(
            select(SessionRow.player_id)
            .join(Hand, Hand.session_id == SessionRow.id)
            .where(Hand.id == record.hand_id)
        )
        if owner != player_id:
            return None
        return (await HandsRepo(self.db).get(record.hand_id)).enriched
```

```python
    async def get(self, prefix: str, key: str) -> float | None:
        """Одно значение по полному ключу `prefix + key`; нет строки — `None`."""
        value = await self.db.scalar(select(CalcCache.value).where(CalcCache.key == f"{prefix}{key}"))
        return None if value is None else float(value)
```

(`EnrichedHand` в импортах `repos.py` есть — его несёт `HandRecord.enriched`; сверить.)

- [ ] **Step 5: Исполнители** (`calcs/__init__.py`; импорты: `asyncio`, `to_notation` из `harness.analysis.charts`, `from harness.analysis.opponent_range import board_at, equity_spot, hero_cards, live_combos, range_spot, start_range`, `from harness.analysis.tools.equity import equity_vs_range`, новые типы из `harness.contracts`, `AnalysesRepo, CalcCacheRepo` из `harness.memory.repos`):

```python
# Потолок итераций Монте-Карло у эквити агента: ход агента ждут до 30 с, а расчёт
# идёт в процессе воркера. Значение — по замеру на стенде (план 2026-10-14, Task 5,
# Step 6). Сид — тот же, что умолчание `equity_vs_range`: один вход — одно число.
AGENT_EQUITY_ITERATIONS = 50_000
AGENT_EQUITY_SEED = 42


def equity_cache_prefix() -> str:
    """Пространство ключей `calc_cache` эквити агента: смена итераций или сида не
    отдаёт число, посчитанное на старых (тот же приём, что у `CalcCacheRepo`)."""
    return f"eqr:v1:{AGENT_EQUITY_ITERATIONS}:{AGENT_EQUITY_SEED}:"


async def _analysed_hand(db: AsyncSession, player_id: int, analysis_id: int) -> CanonicalHand:
    enriched = await AnalysesRepo(db).enriched_hand(analysis_id, player_id)
    if enriched is None:
        raise ValueError("разбора этой руки нет — диапазон соперника считать не по чему")
    return enriched.hand


async def _opponent_range(
    db: AsyncSession, player_id: int, params: OpponentRangeParams
) -> OpponentRangeResult:
    """Стартовый диапазон по чарту в руке разбора; отказы ядра — `ValueError` словами."""
    _check_position(params.opponent_position)
    hand = await _analysed_hand(db, player_id, params.analysis_id)
    start = start_range(range_spot(hand, params.opponent_position))
    dead = [*hero_cards(hand), *board_at(hand, params.street)]
    return OpponentRangeResult(
        calc=params.calc,
        opponent_position=params.opponent_position,
        street=params.street,
        line=start.spot.line,
        opener_position=start.spot.opener_position,
        chart=start.chart,
        chart_seats=start.chart_seats,
        table_seats=start.spot.seats,
        chart_position=start.chart_position,
        vs_chart_position=start.vs_chart_position,
        chart_depth_bb=start.chart_depth_bb,
        range_text=to_notation(start.range),
        hands_share=start.range.fraction_of_hands(),
        live_combos=live_combos(start.range, dead),
    )


async def _equity_vs_range(
    db: AsyncSession, player_id: int, params: EquityVsRangeParams
) -> EquityVsRangeResult:
    """Эквити против суженного: Монте-Карло под потолком итераций, с фиксированным
    сидом и `calc_cache` (`test_equity_is_deterministic_and_cached`). Счёт — в
    потоке: индикатор «печатает» хода агента живёт в том же цикле событий."""
    _check_position(params.opponent_position)
    hand = await _analysed_hand(db, player_id, params.analysis_id)
    spot = equity_spot(hand, params.opponent_position, params.street, params.hands)
    cache = CalcCacheRepo(db)
    prefix = equity_cache_prefix()
    equity = await cache.get(prefix, spot.cache_key)
    if equity is None:
        equity = await asyncio.to_thread(
            equity_vs_range, spot.hero, spot.narrowed, spot.board,
            iterations=AGENT_EQUITY_ITERATIONS, seed=AGENT_EQUITY_SEED,
        )
        await cache.upsert_many(prefix, {spot.cache_key: equity})
    return EquityVsRangeResult(
        calc=params.calc,
        opponent_position=params.opponent_position,
        street=params.street,
        range_text=to_notation(spot.narrowed),
        live_combos=spot.live_combos,
        share_of_start=spot.share_of_start,
        equity=equity,
    )
```

`REGISTRY` += `CalcName.OPPONENT_RANGE: (OpponentRangeParams, _opponent_range)`, `CalcName.EQUITY_VS_RANGE: (EquityVsRangeParams, _equity_vs_range)`; перегрузки `run` — по одной на тип.

`explanation/question.py` (импорт `from harness.explanation.agent_checks import strip_card_tokens` — цикла нет: `agent_checks` не импортирует `question`, сверено):

```python
_CALC_BRIEF[CalcName.OPPONENT_RANGE] = "стартовый диапазон соперника по чарту"   # в литерал словаря
_CALC_BRIEF[CalcName.EQUITY_VS_RANGE] = "эквити против диапазона, суженного агентом"

_STREET_WORD: dict[Street, str] = {
    Street.PREFLOP: "префлоп", Street.FLOP: "флоп", Street.TURN: "тёрн", Street.RIVER: "ривер",
}

_LINE_BRIEF: dict[str, str] = {
    "open_raise": "опен рейзом",
    "open_allin": "опен олл-ином",
    "call": "колл против опена {opener}",
    "three_bet": "3-бет против опена {opener}",
    "allin": "олл-ин против опена {opener}",
}


def _depth_text(value: float, book: NumberBook) -> str:
    return book.count(int(value)) if float(value).is_integer() else book.bb(value)


def _range_text(text: str, book: NumberBook) -> str:
    """Диапазон по классам. Регистрируется то, что останется числом в сверке ответа:
    голые пары (`99`) и доли (`0.5`); классы с буквой сверка вырезает
    (`strip_card_tokens`), их цифры не разрешаются
    (`test_the_range_outputs_register_every_number_that_stays_a_number`)."""
    book.prose(strip_card_tokens(text))
    return text


def _agent_range_lines(result: OpponentRangeResult, book: NumberBook) -> list[str]:
    chart = (
        f"чарт опена {result.chart_position}"
        if result.chart == "open"
        else f"чарт защиты {result.chart_position} против {result.vs_chart_position}"
    )
    source = (
        f"{chart}, {book.count(result.chart_seats)}-max, "
        f"{_depth_text(result.chart_depth_bb, book)} BB"
    )
    if result.chart_seats != result.table_seats:
        source += (
            f"; за столом мест — {book.count(result.table_seats)}: чарт "
            f"{book.count(result.chart_seats)}-max по позиции от конца (правило, не допущение)"
        )
    line = _LINE_BRIEF[result.line].format(opener=result.opener_position)
    return [
        f"Соперник: {result.opponent_position}, линия до флопа — {line}.",
        f"Источник: {source}.",
        f"Стартовый диапазон: {_range_text(result.range_text, book)}.",
        f"Доля рук: {book.pct(100.0 * result.hands_share, trim=True)}%; комбо без ваших карт "
        f"и борда ({_STREET_WORD[result.street]}): {book.count(round(result.live_combos))}.",
    ]


def _agent_equity_lines(result: EquityVsRangeResult, book: NumberBook) -> list[str]:
    return [
        f"Соперник: {result.opponent_position}, улица: {_STREET_WORD[result.street]}.",
        f"Диапазон, суженный агентом: {_range_text(result.range_text, book)}. "
        "Диапазон сужен агентом — это допущение.",
        f"Комбо: {book.count(round(result.live_combos))} — "
        f"{book.pct(100.0 * result.share_of_start, trim=True)}% стартового диапазона.",
        f"Эквити вашей руки при вскрытии: {book.pct(100.0 * result.equity, trim=True)}%.",
    ]
```

`agent_digest`: перед последним `else` — `elif isinstance(result, OpponentRangeResult): lines += _agent_range_lines(result, book)` и `elif isinstance(result, EquityVsRangeResult): lines += _agent_equity_lines(result, book)`. `question_digest`: перед последним `else` — `elif isinstance(result, (OpponentRangeResult, EquityVsRangeResult)): raise ValueError(f"{result.calc.value} — расчёт агента, у /ask его нет")`.

`presentation/messages.py`:

```python
    CalcName.OPPONENT_RANGE: "диапазон соперника по чарту",        # в _CALC_WORD
    CalcName.EQUITY_VS_RANGE: "эквити против суженного диапазона",


def _opponent_range_answer(result: OpponentRangeResult, trim: bool = False) -> list[str]:
    chart = (
        "чарту опена"
        if result.chart == "open"
        else f"чарту защиты {result.chart_position} против {result.vs_chart_position}"
    )
    table = "" if result.chart_seats == result.table_seats else f"; мест за столом: {result.table_seats}"
    return [
        f"Диапазон {result.opponent_position} по {chart} ({result.chart_seats}-max, "
        f"{_bb_number(result.chart_depth_bb)}BB{table}): "
        f"{_calc_pct(100.0 * result.hands_share, trim)} рук, {round(result.live_combos)} комбо."
    ]


def _equity_answer(result: EquityVsRangeResult, trim: bool = False) -> list[str]:
    return [
        f"Эквити против диапазона {result.opponent_position}, суженного по линии (допущение): "
        f"{_calc_pct(100.0 * result.equity, trim)} — {round(result.live_combos)} комбо, "
        f"{_calc_pct(100.0 * result.share_of_start, trim)} стартового."
    ]
```

`_calc_answer`: перед `return _defense_answer(...)` — две ветки `isinstance`. Список агента:

```python
_AGENT_CAN_BE_COUNTED = (
    "Посчитать могу:\n"
    "• ваши частоты — вход в банк, повышение, ре-рейз, продолженная ставка, "
    "бареллы, сдача на продолженную ставку, доход до вскрытия;\n"
    "• те же частоты у поля или у названного оппонента;\n"
    "• вашу частоту против порога — по размеру ставки или по полю;\n"
    "• покрытие разбора и цену расхождений;\n"
    "• типы повторяющихся расхождений;\n"
    "• требуемую частоту защиты против ставки;\n"
    "• в разобранной руке — стартовый диапазон соперника по чарту и эквити вашей руки "
    "против него, суженного по линии (допущение).\n\n"
    "Можно сузить вопрос позицией и вечером."
)
```

`agent_turn_msg` (`:2996`) и `agent_refusal_msg` (`:3033`) — `_AGENT_CAN_BE_COUNTED`; `/ask` (`:3052`, `:3064`) — прежний `_WHAT_CAN_BE_COUNTED`.

- [ ] **Step 6: Замер потолка итераций** (локально; на стенде — Task 11):

```bash
uv run python -c "
import time
from harness.analysis.charts import parse_range
from harness.analysis.tools.equity import equity_vs_range
rng = parse_range('22+, A2s+, K9s+, Q9s+, J9s+, T8s+, 97s+, 87s, A9o+, KTo+, QTo+, JTo')
for n in (20_000, 50_000, 100_000):
    t = time.monotonic(); equity_vs_range(('Ah', '6h'), rng, ['8c', '8s', 'Jd'], iterations=n, seed=42)
    print(n, round(time.monotonic() - t, 2))
"
```

Цель — не больше ~3 с на вызов на стенде (ход агента ждут до 30 с, вызовов до 8). Если 50 000 дольше — снизить константу до значения из замера и назвать числа в отчёте.

- [ ] **Step 7:** `uv run pytest -q -ra` → PASS (особо: `test_the_dictionary_covers_every_name_exactly_once`, `test_every_result_names_the_calculation_that_produced_it`, `test_the_agent_digest_registers_every_number_it_prints` на `ALL_RESULTS` — без изменений ожиданий); `uv run ruff check . && uv run pyright` → чисто.

- [ ] **Step 8: Коммит**

```bash
git add src/harness/contracts/calcs.py src/harness/contracts/__init__.py src/harness/memory/repos.py src/harness/calcs/__init__.py src/harness/explanation/question.py src/harness/presentation/messages.py tests/test_agent_outputs.py tests/test_calcs.py tests/test_presentation.py tests/test_question_text.py tests/test_question_routing.py
git commit -m "$(cat <<'EOF'
Расчёты агента: диапазон соперника по чарту и эквити против суженного — словарь, кэш, выходы

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

**Докстринги, которые становятся ложью:**
- `contracts/calcs.py:21-25` («Оценка только там, где эталон вычислим… третьего источника в наборе нет») — дописать: диапазон соперника берётся из чарта (эталон), эквити против суженного — допущение агента, оценкой не является.
- `calcs/__init__.py:322-324` (`run`: «…а не объединение всех шести») → «…всех восьми».
- `calcs/__init__.py:13-17` («В этом модуле нет ни одной формулы») — остаётся верным (кэш и выбор диапазона — не формулы), сверить.
- `explanation/question.py:252-257` (`question_digest`: «ни у одного из шести расчётов») → «ни у одного из расчётов `/ask`»; `tests/test_question_text.py:127-133` докстринг («на всех шести расчётах разом») — так же.
- `tests/test_question_routing.py:123-127` докстринг — набор `/ask` и набор имён без расчётов агента.

---

### Task 6: Проводка агенту — инструменты, указатель разбора, зона, метаданные

**Агент:** `implementer-core`, затем `reviewer`.

**Files:**
- Modify: `src/harness/calcs/routing.py` — модульный докстринг (`:1-7`), `CalcRun.__init__` (`:168-173`) и докстринг (`:139-166`), `_TOOL_ZONE` (`:243-245`), `_perform` (`:314-333`), `tools_for` (`:359-617`).
- Modify: `src/harness/agent/history.py` — `chart_numbers_in`, `assumed_numbers_in` (после `frequency_numbers`, `:249-252`), `__all__` (`:38-52`), докстринг `_is_output_call` (`:132-135`).
- Modify: `src/harness/agent/run.py` — `run_agent_turn` (`:158-199`).
- Modify: `src/harness/worker/pipeline.py` — `_run_agent`, вызов `run_agent_turn` (`:1343-1352`).
- Modify: `tests/test_agent.py` (`_turn` `:113-120` и новые тесты), `tests/test_question_routing.py:307-315`.

**Interfaces:**
- Produces:
  - `CalcRun(max_calcs: int = 1, sizes_from: frozenset[float] | None = None, analysis_id: int | None = None)`.
  - Инструменты агента `opponent_range(opponent_position: str, street: Street)`, `equity_vs_range(opponent_position: str, street: Street, hands: str)`; `/ask` — прежние шесть.
  - Метаданные `ToolReturn`: `"chart": [доля рук %, комбо]` у `opponent_range`, `"assumed": [эквити %, доля от стартового %]` у `equity_vs_range` (округление — как у `NumberBook.pct`: 0.1).
  - `chart_numbers_in(messages) -> frozenset[float]`, `assumed_numbers_in(messages) -> frozenset[float]`.
  - `run_agent_turn(..., analysis_id: int | None = None)`.
- Consumes: Task 5 (имена, параметры, результаты).

- [ ] **Step 1: Падающие тесты** (`tests/test_agent.py`; `_turn` получает `analysis_id=None` и передаёт его в `run_agent_turn`; импорты: `chart_numbers_in`, `assumed_numbers_in`, `AGENT_ONLY_CALCS`, `tools_for`, `CalcRun`, `harness.analysis.charts.lookup as lookup`, `harness.calcs as calcs_module`, `SessionsRepo`; из `tests.test_calcs` — `stored_analysis`; из `tests.test_opponent_range` — `defend_file`):

```python
_RANGE_CALL = ("opponent_range", {"opponent_position": "BB", "street": "flop"})
_EQUITY_CALL = (
    "equity_vs_range", {"opponent_position": "BB", "street": "flop", "hands": "QQ, 77, 22, AQs"},
)


async def _evening_with_hand(session, tg_user_id: int):
    scope = await _seeded(session, tg_user_id=tg_user_id)
    analysis_id = await stored_analysis(session, session_id=scope[1])
    await SessionsRepo(session).set_last_analysis(scope[1], analysis_id)
    await session.commit()
    return scope, analysis_id


def test_the_agent_tool_set_covers_every_calculation_name_exactly_once():
    tools = tools_for(None, 1, 1, CalcRun(max_calcs=8, sizes_from=frozenset()))  # type: ignore[arg-type]
    assert sorted(tool.name for tool in tools) == sorted(name.value for name in CalcName)


def test_ask_does_not_get_the_agent_only_calculations():
    tools = tools_for(None, 1, 1, CalcRun())  # type: ignore[arg-type]
    assert not {tool.name for tool in tools} & {name.value for name in AGENT_ONLY_CALCS}


async def test_the_agent_reads_the_range_by_the_chart_and_counts_equity_against_its_narrowing(
    db_factory, tmp_path, monkeypatch
):
    monkeypatch.setattr(lookup, "DEFAULT_DEFEND_PATH", defend_file(tmp_path))
    monkeypatch.setattr(calcs_module, "AGENT_EQUITY_ITERATIONS", 2_000)
    async with db_factory() as session:
        scope, analysis_id = await _evening_with_hand(session, tg_user_id=8301)
        script = _TurnScript([([_RANGE_CALL, _EQUITY_CALL], "Ответ без чисел (допущение).")])
        turn = await _turn(db_factory, session, script, scope=scope, analysis_id=analysis_id)
    assert [r.calc for r in turn.results] == [CalcName.OPPONENT_RANGE, CalcName.EQUITY_VS_RANGE]
    assert [c["zone"] for c in turn.calls] == ["strict", "assuming"]
    range_result, equity_result = turn.results
    assert chart_numbers_in(turn.messages) == frozenset(
        {round(100.0 * range_result.hands_share, 1), float(round(range_result.live_combos))}
    )
    assert assumed_numbers_in(turn.messages) == frozenset(
        {round(100.0 * equity_result.equity, 1), round(100.0 * equity_result.share_of_start, 1)}
    )


async def test_without_a_hand_analysis_the_range_tool_says_so(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8302)
        script = _TurnScript([([_RANGE_CALL], _GOOD)])
        turn = await _turn(db_factory, session, script, scope=scope)
    assert turn.results == ()
    assert any("Разбора руки в этом вечере нет" in text for text in script.returns)


async def test_a_refused_narrowing_reaches_the_model_and_spends_no_calculation(
    db_factory, tmp_path, monkeypatch
):
    monkeypatch.setattr(lookup, "DEFAULT_DEFEND_PATH", defend_file(tmp_path))
    async with db_factory() as session:
        scope, analysis_id = await _evening_with_hand(session, tg_user_id=8303)
        bad = ("equity_vs_range", {"opponent_position": "BB", "street": "flop", "hands": "AKo"})
        script = _TurnScript([([bad], "Ответ без чисел (допущение).")])
        turn = await _turn(db_factory, session, script, scope=scope, analysis_id=analysis_id)
    assert turn.results == ()
    assert "нет в стартовом диапазоне" in turn.calls[0]["error"]
```

`tests/test_question_routing.py:307-315` → параметризация по всем `CalcName`, инструменты из `tools_for(None, 1, 1, CalcRun(max_calcs=8, sizes_from=frozenset()))` (у агента набор полный).

- [ ] **Step 2:** `uv run pytest tests/test_agent.py tests/test_question_routing.py -q -ra` → новые FAIL.

- [ ] **Step 3: Реализация** (`calcs/routing.py`):

`CalcRun.__init__` — третий аргумент `analysis_id: int | None = None`, `self.analysis_id = analysis_id`.

```python
# Зона доверия по инструменту (CLAUDE.md): измерения, арифметика и диапазон по
# чарту не опираются на угаданный диапазон; эквити против диапазона, суженного
# агентом, — опирается (спека 2026-10-14, §6.2).
_TOOL_ZONE: dict[CalcName, Zone] = {
    name: Zone.ASSUMING if name is CalcName.EQUITY_VS_RANGE else Zone.STRICT for name in CalcName
}

_NO_HAND = "Разбора руки в этом вечере нет — диапазон соперника считать не по чему."
```

В `_perform` перед `return ToolReturn(...)`:

```python
    chart: list[float] = []
    assumed: list[float] = []
    if isinstance(result, OpponentRangeResult):
        chart = [round(100.0 * result.hands_share, 1), float(round(result.live_combos))]
    elif isinstance(result, EquityVsRangeResult):
        assumed = [round(100.0 * result.equity, 1), round(100.0 * result.share_of_start, 1)]
```

и в `metadata` — `"chart": chart, "assumed": assumed`.

В `tools_for` (перед `in_bb = …`, `:603`):

```python
    async def opponent_range(opponent_position: str, street: Street) -> str | ToolReturn:
        """Стартовый диапазон соперника по чарту в последней разобранной руке.

        Линию соперника до флопа, глубину, число мест и анте код берёт из руки сам.

        Args:
            opponent_position: позиция соперника за столом, как в сводке разбора, например BB.
            street: улица точки решения — по ней из комбо убираются карты борда.
        """
        if run_state.analysis_id is None:
            return _NO_HAND
        return await _perform(
            db, player_id, run_state,
            OpponentRangeParams(
                analysis_id=run_state.analysis_id,
                opponent_position=opponent_position,
                street=street,
            ),
        )

    async def equity_vs_range(opponent_position: str, street: Street, hands: str) -> str | ToolReturn:
        """Эквити вашей руки при вскрытии против диапазона соперника, суженного вами.

        Классы — только из стартового диапазона `opponent_range` этого соперника и
        без весов: доли берутся из чарта. Сужение — допущение.

        Args:
            opponent_position: позиция соперника за столом, как в сводке разбора.
            street: улица точки решения; борд и ваши карты берутся из руки.
            hands: классы рук через запятую, например «AA, KQs, 88-66, A5s-A2s».
        """
        if run_state.analysis_id is None:
            return _NO_HAND
        return await _perform(
            db, player_id, run_state,
            EquityVsRangeParams(
                analysis_id=run_state.analysis_id,
                opponent_position=opponent_position,
                street=street,
                hands=hands,
            ),
        )
```

После словаря `by_name` (`:604-613`):

```python
    if in_bb:
        by_name[CalcName.OPPONENT_RANGE] = opponent_range
        by_name[CalcName.EQUITY_VS_RANGE] = equity_vs_range
```

`agent/history.py`:

```python
def chart_numbers_in(messages: Sequence[ModelMessage]) -> frozenset[float]:
    """Числа выхода `opponent_range` (доля рук, комбо): пересказ стартового диапазона
    по чарту с ними — факт, не допущение (`agent_checks`, Task 7 плана 2026-10-14)."""
    return frozenset(float(n) for meta in _metadata(messages) for n in meta.get("chart", []))


def assumed_numbers_in(messages: Sequence[ModelMessage]) -> frozenset[float]:
    """Доли выхода `equity_vs_range` — эквити и доля от стартового: допущение агента."""
    return frozenset(float(n) for meta in _metadata(messages) for n in meta.get("assumed", []))
```

`agent/run.py`: в `run_agent_turn` — параметр `analysis_id: int | None = None` (последним), `CalcRun(max_calcs=MAX_CALCS_PER_TURN, sizes_from=sizes_from, analysis_id=analysis_id)`.

`worker/pipeline.py`, `_run_agent`: в вызов `run_agent_turn` — `analysis_id=state.last_analysis_id if context is not None else None` (сводки нет — руки тоже нет: указатель на разбор без руки инструменту не нужен).

- [ ] **Step 4:** `uv run pytest tests/test_agent.py tests/test_question_routing.py tests/test_worker_pipeline.py -q -ra` → PASS; `uv run pytest -q -ra`, `uv run ruff check . && uv run pyright` → чисто.

- [ ] **Step 5: Коммит**

```bash
git add src/harness/calcs/routing.py src/harness/agent/history.py src/harness/agent/run.py src/harness/worker/pipeline.py tests/test_agent.py tests/test_question_routing.py
git commit -m "$(cat <<'EOF'
Агент: инструменты диапазона и эквити по указателю разбора; зона assuming у эквити

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

**Докстринги, которые становятся ложью:**
- `calcs/routing.py:3` («Шесть расчётов словаря подключены модели инструментами») → «у `/ask` — шесть, у агента — все имена `CalcName`».
- `calcs/routing.py:243-244` (комментарий `_TOOL_ZONE`: «все шесть расчётов — измерения и арифметика, ни один не опирается на угаданный диапазон») — заменяется комментарием из Step 3.
- `tools_for` (`:362-367`: «Шесть инструментов — по одному на имя из `CalcName`, и ни одного сверх») → «у агента — по одному на каждое имя, у `/ask` — без `AGENT_ONLY_CALCS`» (`test_the_agent_tool_set_covers_every_calculation_name_exactly_once`, `test_the_ask_tool_set_covers_every_ask_calculation_exactly_once`).
- `CalcRun` (`:139-166`): дописать `analysis_id` — указатель разбора вечера от кода, без него инструменты руки отвечают отказом (`test_without_a_hand_analysis_the_range_tool_says_so`).
- `_perform` (`:263-277`): метаданные — плюс `chart` и `assumed`.
- `agent/history.py:133-134` (`_is_output_call`: ссылка на `test_the_tool_set_covers_every_calculation_name_exactly_once`) → новое имя `test_the_agent_tool_set_covers_every_calculation_name_exactly_once`.
- `run_agent_turn` (`:170-177`) — `analysis_id`.

---

### Task 7: Проверка ответа — пересказ чарта как факт, эквити против суженного как допущение

**Агент:** `implementer-core`, затем `reviewer` (мутации на двух новых правилах).

**Files:**
- Modify: `src/harness/explanation/agent_checks.py` — модульный докстринг (`:7-8` и абзац «Допущение — с меткой»), импорт `numbers_in` (`:111`), `_opponent_claim` (`:416-435`), `check_agent_answer` (`:708-769`).
- Modify: `src/harness/agent/run.py` — импорт (`:21-30`), `_violations` (`:109-131`).
- Modify: `tests/test_agent_checks.py`.

**Interfaces:**
- Produces: `check_agent_answer(..., chart_numbers: frozenset[float] = frozenset(), assumed_numbers: frozenset[float] = frozenset())`.
- Consumes: `chart_numbers_in`, `assumed_numbers_in` (Task 6).

- [ ] **Step 1: Падающие тесты** (`tests/test_agent_checks.py`):

```python
_CHART = frozenset({38.2, 410.0})


def test_a_chart_range_sentence_needs_no_mark():
    """Стартовый диапазон по чарту — зона `strict`: пересказ со словом «чарт» и
    числом выхода `opponent_range` метки не требует; без «чарт» или без числа его
    выхода слово диапазона требует её, как раньше."""
    kw = {"allowed": _CHART, "calc_numbers": _CHART, "chart_numbers": _CHART}
    assert _check("Стартовый диапазон BB по чарту — 38.2% рук, 410 комбо.", **kw) == []
    assert any(ASSUMPTION_MARK in v for v in _check("Стартовый диапазон BB — 38.2% рук.", **kw))
    plain = {"allowed": _CHART, "calc_numbers": _CHART}
    assert any(
        ASSUMPTION_MARK in v
        for v in _check("Стартовый диапазон BB по чарту — 38.2% рук, 410 комбо.", **plain)
    )


def test_a_chart_fact_does_not_cover_a_prediction_in_the_same_sentence():
    kw = {"allowed": _CHART, "calc_numbers": _CHART, "chart_numbers": _CHART}
    assert any(
        ASSUMPTION_MARK in v
        for v in _check("По чарту у BB 38.2% рук, и он сбросит всё без пары.", **kw)
    )


def test_an_equity_against_a_narrowed_range_needs_the_mark():
    """Эквити против суженного — допущение даже без слова «диапазон»."""
    eq = frozenset({31.4, 31.8})
    kw = {"allowed": eq, "calc_numbers": eq, "assumed_numbers": eq}
    assert any("суженного" in v for v in _check("Против этих рук у вас 31.4%.", **kw))
    assert _check("Против этих рук у вас 31.4% (допущение).", **kw) == []
    assert _check(
        "Если BB держит только пары и дамы (допущение), эквити 31.4%.", **kw
    ) == []


def test_the_turn_checks_read_chart_and_assumed_numbers_from_history():
    from pydantic_ai.messages import ModelRequest, ToolReturnPart

    from harness.agent.run import _violations

    messages = [ModelRequest(parts=[
        ToolReturnPart(tool_name="opponent_range", content="…", tool_call_id="c1", metadata={
            "calc": "opponent_range", "allowed": [38.2, 410.0], "chart": [38.2, 410.0],
        }),
        ToolReturnPart(tool_name="equity_vs_range", content="…", tool_call_id="c2", metadata={
            "calc": "equity_vs_range", "allowed": [31.4], "assumed": [31.4],
        }),
    ])]
    assert _violations("Стартовый диапазон BB по чарту — 38.2% рук.", messages, None, frozenset()) == []
    assert _violations("У вас 31.4%.", messages, None, frozenset()) != []
```

- [ ] **Step 2:** `uv run pytest tests/test_agent_checks.py -q -ra` → новые FAIL.

- [ ] **Step 3: Реализация** (`agent_checks.py`):

```python
_CHART_WORD_RE = re.compile(r"чарт", re.IGNORECASE)


def _chart_backed(sentence: str, chart_numbers: frozenset[float]) -> bool:
    """Пересказ стартового диапазона из `opponent_range`: слово «чарт» и число
    выхода этого расчёта (`test_a_chart_range_sentence_needs_no_mark`)."""
    if not chart_numbers or not _CHART_WORD_RE.search(sentence):
        return False
    chart = {round(n, 1) for n in chart_numbers}
    return any(round(n, 1) in chart for n in numbers_in(sentence))


def _assumed_share_in(sentence: str, assumed_numbers: frozenset[float]) -> bool:
    """Доля из выхода `equity_vs_range` в предложении."""
    return any(_in(m.group("pct"), assumed_numbers) for m in _PCT_RE.finditer(sentence))
```

`_opponent_claim(sentence, calc_numbers, frequency_numbers, chart_numbers: frozenset[float] = frozenset())`: после `rest = _strip_backed(...)` —

```python
    if _chart_backed(sentence, chart_numbers):
        # Слово диапазона — о чарте, а не догадка; прогноз о сопернике в том же
        # предложении проверяется дальше как обычно
        # (`test_a_chart_fact_does_not_cover_a_prediction_in_the_same_sentence`).
        rest = _RANGE_WORD_RE.sub(" ", rest)
```

`check_agent_answer` — два именованных аргумента с умолчанием `frozenset()`; в цикле по предложениям `_opponent_claim(sentence, calc_numbers, frequency_numbers, chart_numbers)` и:

```python
        if (
            ASSUMPTION_MARK not in sentence
            and assumed_numbers
            and _assumed_share_in(sentence, assumed_numbers)
        ):
            violations.append(
                f"эквити против суженного диапазона без метки {ASSUMPTION_MARK} — «{quoted}»"
            )
```

`agent/run.py`, `_violations` — в вызов `check_agent_answer`: `chart_numbers=chart_numbers_in(messages)`, `assumed_numbers=assumed_numbers_in(messages)`.

- [ ] **Step 4:** `uv run pytest tests/test_agent_checks.py tests/test_agent.py -q -ra` → PASS; все прежние тесты проверок — без правок ожиданий. Мутации ревьюеру: убрать `_CHART_WORD_RE.search` из `_chart_backed` — второй assert `test_a_chart_range_sentence_needs_no_mark` обязан упасть; убрать ветку `assumed` — `test_an_equity_against_a_narrowed_range_needs_the_mark` обязан упасть. `uv run pytest -q -ra`, `uv run ruff check . && uv run pyright` → чисто.

- [ ] **Step 5: Коммит**

```bash
git add src/harness/explanation/agent_checks.py src/harness/agent/run.py tests/test_agent_checks.py
git commit -m "$(cat <<'EOF'
Проверка ответа: пересказ чарта — факт, эквити против суженного — только с меткой

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

**Докстринги, которые становятся ложью:**
- `agent_checks.py:7-8` («В v1 диапазоны не считает ни один расчёт (они — спека №3)») → «стартовый диапазон по чарту считает `opponent_range` (факт), эквити против суженного — `equity_vs_range` (допущение)»; в абзац о метке — два новых правила с именами тестов.
- `_opponent_claim` (`:419-424`) — исключение «чарт + число выхода».
- `check_agent_answer` (`:719-725`) — `chart_numbers`, `assumed_numbers`.

---

### Task 8: Промпт агента — разделы (оркестратор)

**Кто:** оркестратор пишет, владелец правит. **Текст промпта в плане не приводится и в гит не попадает** (`src/harness/agent/prompts/system.md`, закрыт `.gitignore:42` и хуком).

Разделы, которые меняются или появляются (спека §7–§8):

1. **Ответ о прибыльности ставки — до трёх абзацев** в порядке спеки §8: порог из сводки → статистика по уровню данных → вывод «в плюсе / в минусе» только при решённом сравнении → назначение из сводки; эквити при колле — шансы сводки по отдельности, без суммы, «реальный порог ниже» — словами без числа; диапазон соперника — `opponent_range` → сужение классами с «(допущение)» → `equity_vs_range`; нет чарта — одна фраза «чарта для этого спота нет, диапазон не оцениваю».
2. **Порог своей ставки** — из сводки (строка «Назначение, …»); `defense_frequency` — для чужой ставки или произвольного размера.
3. **Чтение диапазона по линии** (раздел уже есть, спека №1 §10) — стартовый только из `opponent_range`; сужение — подмножество стартового, целыми классами, без весов; пересказ стартового — со словом «чарт».
4. **Запрет суммы шансов** («≈4% + ≈3% + ≈6.5%») — шансы пересекаются, оверкарта условна.

- [ ] **Step 1:** Правка файла на машине владельца; `git status --porcelain | grep prompts` → пусто.
- [ ] **Step 2:** `uv run pytest tests/test_agent.py -k system_prompt -q -ra` → PASS (тест не пропущен).
- [ ] **Step 3:** Владельцу на правку; коммита нет.

---

### Task 9: Первый эталон — формат `evals/agent/cases/`, хук, `.gitignore` (оркестратор)

**Кто:** оркестратор; `reviewer` — публикационная политика. Прогонщик и судья — спека №2; здесь только данные и формат.

**Files:**
- Modify: `.githooks/pre-push:21` — `FORBIDDEN_PATHS`.
- Modify: `.gitignore:38-39` — строка `evals/agent/` рядом с `evals/vision/`, `evals/e2e/`.
- Create (не коммитится): `evals/agent/cases/001-flop-cbet/{case.json,hand.json,analysis.json}`.

- [ ] **Step 1: Закрыть каталог ДО первого файла.** В `.githooks/pre-push:21`:

```bash
FORBIDDEN_PATHS='^(evals/vision/|evals/e2e/|evals/agent/|\.claude/screens/|fixtures/hh/.+\.txt$|\.env$|src/harness/[^/]+/prompts/|data/|reports/)'
```

Проверка без пуша:

```bash
FP=$(sed -n "s/^FORBIDDEN_PATHS='\(.*\)'$/\1/p" .githooks/pre-push)
printf 'evals/agent/cases/001-flop-cbet/case.json\n' | grep -Eq "$FP" && echo закрыт
git check-ignore -v evals/agent/cases/001-flop-cbet/case.json   # → .gitignore:…:evals/agent/
```

- [ ] **Step 2: Формат случая** (`case.json`, `schema_version` 1):

```json
{
  "schema_version": 1,
  "id": "001-flop-cbet",
  "created_at": "2026-10-14",
  "hand": {"enriched": "hand.json", "analysis": "analysis.json"},
  "history": [],
  "question": "<вопрос игрока дословно>",
  "reference_answer": "<ответ-образец владельца, исправленный: без суммы шансов, третий абзац с числами расчёта>",
  "required_calls": [
    {"calc": "frequency_vs_threshold", "params": {"subject": "field", "position": "BB"}},
    {"calc": "opponent_range", "params": {"opponent_position": "BB", "street": "flop"}},
    {"calc": "equity_vs_range", "params": {"opponent_position": "BB", "street": "flop"}}
  ],
  "forbidden": [
    {"rule": "sum_of_chances", "why": "шансы пересекаются, оверкарта условна"},
    {"rule": "equity_without_calc", "why": "эквити — только против диапазона и только из ядра"},
    {"rule": "unasked_opponent_values", "why": "величины соперника без вопроса игрока"}
  ],
  "notes": "<откуда случай, решения владельца>"
}
```

`hand.json` — `EnrichedHand` (`model_dump(mode="json")`), `analysis.json` — `AnalysisResult`: из них прогонщик собирает сводку тем же `agent_context`, без стенда (отклонение 6). `required_calls[].params` — подмножество параметров, которое обязано совпасть; остальные параметры свободны.

- [ ] **Step 3: Выгрузка руки со стенда** (номер разбора — из `traces`/`sessions.last_analysis_id` владельца; в репозиторий и в план не пишется):

```bash
docker compose exec -T postgres psql -U harness -d harness -tAc \
  "select h.enriched from analyses a join hands h on h.id = a.hand_id where a.id = <ID>" \
  > evals/agent/cases/001-flop-cbet/hand.json
docker compose exec -T postgres psql -U harness -d harness -tAc \
  "select result from analyses where id = <ID>" > evals/agent/cases/001-flop-cbet/analysis.json
uv run python -c "
import json; from harness.contracts import AnalysisResult, EnrichedHand
from harness.explanation.agent_checks import agent_context
d='evals/agent/cases/001-flop-cbet/'
en=EnrichedHand.model_validate(json.load(open(d+'hand.json'))); r=AnalysisResult.model_validate(json.load(open(d+'analysis.json')))
print(agent_context(r, en).text)"
```

Последняя команда обязана напечатать сводку с новой строкой «Назначение, флоп: …» (Task 3). `reference_answer` — текст владельца, третий абзац с числами, которые выдал живой ход Task 11 (Step 4), а не из головы.

- [ ] **Step 4:** `git status --porcelain evals/` → пусто.
- [ ] **Step 5: Коммит** (только хук и `.gitignore`):

```bash
git add .githooks/pre-push .gitignore
git commit -m "$(cat <<'EOF'
Публикация: evals/agent/ — приватный датасет, закрыт хуком и .gitignore

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Документы (оркестратор)

**Кто:** оркестратор; `reviewer` — проза против кода. Строки — по дереву `e57a324`; каждое утверждение сверить с кодом после задач 1–7.

- `CLAUDE.md:35` («`evals/vision/`, `evals/e2e/` — размеченный eval-датасет») → плюс `evals/agent/`; `docs/publishing-policy.md:25` — то же в таблице.
- `.claude/ARCHITECTURE.md`:
  - `:236-243` (две зоны доверия; «диапазон берётся по приоритету заметки → архетип → популяция») — источник диапазона у агента: стартовый — чарт (опен или защита BB, `strict`), сужение — агент целыми классами (`assuming`, четвёртый источник допущения, всегда с «(допущение)»).
  - `:174-179` (вердикт по чарту) — справочник несёт и чарты защиты BB; разбор их не судит, читает только агент; правило неполного стола одно (`chart_table`, `defend_table`).
  - `:395-428` («Разговорный доступ», проверки ответа) — инструменты `opponent_range`, `equity_vs_range` (только агент); два новых правила проверки (пересказ чарта; эквити против суженного — с меткой); порог из сводки — опора арифметики порога.
  - `:506-511` (открытый вопрос 3, «Диапазон не берётся ниоткуда») — дописать: у агента стартовый диапазон берётся из чарта, сужение — допущение агента; в разборе по-прежнему перебор без диапазона.
- `.claude/EVALS.md`:
  - этаж 5 (`:100-128`): п.3 — первый эталон в `evals/agent/cases/` (формат Task 9, Step 2), прогонщик — спека №2;
  - «Граница механических проверок агента» (`:130-155`) — пересказ стартового диапазона без метки только со словом «чарт» и числом выхода `opponent_range`; доля эквити против суженного — только с меткой; сумму шансов код не ловит — промпт и судья.
- `.claude/SESSIONS_UX.md:507-518` (три состояния) — подписи «диапазон соперника по чарту», «эквити против суженного диапазона»; в состоянии «числа» эквити печатается с «(допущение)»; список «Посчитать могу» у агента длиннее, чем у `/ask`.
- `docs/superpowers/specs/2026-08-28-poker-harness-tech-spec-design.md`:
  - `:72-95` (структура репозитория) — `evals/` += `agent/`; `analysis/charts/data/` — опен и защита;
  - `:166-171` (§5.4 Range) — сужение агентом: подмножество классов стартового, доли чарта;
  - `:715-728` (§10, `evals/`) — `agent/`: случаи маршрутизации и ответа (формат Task 9), прогон — спека №2;
  - `:518` (`calc_cache`) — второе пространство ключей: `eqr:v1:<итерации>:<сид>:` (эквити агента).

- [ ] **Step 1:** Правки по списку.
- [ ] **Step 2:** `grep -rn "ни один расчёт\|не считает ни один\|Шесть расчётов\|все шесть" CLAUDE.md .claude docs/superpowers/specs src/harness` → остались только сознательно оставленные (про `/ask`).
- [ ] **Step 3: Коммит**

```bash
git add CLAUDE.md docs/publishing-policy.md .claude/ARCHITECTURE.md .claude/EVALS.md .claude/SESSIONS_UX.md docs/superpowers/specs/2026-08-28-poker-harness-tech-spec-design.md
git commit -m "$(cat <<'EOF'
Документы: диапазон соперника по чарту и сужение агентом, evals/agent, проверки факта и допущения

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Приёмка — гейты, сетка, развёртывание, живой ход, владелец

**Кто:** оркестратор (скрипт живого хода — `implementer-core` + `reviewer`, он трогает базу стенда); живую проверку делает владелец (`is_dev`).

**Files:**
- Create: `src/harness/worker/agent_probe.py` — живой ход без Телеграма (в образе: `scripts/` в образ не копируется — `Dockerfile:60` копирует только `src`).

- [ ] **Step 1: Гейты.** `uv run pytest -q -ra` — зелено; строку `skipped` назвать и сравнить с исходной (план гейтированных тестов не добавляет). **Зелёный прогон без фикстур не означает, что конвейер проверен** (CLAUDE.md) — на машине с фикстурами `skipped` по фикстурам = 0. `uv run ruff check . && uv run pyright` — чисто. Сетка: вывод скрипта Task 2, Step 4 на HEAD совпадает с выводом на `e57a324`.

- [ ] **Step 2: Скрипт живого хода** (`src/harness/worker/agent_probe.py`):

```python
"""Живой ход агента на стенде без Телеграма: вопрос → ход → журнал и ответ в stdout.

Ничего не отправляет и историю вечера не пишет. Строка `jobs` (статус `done` —
воркер её не возьмёт) и `traces` заводятся, потому что `llm_calls.trace_id` — FK
NOT NULL. Вызовы модели настоящие и платные.

    docker compose exec -T worker python -m harness.worker.agent_probe --tg-user-id N "вопрос"
"""

from __future__ import annotations

import argparse
import asyncio
import json

from harness.agent.history import load_history, trim_history
from harness.agent.run import run_agent_turn
from harness.memory.models import Job, async_session_factory
from harness.memory.repos import PlayersRepo, SessionsRepo
from harness.platform.config import Config
from harness.platform.llm import LLM
from harness.platform.trace import Trace
from harness.presentation import agent_turn_msg
from harness.worker.pipeline import _agent_context


async def _probe(tg_user_id: int, question: str, *, fresh: bool) -> None:
    cfg = Config.from_env()
    factory = async_session_factory(cfg.database_url)
    async with factory() as session:
        player = await PlayersRepo(session).find(tg_user_id)
        if player is None:
            raise SystemExit(f"игрока tg_user_id={tg_user_id} нет")
        evening = await SessionsRepo(session).active_or_create(player.id)
        state = await SessionsRepo(session).agent_state(evening.id)
        job = Job(type="agent", status="done", player_id=player.id, session_id=evening.id,
                  payload={"text": question, "probe": True})
        session.add(job)
        await session.commit()
        job_id, session_id = job.id, evening.id
    trace = Trace(factory)
    await trace.open(job_id)
    async with factory() as session:
        context = await _agent_context(session, state.last_analysis_id)
        history = [] if fresh else trim_history(load_history(state.history), cfg.agent_history_max_tokens)
        turn = await run_agent_turn(
            session, LLM(cfg, factory), player_id=player.id, session_id=session_id, text=question,
            trace_id=trace.trace_id, history=history, context=context,
            analysis_id=state.last_analysis_id if context is not None else None,
        )
        await session.commit()
    trace.record("agent_turn", question=question, **turn.journal())
    await trace.flush()
    msg = agent_turn_msg(turn.state, turn.prose, list(turn.results), from_context=turn.from_context,
                         from_history=turn.from_history, job_id=job_id)
    print(json.dumps(turn.journal(), ensure_ascii=False, indent=2))
    print("\n--- игрок увидел бы ---\n" + msg.text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tg-user-id", type=int, required=True)
    parser.add_argument("--fresh", action="store_true", help="без истории вечера")
    parser.add_argument("question")
    args = parser.parse_args()
    asyncio.run(_probe(args.tg_user_id, args.question, fresh=args.fresh))


if __name__ == "__main__":
    main()
```

Сверить до запуска: имена `Job`, `async_session_factory` (`memory/models.py:680`), `PlayersRepo.find` (`repos.py:115`), `Trace.open` (`trace.py:73`), `Trace.flush` (`trace.py:121`), `_agent_context` (`pipeline.py:1313`). Коммит:

```bash
git add src/harness/worker/agent_probe.py
git commit -m "$(cat <<'EOF'
Воркер: живой ход агента без Телеграма — для приёмки на стенде

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: Сборка и подъём** (`docs/deploy-runbook.md:291-298`; миграций в срезе нет — `alembic_version` не меняется). Промпт на месте до сборки (`ls src/harness/agent/prompts/system.md`); файл `defend_8max.json`, если владелец его дал и не коммитит, — на машине сборки до `build` (открытый вопрос 2).

```bash
docker compose build --build-arg GIT_SHA=$(git rev-parse HEAD)
docker compose up -d
git rev-parse HEAD
docker inspect --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}' $(docker compose ps -q worker)
docker inspect --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}' $(docker compose ps -q bot)
docker compose exec -T worker python -c "import harness.analysis.opponent_range, harness.worker.agent_probe; print('ok')"
```

Метки обоих контейнеров = `git rev-parse HEAD`; только после этого докладывать.

- [ ] **Step 4: Замер и живой ход на стенде.** Время эквити в контейнере — командой Task 5, Step 6 через `docker compose exec -T worker python -c …`; если вызов дольше ~3 с — константа вниз, пересборка. Затем живой ход по эталонному вопросу на руке владельца (вечер владельца, указатель на разбор этой руки):

```bash
docker compose exec -T worker python -m harness.worker.agent_probe --tg-user-id <владелец> --fresh "<вопрос эталона>"
```

Смотреть в журнале: вызваны `opponent_range` (зона `strict`) и `equity_vs_range` (`assuming`); `state` = `prose`; в ответе нет суммы шансов; стартовый диапазон назван со словом «чарт»; эквити — с «(допущение)». Без файла защиты — `opponent_range` отвечает «чарта для этого спота нет», и третий абзац — одна фраза (§8). Задержку хода — числом из `llm_calls` (`select purpose, status, latency_ms from llm_calls where purpose='agent' order by id desc limit 10;`).

- [ ] **Step 5: Живая проверка владельцем** (Телеграм, `is_dev`): тот же вопрос на той же руке; «не то посчитано» под ответом пишет `eval_cases` как раньше. Итог владельцу: что прошло, что нет, задержка числом, состояние ответа; числа живого хода — в `reference_answer` эталона (Task 9, Step 3).

---

## Самопроверка плана

- **Покрытие спеки:** §4 — Task 1; §5 — Task 2 (+ защита в Task 4); §6.1 — Tasks 4–6; §6.2 — Tasks 4–7; §7 — Task 3; §8 — Task 8; §9 — Task 9; §10 тесты — Tasks 1–7, сетка — Task 2/11, живой ход — Task 11; §11 — открытые вопросы ниже. Документы — Task 10.
- **Имена между задачами:** `DefendKey`/`DefendStrategy`/`load_defend_book` (T1) → `start_range` (T4); `defend_table` (T2) → T4; `RangeLine` (T4) → результаты (T5); `range_spot`/`start_range`/`equity_spot`/`board_at`/`hero_cards`/`live_combos` (T4) → исполнители (T5); `AGENT_ONLY_CALCS`, `OpponentRangeParams`, `EquityVsRangeParams` (T5) → инструменты (T6); метаданные `chart`/`assumed` и `chart_numbers_in`/`assumed_numbers_in` (T6) → проверки (T7); `AgentContext.threshold_numbers` (T3) → `_violations` (T3); `run_agent_turn(..., analysis_id=…)` (T6) → пробник (T11).
- **Review Focus:** все пять пунктов закреплены названными тестами в задачах 2, 3, 4, 5, 7.
- **Заглушки:** `<ID>`, `<вопрос эталона>`, `<владелец>` — данные стенда и владельца, в план и в гит не пишутся намеренно.

## Открытые вопросы владельцу

1. **Чарт защиты BB против BTN на 25 BB** (спека §11.1) — без него `opponent_range` для BB отвечает «чарта нет», и живой ход Task 11 проверит только эту ветку. Соседние глубины — по возможности.
2. **Коммитить ли `defend_8max.json`** (§11.2). Если нет — строка в `.gitignore`, `.dockerignore` не трогать (файл должен попасть в образ из рабочего дерева), шаг «приватные файлы» в `docs/deploy-runbook.md`.
3. **Отклонения 2–4** — новые правила проверки ответа (пересказ чарта без метки; эквити против суженного — только с меткой; порог сводки — опора арифметики).
4. **Лимп SB** — в опен-чарте есть `limp`, спека §6.1 велит отказ. Оставить отказ или отдавать диапазон лимпа?
5. **Потолок итераций** — 50 000 до замера; назвать число после Task 11, Step 4.
6. **Голые пары и доли чарта в выходе** — разрешены числами на весь вечер (отклонение 5); если на стенде это пропускает выдумки — печатать диапазон без долей.
