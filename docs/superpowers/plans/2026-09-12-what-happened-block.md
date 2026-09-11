# Блок «Что было»: реализация §5.6

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ход раздачи прозой в ББ встаёт первым в сообщении разбора, а модель перестаёт быть слепой — получает карты, борд, позицию и состав допущенного диапазона.

**Architecture:** Реплей остаётся чистым кодом (`explanation/hand_replay.py`, ноль токенов) и меняет только формат: проза вместо строк по улицам, ББ вместо фишек, метка и частоты оппонента, цена решения в конце. Контекст спота едет к модели не сменой сигнатуры `verdict_text`, а тремя необязательными полями `PointVerdict`, которые заполняет `analyze_hand`: так eval-кейсы и все три места вызова остаются рабочими, а свойство «`verdict_text` физически не видит раздачу» сохраняется.

**Tech Stack:** Python 3.12, pydantic 2, pytest, aiogram-совместимый слой `presentation` (чистые функции, возвращают `Msg`).

**Spec:** [docs/superpowers/specs/2026-08-28-poker-harness-tech-spec-design.md](../specs/2026-08-28-poker-harness-tech-spec-design.md), §5.6 «Блок „Что было“ — реплей руки, собранный кодом»

## Global Constraints

Требования ниже действуют в КАЖДОЙ задаче плана.

- **Правило зависимостей (CLAUDE.md).** `contracts`, `parsers`, `normalizer`, `engine`, `analysis`, `explanation`, `presentation` не импортируют Телеграм, БД и `harness.platform`. Данные приходят аргументами, результат возвращается значением. Про инфраструктуру знают только `bot`, `worker`, `memory`, `platform`.
- **Никогда не выдумывать числа о деньгах (CLAUDE.md).** Расхождение — эскалация или отказ с логом, но не «подправить, чтобы сошлось».
- **Судить решение против диапазона, а не против вскрытой карты (CLAUDE.md).** Правильный вход, проигравший по случайности, ошибкой не считается.
- **Все суммы блока — в ББ, с одним знаком после точки** (спека §5.6). Фишек в блоке нет. Приблизительности нет: «~5.5» обещало бы неуверенность, которой у кода нет.
- **Масти — символом, никогда буквами** (спека §5.6). За каждым символом стоит селектор эмодзи-презентации U+FE0F (`️`).
- **К игроку обращаются на «вы»** — так же, как во всех трёх промптах изложения.
- **Бюджет сообщения — 4096 символов**, `sendMessage` длиннее не отправляет вовсе. При нехватке места режется проза модели, а не блок «Что было» и не числа. Обрезка называется вслух.
- **Частот ровно две — VPIP и PFR.** Частота с нулевым знаменателем не печатается; пустая скобка не печатается вовсе.
- **Команды проверки:** `uv run pytest -q -ra` (флаг `-ra` обязателен — показывает пропуски), затем `uv run ruff check . && uv run pyright`.
- **Докстринг утверждает только то, что закреплено тестом.** Не писать в прозе обещаний, которых тест не проверяет.

---

### Task 1: Нотация карт перестаёт читаться как число

Проверка верности читает `A5s` как число 5, `99` как 99, `T9o` как 9. Пока это так, ни карты, ни состав диапазона в выжимку отдать нельзя: модель процитирует нас же и будет за это отбракована. Гнать карты через `NumberBook.token` нельзя — это разрешило бы модели писать «теряет 5 bb» свободно.

**Files:**
- Modify: `src/harness/explanation/faithfulness.py` (рядом с `_NUMBER_RE`, строка 48, и `numbers_in`, строка 112)
- Test: `tests/test_faithfulness.py`

**Interfaces:**
- Consumes: ничего из предыдущих задач (первая)
- Produces: `numbers_in(text: str) -> list[float]` с прежней сигнатурой и новым поведением; приватная `_CARD_RE` для тестов не нужна

- [ ] **Step 1: Написать падающие тесты**

```python
def test_a_hand_class_is_not_a_number():
    """`A5s`, `T9o`, `99` — имена рук, а не величины: иначе модель наказана за
    то, что процитировала состав диапазона, который мы сами ей показали."""
    assert numbers_in("отвечает только AA, KK и 99") == []
    assert numbers_in("в диапазоне есть A5s и T9o") == []


def test_a_card_is_not_a_number():
    """Карта и борд — факты руки. Масть пишется символом с селектором U+FE0F."""
    assert numbers_in("у вас J♥️9♥️") == []
    assert numbers_in("борд K♥️ J♦️ 2♣️") == []


def test_a_real_quantity_next_to_a_card_is_still_read():
    """Вырезание карт не имеет права глотать соседнюю величину — иначе проверка
    перестанет ловить выдуманные числа в той же фразе."""
    assert numbers_in("с A5s вы теряете 1.2 bb") == [1.2]
    assert numbers_in("борд K♥️, потеря 3.9 bb") == [3.9]


def test_a_bare_number_that_looks_like_a_rank_is_still_read():
    """«9 bb» — величина, а не карта: вырезается нотация, а не цифра вообще."""
    assert numbers_in("теряет 9 bb") == [9.0]
    assert unsupported_numbers("теряет 5 bb", allowed=[1.2]) == [5.0]
```

- [ ] **Step 2: Запустить и убедиться, что падают**

Run: `uv run pytest tests/test_faithfulness.py -k "card or hand_class or rank" -v`
Expected: FAIL — `numbers_in("отвечает только AA, KK и 99")` вернёт `[99.0]`, остальные тоже вернут лишние числа.

- [ ] **Step 3: Реализовать вырезание нотации**

В `src/harness/explanation/faithfulness.py`, рядом с `_NUMBER_RE`:

```python
# Нотация карт и классов рук — НЕ величины, и до счёта чисел она вырезается
# (задача 1 плана 2026-09-12). Иначе состав диапазона в выжимке (`A5s`, `99`)
# и карты руки (`J♥️9♥️`) прочтутся как выдуманные числа, и текст, честно
# процитировавший наш же промпт, будет отбракован целиком.
#
# Через `NumberBook.token` их гнать нельзя: это зарегистрировало бы 5 и 9 как
# разрешённые величины, и «теряет 5 bb» прошло бы проверку на любом разборе.
#
# Порядок в чередовании значим: класс руки (две карты подряд) идёт ПЕРВЫМ,
# иначе одиночная карта съест его первую половину и оставит хвост.
_RANK = "[2-9TJQKA]"
_SUIT_SYMBOL_CLASS = "[♠♡♢♣♤♥♦♧]️?"
_CARD_NOTATION_RE = re.compile(
    rf"{_RANK}{_RANK}[so]\b"           # класс руки: A5s, T9o
    rf"|{_RANK}{_RANK}\b"              # пара: AA, 99
    rf"|{_RANK}{_SUIT_SYMBOL_CLASS}",  # карта с мастью: J♥️
)
```

И в `numbers_in`:

```python
def numbers_in(text: str) -> list[float]:
    """Все числа текста в порядке появления, с учётом разрядов и запятой-дроби.

    Нотация карт вырезается до счёта (`_CARD_NOTATION_RE`): карта — не
    количество. Закреплено `test_a_hand_class_is_not_a_number` и
    `test_a_real_quantity_next_to_a_card_is_still_read`.
    """
    joined = _THOUSANDS_RE.sub("", text)
    joined = _CARD_NOTATION_RE.sub(" ", joined)
    return [
        float(match.group().replace(",", ".").replace("−", "-"))
        for match in _NUMBER_RE.finditer(joined)
    ]
```

- [ ] **Step 4: Запустить тесты**

Run: `uv run pytest tests/test_faithfulness.py -q -ra`
Expected: PASS, включая все прежние тесты файла.

- [ ] **Step 5: Прогнать весь набор — проверка верности общая для трёх промптов**

Run: `uv run pytest -q -ra`
Expected: PASS. Смотреть на строку `skipped`: без фикстур пропускается 5 тестов, это норма; больше — регрессия.

- [ ] **Step 6: Коммит**

```bash
git add src/harness/explanation/faithfulness.py tests/test_faithfulness.py
git commit -m "Карта — не количество: нотация перестала читаться как число"
```

---

### Task 2: Контекст спота едет в `PointVerdict`, а не в сигнатуру

Модель слепа: выжимка знает улицу и вид спота, но не знает ни карт, ни борда, ни позиции. Отдать их сменой сигнатуры `verdict_text` нельзя дёшево — eval-кейсы хранят только `AnalysisResult` (`platform/eval_runner.py`, `_cases_from_dir`), и раздачи у них нет. Поэтому три необязательных поля едут в `PointVerdict`, а заполняет их `analyze_hand`, у которого есть и рука, и точки.

Умолчания обязательны: `AnalysisResult` лежит в таблице `analyses` и в eval-кейсах, и старый JSON обязан читаться тем же типом.

**Files:**
- Modify: `src/harness/contracts/analysis.py:186-197` (класс `PointVerdict`)
- Modify: `src/harness/analysis/__init__.py:37-44` (`analyze_hand`)
- Test: `tests/test_contracts.py`, `tests/test_preflop_analysis.py`

**Interfaces:**
- Consumes: ничего из Task 1
- Produces: `PointVerdict.hero_cards: list[str]`, `PointVerdict.board: list[str]`, `PointVerdict.position: str` — все с умолчаниями. Формат карты — как в `CanonicalHand.dealt`: двухсимвольная строка ранга и масти буквой (`"Jh"`), символ рисует изложение.

- [ ] **Step 1: Написать падающие тесты**

В `tests/test_contracts.py`:

```python
def test_a_point_without_spot_context_still_loads():
    """Разборы, записанные до появления полей, читаются тем же типом: они лежат
    в `analyses` и в eval-кейсах, и миграции у jsonb нет."""
    old = {
        "dp_index": 0, "street": "preflop", "spot": "pushfold_unopened",
        "zone": "strict", "action_taken": "fold", "best_action": "shove",
        "ev_diff_bb": -3.9,
    }
    point = PointVerdict.model_validate(old)
    assert point.hero_cards == []
    assert point.board == []
    assert point.position == ""
```

В `tests/test_preflop_analysis.py` (рядом с прочими тестами `analyze_hand`):

```python
def test_analyze_hand_fills_the_spot_context_on_every_point():
    """Карты, борд и позиция приходят к изложению через разбор, а не через
    смену сигнатуры: eval-кейсы хранят только `AnalysisResult`."""
    en = enrich(normalize(parse_hand(SAMPLE, source_ref="x")))  # приём файла, строка 357
    res = analyze_hand(en)
    assert res.points, "образец обязан дать хотя бы одну точку"
    for point in res.points:
        assert point.hero_cards == en.hand.dealt[en.hand.hero_label]
        assert point.position != ""
```

- [ ] **Step 2: Запустить и убедиться, что падают**

Run: `uv run pytest tests/test_contracts.py::test_a_point_without_spot_context_still_loads tests/test_preflop_analysis.py -k spot_context -v`
Expected: FAIL с `AttributeError: 'PointVerdict' object has no attribute 'hero_cards'`.

- [ ] **Step 3: Добавить поля в контракт**

В `src/harness/contracts/analysis.py`, в `PointVerdict`, после `detail`:

```python
    # Контекст спота для изложения (план 2026-09-12, задача 2): карты героя,
    # борд на момент решения и позиция героя. Это ФАКТЫ руки, а не величины
    # расчёта, и на границу «точное считает код» они не посягают — модель
    # получает их, чтобы объяснить решение, а не чтобы что-то посчитать.
    #
    # Поля здесь, а не аргументом `verdict_text`: eval-кейсы хранят только
    # `AnalysisResult` (`platform/eval_runner._cases_from_dir`), и раздачи у
    # них нет. Умолчания обязательны — тот же тип читает jsonb, записанный до
    # появления полей (`test_a_point_without_spot_context_still_loads`).
    #
    # Формат карты — как в `CanonicalHand.dealt`: ранг и масть буквой (`"Jh"`).
    # Символ масти рисует изложение, контракт хранит источник.
    hero_cards: list[str] = []
    board: list[str] = []
    position: str = ""
```

- [ ] **Step 4: Заполнить их в одном месте**

В `src/harness/analysis/__init__.py`:

```python
def _with_spot_context(point: PointVerdict, en: EnrichedHand) -> PointVerdict:
    """Карты, борд и позицию дописывает разбор, а не каждый строитель вердикта.

    Одно место, а не пять: `PointVerdict` конструируют `preflop` (трижды),
    `river` и `classifier`, и пятикратное повторение развело бы формат при
    первой же правке. Борд берётся по улице точки — на префлопе его нет.
    """
    hand = en.hand
    board: list[str] = []
    for street in Street:
        board.extend(hand.boards.get(street, []))
        if street is point.street:
            break
    return point.model_copy(
        update={
            "hero_cards": list(hand.dealt.get(hand.hero_label, [])),
            "board": board,
            "position": next(
                (p.position for p in hand.players if p.label == hand.hero_label), ""
            ),
        }
    )


def analyze_hand(en: EnrichedHand) -> AnalysisResult:
    """Разобрать все точки решения героя в одной руке."""
    points = [_with_spot_context(verdict_for(dp, en), en) for dp in en.report.decision_points]
    return AnalysisResult(
```

Импорты в шапке файла: добавить `PointVerdict` и `Street` к существующему `from harness.contracts import ...`.

- [ ] **Step 5: Запустить тесты**

Run: `uv run pytest tests/test_contracts.py tests/test_preflop_analysis.py tests/test_river_analysis.py -q -ra`
Expected: PASS.

- [ ] **Step 6: Прогнать весь набор**

Run: `uv run pytest -q -ra`
Expected: PASS, `skipped` не вырос.

- [ ] **Step 7: Коммит**

```bash
git add src/harness/contracts/analysis.py src/harness/analysis/__init__.py tests/test_contracts.py tests/test_preflop_analysis.py
git commit -m "Карты, борд и позиция едут к изложению через разбор"
```

---

### Task 3: Выжимка называет карты, борд и состав допущенного диапазона

Текст выходил пересказом собственных цифр, потому что кроме цифр модель ничего не знала. Теперь у точки есть контекст спота, а у допущения — диапазон, из которого выжимка печатала только долю в процентах. «Предполагая, что он отвечает только тузами и половиной королей» и есть существо пуш-фолд вердикта.

**Files:**
- Modify: `src/harness/explanation/verdict_text.py` (`_point_lines`, строка 228; `_detail_lines`, строка 204)
- Test: `tests/test_verdict_text.py`

**Interfaces:**
- Consumes: `PointVerdict.hero_cards`, `.board`, `.position` из Task 2; `numbers_in` из Task 1
- Produces: `verdict_digest(res: AnalysisResult) -> Digest` — сигнатура НЕ меняется. Публикует
  `hand_replay.cards_text` и `hand_replay.board_text` (переименование приватных `_cards`/`_board`),
  которыми Task 5 пользуется дальше: формат карты обязан быть один на оба выхода изложения.

- [ ] **Step 1: Написать падающие тесты**

```python
def test_the_digest_names_the_cards_and_the_position():
    """Модель обязана знать, чем и откуда сыграно, иначе объяснять ей нечем."""
    res = _result([_point(dp_index=0, ev_diff_bb=-3.9)])
    res.points[0] = res.points[0].model_copy(
        update={"hero_cards": ["Jh", "9h"], "position": "SB", "board": []}
    )
    text = verdict_digest(res).text
    assert "J♥️9♥️" in text
    assert "SB" in text


def test_the_digest_names_the_assumed_range_not_only_its_share():
    """Доля в процентах не объясняет ничего: «0.7% всех рук» нельзя пересказать
    словами, а «только тузы и половина королей» — можно."""
    res = _result([_point(dp_index=0, ev_diff_bb=-3.9, zone=Zone.ASSUMING)])
    text = verdict_digest(res).text
    assert "AA" in text and "KK" in text


def test_the_cards_in_the_digest_add_no_allowed_numbers():
    """Карта не величина: показав `99`, мы не имеем права разрешить модели
    писать «99 bb». Держится вырезанием нотации (задача 1), а не реестром."""
    res = _result([_point(dp_index=0, ev_diff_bb=-3.9)])
    res.points[0] = res.points[0].model_copy(update={"hero_cards": ["9h", "9s"]})
    digest = verdict_digest(res)
    assert 9.0 not in digest.allowed


def test_the_digest_registers_every_number_it_prints_with_cards():
    """Прежний инвариант файла обязан держаться и с картами в выжимке: числа
    промпта минус разрешённые дают пустоту."""
    res = _result([_point(dp_index=0, ev_diff_bb=-3.9, zone=Zone.ASSUMING)])
    res.points[0] = res.points[0].model_copy(
        update={"hero_cards": ["Jh", "9h"], "board": ["Kh", "Jd", "2c"], "position": "SB"}
    )
    digest = verdict_digest(res)
    assert unsupported_numbers(digest.text, digest.allowed) == []
```

- [ ] **Step 2: Запустить и убедиться, что падают**

Run: `uv run pytest tests/test_verdict_text.py -k "cards or assumed_range" -v`
Expected: FAIL — в выжимке нет ни карт, ни состава диапазона.

- [ ] **Step 3: Реализовать**

Сначала — в `src/harness/explanation/hand_replay.py` — сделать рисование карт публичным, без правки
тела: `_cards` → `cards_text`, `_board` → `board_text`, оба в `__all__`. Формат карты обязан быть
один на оба выхода изложения, а Task 5 перепишет вокруг них всё остальное.

Затем в `src/harness/explanation/verdict_text.py`:

```python
from harness.contracts import Range
from harness.explanation.hand_replay import board_text, cards_text
```

Потолок на состав диапазона и его печать:

```python
# Сколько классов диапазона называется поимённо. Потолок, а не весь состав:
# широкий диапазон занял бы половину промпта списком, который модель всё равно
# не перескажет. Порядок — по весу, потом по имени: два прогона одной руки
# обязаны дать одну выжимку (`test_the_range_listing_is_stable`).
_MAX_NAMED_CLASSES = 8


def _range_text(rng: Range) -> str:
    """Состав диапазона словами: классы по убыванию веса, половинные — с долей."""
    ordered = sorted(rng.weights.items(), key=lambda kv: (-kv[1], kv[0]))
    named = [
        name if weight >= 1.0 else f"{name} наполовину"
        for name, weight in ordered[:_MAX_NAMED_CLASSES]
    ]
    tail = "" if len(ordered) <= _MAX_NAMED_CLASSES else " и другие"
    return ", ".join(named) + tail
```

В `_point_lines`, первой строкой точки, после заголовка — контекст спота:

```python
    if point.position or point.hero_cards:
        where = f"позиция {point.position}" if point.position else ""
        what = f"карты {cards_text(point.hero_cards)}" if point.hero_cards else ""
        lines.append("  " + "; ".join(part for part in (where, what) if part) + ".")
    if point.board:
        lines.append(f"  борд: {board_text(point.board)}.")
```

В блоке допущения (`_point_lines`, где печатается `share`) — дописать состав:

```python
        lines.append(
            f"  допущение о диапазоне оппонента{note}; в нём {share}% всех рук: "
            f"{_range_text(assumption.range)}."
        )
```

- [ ] **Step 4: Запустить тесты**

Run: `uv run pytest tests/test_verdict_text.py -q -ra`
Expected: PASS, включая прежний `test_the_digest_registers_every_number_it_prints`.

- [ ] **Step 5: Коммит**

```bash
git add src/harness/explanation/verdict_text.py tests/test_verdict_text.py
git commit -m "Выжимка называет карты, борд и состав допущенного диапазона"
```

---

### Task 4: Промпт вердикта снимает запрет на карты и требует назвать диапазон

Пункт 5 промпта запрещает называть карты числом, потому что их не было в выжимке. Теперь они там есть, и запрет надо переписать, иначе модель промолчит о том, что мы ей показали.

**Files:**
- Modify: `src/harness/explanation/prompts/verdict.md` (пункт 5 раздела «Что нельзя», раздел «Что обязательно»)
- Test: `tests/test_verdict_text.py` (тест на связность промпта и выжимки)

**Interfaces:**
- Consumes: выжимку из Task 3
- Produces: ничего программного — промпт читается `read_prompt(_PROMPT_PATH)`

- [ ] **Step 1: Переписать пункт 5**

Было:

```
5. **Банк, позицию, стек и карты игрок видит в реплее над твоим текстом —
   не называй их числом.** В выжимке их нет, и любая такая цифра будет твоей
   выдумкой, даже если ты угадаешь.
```

Стало:

```
5. **Банк и стек не называй числом.** В выжимке их нет, и любая такая цифра
   будет твоей выдумкой, даже если ты угадаешь. Карты, борд и позицию называть
   МОЖНО и нужно — они в выжимке есть.
```

- [ ] **Step 2: Дописать обязательство про диапазон**

В раздел «Что обязательно», после пункта про зону «предполагая»:

```
* Если у точки есть состав диапазона — назови его руками, а не долей.
  «Если он отвечает только тузами и половиной королей» объясняет решение;
  «в его диапазоне 0.7% рук» не объясняет ничего.
```

- [ ] **Step 3: Проверить связность промпта и выжимки**

```python
def test_the_prompt_no_longer_forbids_naming_cards():
    """Промпт и выжимка обязаны говорить одно: карты показаны — значит разрешены."""
    prompt = read_prompt(_PROMPT_PATH)
    assert "Карты, борд и позицию называть" in prompt
```

Run: `uv run pytest tests/test_verdict_text.py -q -ra`
Expected: PASS.

- [ ] **Step 4: Коммит**

```bash
git add src/harness/explanation/prompts/verdict.md tests/test_verdict_text.py
git commit -m "Промпт вердикта снимает запрет на карты и требует назвать диапазон"
```

**Внимание:** каталог промптов не пушится (CLAUDE.md, «Прежде чем что-либо публиковать»). Коммит локальный; решение о публикации принимается на пуше хуком `.githooks/pre-push`.

---

### Task 5: Реплей — проза, ББ, цена решения

Формат меняется целиком, поэтому тесты файла переписываются вместе с ним. Прежние `_lines(...) <= 5` и `<= 8` уходят: строк теперь две-три, и меряется длина прозы, а не число строк.

**Files:**
- Modify: `src/harness/explanation/hand_replay.py` (весь модуль)
- Test: `tests/test_hand_replay.py`

**Interfaces:**
- Consumes: ничего из предыдущих задач
- Produces: `hand_replay(en: EnrichedHand, *, ev_loss_bb: float | None = None) -> HandReplay`; `cards_text(cards: list[str]) -> str` и `board_text(cards: list[str]) -> str` — публичные, их зовёт Task 3; `chips()` остаётся ради `presentation` риверной точки

- [ ] **Step 1: Написать падающие тесты**

```python
def test_the_replay_speaks_in_big_blinds_not_chips():
    """Вердикт ниже говорит в ББ, и два масштаба в одном сообщении заставляют
    читателя пересчитывать."""
    text = hand_replay(_postflop_hand()).plain
    assert "250" not in text, "фишки остались в тексте"
    assert "ББ" in text


def test_the_replay_is_prose_not_one_line_per_street():
    """Построчный формат и был причиной, по которой блок прятали под кнопку."""
    assert len(_lines(hand_replay(_postflop_hand()).plain)) <= 4


def test_the_replay_ends_with_the_cost_when_it_is_known():
    """Цена — число ядра, а не суждение: спека §5.6 требовала её с первой редакции."""
    text = hand_replay(_postflop_hand(), ev_loss_bb=-3.9).plain
    assert "Потеря 3.9 ББ" in text


def test_the_replay_without_a_cost_says_nothing_about_it():
    """Рука без судимых точек не получает строки «потеря 0.0»: нуля расчёт
    не выносил, он просто ничего не судил."""
    assert "Потеря" not in hand_replay(_postflop_hand()).plain


def test_a_printed_amount_is_the_whole_bet_not_the_increment():
    """Спека §5.6: `бет 1.9` значит 1.9 ББ в банке от этого игрока на этой
    улице. Берётся `committed_after` — накопленное, а не разница с предыдущим
    действием, иначе рейз после бета показал бы добавку и читался бы вдвое дешевле."""
    en = _postflop_hand()
    raise_action = next(a for a in en.hand.actions if a.kind is ActionKind.RAISE)
    assert f"{raise_action.committed_after / en.hand.bb:.1f}" in hand_replay(en).plain


def test_the_replay_addresses_the_player_as_you():
    """Блок и текст под ним не имеют права говорить с игроком по-разному."""
    text = hand_replay(_postflop_hand()).plain
    assert "Hero" not in text
    assert "вы" in text.lower()
```

Плюс сохранить из прежнего файла: `test_the_replay_prints_no_number_the_hand_does_not_contain`, тест на слипание фолдов, тест на вскрытие без комбинаций, тест на селектор U+FE0F.

- [ ] **Step 2: Запустить и убедиться, что падают**

Run: `uv run pytest tests/test_hand_replay.py -q -ra`
Expected: FAIL — формат прежний.

- [ ] **Step 3: Переписать модуль**

Ключевые изменения, по одному:

```python
def bb(value_chips: int, big_blind: int) -> str:
    """Сумма в ББ, одним знаком. Единственный формат величин блока (спека §5.6).

    Приблизительности нет: движок считает банк точно, и «~5.5» обещало бы
    неуверенность, которой у кода нет.
    """
    return f"{value_chips / big_blind:.1f}"


def cards_text(cards: list[str]) -> str:
    """Карманные карты подряд: `J♥️9♥️` — так одномастность видна разом.

    Публична: тот же формат печатает выжимка вердикта (`explanation.verdict_text`),
    и две копии дали бы два вида одной карты в одном сообщении.
    """
    return "".join(_card(card) for card in cards)


def board_text(cards: list[str]) -> str:
    """Борд через пробел: это три отдельные карты, а не рука. Публична — см. `cards_text`."""
    return " ".join(_card(card) for card in cards)
```

`_action_text` — «вы» вместо `Hero`, суммы в ББ, без скобок с глубиной:

```python
    who = "вы" if action.label == hand.hero_label else _position(hand, action.label)
    if not show_amount:
        return f"{who} {word}"
    return f"{who} {word} {bb(action.committed_after, hand.bb)}"
```

`hand_replay` — шапка одной строкой, улицы через точку внутри абзаца:

```python
def hand_replay(en: EnrichedHand, *, ev_loss_bb: float | None = None) -> HandReplay:
    """Реплей одной руки: шапка строкой, ход раздачи прозой, цена в конце.

    `ev_loss_bb` — суммарная цена расхождений (`AnalysisResult.total_ev_loss_bb`).
    `None` значит «судимых точек нет», и строки о потере не будет вовсе: нуля
    расчёт не выносил (`test_the_replay_without_a_cost_says_nothing_about_it`).
    """
```

Названия улиц в прозе идут с прописной, а не капсом: капс был заголовком строки, внутри фразы он кричит.

```python
_STREET_TITLE: dict[Street, str] = {
    Street.PREFLOP: "Префлоп",
    Street.FLOP: "Флоп",
    Street.TURN: "Тёрн",
    Street.RIVER: "Ривер",
}


def hand_replay(en: EnrichedHand, *, ev_loss_bb: float | None = None) -> HandReplay:
    """Реплей одной руки: шапка строкой, ход раздачи прозой, цена в конце.

    `ev_loss_bb` — суммарная цена расхождений (`AnalysisResult.total_ev_loss_bb`).
    `None` значит «судимых точек нет», и строки о потере не будет вовсе: нуля
    расчёт не выносил (`test_the_replay_without_a_cost_says_nothing_about_it`).
    """
    hand = en.hand
    hero = _hero(hand)
    spans: list[ReplaySpan] = []

    hero_cards = hand.dealt.get(hand.hero_label, [])
    cards_part = f", {cards_text(hero_cards)}" if hero_cards else ""
    spans.append(
        ReplaySpan(text=f"Вы на {hero.position}{cards_part}, {bb(hero.stack, hand.bb)} ББ.\n")
    )

    decisions = _hero_decision_indices(en)
    pot_before = _dead_before_deal(hand)
    first = True
    for street in Street:
        actions = _street_actions(hand, street)
        board = hand.boards.get(street, [])
        if not actions:
            # Улица без ходов: либо её не было вовсе, либо все чекнули.
            if street is not Street.PREFLOP and board:
                spans.append(
                    ReplaySpan(text=f"{'' if first else ' '}{_STREET_TITLE[street]} "
                                    f"{board_text(board)} чек-чек.")
                )
                first = False
            continue
        head = "" if street is Street.PREFLOP else (
            f"{_STREET_TITLE[street]} {board_text(board)}, банк {bb(pot_before, hand.bb)}. "
        )
        spans.append(ReplaySpan(text=f"{'' if first else ' '}{head}"))
        spans.extend(_street_flow(hand, actions, decisions))
        spans.append(ReplaySpan(text="."))
        first = False
        pot_before = en.report.pot_by_street.get(street, pot_before)

    if ev_loss_bb is not None:
        spans.append(ReplaySpan(text=f" Потеря {abs(ev_loss_bb):.1f} ББ."))

    showdown = _showdown_line(hand)
    if showdown is not None:
        spans.append(ReplaySpan(text=f" {showdown}"))

    return HandReplay(spans=spans)
```

Итоговый банк отдельной строкой больше не печатается: `_MATERIAL_POT_GROWTH` и `_last_street_with_actions` уходят вместе с построчным форматом — банк каждой улицы и так стоит в начале её фразы.

- [ ] **Step 4: Запустить тесты**

Run: `uv run pytest tests/test_hand_replay.py -q -ra`
Expected: PASS.

- [ ] **Step 5: Коммит**

```bash
git add src/harness/explanation/hand_replay.py tests/test_hand_replay.py
git commit -m "Реплей говорит прозой, в ББ и называет цену решения"
```

---

### Task 6: Метка и частоты оппонента в реплее

`explanation` не имеет права импортировать `memory` (правило зависимостей). Статистика приходит аргументом, добывает её вызывающий.

**Files:**
- Modify: `src/harness/explanation/hand_replay.py`
- Test: `tests/test_hand_replay.py`

**Interfaces:**
- Consumes: `hand_replay` из Task 5
- Produces: `hand_replay(en, *, ev_loss_bb=None, stats: Mapping[str, PlayerStats] | None = None)` — ключ словаря это `CanonicalPlayer.label`, тот же, что у `player_stats_by_label`

- [ ] **Step 1: Написать падающие тесты**

```python
def test_an_opponent_carries_its_label_and_two_frequencies():
    stats = {"p2": PlayerStats(hands=40, vpip=10, pfr=7)}
    text = hand_replay(_postflop_hand(), stats=stats).plain
    assert "VPIP 25%" in text and "PFR 18%" in text


def test_an_opponent_without_a_sample_carries_no_brackets():
    """Скрин даёт одну руку, знаменателя нет. «VPIP 0%» — утверждение, которого
    никто не измерял; пустая скобка не печатается вовсе."""
    text = hand_replay(_postflop_hand(), stats={"p2": PlayerStats()}).plain
    assert "VPIP" not in text and "()" not in text


def test_a_measured_zero_is_printed_because_it_was_measured():
    """У VPIP и PFR знаменатель ОБЩИЙ (`PlayerStats.hands`), поэтому они
    появляются и исчезают вместе, а случая «в скобке одна из двух» нет.
    Ноль при непустом знаменателе — измеренный ноль: сорок раздач без единого
    рейза говорят об игроке ровно то, что должны."""
    stats = {"p2": PlayerStats(hands=40, vpip=10, pfr=0)}
    text = hand_replay(_postflop_hand(), stats=stats).plain
    assert "VPIP 25%" in text and "PFR 0%" in text
```

- [ ] **Step 2: Запустить и убедиться, что падают**

Run: `uv run pytest tests/test_hand_replay.py -k opponent -v`
Expected: FAIL — аргумента `stats` нет.

- [ ] **Step 3: Реализовать**

```python
def _opponent_mark(label: str, stats: Mapping[str, PlayerStats] | None) -> str:
    """Метка оппонента и две префлоп-частоты — те, у которых есть знаменатель.

    `PlayerStats._share` возвращает `None` при нулевом знаменателе, и это
    единственно честное поведение: «VPIP 0%» по нулю раздач никто не измерял.
    Пустая скобка не печатается вовсе (`test_an_opponent_without_a_sample_...`).
    """
    if stats is None or label not in stats:
        return ""
    row = stats[label]
    parts = [
        f"{name} {value:.0f}%"
        for name, value in (("VPIP", row.vpip_pct), ("PFR", row.pfr_pct))
        if value is not None
    ]
    return f" ({label}, {', '.join(parts)})" if parts else ""
```

Зовётся из `_action_text` для первого действия оппонента в руке — метка ставится один раз, а не у каждого хода.

- [ ] **Step 4: Запустить тесты**

Run: `uv run pytest tests/test_hand_replay.py -q -ra`
Expected: PASS.

- [ ] **Step 5: Коммит**

```bash
git add src/harness/explanation/hand_replay.py tests/test_hand_replay.py
git commit -m "Рядом с оппонентом — метка и две частоты, если есть знаменатель"
```

---

### Task 7: Блок встаёт первым в сообщении разбора, с бюджетом

Реплей возвращается в основное сообщение, откуда его увёл лимит 4096. Прозаический формат укладывается с запасом, но запас не гарантия: при нехватке места режется проза модели, а не блок и не числа.

**Files:**
- Modify: `src/harness/presentation/messages.py` (`deep_dive_msg`, строка 699)
- Modify: `src/harness/worker/pipeline.py` (два места вызова `deep_dive_msg`: около строк 1138 и 1225)
- Test: `tests/test_presentation.py`, `tests/test_worker_pipeline.py`

**Interfaces:**
- Consumes: `hand_replay` из Task 5 и 6
- Produces: `deep_dive_msg(..., replay: HandReplay | None = None)` — новый необязательный аргумент, первым печатается его текст

- [ ] **Step 1: Написать падающие тесты**

```python
def test_the_deep_dive_opens_with_what_happened():
    msg = deep_dive_msg(_res(), elapsed_s=12, zone=Zone.STRICT, quota_left=17,
                        quota_total=50, replay=_replay())
    assert msg.text.startswith("Что было")


def test_the_model_prose_is_cut_before_the_replay_is():
    """Слова необязательны, числа обязательны: в тесноте режется проза."""
    long_prose = VerdictTextOut(
        points=[PointText(dp_index=0, verdict_label="mistake", text="я" * 4000)],
        summary="",
    )
    msg = deep_dive_msg(_res(), elapsed_s=12, zone=Zone.STRICT, quota_left=17,
                        quota_total=50, replay=_replay(), verdict=long_prose)
    assert len(msg.text) <= 4096
    assert "Что было" in msg.text
    assert "показано не целиком" in msg.text
```

- [ ] **Step 2: Запустить и убедиться, что падают**

Run: `uv run pytest tests/test_presentation.py -k deep_dive -v`
Expected: FAIL — аргумента `replay` нет.

- [ ] **Step 3: Реализовать**

Строки собираются парами «текст, это ли проза модели», и при нехватке бюджета режется только проза.

```python
def _shrink_prose(rows: list[tuple[str, bool]], budget: int) -> list[tuple[str, bool]]:
    """Уместить сообщение в бюджет, срезая ТОЛЬКО прозу модели.

    Слова необязательны, числа обязательны — то же правило, по которому
    `worker.pipeline` отдаёт разбор без прозы, когда модель не ответила.
    Блок «Что было» и строки с числами не трогаются вовсе. Обрезка называется
    вслух маркером `_fitted` (`test_the_model_prose_is_cut_before_the_replay_is`).
    """
    fixed = sum(len(text) + 1 for text, is_prose in rows if not is_prose)
    left = budget - fixed
    out: list[tuple[str, bool]] = []
    for text, is_prose in rows:
        if not is_prose:
            out.append((text, is_prose))
            continue
        if left <= 1:
            continue  # места не осталось совсем — строка пропадает целиком
        out.append((_fitted(text, left - 1), True))
        left -= len(out[-1][0]) + 1
    return out
```

В самом `deep_dive_msg`: блок печатается первым, выделение точки решения — тем же `<b>` при
`parse_mode="HTML"`, что было в `replay_msg`, с экранированием остального текста через
`_html_escape`. Прежний абзац докстринга «**Реплея здесь нет — он под кнопкой «Подробнее»**» теперь
ложь и обязан уйти вместе с кодом; на его место — почему блок вернулся и что режется в тесноте.

```python
    head = "Что было\n" + "".join(
        f"<b>{_html_escape(span.text)}</b>" if span.emphasis else _html_escape(span.text)
        for span in replay.spans
    ) if replay is not None else ""
```

В `worker/pipeline.py` — оба места: посчитать реплей и передать.

```python
                verdict = await _verdict_prose(deps, trace, result)
                replay = hand_replay(
                    hand.enriched,
                    ev_loss_bb=result.total_ev_loss_bb if result.ranked else None,
                    stats=await _tournament_stats(session, hand),
                )
```

`_tournament_stats` — новая приватная функция воркера. Она живёт именно здесь, а не в `explanation`:
правило зависимостей запрещает изложению знать про БД (Global Constraints).

```python
async def _tournament_stats(
    session: AsyncSession, hand: HandRecord
) -> dict[str, PlayerStats] | None:
    """Частоты соседей по столу — или `None`, если считать их не по чему.

    Провенанс решает (спека §5.6): метка участника сквозная только внутри
    турнира, поэтому на HH-входе частоты набираются по рукам турнира, а на
    скриншоте их нет вовсе — одна рука не даёт знаменателя. `None`, а не пустой
    словарь: «не считали» и «посчитали, вышло пусто» — разные вещи, и реплей
    печатает скобку только по первому.
    """
    if hand.tournament_id is None:
        return None
    records = await HandsRepo(session).list_by_tournament(hand.tournament_id)
    hands = [record.canonical for record in records if record.canonical is not None]
    return player_stats_by_label(hands) if hands else None
```

Цена вопроса: один запрос на разбор, по индексу `tournament_id`, и разбор канонических рук турнира
в памяти. Турнир — сотни рук, не миллионы; если это когда-нибудь станет заметно, кэш считается по
`tournaments`, а не по каждому разбору (SCALING.md: сначала телеметрия, потом кэш).

- [ ] **Step 4: Запустить тесты**

Run: `uv run pytest tests/test_presentation.py tests/test_worker_pipeline.py -q -ra`
Expected: PASS.

- [ ] **Step 5: Прогнать весь набор**

Run: `uv run pytest -q -ra`
Expected: PASS, `skipped` не вырос.

- [ ] **Step 6: Коммит**

```bash
git add src/harness/presentation/messages.py src/harness/worker/pipeline.py tests/test_presentation.py tests/test_worker_pipeline.py
git commit -m "Блок «Что было» встаёт первым, а в тесноте режется проза"
```

---

### Task 8: «Подробнее» уходит из клавиатуры

Реплей теперь в основном сообщении, и показывать его второй раз незачем. Кнопки, не показывающей ничего, в интерфейсе быть не может, поэтому она уходит; слот остаётся за владельцем.

**Files:**
- Modify: `src/harness/presentation/keyboards.py:79` (строка кнопки внутри `verdict_buttons`), `:61` и `:201` (`DETAIL_PREFIX`). **Не трогать `deep_dive_button` (строка 65)** — это кнопка «разобрать» под строкой скана, другой путь.
- Modify: `src/harness/bot/handlers.py:1051` (ветка роутера), `:1132-1143` (`_replay_reply`)
- Modify: `src/harness/presentation/messages.py` (`replay_msg`, `replay_unavailable_msg`)
- Modify: `.claude/SESSIONS_UX.md`, раздел «Под вердиктом — инлайн-кнопки»
- Test: `tests/test_presentation.py`, `tests/test_bot_handlers.py`

**Interfaces:**
- Consumes: Task 7 (блок уже в основном сообщении — иначе кнопку убирать нельзя)
- Produces: клавиатура из двух кнопок

- [ ] **Step 1: Написать падающий тест**

```python
def test_the_verdict_buttons_are_two():
    """Реплей переехал в основное сообщение (план 2026-09-12, задача 7), и
    кнопка, показывающая его второй раз, осталась бы без содержания.

    Трогается ТОЛЬКО `verdict_buttons`. Однонамённая `deep_dive_button`
    («разобрать» под строкой скана) — другая кнопка и другой путь, её этот
    план не касается."""
    labels = [btn.text for btn in verdict_buttons("TM99")]
    assert labels == ["🎯 Диапазоны", "✋ Не согласен"]
```

- [ ] **Step 2: Запустить и убедиться, что падает**

Run: `uv run pytest tests/test_presentation.py -k keyboard -v`
Expected: FAIL — кнопок три.

- [ ] **Step 3: Убрать кнопку и мёртвый путь**

Удалить: строку кнопки в `keyboards.py`, `DETAIL_PREFIX` из констант и `__all__`, ветку `DETAIL_PREFIX` в роутере `handlers.py`, `_replay_reply`, `replay_msg`, `replay_unavailable_msg` и их тесты.

Проверить, что `_html_escape`-разметка выделения не потерялась: она нужна в `deep_dive_msg` (Task 7), туда и переехала.

- [ ] **Step 4: Привести SESSIONS_UX в соответствие**

В `.claude/SESSIONS_UX.md`, раздел «Под вердиктом — инлайн-кнопки»:

```
🎯 Диапазоны     ✋ Не согласен
```

Убрать строку про «Подробнее». Дописать: ход раздачи печатается блоком «Что было» в самом разборе, первым; слот третьей кнопки зарезервирован.

- [ ] **Step 5: Запустить весь набор**

Run: `uv run pytest -q -ra`
Expected: PASS. `skipped` не вырос.

- [ ] **Step 6: Линтеры и типы**

Run: `uv run ruff check . && uv run pyright`
Expected: чисто. Мёртвые импорты после удаления `replay_msg` — частая находка именно здесь.

- [ ] **Step 7: Коммит**

```bash
git add src/harness/presentation/keyboards.py src/harness/presentation/messages.py src/harness/bot/handlers.py .claude/SESSIONS_UX.md tests/test_presentation.py tests/test_bot_handlers.py
git commit -m "«Подробнее» уходит: реплей теперь в самом разборе"
```

---

## Приёмка плана целиком

- [ ] `uv run pytest -q -ra` — зелено, `skipped` равен пяти (без фикстур). **Зелёный прогон без фикстур не означает, что конвейер проверен** (CLAUDE.md): регрессионная сетка на 318 руках — главный приёмочный гейт, и без реальных HH она пропускается.
- [ ] `uv run ruff check . && uv run pyright` — чисто.
- [ ] Глазами: разбор одной руки со скрина (без частот) и из HH (с частотами) укладывается в 4096 символов и открывается блоком «Что было».
- [ ] Спека §5.6 и `.claude/SESSIONS_UX.md` описывают то, что в коде. Расхождение, найденное на этом шаге, правится документом, а не забывается — именно так разошлась первая редакция раздела.
