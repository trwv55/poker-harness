# Черновик: раздача в контексте модели вердикта (спека §5.7, отложено)

> **Статус: НЕ В РАБОТЕ.** Решение владельца 2026-09-12: эти четыре задачи вынесены из плана
> «Блок „Что было“» и ждут обсуждения §5.7. Они написаны под узкий вариант — карты, борд и
> позиция через поля `PointVerdict` плюс состав диапазона в выжимке. Направление владельца шире:
> модель должна видеть раздачу целиком, как связующее звено между игроком и системой. После
> обсуждения черновик **переписывается, а не исполняется**.

**Spec:** [§5.7](../specs/2026-08-28-poker-harness-tech-spec-design.md) «Раздача в контексте модели вердикта — отложено, обсуждается».

**Что здесь ценного и почему сохранено.** Два круга ревью (сессия 2026-09-11/12) нашли в этих
задачах дефекты, которые повторятся при любом дизайне, и их стоит помнить:

- цифровые пары `22`…`99` неотличимы от сумм в ББ никаким регэкспом — дыра в `unsupported_numbers`
  открывается одной строкой;
- точка хранится колонками `decision_points` (`_POINT_COLUMNS`, тест-страж) — новое поле
  `PointVerdict` без миграции теряется молча, а колонка `position` уже занята;
- состав диапазона словами: порядок по силе, а не по алфавиту; «наполовину» только при весе 0.5
  (веса бывают любые); число в хвосте («и ещё N») становится разрешённым для модели;
- eval-кейсы вердикта надо перегенерировать, когда меняется вход модели.

Нумерация задач оставлена исходной (1–4), чтобы совпадать с отчётами ревью.

---

### Task 1: Нотация карт перестаёт читаться как число — без дыры для двузначных сумм

Проверка верности читает `A5s` как 5, `J♥️` — как ничего (буква), `99` — как 99. Пока так, карты и состав диапазона в выжимку отдавать нельзя: модель процитирует нас и будет отбракована.

**Решение, принятое здесь, а не регэкспом.** Чисто цифровые пары (`22`…`99`) неотличимы от величин никаким шаблоном: «теряет 33 bb» и класс `33` — одна и та же строка. Регэксп вырезает только ОДНОЗНАЧНУЮ нотацию — карту с мастью и класс с суффиксом `s`/`o`. Цифровые пары выжимка регистрирует через `NumberBook.token` ровно тогда, когда печатает их в составе диапазона (Task 3): утечка ограничена парами, которые реально названы, и «99 bb» как сумма неправдоподобна. Это осознанный компромисс, и он записан в докстринге `numbers_in`.

**Files:**
- Modify: `src/harness/explanation/faithfulness.py` (`_NUMBER_RE` — строка 48; `numbers_in` — строка 112)
- Test: `tests/test_faithfulness.py`

**Interfaces:**
- Consumes: —
- Produces: `numbers_in(text: str) -> list[float]` — сигнатура прежняя, поведение новое.

- [ ] **Step 1: Написать падающие тесты**

```python
def test_a_suited_or_offsuit_class_is_not_a_number():
    """`A5s`, `T9o` — имена рук, а не величины: иначе модель наказана за то,
    что процитировала состав диапазона, который мы сами ей показали."""
    assert numbers_in("в диапазоне есть A5s и T9o") == []
    assert numbers_in("с 75s вы теряете 1.2 bb") == [1.2]


def test_a_card_with_a_suit_is_not_a_number():
    """Карта и борд — факты руки. Масть пишется символом с селектором U+FE0F."""
    assert numbers_in("у вас J♥️9♥️") == []
    assert numbers_in("борд K♥️ J♦️ 2♣️, потеря 3.9 bb") == [3.9]


def test_a_two_digit_amount_is_still_read_as_a_number():
    """Дыра, закрытая на ревью: пара `33` и сумма 33 bb — одна строка. Регэксп
    НЕ вырезает цифровые пары; их регистрирует выжимка, когда печатает.
    Иначе любая сумма от 22 до 99 bb уходила бы без отбраковки."""
    assert numbers_in("теряет 33 bb") == [33.0]
    assert numbers_in("банк 225") == [225.0]
    assert numbers_in("1.25 bb") == [1.25]
    assert unsupported_numbers("теряет 55 bb", allowed=[1.2]) == [55.0]


def test_a_bare_rank_digit_is_still_read():
    """«9 bb» — величина: вырезается нотация, а не цифра вообще."""
    assert numbers_in("теряет 9 bb") == [9.0]
```

- [ ] **Step 2: Запустить и убедиться, что падают**

Run: `uv run pytest tests/test_faithfulness.py -k "suited or suit_is_not or two_digit or bare_rank" -v`
Expected: FAIL на первых двух (`A5s` даст `[5.0]`; `J♥️9♥️` даст `[9.0]`; строка с бордом — `[2.0, 3.9]`); третий и четвёртый ПРОХОДЯТ уже сейчас — они страхуют от регрессии, и это нормально.

- [ ] **Step 3: Реализовать**

В `src/harness/explanation/faithfulness.py`, после `_NUMBER_RE`:

```python
# Нотация карт — НЕ величина, и до счёта чисел вырезается (план 2026-09-12,
# задача 1). Вырезается только ОДНОЗНАЧНАЯ: карта с мастью (`J♥️`) и класс с
# суффиксом (`A5s`, `T9o`). Чисто цифровые пары (`22`…`99`) не вырезаются —
# они неотличимы от сумм («теряет 33 bb»), и шаблон, съедающий их, открыл бы
# модели любую двузначную сумму без отбраковки
# (`test_a_two_digit_amount_is_still_read_as_a_number`). Такие пары
# регистрирует выжимка через `NumberBook.token` ровно тогда, когда печатает их
# в составе диапазона (`verdict_text._range_text`).
_CARD_NOTATION_RE = re.compile(
    r"\b[2-9TJQKA]{2}[so]\b"          # класс руки с суффиксом: A5s, T9o
    r"|[2-9TJQKA][♠♥♦♣]️?"      # карта с мастью: J♥️, 2♣
)
```

```python
def numbers_in(text: str) -> list[float]:
    """Все числа текста в порядке появления, с учётом разрядов и запятой-дроби.

    Однозначная нотация карт вырезается до счёта (`_CARD_NOTATION_RE`):
    карта — не количество. Цифровые пары НЕ вырезаются — см. комментарий у
    шаблона и `test_a_two_digit_amount_is_still_read_as_a_number`.
    """
    joined = _THOUSANDS_RE.sub("", text)
    joined = _CARD_NOTATION_RE.sub(" ", joined)
    return [
        float(match.group().replace(",", ".").replace("−", "-"))
        for match in _NUMBER_RE.finditer(joined)
    ]
```

- [ ] **Step 4: Запустить тесты файла, затем весь набор**

Run: `uv run pytest tests/test_faithfulness.py -q -ra` → PASS.
Run: `uv run pytest -q -ra` → PASS, число `skipped` прежнее.

- [ ] **Step 5: Коммит**

```bash
git add src/harness/explanation/faithfulness.py tests/test_faithfulness.py
git commit -m "Карта с мастью и класс с суффиксом — не число; цифровые пары остаются суммами"
```

---

### Task 2: Контекст спота — три поля `PointVerdict`, колонки и миграция 0011

Модель не знает ни карт, ни борда, ни позиции. Отдать их сменой сигнатуры `verdict_text` нельзя: eval-кейсы хранят только `AnalysisResult` (`platform/eval_runner._cases_from_dir`, строки 129-137). Поэтому поля едут в `PointVerdict`, заполняет их `analyze_hand`.

**Точка хранится строкой `decision_points`, а не в `analyses.result`** (`AnalysesRepo.save` пишет документ с `exclude={"points"}`, `repos.py:730`; `get_by_hand` собирает точки ТОЛЬКО по карте `_POINT_COLUMNS`, `repos.py:814-821`). Тест-страж `tests/test_memory.py:1732-1745` требует `set(PointVerdict.model_fields) == set(_POINT_COLUMNS)`. Значит: три новых поля = три колонки = миграция, иначе на пути повтора после падения (`existing.result`, `pipeline.py:1118-1120, 1196-1198`) модель получит точки без карт.

**Имя поля позиции — `hero_position`, не `position`:** колонка `position` в `decision_points` УЖЕ есть, это обстановка из `DecisionPoint` (`_CONTEXT_COLUMNS`, `repos.py:688`), и одноимённое поле вердикта столкнулось бы с ней в `row |=` (`repos.py:765-768`).

**Files:**
- Modify: `src/harness/contracts/analysis.py:186-197` (`PointVerdict`)
- Modify: `src/harness/analysis/__init__.py:37-44` (`analyze_hand`)
- Modify: `src/harness/memory/models.py` (класс `DecisionPointRow`, строки колонок 86-89 по счёту внутри класса; докстринг класса — список полей `PointVerdict`)
- Modify: `src/harness/memory/repos.py:669-681` (`_POINT_COLUMNS`)
- Create: `migrations/versions/0011_decision_point_spot_context.py`
- Test: `tests/test_contracts.py`, `tests/test_preflop_analysis.py`, `tests/test_memory.py`

**Interfaces:**
- Consumes: —
- Produces: `PointVerdict.hero_cards: list[str] = []`, `PointVerdict.board: list[str] = []`, `PointVerdict.hero_position: str = ""`. Формат карты — как в `CanonicalHand.dealt`: `"Jh"` (ранг, масть буквой); символ рисует изложение. Колонки `decision_points.hero_cards jsonb NOT NULL DEFAULT '[]'`, `board jsonb NOT NULL DEFAULT '[]'`, `hero_position varchar(16) NOT NULL DEFAULT ''`.

- [ ] **Step 1: Написать падающие тесты**

`tests/test_contracts.py` (`PointVerdict` на уровне модуля НЕ импортирован — добавить `from harness.contracts import PointVerdict` к импортам файла):

```python
def test_a_point_without_spot_context_still_loads():
    """Точки, записанные до появления полей, читаются тем же типом: они лежат
    в `decision_points` и в eval-кейсах, и умолчание колонки — пустое."""
    old = {
        "dp_index": 0, "street": "preflop", "spot": "pushfold_unopened",
        "zone": "strict", "action_taken": "fold", "best_action": "shove",
        "ev_diff_bb": -3.9,
    }
    point = PointVerdict.model_validate(old)
    assert point.hero_cards == [] and point.board == [] and point.hero_position == ""
```

`tests/test_preflop_analysis.py` (образец и импорты `enrich`, `normalize`, `parse_hand`, `SAMPLE` уже есть в файле — см. строку 357):

```python
def test_analyze_hand_fills_the_spot_context_on_every_point():
    """Карты, борд и позиция приходят к изложению через разбор, а не через
    смену сигнатуры: eval-кейсы хранят только `AnalysisResult`."""
    en = enrich(normalize(parse_hand(SAMPLE, source_ref="x")))
    res = analyze_hand(en)
    assert res.points, "образец обязан дать хотя бы одну точку"
    hero = next(p for p in en.hand.players if p.label == en.hand.hero_label)
    for point in res.points:
        assert point.hero_cards == en.hand.dealt[en.hand.hero_label]
        assert point.hero_position == hero.position
        if point.street is Street.PREFLOP:
            assert point.board == []
```

`tests/test_memory.py`, рядом с `test_every_field_of_a_point_verdict_has_its_column`. Фикстура файла — `db` (сессия), не `db_factory`; фабрики — `_player_with_session(db, tg_user_id)` (строка 421), `_save_hand_in(db, *, session_id, tournament_id, hand_no)` (427), `_verdict(spot, taken, best, ev_diff_bb, **over)` (651). Образец использования — тест на строке ~674.

```python
async def test_spot_context_survives_the_round_trip_through_decision_points(db):
    """Путь повтора после падения читает `existing.result` из базы: без колонок
    карты пропали бы молча, и модель на ретрае получила бы слепую выжимку."""
    _player_id, session_id = await _player_with_session(db, tg_user_id=7001)
    # `hands.tournament_id` — FK на `tournaments.id`: турнир создаётся первым,
    # как в `test_player_hands_come_grouped_by_tournament` (строка ~441).
    tournament_id = await TournamentsRepo(db).create(session_id=session_id, source_file="t.txt")
    hand_id = await _save_hand_in(db, session_id=session_id, tournament_id=tournament_id, hand_no="TM1")
    point = _verdict(
        SpotKind.PUSHFOLD_UNOPENED, "fold", "shove", -1.0,
        hero_cards=["Jh", "9h"], board=["Kh", "Jd", "2c"], hero_position="SB",
    )
    res = AnalysisResult(hand_no="TM1", points=[point], ranked=[0], total_ev_loss_bb=-1.0)
    await AnalysesRepo(db).save(hand_id=hand_id, result=res, decision_points=[])
    await db.commit()
    record = await AnalysesRepo(db).get_by_hand(hand_id)
    assert record is not None
    back = record.result.points[0]
    assert (back.hero_cards, back.board, back.hero_position) == (["Jh", "9h"], ["Kh", "Jd", "2c"], "SB")
```

`TournamentsRepo.create(*, session_id, source_file) -> int` — `repos.py:839`.

- [ ] **Step 2: Запустить и убедиться, что падают**

Run: `uv run pytest tests/test_contracts.py -k spot_context tests/test_preflop_analysis.py -k spot_context -v`
Expected: FAIL, `AttributeError: 'PointVerdict' object has no attribute 'hero_cards'`.

- [ ] **Step 3: Контракт**

В `PointVerdict`, после `detail`:

```python
    # Контекст спота для изложения (план 2026-09-12, задача 2): карты героя,
    # борд на момент решения и позиция героя. Это ФАКТЫ руки, а не величины
    # расчёта: модель получает их, чтобы объяснить решение, а не считать.
    #
    # Поля здесь, а не аргументом `verdict_text`: eval-кейсы хранят только
    # `AnalysisResult` (`platform/eval_runner._cases_from_dir`). Умолчания
    # обязательны — тот же тип читает строки `decision_points`, записанные до
    # миграции 0011 (`test_a_point_without_spot_context_still_loads`).
    #
    # `hero_position`, а не `position`: колонка `position` в `decision_points`
    # уже занята обстановкой из `DecisionPoint` (`repos._CONTEXT_COLUMNS`).
    #
    # Формат карты — как в `CanonicalHand.dealt` (`"Jh"`); символ масти рисует
    # изложение, контракт хранит источник.
    hero_cards: list[str] = []
    board: list[str] = []
    hero_position: str = ""
```

- [ ] **Step 4: Заполнение в одном месте**

`src/harness/analysis/__init__.py` — в импорт `from harness.contracts import ...` добавить `PointVerdict, Street`:

```python
def _with_spot_context(point: PointVerdict, en: EnrichedHand) -> PointVerdict:
    """Карты, борд и позицию дописывает разбор, а не каждый строитель вердикта.

    Одно место, а не пять: `PointVerdict` конструируют `preflop` (трижды),
    `river` и `classifier`. Борд — накопленный до улицы точки включительно; на
    префлопе его нет. `model_copy` валидаторов не гоняет, и это безопасно:
    `zone`/`assumption` здесь не меняются.
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
            "hero_position": next(
                (p.position for p in hand.players if p.label == hand.hero_label), ""
            ),
        }
    )


def analyze_hand(en: EnrichedHand) -> AnalysisResult:
    """Разобрать все точки решения героя в одной руке."""
    points = [_with_spot_context(verdict_for(dp, en), en) for dp in en.report.decision_points]
    return AnalysisResult(
```

- [ ] **Step 5: Запустить — контракт и разбор зелёные, страж хранения КРАСНЫЙ**

Run: `uv run pytest tests/test_contracts.py tests/test_preflop_analysis.py tests/test_memory.py -q -ra`
Expected: `test_every_field_of_a_point_verdict_has_its_column` FAIL — поля есть, колонок нет. Это и есть страж; дальше — колонки.

- [ ] **Step 6: Колонки в модели, карта, миграция**

`src/harness/memory/models.py`, `DecisionPointRow`, после `detail`:

```python
    # Контекст спота из `PointVerdict` (миграция 0011): карты и борд — списки,
    # как `tools`; позиция героя — `hero_position`, потому что `position` уже
    # занята обстановкой из `DecisionPoint`.
    hero_cards: Mapped[Any] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    board: Mapped[Any] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    hero_position: Mapped[str] = mapped_column(String(16), nullable=False, server_default="")
```

В докстринге класса, в списке «`PointVerdict` целиком, поле в поле», дописать `hero_cards`, `board`, `hero_position`.

`src/harness/memory/repos.py`, `_POINT_COLUMNS`:

```python
    "detail": "detail",
    "hero_cards": "hero_cards",
    "board": "board",
    "hero_position": "hero_position",
}
```

`migrations/versions/0011_decision_point_spot_context.py` — по образцу 0010:

```python
"""контекст спота в строке точки решения

Карты героя, борд и позиция едут к изложению полями `PointVerdict` (план
2026-09-12, задача 2). Точка хранится ТОЛЬКО в `decision_points`, поле в
поле, — значит, у каждого поля обязана быть колонка, иначе путь повтора
после падения (`existing.result`) читает точки без карт.

Уже записанные строки получают пустые умолчания: карты для них не
восстанавливаются задним числом — они есть в `hands.enriched`, и переливка
была бы отдельным решением, не миграцией схемы.

Revision ID: 0011
Revises: 0010
Create Date: 2026-09-12 12:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = '0011'
down_revision: str | Sequence[str] | None = '0010'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('decision_points', sa.Column(
        'hero_cards', postgresql.JSONB(astext_type=sa.Text()),
        nullable=False, server_default=sa.text("'[]'::jsonb")))
    op.add_column('decision_points', sa.Column(
        'board', postgresql.JSONB(astext_type=sa.Text()),
        nullable=False, server_default=sa.text("'[]'::jsonb")))
    op.add_column('decision_points', sa.Column(
        'hero_position', sa.String(length=16), nullable=False, server_default=''))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('decision_points', 'hero_position')
    op.drop_column('decision_points', 'board')
    op.drop_column('decision_points', 'hero_cards')
```

- [ ] **Step 7: Запустить тесты хранения и весь набор**

Run: `uv run pytest tests/test_memory.py -q -ra` → PASS, включая страж и round-trip (тесты БД поднимают Postgres через `alembic_config` из `conftest.py` — миграция накатывается там же).
Run: `uv run pytest -q -ra` → PASS.

- [ ] **Step 8: Коммит**

```bash
git add src/harness/contracts/analysis.py src/harness/analysis/__init__.py src/harness/memory/models.py src/harness/memory/repos.py migrations/versions/0011_decision_point_spot_context.py tests/test_contracts.py tests/test_preflop_analysis.py tests/test_memory.py
git commit -m "Карты, борд и позиция едут к изложению через разбор и хранятся колонками"
```

---

### Task 3: Выжимка называет карты, борд, позицию и состав допущенного диапазона

**Порядок классов — по силе, и это решение ревью.** Ключ `(-weight, name)` при равных весах сортирует по ASCII: цифры раньше букв, и для любого диапазона шире дюжины классов выжимка печатала бы `22, 33, …, 99, A2o, …` — низ диапазона вместо его верха, и регистрировала бы все восемь цифровых пар (компромисс Task 1 превращался в дыру: «теряет 55 bb» проходило бы). Готового статического порядка по силе в проекте нет; `analysis.tools.pushfold.classes_by_equity_against` — расчёт, и звать его из изложения нельзя. Поэтому в `contracts/ranges.py` заводится **детерминированный порядок для показа** по формуле Чена (высшая карта, пара ×2, +2 за одномастность, штраф за разрыв, бонус за коннектор). Это не оценка EV и не претензия на точность — только воспроизводимый порядок «сильнее раньше», закреплённый тестами на СВОЙСТВА (AA первый, пары выше своих коннекторов, suited выше offsuit тех же рангов, 72o в хвосте), а не на числа.

**Доля веса печатается как доля, а не как «наполовину».** Веса бывают любыми (`multiway._range_of` округляет до 6 знаков, `_average_range` усредняет): класс с весом 0.33, названный «наполовину», — выдуманная величина в самой выжимке. Печатается `на {book.pct(100*w)}%`, число регистрируется как любое число выжимки.

**Хвост — без числа.** `book.count(rest)` регистрировал бы малое целое (3, 5, 7) как разрешённое — ровно тот дефект, от которого файл защищён (`test_a_small_round_number_is_not_allowed_by_the_point_numbering`). Хвост — словами: «и другие, слабее».

**Files:**
- Modify: `src/harness/contracts/ranges.py` — новая `strength_order() -> tuple[str, ...]` и `chen_score(cls: str) -> float`
- Modify: `src/harness/explanation/hand_replay.py` — только переименование `_cards` → `cards_text`, `_board` → `board_text` (и ВСЕ их вызовы внутри файла: `hand_replay.py:284, 316` и рядом; проверить `grep -n '_cards\|_board' src/harness/explanation/hand_replay.py`), добавить оба в `__all__`
- Modify: `src/harness/explanation/verdict_text.py` (`_point_lines` — строка 230; `_detail_lines` — строка 206; импорты)
- Test: `tests/test_contracts.py`, `tests/test_verdict_text.py`

**Interfaces:**
- Consumes: `PointVerdict.hero_cards/.board/.hero_position` (Task 2); `numbers_in` (Task 1)
- Produces: `verdict_digest(res) -> Digest` — сигнатура прежняя; публичные `hand_replay.cards_text(cards: list[str]) -> str`, `hand_replay.board_text(cards: list[str]) -> str`, которыми пользовался бы реплей (в плане §5.6 он остался на приватных `_cards`/`_board` того же файла; переименование понадобится только здесь).

- [ ] **Step 1: Написать падающие тесты**

```python
def test_the_digest_names_the_cards_and_the_position():
    """Модель обязана знать, чем и откуда сыграно, иначе объяснять ей нечем."""
    res = _result([_point(dp_index=0, ev_diff_bb=-3.9)])
    res.points[0] = res.points[0].model_copy(
        update={"hero_cards": ["Jh", "9h"], "hero_position": "SB"}
    )
    text = verdict_digest(res).text
    assert "J♥️9♥️" in text and "SB" in text


def test_the_digest_names_the_board_for_a_postflop_point():
    res = _result([_point(dp_index=0, ev_diff_bb=-1.0, spot=SpotKind.POSTFLOP)])
    res.points[0] = res.points[0].model_copy(update={"board": ["Kh", "Jd", "2c"]})
    assert "K♥️ J♦️ 2♣️" in verdict_digest(res).text


def test_the_digest_names_the_assumed_range_not_only_its_share():
    """Доля в процентах не объясняет ничего: «0.7% всех рук» нельзя пересказать
    словами, а «AA, KK наполовину» — можно."""
    res = _result([_point(dp_index=0, ev_diff_bb=-3.9, zone=Zone.ASSUMING)])
    text = verdict_digest(res).text
    assert "AA" in text and "KK наполовину" in text


def test_the_range_listing_is_stable_and_ordered_by_strength():
    """Два прогона одной руки обязаны дать одну выжимку, и порядок — по силе
    (`contracts.ranges.strength_order`), а не по порядку ключей и не по ASCII."""
    a = Range(weights={"KK": 1.0, "AA": 1.0, "QQ": 0.5, "22": 1.0})
    b = Range(weights={"22": 1.0, "QQ": 0.5, "AA": 1.0, "KK": 1.0})
    book = NumberBook()
    assert _range_text(a, book) == _range_text(b, book) == "AA, KK, QQ на 50.0%, 22"


def test_a_partial_weight_is_printed_as_its_share_not_as_a_half():
    """Вес 0.33 — не «наполовину»: доля печатается числом и регистрируется."""
    book = NumberBook()
    assert _range_text(Range(weights={"AA": 0.33}), book) == "AA на 33.0%"
    assert 33.0 in book.allowed


def test_a_wide_range_shows_its_top_and_names_the_tail_without_a_number():
    """Широкий диапазон: верх по силе, хвост словами. Ни одного малого целого
    в реестре от хвоста — иначе «теряет 5 bb» прошло бы проверку."""
    wide = Range(weights=dict.fromkeys(all_classes()[:40], 1.0))
    book = NumberBook()
    text = _range_text(wide, book)
    assert text.startswith("AA, KK, QQ") and text.endswith(" и другие, слабее")
    assert not {float(k) for k in range(2, 13)} & book.allowed
    digit_pairs_named = [t for t in text.replace(",", " ").split() if t.isdigit()]
    assert len(digit_pairs_named) <= 2, f"утечка цифровых пар: {digit_pairs_named}"


def test_a_digit_pair_in_the_range_is_registered_but_a_suited_class_is_not():
    """Компромисс задачи 1: `99` регистрируется как разрешённое (иначе цитата
    состава отбракуется), `A5s` — нет (его вырезает нотация)."""
    res = _result([_point(dp_index=0, ev_diff_bb=-3.9, zone=Zone.ASSUMING)])
    res.points[0] = res.points[0].model_copy(update={
        "assumption": res.points[0].assumption.model_copy(
            update={"range": Range(weights={"99": 1.0, "A5s": 1.0})})
    })
    digest = verdict_digest(res)
    assert 99.0 in digest.allowed and 5.0 not in digest.allowed


def test_the_digest_registers_every_number_it_prints_with_context():
    """Прежний инвариант файла держится и с картами, бордом и составом."""
    res = _result([_point(dp_index=0, ev_diff_bb=-3.9, zone=Zone.ASSUMING)])
    res.points[0] = res.points[0].model_copy(
        update={"hero_cards": ["Jh", "9h"], "board": ["Kh", "Jd", "2c"], "hero_position": "SB"}
    )
    digest = verdict_digest(res)
    assert unsupported_numbers(digest.text, digest.allowed) == []
```

Импорты в тесте: `Range`, `all_classes` (из `harness.contracts`), `NumberBook` (из `harness.explanation.digest`), `_range_text` через `verdict_text_module` (модуль импортируется в файле под этим именем, строка 43).

`tests/test_contracts.py` — свойства порядка, не числа:

```python
def test_strength_order_has_the_properties_a_display_order_needs():
    """Порядок для показа состава диапазона: воспроизводим, «сильнее раньше».
    Проверяются свойства, а не значения формулы — это не оценка EV."""
    order = strength_order()
    assert len(order) == 169 and len(set(order)) == 169
    assert order[0] == "AA"
    pos = {cls: i for i, cls in enumerate(order)}
    assert pos["KK"] < pos["AKs"] < pos["AKo"]          # пара выше своих коннекторов
    assert pos["QJs"] < pos["QJo"]                        # suited выше offsuit
    assert pos["TT"] < pos["99"] < pos["22"]              # пары по рангу
    assert pos["72o"] > 150                                # худшие — в хвосте
    assert order == strength_order()                       # детерминизм
```

- [ ] **Step 2: Запустить и убедиться, что падают**

Run: `uv run pytest tests/test_contracts.py -k strength_order tests/test_verdict_text.py -k "names_the or range_listing or partial_weight or wide_range or digit_pair or with_context" -v`
Expected: FAIL (нет `strength_order`; нет карт в выжимке; `_range_text` не существует).

- [ ] **Step 3: Реализовать**

`hand_replay.py`: переименовать `_cards` → `cards_text`, `_board` → `board_text`, поправить вызовы, добавить в `__all__`.

`contracts/ranges.py`:

```python
_CHEN_HIGH = {"A": 10.0, "K": 8.0, "Q": 7.0, "J": 6.0}


def chen_score(cls: str) -> float:
    """Формула Чена — порядок ДЛЯ ПОКАЗА, не оценка EV (план 2026-09-12, Task 3).

    Высшая карта (A=10, K=8, Q=7, J=6, иначе ранг/2), пара — удвоить (не меньше
    5), одномастность +2, штраф за разрыв (1→−1, 2→−2, 3→−4, ≥4→−5), коннектор
    ниже Q +1. Свойства закреплены
    `test_strength_order_has_the_properties_a_display_order_needs`; точные
    значения нигде не утверждаются и ничего в расчёте не решают.
    """
    hi, lo = cls[0], cls[1]
    value = lambda r: _CHEN_HIGH.get(r, (RANKS[::-1].index(r) + 2) / 2)  # noqa: E731
    score = value(hi)
    if hi == lo:
        return max(score * 2, 5.0)
    gap = RANKS.index(lo) - RANKS.index(hi) - 1
    score -= {0: 0.0, 1: 1.0, 2: 2.0, 3: 4.0}.get(gap, 5.0)
    if gap <= 1 and RANKS.index(hi) > RANKS.index("Q"):
        score += 1.0
    if cls.endswith("s"):
        score += 2.0
    return score


def strength_order() -> tuple[str, ...]:
    """169 классов, сильнее раньше, по `chen_score`; имя — вторичный ключ."""
    return tuple(sorted(all_classes(), key=lambda c: (-chen_score(c), c)))
```

`verdict_text.py`, импорты:

```python
from harness.contracts import Range, strength_order  # к существующему списку
from harness.explanation.digest import NumberBook  # уже импортирован — проверить
from harness.explanation.hand_replay import board_text, cards_text
```

```python
# Сколько классов диапазона называется поимённо. Порядок — по силе
# (`contracts.ranges.strength_order`), затем по имени. Потолок, а не весь
# состав: широкий диапазон занял бы половину промпта; хвост — словами, БЕЗ
# числа: `book.count` зарегистрировал бы малое целое как разрешённое
# (`test_a_small_round_number_is_not_allowed_by_the_point_numbering`).
_MAX_NAMED_CLASSES = 12
_RANK_OF = {cls: i for i, cls in enumerate(strength_order())}


def _range_text(rng: Range, book: NumberBook) -> str:
    """Состав диапазона словами: верх по силе, доли весом, хвост без числа.

    Цифровые пары (`99`) — через `book.token`: нотация их не вырезает (Task 1),
    и без регистрации цитата состава была бы отбракована. Утечка реестра —
    до двух-трёх старших пар в широком диапазоне (порядок по силе держит
    младшие за потолком; `test_a_wide_range_shows_its_top_...`). Доля веса —
    числом через `book.pct`: «наполовину» при весе 0.33 было бы выдумкой.
    """
    ordered = sorted(rng.weights.items(), key=lambda kv: (_RANK_OF[kv[0]], kv[0]))
    named: list[str] = []
    for name, weight in ordered[:_MAX_NAMED_CLASSES]:
        shown = book.token(name) if name.isdigit() else name
        named.append(shown if weight >= 1.0 else f"{shown} на {book.pct(100.0 * weight)}%")
    tail = " и другие, слабее" if len(ordered) > _MAX_NAMED_CLASSES else ""
    return ", ".join(named) + tail
```

В `_point_lines`, сразу после строки-заголовка точки:

```python
    context: list[str] = []
    if point.hero_position:
        context.append(f"позиция {point.hero_position}")
    if point.hero_cards:
        context.append(f"карты {cards_text(point.hero_cards)}")
    if context:
        lines.append("  " + "; ".join(context) + ".")
    if point.board:
        lines.append(f"  борд: {board_text(point.board)}.")
```

В блоке допущения — дописать состав:

```python
        lines.append(
            f"  допущение о диапазоне оппонента{note}; в нём {share}% всех рук: "
            f"{_range_text(assumption.range, book)}."
        )
```

Докстринг `_detail_lines` (строки ~215-217) — фраза «Глубины САМОГО героя, банка и позиции здесь нет намеренно» теперь неверна про позицию: убрать слово «позиции», оставить банк и глубину героя.

- [ ] **Step 4: Запустить**

Run: `uv run pytest tests/test_verdict_text.py tests/test_hand_replay.py -q -ra` → PASS.

- [ ] **Step 5: Коммит**

```bash
git add src/harness/contracts/ranges.py src/harness/explanation/verdict_text.py src/harness/explanation/hand_replay.py tests/test_contracts.py tests/test_verdict_text.py
git commit -m "Выжимка называет карты, борд, позицию и состав диапазона — по силе, долями, без чисел в хвосте"
```

---

### Task 4: Промпт вердикта — карты разрешены, диапазон обязателен

**Files:**
- Modify: `src/harness/explanation/prompts/verdict.md` (строка 4; пункт 5 «Что нельзя», строки 20-22; раздел «Что обязательно»)
- Test: `tests/test_verdict_text.py`

**Interfaces:**
- Consumes: выжимку из Task 3
- Produces: —

- [ ] **Step 1: Правки промпта**

Строка 4, было: `Ход раздачи игрок видит отдельно, пересказывать его не надо.`
Стало: `Ход раздачи игрок видит в блоке «Что было» над твоим текстом, пересказывать его не надо.`

Пункт 5, было (дословно, `verdict.md:20-22`):

```
5. **Банк, позицию, стек и карты игрок видит в реплее над твоим текстом —
   не называй их числом.** В выжимке их нет, и любая такая цифра будет твоей
   выдумкой, даже если ты угадаешь.
```

Стало:

```
5. **Банк и стек не называй числом.** В выжимке их нет, и любая такая цифра
   будет твоей выдумкой, даже если ты угадаешь. Карты, борд и позицию называть
   можно и нужно — они в выжимке есть.
```

Раздел «Что обязательно», после пункта про зону «предполагая»:

```
* Если у точки назван состав диапазона — пересказывай его руками, а не долей.
  «Если он отвечает только тузами и половиной королей» объясняет решение;
  «в его диапазоне 0.7% рук» не объясняет ничего.
```

- [ ] **Step 2: Тест связности (под гейтом — промпт есть не в каждом клоне)**

```python
@requires_prompts
def test_the_prompt_no_longer_forbids_naming_cards():
    """Промпт и выжимка обязаны говорить одно: карты показаны — значит разрешены."""
    prompt = verdict_text_module.read_prompt(verdict_text_module._PROMPT_PATH)
    assert "Карты, борд и позицию называть" in prompt
    assert "видит отдельно" not in prompt
```

Run: `uv run pytest tests/test_verdict_text.py -q -ra` → PASS (или skip в клоне без промптов — тогда строка `skipped` это покажет).

- [ ] **Step 3: Коммит**

```bash
git add src/harness/explanation/prompts/verdict.md tests/test_verdict_text.py
git commit -m "Промпт вердикта: карты и борд разрешены, состав диапазона обязателен"
```

Промпты не пушатся; решение о публикации — на пуше, хуком.

---

## Приёмка (когда черновик станет планом)

- [ ] `alembic upgrade head` на локальной базе — миграция накатывается и откатывается (если данные
  по-прежнему едут через `PointVerdict`).
- [ ] **Eval-кейсы вердикта перегенерировать** (`evals/verdict/cases/`, вне репозитория, делает
  владелец): в них `AnalysisResult` без новых полей, и eval-прогон промпта на них проверял бы не то.
  Перегенерация — тем же `analyze_hand` через `eval_runner --hh` (EVALS.md: eval модели — отдельный
  этаж от тестов кода).
