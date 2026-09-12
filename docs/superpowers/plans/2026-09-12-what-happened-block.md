# Блок «Что было»: реплей в разборе (спека §5.6)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ход раздачи прозой в ББ, с меткой и частотами оппонента и ценой решения, встаёт первым в сообщении разбора; кнопка «Подробнее» уходит.

**Architecture:** Реплей уже есть и остаётся чистым кодом (`explanation/hand_replay.py`, ноль токенов). Меняется его формат: проза вместо строк по улицам, ББ вместо фишек, «вы» вместо `Hero`, метка и частоты оппонента, цена решения в конце. `presentation.deep_dive_msg` ставит блок первым, воркер начинает пробрасывать `parse_mode`, бюджет 4096 держится по итоговому тексту. Что при этом видит модель вердикта, план не трогает — это §5.7, отложено.

**Tech Stack:** Python 3.12, pydantic 2, SQLAlchemy 2 async (Postgres), pytest; слой `presentation` — чистые функции, возвращающие `Msg`.

**Spec:** [docs/superpowers/specs/2026-08-28-poker-harness-tech-spec-design.md](../specs/2026-08-28-poker-harness-tech-spec-design.md), §5.6 «Блок „Что было“ — реплей руки, собранный кодом»

**История плана.** Первая редакция была на восемь задач и прошла два круга ревью (шесть, затем пятнадцать находок — почти все утверждения о коде по памяти). 2026-09-12 владелец разрезал её: задачи про контекст модели (правка проверки верности, поля `PointVerdict`, миграция, состав диапазона, промпт) ушли в [черновик §5.7](2026-09-12-verdict-context-draft.md) и ждут обсуждения; здесь остались четыре задачи про сам реплей, бывшие 5–8. Третий круг ревью по коммиту `35bf39a` не состоялся (лимит сессии), так что закрытие находок второго круга по этим задачам не подтверждено. Имплементатор: **не доверяй ни одному утверждению плана о коде, которого не видишь в открытом файле** — открой и сверь.

## Global Constraints

Действуют в КАЖДОЙ задаче.

- **Правило зависимостей (CLAUDE.md).** `contracts`, `parsers`, `normalizer`, `engine`, `analysis`, `explanation`, `presentation` не импортируют Телеграм, БД и `harness.platform`. Данные — аргументами, результат — значением. Про инфраструктуру знают только `bot`, `worker`, `memory`, `platform`.
- **Никогда не выдумывать числа о деньгах (CLAUDE.md).** Реплей печатает только числа, которые есть в руке и в `AnalysisResult`; проверку `unsupported_numbers` план не трогает.
- **Судить решение против диапазона, а не против вскрытой карты (CLAUDE.md).**
- **Все суммы блока — в ББ, с одним знаком после точки** (спека §5.6). Фишек в блоке нет. Приблизительности («~») нет.
- **Масти — символом, никогда буквами**, за символом селектор эмодзи-презентации U+FE0F.
- **К игроку обращаются на «вы».**
- **Бюджет сообщения — 4096 символов**, `sendMessage` длиннее не отправляет. При нехватке режется проза модели, не блок и не числа. Обрезка называется вслух маркером `_fitted`.
- **Частот ровно две — VPIP и PFR**, у них общий знаменатель `PlayerStats.hands`: появляются и исчезают вместе. Измеренный ноль печатается.
- **Улицы без ходов — прогон борда, печатается только борд.** Чек — действие движка (`ActionKind.CHECK`); «чек-чек» без действий в руке — выдумка.
- **Промпты не пушатся** (CLAUDE.md, «Прежде чем что-либо публиковать»); тесты, читающие промпт, гейтятся `@requires_prompts` из `tests/conftest.py`.
- **Команды проверки:** `uv run pytest -q -ra` (смотреть строку `skipped`: без фикстур и промптов пропуски есть, их число не должно расти), затем `uv run ruff check . && uv run pyright`.
- **Докстринг утверждает только закреплённое тестом.** Докстринги, которые задача делает ложью, правятся той же задачей — список в каждой задаче.

---
### Task 1: Реплей — проза, ББ, «вы», цена решения; прогон борда без выдуманных чеков

> **Контракт этой задачи изменён после её сдачи (решение владельца 2026-09-12).** Цена решения из
> блока убрана совсем: последняя фраза абзаца говорит об исходе раздачи в фишках («Забираете /
> Отдаёте N ББ»), а цена живёт строкой ниже, в разборе точки. Поэтому сигнатура ниже
> (`ev_loss_bb`), докстринги про «Потеря N ББ» и тесты `test_the_replay_ends_with_the_cost…` —
> история задачи, а не текущее положение дел. Действующее описание — спека §5.6, «Последняя фраза
> абзаца — исход раздачи, а не цена решения».

Формат меняется целиком, тесты файла переписываются вместе с ним.

**Files:**
- Modify: `src/harness/explanation/hand_replay.py` (весь модуль, включая модульный докстринг — строки 19-22 «Вердиктов и цен в bb здесь нет» становятся ложью)
- Test: `tests/test_hand_replay.py`

**Interfaces:**
- Consumes: `_cards`, `_board` (уже в `hand_replay.py:140, 145` — переименование в публичные не нужно)
- Produces: `hand_replay(en: EnrichedHand, *, ev_loss_bb: float | None = None) -> HandReplay`; `bb(value_chips: int, big_blind: int) -> str`; `chips()` ОСТАЁТСЯ (его зовёт `presentation/messages.py:656-657, 2201` для риверной точки). `ReplaySpan.emphasis` остаётся — Task 3 рисует его `<b>`.

- [ ] **Step 1: Переписать тесты файла**

Оставить (с правками) из прежнего файла:

- `test_hero_decision_is_emphasised_inside_the_flow_not_on_its_own_line` (строки 267-274) — выделение внутри потока сохраняется, но тест утверждает `"Hero" in span` — заменить на `"вы" in span.lower()`.
- `test_header_is_two_lines_with_level_blinds_and_hero` (223-232, `startswith("TSYN")`, `"Hero SB"`) — переписать в `test_header_is_one_line_with_position_cards_and_stack`: `startswith("Вы на SB, J♥️9♥️, 10.0 ББ.\n")`.
- `test_raise_over_a_raise_is_called_a_3bet_preflop` (298-328, `"рейз 250"`, `"3-бет 700"`) — суммы в ББ и «опен»: `"опен 2.5"`, `"3-бет 7.0"`.
- `test_the_replay_prints_no_number_the_hand_does_not_contain` (строки 355-383) — расширить `allowed` величинами в ББ, которые печатает новый формат:

```python
    allowed |= {round(v / hand.bb, 1) for v in en.report.pot_by_street.values()}
    allowed |= {round(sum(post.amount for post in hand.posts) / hand.bb, 1)}
    allowed |= {round(p.stack / hand.bb, 1) for p in hand.players}
```

- `test_showdown_line_shows_the_cards_that_were_actually_shown` (строки 386-392) — заменить `assert "Hero" in line` на `assert "вы J♥️9♥️" in line`.
- тест на селектор U+FE0F, на слипание фолдов (`UTG/HJ фолд`), на отсутствие комбинаций — оставить.
- Удалить: тесты на число строк `<= 5`/`<= 8`, на «итоговый банк отдельной строкой», на капс `ПРЕФЛОП`/`ТЁРН` в строке.

Добавить:

```python
def test_the_replay_speaks_in_big_blinds_not_chips():
    text = hand_replay(_postflop_hand()).plain
    assert "250" not in text and "300" not in text, "фишки остались в тексте"
    assert "ББ" in text


def test_the_replay_is_a_short_paragraph():
    """Построчный формат и был причиной, по которой блок прятали под кнопку."""
    assert len(_lines(hand_replay(_postflop_hand()).plain)) <= 3


def test_the_replay_addresses_the_player_as_you_everywhere():
    """Блок и текст под ним не имеют права говорить с игроком по-разному —
    включая строку вскрытия (`_preflop_shove_hand` её имеет)."""
    for en in (_postflop_hand(), _preflop_shove_hand()):
        text = hand_replay(en).plain
        assert "Hero" not in text
        assert "вы" in text.lower()


def test_a_printed_amount_is_the_whole_bet_not_the_increment():
    """Спека §5.6: `бет 1.9` — 1.9 ББ от этого игрока на этой улице целиком.
    Берётся `committed_after`, а не разница с предыдущим действием."""
    en = _postflop_hand()
    raise_action = next(a for a in en.hand.actions if a.kind is ActionKind.RAISE)
    assert f"опен {raise_action.committed_after / en.hand.bb:.1f}" in hand_replay(en).plain


def test_a_run_out_street_prints_only_its_board():
    """После олл-ина никто не ходит. Печатается борд — и НИКАКИХ «чек-чек»:
    приписать игрокам действия, которых не было, значит выдумать ход руки."""
    text = hand_replay(_preflop_shove_hand()).plain
    assert "чек" not in text.lower()
    assert "Флоп 6♠️ J♦️ Q♦️ · Тёрн 7♥️ · Ривер A♥️." in text


def test_the_replay_ends_with_the_cost_when_it_is_known():
    text = hand_replay(_postflop_hand(), ev_loss_bb=-3.9).plain
    assert text.rstrip().endswith("Потеря 3.9 ББ.")


def test_a_measured_zero_cost_is_printed():
    """`ranked` непустой при сумме 0.0 — измеренный ноль, и он печатается."""
    assert "Потеря 0.0 ББ." in hand_replay(_postflop_hand(), ev_loss_bb=0.0).plain


def test_the_replay_without_a_cost_says_nothing_about_it():
    assert "Потеря" not in hand_replay(_postflop_hand()).plain
```

Ожидаемый вид `_postflop_hand()` после правки (для сверки глазами): фикстура — Hero SB с J♥️9♥️ и стеком 1000 при bb 100, CO (P5) рейзит до 250, Hero коллирует, BB (P2) фолдит, на флопе Hero чек, CO бет 300, Hero фолд:

```
Вы на SB, J♥️9♥️, 10.0 ББ.
UTG/HJ фолд → CO опен 2.5 → BTN фолд → вы колл → BB фолд. Флоп 6♠️ J♦️ Q♦️, банк 6.6. Вы чек → CO бет 3.0 → вы фолд.
```

Первый шаг потока каждой улицы начинает предложение — с прописной («Вы чек»); внутри потока «вы» строчное. Закрепить:

```python
def test_a_street_sentence_starts_with_a_capital_even_when_it_is_you():
    text = hand_replay(_postflop_hand()).plain
    assert "банк 6.6. Вы чек" in text and "→ вы фолд" in text
```

- [ ] **Step 2: Запустить и убедиться, что падают**

Run: `uv run pytest tests/test_hand_replay.py -q -ra` → FAIL (старый формат).

- [ ] **Step 3: Переписать модуль**

Модульный докстринг: абзац «Вердиктов и цен в bb здесь нет» заменить на: «Вердиктов словами здесь нет; цена решения в ББ печатается последней фразой — это число ядра (`AnalysisResult.total_ev_loss_bb`), а не суждение (`test_the_replay_ends_with_the_cost_when_it_is_known`)». Абзац про «строки по улицам» заменить на прозу: одна шапка, один абзац.

Названия улиц — с прописной, не капсом; первый рейз префлопа — «опен»:

```python
_STREET_TITLE: dict[Street, str] = {
    Street.PREFLOP: "Префлоп",
    Street.FLOP: "Флоп",
    Street.TURN: "Тёрн",
    Street.RIVER: "Ривер",
}

# Порядковые имена рейзов префлопа: первый — «опен» (слово владельца, спека
# §5.6), дальше 3-бет, 4-бет. Счёт уже записанных действий, не новая величина.
_RERAISE_WORD: dict[int, str] = {1: "опен", 2: "3-бет", 3: "4-бет", 4: "5-бет"}


def bb(value_chips: int, big_blind: int) -> str:
    """Сумма в ББ, одним знаком — единственный формат величин блока (спека §5.6).
    Приблизительности нет: движок считает точно, «~5.5» обещало бы неуверенность."""
    return f"{value_chips / big_blind:.1f}"
```

`_action_word`: в ветке `RAISE and PREFLOP` — `_RERAISE_WORD.get(raise_ordinal, "рейз")` (теперь и ordinal 1 даёт «опен»).

`_action_text` — «вы» вместо `Hero`, суммы в ББ, без скобок глубины:

```python
def _action_text(hand: CanonicalHand, action: CanonicalAction, raise_ordinal: int) -> str:
    """Один ход: кто (позицией; герой — «вы»), что сделал и — у ставок — на сколько в ББ."""
    word = _action_word(action, raise_ordinal)
    who = "вы" if action.label == hand.hero_label else _position(hand, action.label)
    show_amount = action.is_all_in or action.kind in _ACTIONS_WITH_AMOUNT
    if not show_amount:
        return f"{who} {word}"
    return f"{who} {word} {bb(action.committed_after, hand.bb)}"
```

`_showdown_line` — `'вы' if entry.label == hand.hero_label else _position(...)`.

Удалить: `_MATERIAL_POT_GROWTH`, `_last_street_with_actions`, `_bb`; `_THIN` оставить (нужен `chips`). В `__all__` добавить `bb`.

Известное умолчание, не дефект: у руки, закончившейся олл-ином на префлопе, итоговый банк в блоке не печатается (банк стоит в начале фразы улицы с ходами, а постфлоп-ходов нет). Спека этого не требует; если владелец захочет — отдельная фраза «Банк N ББ.» после префлопа, когда дальше ходов нет.

```python
def hand_replay(en: EnrichedHand, *, ev_loss_bb: float | None = None) -> HandReplay:
    """Реплей одной руки: шапка строкой, ход раздачи прозой одним абзацем.

    `ev_loss_bb` — `AnalysisResult.total_ev_loss_bb`; `None` значит «судимых
    точек нет», и фразы о потере не будет: нуля расчёт не выносил
    (`test_the_replay_without_a_cost_says_nothing_about_it`). Ноль при
    непустом `ranked` — измеренный и печатается.

    Улица без ходов — прогон борда после олл-ина: печатается только борд, через
    ` · ` с соседними такими же (`test_a_run_out_street_prints_only_its_board`).
    Чек — действие движка и печатается как ход, «чек-чек» здесь не выдумывается.
    """
    hand = en.hand
    hero = _hero(hand)
    spans: list[ReplaySpan] = []

    hero_cards = hand.dealt.get(hand.hero_label, [])
    cards_part = f", {_cards(hero_cards)}" if hero_cards else ""
    spans.append(ReplaySpan(text=f"Вы на {hero.position}{cards_part}, {bb(hero.stack, hand.bb)} ББ.\n"))

    decisions = _hero_decision_indices(en)
    pot_before = _dead_before_deal(hand)
    quiet: list[str] = []
    first = True

    def sep() -> str:
        nonlocal first
        s = "" if first else " "
        first = False
        return s

    def flush_quiet() -> None:
        if quiet:
            spans.append(ReplaySpan(text=f"{sep()}{' · '.join(quiet)}."))
            quiet.clear()

    for street in Street:
        actions = _street_actions(hand, street)
        board = hand.boards.get(street, [])
        if not actions:
            if street is not Street.PREFLOP and board:
                quiet.append(f"{_STREET_TITLE[street]} {_board(board)}")
            continue
        flush_quiet()
        if street is not Street.PREFLOP:
            spans.append(ReplaySpan(
                text=f"{sep()}{_STREET_TITLE[street]} {_board(board)}, банк {bb(pot_before, hand.bb)}. "
            ))
        else:
            spans.append(ReplaySpan(text=sep()))
        flow = _street_flow(hand, actions, decisions)
        first_step = flow[0]
        flow[0] = first_step.model_copy(update={"text": first_step.text[:1].upper() + first_step.text[1:]})
        spans.extend(flow)
        spans.append(ReplaySpan(text="."))
        pot_before = en.report.pot_by_street.get(street, pot_before)
    flush_quiet()

    showdown = _showdown_line(hand)
    if showdown is not None:
        spans.append(ReplaySpan(text=f" {showdown}."))
    if ev_loss_bb is not None:
        spans.append(ReplaySpan(text=f" Потеря {abs(ev_loss_bb):.1f} ББ."))
    return HandReplay(spans=spans)
```

Вскрытие идёт ДО цены: спека — «завершает абзац цена решения».

- [ ] **Step 4: Запустить**

Run: `uv run pytest tests/test_hand_replay.py tests/test_verdict_text.py -q -ra` → PASS.

- [ ] **Step 5: Коммит**

```bash
git add src/harness/explanation/hand_replay.py tests/test_hand_replay.py
git commit -m "Реплей говорит прозой, в ББ, на «вы» и называет цену; прогон борда без выдуманных чеков"
```

---

### Task 2: Метка и частоты оппонента в реплее

`explanation` не импортирует `memory`: статистика приходит аргументом. Метка — `PlayerState.label` (`contracts/canonical.py`), регистр как в источнике (`P5`, не `p5`). Ставится один раз — при первом ходе оппонента, который дошёл до `_action_text`; слипшиеся фолды туда не доходят, и это правильно: у пасующего сказать нечего.

**Files:**
- Modify: `src/harness/explanation/hand_replay.py` (`hand_replay`, `_street_flow`, `_action_text`)
- Test: `tests/test_hand_replay.py`

**Interfaces:**
- Consumes: `hand_replay` (Task 1); `PlayerStats` из `harness.contracts`
- Produces: `hand_replay(en, *, ev_loss_bb=None, stats: Mapping[str, PlayerStats] | None = None)`; ключ словаря — `PlayerState.label`, тот же, что у `analysis.player_stats.player_stats_by_label`.

- [ ] **Step 1: Написать падающие тесты**

Активный оппонент `_postflop_hand` — `P5` (CO; `_SEATS` в файле, строка ~40; `P2` — BB, который только фолдит и в поток не попадает).

```python
def test_an_opponent_carries_its_label_and_both_frequencies_once():
    stats = {"P5": PlayerStats(hands=40, vpip=10, pfr=7)}
    text = hand_replay(_postflop_hand(), stats=stats).plain
    assert "CO (P5, VPIP 25%, PFR 18%) опен 2.5" in text
    assert text.count("P5") == 1, "метка ставится один раз, не у каждого хода"


def test_an_opponent_without_a_sample_carries_no_brackets():
    """Скрин даёт одну руку, знаменателя нет. «VPIP 0%» никто не измерял;
    пустая скобка не печатается вовсе."""
    text = hand_replay(_postflop_hand(), stats={"P5": PlayerStats()}).plain
    assert "VPIP" not in text and "(" not in text


def test_a_measured_zero_is_printed_because_it_was_measured():
    """У VPIP и PFR знаменатель ОБЩИЙ (`PlayerStats.hands`): появляются и
    исчезают вместе. Ноль по сорока раздачам — измеренный."""
    stats = {"P5": PlayerStats(hands=40, vpip=10, pfr=0)}
    assert "VPIP 25%, PFR 0%" in hand_replay(_postflop_hand(), stats=stats).plain


def test_a_frequency_rounds_half_up():
    stats = {"P5": PlayerStats(hands=8, vpip=1, pfr=1)}  # 12.5%
    assert "VPIP 13%, PFR 13%" in hand_replay(_postflop_hand(), stats=stats).plain


def test_a_folding_opponent_gets_no_label():
    """Слипшиеся фолды (`UTG/HJ фолд`) не несут ни метки, ни частот."""
    stats = {"P3": PlayerStats(hands=40, vpip=10, pfr=7)}
    assert "P3" not in hand_replay(_postflop_hand(), stats=stats).plain
```

- [ ] **Step 2: Запустить и убедиться, что падают**

Run: `uv run pytest tests/test_hand_replay.py -k opponent -v` → FAIL (нет аргумента `stats`).

- [ ] **Step 3: Реализовать**

```python
def _opponent_mark(label: str, stats: Mapping[str, PlayerStats] | None) -> str:
    """` (P5, VPIP 25%, PFR 18%)` — или пустая строка.

    `vpip_pct`/`pfr_pct` возвращают `None` при `hands == 0`, и это единственно
    честное поведение: «VPIP 0%» по нулю раздач никто не измерял. Знаменатель
    у них общий, поэтому либо обе, либо ни одной; отдельной ветки «одна из
    двух» нет — её не существует.
    """
    if stats is None or label not in stats:
        return ""
    row = stats[label]
    if row.vpip_pct is None or row.pfr_pct is None:
        return ""
    return f" ({label}, VPIP {_half_up(row.vpip_pct)}%, PFR {_half_up(row.pfr_pct)}%)"


def _half_up(pct: float) -> int:
    """Половина — вверх: `:.0f` и `round` округляют банковски (12.5 → 12), и
    правило нигде не было закреплено (`test_a_frequency_rounds_half_up`)."""
    return int(pct + 0.5)
```

Состояние «кому метка уже поставлена» живёт в `hand_replay` (тот зовётся один раз на руку) и передаётся вниз:

- `hand_replay(..., stats=None)`: `marked: set[str] = set()`; в цикле — `_street_flow(hand, actions, decisions, stats, marked)`.
- `_street_flow(hand, actions, hero_decisions, stats, marked)`: передаёт оба в `_action_text`.
- `_action_text(hand, action, raise_ordinal, stats, marked)`: для оппонента —

```python
    who = _position(hand, action.label)
    if action.label not in marked:
        who += _opponent_mark(action.label, stats)
        marked.add(action.label)
```

(герой — «вы», без метки; `marked.add` только для оппонентов.)

- [ ] **Step 4: Запустить**

Run: `uv run pytest tests/test_hand_replay.py -q -ra` → PASS.

- [ ] **Step 5: Коммит**

```bash
git add src/harness/explanation/hand_replay.py tests/test_hand_replay.py
git commit -m "Рядом с оппонентом — метка и обе частоты, один раз, если есть знаменатель"
```

---

### Task 3: Блок первым в сообщении, HTML до воркера, бюджет по итоговому тексту, статистика одной колонкой

Реплей возвращается в основное сообщение, откуда его увёл лимит 4096 (`deep_dive_msg`, докстринг строк 746-751). Три вещи, которых первая редакция плана не знала:

1. **Разбор отправляет воркер, а его `_payload` (`worker/main.py:128-136`) не передаёт `parse_mode`.** `parse_mode` пробрасывает только `bot/router.py:104`. Без правки воркера игрок увидит `<b>` и `&amp;` буквально.
2. **Бюджет меряется по ИТОГОВОМУ `msg.text`** — после экранирования (`&`→`&amp;` ×5) и `<b></b>` (+7 на span), иначе 4096 пробивается.
3. **`HandsRepo.list_by_tournament` десериализует `raw`, `canonical` И `enriched`** (`_to_record`, `repos.py:627-646`). Для частот нужна одна колонка — образец `player_hands_by_tournament` (`repos.py:525-565`).

**Files:**
- Modify: `src/harness/presentation/messages.py` (`deep_dive_msg`, строка 699 и докстринг 746-751; новая `_shrink_prose`)
- Modify: `src/harness/worker/main.py:128-136` (`_payload`)
- Modify: `src/harness/memory/repos.py` (`HandsRepo` — новый `canonical_by_tournament`)
- Modify: `src/harness/worker/pipeline.py` (скриншотный путь ~1146-1156: переменные `enriched`, `record`; путь разбора ~1225-1248: переменные `hand`)
- Test: `tests/test_presentation.py`, `tests/test_worker_pipeline.py`, `tests/test_memory.py`
- Modify (не пушится): `src/harness/explanation/prompts/verdict.md`, строка 4 «Ход раздачи игрок видит отдельно» — после этой задачи он в том же сообщении, над текстом; заменить на «над твоим текстом». Пункт 5 промпта уже так и говорит, его не трогать. Тесты промпта не нужны: смысл инструкции не меняется.

**Interfaces:**
- Consumes: `hand_replay` (Tasks 1-2), `player_stats_by_label` (существует, `analysis/player_stats.py:323`)
- Produces: `deep_dive_msg(..., replay: HandReplay | None = None) -> Msg` с `parse_mode="HTML"` при `replay is not None`; `HandsRepo.canonical_by_tournament(tournament_id: int) -> list[CanonicalHand]`; `worker.pipeline._tournament_stats(session, tournament_id: int | None) -> dict[str, PlayerStats] | None`.

- [ ] **Step 1: Написать падающие тесты**

`tests/test_presentation.py` (фабрики файла: `_prose_result()` — строка 1062, `_replay()` — 1093; `_replay` вернуть в новом виде спанов, с `emphasis=True` на одном):

```python
def _replay() -> HandReplay:
    return HandReplay(spans=[
        ReplaySpan(text="Вы на SB, J♥️9♥️, 10.0 ББ.\nUTG фолд → "),
        ReplaySpan(text="вы олл-ин 9.9", emphasis=True),
        ReplaySpan(text=" → BB & CO фолд."),
    ])


def test_the_deep_dive_opens_with_what_happened_in_html():
    msg = deep_dive_msg(_prose_result(), 12, Zone.STRICT, 17, 50, replay=_replay())
    assert msg.parse_mode == "HTML"
    assert msg.text.startswith("Что было\n")
    assert "<b>вы олл-ин 9.9</b>" in msg.text
    assert "BB &amp; CO" in msg.text, "остальной текст экранируется"


def test_without_a_replay_the_deep_dive_stays_plain():
    msg = deep_dive_msg(_prose_result(), 12, Zone.STRICT, 17, 50)
    assert msg.parse_mode is None and "Что было" not in msg.text


def test_the_model_prose_is_cut_before_the_replay_is():
    """Слова необязательны, числа обязательны: в тесноте режется проза, и
    инвариант меряется по ИТОГОВОМУ тексту — после экранирования и разметки."""
    res = _prose_result()
    long_prose = VerdictTextOut(
        points=[PointText(dp_index=p.dp_index, verdict_label="mistake", text="&" * 3000)
                for p in res.points],
        summary="я" * 2000,
    )
    msg = deep_dive_msg(res, 12, Zone.STRICT, 17, 50, replay=_replay(), verdict=long_prose)
    assert len(msg.text) <= 4096
    assert msg.text.startswith("Что было\n") and "<b>вы олл-ин 9.9</b>" in msg.text
    assert "показано не целиком" in msg.text
    assert "разборов 17/50" in msg.text, "статус-строка не режется"
```

`tests/test_memory.py`:

```python
async def test_canonical_by_tournament_reads_only_hands_with_a_canonical_checkpoint(db):
    """Одна колонка, как у `player_hands_by_tournament`: `raw` и `enriched`
    весят кратно больше, а частотам нужен только `canonical`."""
    _player_id, session_id = await _player_with_session(db, tg_user_id=7002)
    tid = await TournamentsRepo(db).create(session_id=session_id, source_file="t.txt")  # FK
    with_canon = await _save_hand_in(db, session_id=session_id, tournament_id=tid, hand_no="C1")
    raw_only = await _save_hand_in(db, session_id=session_id, tournament_id=tid, hand_no="C2")
    # `_save_hand_in` пишет и canonical — у второй руки его снять: открыть
    # фабрику (строка 427) и повторить её без `save_canonical`, либо обнулить
    # колонку прямым UPDATE; не выдумывать метод, которого нет.
    hands = await HandsRepo(db).canonical_by_tournament(tid)
    assert [h.hand_no for h in hands] == ["C1"]
```

Подготовка фикстуры (турнир, руки) — той же, что у теста `player_hands_by_tournament` в этом файле (найти по имени, повторить).

`tests/test_worker_pipeline.py` — в `test_deep_dive_saves_the_model_text_and_shows_it_to_the_player` (строки ~1399-1407) заменить две последние проверки:

```python
    # Ход раздачи — блоком «Что было» первым в самом разборе (план 2026-09-12).
    assert any(text.startswith("Что было\n") for text in texts)
    assert not any("ПРЕФЛОП" in text for text in texts)
```

(Проверку кнопки `detail:` — удалить здесь, а не в Task 4: после этой задачи кнопка ещё есть, но утверждение о ней уже не про этот тест.)

- [ ] **Step 2: Запустить и убедиться, что падают**

Run: `uv run pytest tests/test_presentation.py -k "deep_dive or prose_is_cut" -v` → FAIL (нет аргумента `replay`).

- [ ] **Step 3: `_payload` в воркере**

```python
def _payload(msg: Msg, **fields: object) -> dict[str, object]:
    """Тело запроса к Bot API: обязательные поля, `reply_markup` — когда кнопки
    есть, `parse_mode` — когда сообщение несёт разметку (`Msg.parse_mode`).
    Без последнего разбор с блоком «Что было» ушёл бы игроку с `<b>` буквально
    (`test_the_payload_carries_parse_mode_when_the_message_has_markup`).
    """
    body: dict[str, object] = {**fields, "text": msg.text}
    if msg.parse_mode is not None:
        body["parse_mode"] = msg.parse_mode
    markup = _keyboard(msg.buttons)
    if markup is not None:
        body["reply_markup"] = markup
    return body
```

Тестов на `_payload` в репозитории НЕТ (проверено `grep -rn "_payload\|parse_mode" tests/` — пусто). Добавить в `tests/test_worker_pipeline.py`, импортировав `from harness.worker.main import _payload`:

```python
def test_the_payload_carries_parse_mode_when_the_message_has_markup():
    """Разбор с блоком «Что было» едет в HTML; без этого поля Bot API показал
    бы `<b>` и `&amp;` буквально. Без разметки поля нет — как и раньше."""
    assert _payload(Msg(text="x", parse_mode="HTML"), chat_id=1)["parse_mode"] == "HTML"
    assert "parse_mode" not in _payload(Msg(text="x"), chat_id=1)
```

- [ ] **Step 4: `_shrink_prose` и `deep_dive_msg`**

Резать надо СЫРОЙ текст, а экранировать после: разрез по экранированному с вероятностью 4/5 попал бы внутрь `&amp;`, и что Telegram сделает с `&am […]` — неизвестно. Поэтому усадка идёт по сырым строкам с бюджетом, а итог меряется после рендера; если экранирование раздуло текст за предел — бюджет уменьшается на перелёт и усадка повторяется (сходится за считанные итерации: перелёт монотонно убывает).

```python
_MARKER = " […показано не целиком]"


def _shrink_prose(rows: list[tuple[str, bool]], budget: int) -> list[tuple[str, bool]]:
    """Уместить строки в бюджет, срезая ТОЛЬКО прозу модели (сырую, до экранирования).

    Слова необязательны, числа обязательны — то же правило, по которому
    `worker.pipeline` отдаёт разбор без прозы, когда модель не ответила.
    Строка, которая влезает целиком, не трогается; которой не хватает места
    даже на маркер — пропадает целиком (иначе `_fitted` вернул бы один маркер
    и пробил бюджет на его длину).
    """
    fixed = sum(len(text) + 1 for text, is_prose in rows if not is_prose)
    left = budget - fixed
    out: list[tuple[str, bool]] = []
    for text, is_prose in rows:
        if not is_prose:
            out.append((text, False))
            continue
        if len(text) + 1 <= left:
            out.append((text, True))
            left -= len(text) + 1
            continue
        if left <= len(_MARKER) + 1:
            continue
        cut = _fitted(text, left - 1)
        out.append((cut, True))
        left -= len(cut) + 1
    return out


def _render_html(head: str, rows: list[tuple[str, bool]]) -> str:
    """Блок уже с разметкой; остальные строки экранируются здесь, ПОСЛЕ усадки."""
    return "\n".join([head, *(_html_escape(text) for text, _ in rows)])


def _fit_html(head: str, rows: list[tuple[str, bool]], limit: int) -> str:
    budget = limit - len(head) - 1
    text = _render_html(head, rows)
    for _ in range(8):
        if len(text) <= limit:
            return text
        budget -= len(text) - limit
        text = _render_html(head, _shrink_prose(rows, budget))
    return _render_html(head, [(t, p) for t, p in rows if not p])  # крайний случай: без прозы
```

Тест плана с прозой из `"&"` — ровно худший случай: закрепить, что в итоге нет разорванной сущности: `assert "&am" not in msg.text.replace("&amp;", "")`.

В `deep_dive_msg(..., replay: HandReplay | None = None)`:

- если `replay is None` — прежнее поведение, `parse_mode=None`, текст не экранируется;
- иначе: `parse_mode="HTML"`, блок — первым, остальные строки собираются СЫРЫМИ парами `(text, is_prose)` и экранируются внутри `_fit_html` после усадки:

```python
    head = "Что было\n" + "".join(
        f"<b>{_html_escape(s.text)}</b>" if s.emphasis else _html_escape(s.text)
        for s in replay.spans
    )
    rows: list[tuple[str, bool]] = [("", False)]  # пустая строка после блока; сам блок — `head`
```

- дальше прежняя сборка `lines`, но в `rows` с флагом: строки из `_prose_lines(...)` и `verdict.summary` — `True`, всё остальное — `False`; статус-строка и кнопки — как были;
- `text = _fit_html(head, rows, _TELEGRAM_TEXT_LIMIT)`.

Докстринг `deep_dive_msg`: абзац 746-751 («Реплея здесь нет…») заменить на: «Блок «Что было» — первым (спека §5.6, план 2026-09-12): проза короче построчного реплея, ради которого его когда-то прятали за кнопку; в тесноте режется проза модели, не блок (`_shrink_prose`). `parse_mode="HTML"` только при наличии блока — иначе экранировать пришлось бы весь текст всюду».

- [ ] **Step 5: `canonical_by_tournament` и воркер**

`HandsRepo`:

```python
    async def canonical_by_tournament(self, tournament_id: int) -> list[CanonicalHand]:
        """Канонические руки одного турнира — одной колонкой.

        Вход частот оппонентов для блока «Что было» (план 2026-09-12). Читается
        только `canonical`, как в `player_hands_by_tournament`: `raw` и
        `enriched` весят кратно больше, а `player_stats_by_label` нужен лишь
        канон. Руки без чекпоинта пропускаются
        (`test_canonical_by_tournament_reads_only_hands_with_a_canonical_checkpoint`).
        """
        stmt = (
            select(Hand.canonical)
            .where(Hand.tournament_id == tournament_id, Hand.canonical.is_not(None))
            .order_by(Hand.id)
        )
        return [CanonicalHand.model_validate(row) for row in await self.db.scalars(stmt)]
```

`worker/pipeline.py`:

```python
async def _tournament_stats(
    session: AsyncSession, tournament_id: int | None
) -> dict[str, PlayerStats] | None:
    """Частоты соседей по столу — или `None`, если считать их не по чему.

    Провенанс решает (спека §5.6): метка участника сквозная внутри турнира,
    поэтому на HH-входе частоты набираются по рукам турнира, а у скриншота
    `tournament_id` нет — и частот нет. `None`, не пустой словарь: «не считали»
    и «посчитали, вышло пусто» — разные вещи.
    """
    if tournament_id is None:
        return None
    hands = await HandsRepo(session).canonical_by_tournament(tournament_id)
    return player_stats_by_label(hands) if hands else None
```

Скриншотный путь (`~1146`, переменные `enriched`, `record`):

```python
        replay = hand_replay(
            enriched,
            ev_loss_bb=result.total_ev_loss_bb if result.ranked else None,
            stats=None,  # скрин: одна рука, знаменателя нет
        )
        msg = deep_dive_msg(result, ..., verdict=verdict, replay=replay, ...)
```

Путь разбора (`~1225`, переменная `hand: HandRecord`):

```python
        replay = hand_replay(
            hand.enriched,
            ev_loss_bb=result.total_ev_loss_bb if result.ranked else None,
            stats=await _tournament_stats(session, hand.tournament_id),
        )
```

(`hand.enriched` может быть `None` по контракту `HandRecord` — на этом пути он уже проверен выше; если нет, `replay=None`.)

- [ ] **Step 6: Запустить и весь набор**

Run: `uv run pytest tests/test_presentation.py tests/test_worker_pipeline.py tests/test_memory.py -q -ra` → PASS.
Run: `uv run pytest -q -ra` → PASS, `skipped` прежний. Тест воркера гейтится фикстурами — без них он пропущен, и это надо ВИДЕТЬ в сводке.

- [ ] **Step 7: Коммит**

```bash
git add src/harness/presentation/messages.py src/harness/worker/main.py src/harness/worker/pipeline.py src/harness/memory/repos.py tests/test_presentation.py tests/test_worker_pipeline.py tests/test_memory.py
git commit -m "Блок «Что было» первым: HTML доезжает через воркер, в тесноте режется проза"
```

---

### Task 4: «Подробнее» уходит из клавиатуры — полный перечень мест

**Files:**
- Modify: `src/harness/presentation/keyboards.py:79` (строка кнопки в `verdict_buttons`), `:61` и `:201` (`DETAIL_PREFIX`). **Не трогать `deep_dive_button` (строка 65)** — «разобрать» под строкой скана, другой путь.
- Modify: `src/harness/presentation/__init__.py:11, 85-86, 107, 178-179` (реэкспорт `DETAIL_PREFIX`, `replay_msg`, `replay_unavailable_msg`)
- Modify: `src/harness/presentation/messages.py` (`replay_msg`, `replay_unavailable_msg` — удалить; `_html_escape` ОСТАЁТСЯ, его зовёт `deep_dive_msg`)
- Modify: `src/harness/bot/handlers.py:61` (импорт), `:167-179` (`UI_CALLBACK_PREFIXES` — убрать `DETAIL_PREFIX`), `:1050-1051` (ветка), `:1132-1143` (`_replay_reply`)
- Modify: `.claude/SESSIONS_UX.md`, раздел «Под вердиктом — инлайн-кнопки»
- Test — точные места, где `detail:` или `replay_msg` живут сейчас (проверено grep):
  - `tests/test_presentation.py:65` — импорт `replay_msg`, убрать;
  - `tests/test_presentation.py:485` — `["ranges:H99", "detail:H99", "disagree:H99"]` → две;
  - `tests/test_presentation.py:1139` — `any(... startswith("detail:") ...)` под `deep_dive_msg`, убрать утверждение;
  - `tests/test_presentation.py:1142-1160` — `test_replay_msg_marks_the_hero_decision_in_bold`, `test_replay_msg_escapes_a_nickname_that_looks_like_a_tag`: перенести в тесты `deep_dive_msg` с `replay=` (жирный и экранирование теперь там), сами тесты `replay_msg` удалить;
  - `tests/test_bot_handlers.py:945, 972` — докстринги перечисляют три префикса, поправить на два;
  - `tests/test_bot_handlers.py:1953` — `handle_ui_callback(deps, ..., "detail:RC1234")`: `handle_ui_callback` на неизвестный префикс возвращает `None` (`handlers.py:1053`), а «Эта кнопка не работает.» отдаёт `on_unhandled_callback` в `router.py:293`. Заменить на `assert await handle_ui_callback(deps, _TG_USER_ID, "detail:RC1234") is None` с докстрингом про старые сообщения; сквозной путь до текста уже закреплён `test_a_button_without_a_handler_still_gets_an_answer_not_a_spinner` (`:942`).
  - `tests/test_bot_handlers.py:1960-1971` — `test_the_details_button_of_an_unfinished_hand_says_so` импортирует `replay_unavailable_msg` — удалить целиком: сценария «рука без хода» у кнопки больше нет.

**Interfaces:**
- Consumes: Task 3 (блок уже в разборе)
- Produces: `verdict_buttons(hand_no) -> list[Btn]` из двух кнопок.

- [ ] **Step 1: Падающий тест**

```python
def test_the_verdict_buttons_are_two():
    """Реплей переехал в разбор (задача 3); кнопка, показывающая его второй раз,
    осталась бы без содержания. `deep_dive_button` («разобрать» под сканом) —
    другая кнопка, её план не касается."""
    assert [b.text for b in verdict_buttons("TM99")] == ["🎯 Диапазоны", "✋ Не согласен"]
```

- [ ] **Step 2: Запустить** → FAIL (три кнопки).

- [ ] **Step 3: Удалить кнопку и мёртвый путь** — по списку файлов выше. Побочный эффект, который надо назвать в докстринге `verdict_buttons`: кнопка «Подробнее» в УЖЕ отправленных сообщениях остаётся и теперь попадает в `on_unhandled_callback` → «Эта кнопка не работает.» (`messages.py:855`). Это ожидаемо и дешевле переписывания старых сообщений.

- [ ] **Step 4: SESSIONS_UX**

Раздел «Под вердиктом — инлайн-кнопки»:

```
🎯 Диапазоны     ✋ Не согласен
```

Убрать строку «**Подробнее** — развёрнутый разбор…». Дописать: ход раздачи печатается блоком «Что было» первым в самом разборе (спека §5.6); слот третьей кнопки зарезервирован, содержания у него пока нет (решение владельца 2026-09-12). **Таблицу станций прогресса (строки 119-123) НЕ трогать:** строка `подробнее: Считаю эквити… → Формулирую…` — это станции задачи `deep_dive` (кнопка «разобрать» под строкой скана), а не кнопка под вердиктом.

- [ ] **Step 5: Весь набор, линтеры, типы**

Run: `uv run pytest -q -ra` → PASS. Run: `uv run ruff check . && uv run pyright` → чисто (мёртвые импорты после удаления — частая находка здесь).

- [ ] **Step 6: Коммит**

```bash
git add src/harness/presentation/keyboards.py src/harness/presentation/__init__.py src/harness/presentation/messages.py src/harness/bot/handlers.py .claude/SESSIONS_UX.md tests/test_presentation.py tests/test_bot_handlers.py
git commit -m "«Подробнее» уходит: реплей теперь в самом разборе"
```

---

## Приёмка плана целиком

- [ ] `uv run pytest -q -ra` — зелено; число `skipped` равно исходному (без фикстур и промптов). **Зелёный прогон без фикстур не означает, что конвейер проверен** (CLAUDE.md): регрессионная сетка на 318 руках и тест воркера с блоком «Что было» без реальных HH пропускаются.
- [ ] `uv run ruff check . && uv run pyright` — чисто.
- [ ] Глазами, на живом Telegram: разбор со скрина (без частот) и из HH (с частотами) открывается блоком «Что было», укладывается в 4096, `<b>` не виден буквально; разбор с намеренно длинной прозой (заглушка модели) приходит с маркером обрезки и без обрывков `&am`.
- [ ] Спека §5.6 и `.claude/SESSIONS_UX.md` описывают то, что в коде. Расхождение, найденное здесь, правится документом той же задачей.
