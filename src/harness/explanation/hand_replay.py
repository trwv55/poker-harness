"""Реплей руки прозой — скелет раздачи, собранный кодом (спека §5.6).

Третий выход изложения рядом с текстом вердикта (LLM) и матрицей диапазонов
(код). **Ноль токенов и ноль новых расчётов**: почти всё, что здесь печатается,
`EnrichedHand` уже содержит — действия по улицам в порядке хода, банк каждой
улицы, стеки, точки решения героя; цена решения приходит аргументом из
`AnalysisResult`. Модуль только форматирует.

**Зачем он есть.** Разбор без хода руки нечитаем: читатель не может ни
восстановить раздачу, ни проверить вывод. И следствие для второго выхода: раз
ход руки показан кодом, промпт вердикта не пересказывает раздачу — класс ошибок
«модель переврала ход руки» становится негде совершить.

**Границы, которые модуль себе ставит.**

* Комбинации на вскрытии не называются («пара валетов») и эквити не печатается:
  и то и другое — оценка руки, то есть расчёт, а расчёт живёт в `analysis`.
  Печатаются только карты, которые игроки действительно показали
  (`test_showdown_line_shows_the_cards_that_were_actually_shown`).
* Вердиктов словами здесь нет; цена решения в ББ печатается последней фразой —
  это число ядра (`AnalysisResult.total_ev_loss_bb`), а не суждение
  (`test_the_replay_ends_with_the_cost_when_it_is_known`). Чего с ходом руки не
  так — соседние строки разбора (`presentation.deep_dive_msg`) и текст модели.
  Точка решения героя при этом выделена (`spans`), то есть найти её в потоке
  можно без единого числа отсюда.
* Ни одного числа, которого нет в руке: пришпилено
  `test_the_replay_prints_no_number_the_hand_does_not_contain`. Два числа
  приходят аргументами и потому этому запрету не противоречат — цена решения
  (`ev_loss_bb`) и префлоп-частоты оппонента (`stats`): обе величины посчитаны
  снаружи, реплей их только печатает.

**Форма — одна шапка и один абзац** (спека §5.6): позиция, карты и стек строкой,
дальше ход раздачи прозой — действия через `→`, улицы разделены точкой, борд
назван в начале своей улицы, банк один раз на улицу. Построчный формат по улицам
и был причиной, по которой блок прятали под кнопку: рука на шесть-восемь строк
занимает экран телефона целиком (`test_the_replay_is_a_short_paragraph`).

**Почему `HandReplay`, а не голая строка.** Спека требует выделить точку решения
героя жирным ПРЯМО в потоке действий, а разметка Телеграма — дело
`presentation`, не изложения (правило «`explanation` — покерное содержание,
`presentation` — вид в Телеграме»). Поэтому реплей возвращает последовательность
кусков с флагом выделения, а во что превратится выделение — решает тот, кто
собирает сообщение. `plain` — тот же текст без разметки, для тестов, отчётов и
любого канала, где выделять нечем.

**Масти — символом, никогда буквами** (спека §5.6: от скорости чтения
одномастности зависит оценка дро, и `Qs 8s` её не даёт). За каждым символом
стоит селектор эмодзи-презентации U+FE0F — им спека и требует цвет; каким
именно выйдет цвет, решает шрифт клиента, и проверить это отсюда нельзя.
Тест пришпиливает то, что в нашей власти: селектор стоит, букв нет.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel

from harness.contracts import (
    ActionKind,
    CanonicalAction,
    CanonicalHand,
    EnrichedHand,
    PlayerStats,
    Street,
)

__all__ = ["HandReplay", "ReplaySpan", "bb", "chips", "hand_replay"]

# Неразрывный пробел разделяет разряды: «31 250» в русском тексте, но перенос
# строки внутри числа невозможен. Запятая как разделитель разрядов исключена
# намеренно — «31,250» в русском читается как 31.25.
_THIN = " "

# Селектор эмодзи-презентации: без него ♥ и ♦ рисуются тем же чёрным глифом, что
# ♠ и ♣, и требование «символ плюс цвет» выполняется только на бумаге.
_VS16 = "️"

_SUIT_SYMBOL: dict[str, str] = {
    "s": f"♠{_VS16}",
    "h": f"♥{_VS16}",
    "d": f"♦{_VS16}",
    "c": f"♣{_VS16}",
}

_STREET_TITLE: dict[Street, str] = {
    Street.PREFLOP: "Префлоп",
    Street.FLOP: "Флоп",
    Street.TURN: "Тёрн",
    Street.RIVER: "Ривер",
}

# Слова действий — язык игрока, не токены движка (тот же словарь по смыслу, что
# `presentation._ACTION_WORD`, но здесь именуются другие вещи: там решение
# героя, тут ход за столом).
_ACTION_WORD: dict[ActionKind, str] = {
    ActionKind.FOLD: "фолд",
    ActionKind.CHECK: "чек",
    ActionKind.CALL: "колл",
    ActionKind.BET: "бет",
    ActionKind.RAISE: "рейз",
}

# Порядковые имена рейзов префлопа: первый — «опен» (слово владельца, спека
# §5.6), дальше 3-бет, 4-бет. Счёт уже записанных действий, не новая величина.
_RERAISE_WORD: dict[int, str] = {1: "опен", 2: "3-бет", 3: "4-бет", 4: "5-бет"}

# Суммы, которые показываются: у ставки видно, СКОЛЬКО поставлено, у колла и
# паса показывать нечего — сумма колла равна названной до него (спека §5.6:
# «реплей это скелет раздачи, а не протокол»).
_ACTIONS_WITH_AMOUNT = frozenset({ActionKind.BET, ActionKind.RAISE})


class ReplaySpan(BaseModel):
    """Кусок текста реплея и признак того, что он выделяется (точка решения героя)."""

    text: str
    emphasis: bool = False


class HandReplay(BaseModel):
    """Реплей как последовательность кусков; `plain` — тот же текст без разметки."""

    spans: list[ReplaySpan]

    @property
    def plain(self) -> str:
        return "".join(span.text for span in self.spans)


def chips(amount: int) -> str:
    """Сумма в фишках: разряды через неразрывный пробел.

    Публична ради второго читателя — `presentation.messages` печатает суммы
    риверной точки в том же сообщении, что и реплей; две копии формата дали бы
    два вида одного числа в двух соседних сообщениях.
    """
    return f"{amount:,}".replace(",", _THIN)


def bb(value_chips: int, big_blind: int) -> str:
    """Сумма в ББ, одним знаком — единственный формат величин блока (спека §5.6).
    Приблизительности нет: движок считает точно, «~5.5» обещало бы неуверенность."""
    return f"{value_chips / big_blind:.1f}"


def _card(card: str) -> str:
    return f"{card[0]}{_SUIT_SYMBOL.get(card[1], card[1])}"


def _cards(cards: list[str]) -> str:
    """Карманные карты подряд, без пробелов: `J♥️9♥️` — так одномастность видна разом."""
    return "".join(_card(card) for card in cards)


def _board(cards: list[str]) -> str:
    """Борд — через пробел (`6♠️ J♦️ Q♦️`): это три отдельные карты, а не рука."""
    return " ".join(_card(card) for card in cards)


def _position(hand: CanonicalHand, label: str) -> str:
    for player in hand.players:
        if player.label == label:
            return player.position
    return label


def _hero(hand: CanonicalHand):
    for player in hand.players:
        if player.label == hand.hero_label:
            return player
    raise ValueError(f"в раздаче {hand.hand_no} нет места героя ({hand.hero_label})")


def _half_up(pct: float) -> int:
    """Половина — вверх: `:.0f` и `round` округляют банковски (12.5 → 12), и
    правило нигде не было закреплено (`test_a_frequency_rounds_half_up`)."""
    return int(pct + 0.5)


def _opponent_mark(label: str, stats: Mapping[str, PlayerStats] | None) -> str:
    """` (P5, VPIP 25%, PFR 18%)` — или пустая строка.

    `vpip_pct`/`pfr_pct` возвращают `None` при `hands == 0`, и это единственно
    честное поведение: «VPIP 0%» по нулю раздач никто не измерял. Знаменатель
    у них общий, поэтому либо обе, либо ни одной; отдельной ветки «одна из
    двух» нет — её не существует
    (`test_an_opponent_without_a_sample_carries_no_brackets`).
    """
    if stats is None or label not in stats:
        return ""
    row = stats[label]
    if row.vpip_pct is None or row.pfr_pct is None:
        return ""
    return f" ({label}, VPIP {_half_up(row.vpip_pct)}%, PFR {_half_up(row.pfr_pct)}%)"


def _action_word(action: CanonicalAction, raise_ordinal: int) -> str:
    """Слово действия: рейзы на префлопе получают порядковое имя (опен, 3-бет)."""
    if action.is_all_in and action.kind in (ActionKind.BET, ActionKind.RAISE, ActionKind.CALL):
        return "олл-ин"
    if action.kind is ActionKind.RAISE and action.street is Street.PREFLOP:
        return _RERAISE_WORD.get(raise_ordinal, "рейз")
    return _ACTION_WORD[action.kind]


def _action_text(
    hand: CanonicalHand,
    action: CanonicalAction,
    raise_ordinal: int,
    stats: Mapping[str, PlayerStats] | None,
    marked: set[str],
) -> str:
    """Один ход: кто (позицией; герой — «вы»), что сделал и — у ставок — на сколько в ББ.

    Метка и частоты оппонента печатаются при его ПЕРВОМ ходе, дошедшем сюда, и
    больше не повторяются: `marked` копит уже помеченных
    (`test_an_opponent_carries_its_label_and_both_frequencies_once`). У героя
    метки нет — к нему обращаются «вы».
    """
    word = _action_word(action, raise_ordinal)
    if action.label == hand.hero_label:
        who = "вы"
    else:
        who = _position(hand, action.label)
        if action.label not in marked:
            who += _opponent_mark(action.label, stats)
            marked.add(action.label)
    show_amount = action.is_all_in or action.kind in _ACTIONS_WITH_AMOUNT
    if not show_amount:
        return f"{who} {word}"
    return f"{who} {word} {bb(action.committed_after, hand.bb)}"


def _street_flow(
    hand: CanonicalHand,
    actions: list[tuple[int, CanonicalAction]],
    hero_decisions: set[int],
    stats: Mapping[str, PlayerStats] | None,
    marked: set[str],
) -> list[ReplaySpan]:
    """Поток действий улицы: шаги через `→`, слипшиеся фолды, выделенный герой.

    Подряд идущие пасы сливаются в один шаг (`UTG/HJ фолд`) — они одинаковы по
    смыслу и занимают место, которого у сообщения нет. Пас героя в слипание не
    попадает: его решение обязано остаться видимым отдельно. Слипшийся пас не
    доходит до `_action_text`, а значит и метки оппонента не несёт — у
    пасующего сказать нечего (`test_a_folding_opponent_gets_no_label`).

    `marked` живёт выше по стеку (`hand_replay`), потому что метка ставится
    один раз на раздачу, а не один раз на улицу.
    """
    spans: list[ReplaySpan] = []
    folds: list[str] = []

    def step(text: str, *, emphasis: bool = False) -> None:
        if spans:
            spans.append(ReplaySpan(text=" → "))
        spans.append(ReplaySpan(text=text, emphasis=emphasis))

    def flush_folds() -> None:
        if folds:
            step(f"{'/'.join(folds)} фолд")
            folds.clear()

    raises_so_far = 0
    for index, action in actions:
        if action.kind is ActionKind.RAISE:
            raises_so_far += 1
        is_hero = action.label == hand.hero_label
        if action.kind is ActionKind.FOLD and not is_hero:
            folds.append(_position(hand, action.label))
            continue
        flush_folds()
        step(
            _action_text(hand, action, raises_so_far, stats, marked),
            emphasis=is_hero and index in hero_decisions,
        )
    flush_folds()
    return spans


def _hero_decision_indices(en: EnrichedHand) -> set[int]:
    """Номера действий героя, которые движок назвал точками решения.

    Сопоставление по порядку: `EngineReport.decision_points` перечисляет решения
    героя в том же порядке, в каком его действия стоят в руке, поэтому k-й точке
    отвечает k-е действие героя. Отдельного индекса действия в контракте нет, а
    заводить его ради выделения в тексте — менять контракт ради оформления.
    """
    hand = en.hand
    hero_action_indices = [
        i for i, action in enumerate(hand.actions) if action.label == hand.hero_label
    ]
    return {
        hero_action_indices[k]
        for k in range(min(len(hero_action_indices), len(en.report.decision_points)))
    }


def _street_actions(hand: CanonicalHand, street: Street) -> list[tuple[int, CanonicalAction]]:
    """Действия улицы вместе с их номерами В РУКЕ — номер и есть ключ, по которому
    поток узнаёт точку решения героя (`_hero_decision_indices`)."""
    return [(i, action) for i, action in enumerate(hand.actions) if action.street is street]


def _dead_before_deal(hand: CanonicalHand) -> int:
    """Банк до первого хода: блайнды и анте, как их записал источник.

    Число не из отчёта движка — и взято оно не как оценка, а как сумма записанных
    постов: то, что лежит в банке ДО первого действия, движок отдельной величиной
    не публикует (`pot_by_street` — итог улицы). Второе такое число блока — цена
    решения, и та приходит аргументом.
    """
    return sum(post.amount for post in hand.posts)


def _showdown_line(hand: CanonicalHand) -> str | None:
    """Строка вскрытия: кто что показал. Комбинации не называются (см. докстринг)."""
    if not hand.showdowns:
        return None
    shown = [
        f"{'вы' if entry.label == hand.hero_label else _position(hand, entry.label)} "
        f"{_cards(entry.cards)}"
        for entry in hand.showdowns
        if entry.cards
    ]
    return f"Вскрытие: {' vs '.join(shown)}" if shown else None


def hand_replay(
    en: EnrichedHand,
    *,
    ev_loss_bb: float | None = None,
    stats: Mapping[str, PlayerStats] | None = None,
) -> HandReplay:
    """Реплей одной руки: шапка строкой, ход раздачи прозой одним абзацем.

    `ev_loss_bb` — `AnalysisResult.total_ev_loss_bb`; `None` значит «судимых
    точек нет», и фразы о потере не будет: нуля расчёт не выносил
    (`test_the_replay_without_a_cost_says_nothing_about_it`). Ноль при
    непустом `ranked` — измеренный и печатается.

    `stats` — статистика мест по ключу `PlayerState.label` (тот же ключ, что у
    `analysis.player_stats.player_stats_by_label`); её добывает вызывающий, и
    правило зависимостей не нарушается — `explanation` не знает `memory`. У
    оппонента, чья метка в словаре есть, при первом его ходе печатается скобка
    с меткой и парой префлоп-частот
    (`test_an_opponent_carries_its_label_and_both_frequencies_once`).

    Улица без ходов — прогон борда после олл-ина: печатается только борд, через
    ` · ` с соседними такими же (`test_a_run_out_street_prints_only_its_board`).
    Чек — действие движка и печатается как ход, «чек-чек» здесь не выдумывается.
    """
    hand = en.hand
    hero = _hero(hand)
    spans: list[ReplaySpan] = []

    hero_cards = hand.dealt.get(hand.hero_label, [])
    cards_part = f", {_cards(hero_cards)}" if hero_cards else ""
    spans.append(
        ReplaySpan(text=f"Вы на {hero.position}{cards_part}, {bb(hero.stack, hand.bb)} ББ.\n")
    )

    decisions = _hero_decision_indices(en)
    marked: set[str] = set()
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
            head = f"{_STREET_TITLE[street]} {_board(board)}, банк {bb(pot_before, hand.bb)}"
            spans.append(ReplaySpan(text=f"{sep()}{head}. "))
        else:
            spans.append(ReplaySpan(text=sep()))
        flow = _street_flow(hand, actions, decisions, stats, marked)
        first_step = flow[0]
        flow[0] = first_step.model_copy(
            update={"text": first_step.text[:1].upper() + first_step.text[1:]}
        )
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
