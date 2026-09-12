"""Реплей руки прозой — скелет раздачи, собранный кодом (спека §5.6).

Третий выход изложения рядом с текстом вердикта (LLM) и матрицей диапазонов
(код). **Ноль токенов и ноль новых расчётов**: почти всё, что здесь печатается,
`EnrichedHand` уже содержит — действия по улицам в порядке хода, банк каждой
улицы, стеки, точки решения героя, исход раздачи; аргументом приходят только
префлоп-частоты оппонента. Модуль только форматирует.

**Зачем он есть.** Разбор без хода руки нечитаем: читатель не может ни
восстановить раздачу, ни проверить вывод. И следствие для второго выхода: раз
ход руки показан кодом, промпт вердикта не пересказывает раздачу — класс ошибок
«модель переврала ход руки» становится негде совершить.

**Границы, которые модуль себе ставит.**

* Комбинации на вскрытии не называются («пара валетов») и эквити не печатается:
  и то и другое — оценка руки, то есть расчёт, а расчёт живёт в `analysis`.
  Печатаются только карты, которые игроки действительно показали
  (`test_showdown_line_shows_the_cards_that_were_actually_shown`). Откуда взялась
  запись о показанных картах — вскрытие, добровольный показ спасовавшего или
  чтение карт героя с экрана — различает `_showdown_line`, и печатает их
  по-разному.
* Вердиктов словами здесь нет, и **цены решения тоже нет** (решение владельца
  2026-09-12, спека §5.6): она не движение фишек, а расчёт против диапазона, и
  живёт строкой ниже — в разборе точки (`presentation.deep_dive_msg`). Блок
  заканчивается ИСХОДОМ раздачи в фишках (`_outcome_line`). Судить решение блок
  не берётся вовсе, поэтому правило «против диапазона, а не против вскрытой
  карты» он не нарушает: он говорит только, что было. Точка решения героя при
  этом выделена (`spans`), то есть найти её в потоке можно без единого числа
  отсюда.
* Ни одного числа, которого нет в руке: пришпилено
  `test_the_replay_prints_no_number_the_hand_does_not_contain`. Одна величина
  приходит аргументом и потому этому запрету не противоречит — префлоп-частоты
  оппонента (`stats`): они посчитаны снаружи, реплей их только печатает.

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
    Provenance,
    Street,
    hero_stack_delta_bb,
    went_to_showdown,
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


def _sentence_start(text: str) -> str:
    """Первая буква предложения — заглавная.

    Один способ на весь модуль: так поднимается и первый шаг улицы («вы чек» →
    «Вы чек»), и самостоятельная строка показа карт
    (`test_a_street_sentence_starts_with_a_capital_even_when_it_is_you`,
    `test_a_show_without_a_showdown_is_not_called_a_showdown`). Обращение «вы»
    попадает в начало предложения в обоих местах, и разъехаться им негде.
    """
    return text[:1].upper() + text[1:]


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
    mark: str,
) -> str:
    """Один ход: кто (позицией; герой — «вы»), что сделал и — у ставок — на сколько в ББ.

    `mark` — готовая скобка с меткой и частотами оппонента (`_opponent_mark`)
    либо пустая строка; КОМУ и когда она достаётся, решает `_street_flow`:
    правило «при первом ходе, дошедшем до потока» неотделимо от слипания
    фолдов, которым владеет он. Здесь скобка только приписывается к тому, кто
    ходит.
    """
    word = _action_word(action, raise_ordinal)
    who = "вы" if action.label == hand.hero_label else _position(hand, action.label) + mark
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
    попадает: его решение обязано остаться видимым отдельно.

    Здесь же решается, кому достанется метка с частотами: её получает ПЕРВЫЙ
    ход оппонента, ставший отдельным шагом (`marked` копит уже помеченных,
    `test_an_opponent_carries_its_label_and_both_frequencies_once`). Правило
    стоит рядом со слипанием не случайно — оно от него и зависит: слипшийся пас
    отдельным шагом не становится и метки не несёт, у пасующего сказать нечего
    (`test_a_folding_opponent_gets_no_label`). У героя метки нет — к нему
    обращаются «вы».

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
        mark = ""
        if not is_hero and action.label not in marked:
            mark = _opponent_mark(action.label, stats)
            marked.add(action.label)
        step(
            _action_text(hand, action, raises_so_far, mark),
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
    не публикует (`pot_by_street` — итог улицы). Второе число БАНКА не от движка —
    банк, ушедший герою (`_outcome_line`): его пишет рум.
    """
    return sum(post.amount for post in hand.posts)


def _showdown_line(hand: CanonicalHand) -> str | None:
    """Строка вскрытия: кто что показал. Комбинации не называются (см. докстринг).

    Запись в `showdowns` бывает трёх происхождений, и печатаются они по-разному.
    Дошедшие до вскрытия (правило `contracts.went_to_showdown` — оно же считает
    долю вскрытий в статистике) стоят в ряд через ` vs `. Спасовавший, чью карту
    источник назвал сам — GG пишет добровольный показ отдельной строкой, — идёт
    после них с пометкой `(игрок показал)`: она принадлежит ИМЕННО ему, потому
    что `vs` между ним и вскрывшимися утверждало бы, что он с ними мерился
    (`test_a_card_shown_after_a_fold_is_marked_as_a_show`). Слово «Вскрытие»
    поэтому стоит только там, где вскрытие было: показ без вскрытия печатается
    сам по себе, с заглавной буквы (`_sentence_start`).

    **У героя показ — не пометка, а фраза: «вы показали J♥️»** (решение владельца
    2026-09-12). Скобка третьего лица у обращения во втором («вы J♥️ (игрок
    показал)») по-русски не читается, а блок говорит с игроком на «вы» везде
    (`test_a_show_without_a_showdown_is_not_called_a_showdown`,
    `test_a_hero_show_beside_a_real_showdown_stays_inside_the_line`).

    Третье происхождение — скриншот: карманные карты героя видны на экране
    ВСЕГДА, в том числе в раздаче, где он спасовал, и зрение честно записывает
    прочитанное. Карт спасовавшего соперника на экране не видно, поэтому запись
    о спасовавшем на скрине показом быть не может — она не печатается вовсе
    (`test_cards_of_a_folded_player_read_off_a_screenshot_are_not_printed`), а
    карты героя и так стоят в шапке блока. Чем платим за это правило и на какой
    базе оно измерено — спека §5.6.
    """
    seen: list[str] = []
    shown: list[str] = []
    for entry in hand.showdowns:
        if not entry.cards:
            continue
        is_hero = entry.label == hand.hero_label
        who = "вы" if is_hero else _position(hand, entry.label)
        cards = _cards(entry.cards)
        if went_to_showdown(hand, entry.label):
            seen.append(f"{who} {cards}")
        elif hand.provenance is not Provenance.SCREENSHOT:
            shown.append(f"вы показали {cards}" if is_hero else f"{who} {cards} (игрок показал)")
    if seen:
        tail = f"; {', '.join(shown)}" if shown else ""
        return f"Вскрытие: {' vs '.join(seen)}{tail}"
    return _sentence_start(", ".join(shown)) if shown else None


def _outcome_line(en: EnrichedHand) -> str | None:
    """Последняя фраза абзаца: исход раздачи в фишках, в ББ её уровня.

    Решение владельца 2026-09-12. Раньше абзац заканчивался ценой решения, и она
    обманывала: на реальной руке расхождение стоило 1.8 ББ по расчёту, а из
    стека ушло 0.1 ББ — одно анте. Блок «Что было» говорит о ФИШКАХ РАЗДАЧИ;
    цена решения осталась там, где живёт, — строкой ниже, в разборе точки.

    **Знак берёт изменение стека** (`contracts.hero_stack_delta_bb` — счёт
    движка), а не наличие записи `collected`: бывает раздача, где герой что-то
    собрал (сплит, сайд-пот), а стек всё равно уменьшился, и по записи выплаты
    блок сказал бы «Забираете» на проигранной раздаче
    (`test_the_sign_comes_from_the_stack_not_from_the_payout_record`). Цена
    рулинга: на сплите игрок увидит «Отдаёте» там, где часть банка он всё же
    взял.

    **Асимметрия названа вслух, потому что она есть.** При выигрыше печатается
    доля героя в банке ЦЕЛИКОМ — вместе с фишками, которые он положил в неё сам
    (на сплите и сайд-поте это его доля, а не банк стола); при
    проигрыше — чистая убыль стека. По скрину владельца это 2.9 ББ банка против
    2.3 ББ чистого прироста; в синтетике той же раздачи
    (`_folded_through_shove_hand`, структура постов там своя) — 2.9 против 1.8,
    и пришпилена тестом ПАРА, а не одно число
    (`test_a_won_hand_ends_with_the_whole_pot`): иначе асимметрия держалась бы
    на прозе. Владелец выбрал банк, увидев обе величины. Числа при
    этом приходят из РАЗНЫХ источников: банк — запись рума (`CanonicalHand.
    collected`), убыль — счёт движка. Доли героя в банке движок отдельной
    величиной не публикует, поэтому взять оба числа у него нельзя, не заводя
    нового поля в `EngineReport`.

    Стек не изменился (округляется до 0.0) — фразы нет вовсе: «Отдаёте 0.0 ББ»
    не событие раздачи. Её нет и там, где стек вырос, а записи о банке источник
    не дал: печатать под словом «Забираете» ноль значило бы назвать выигранную
    раздачу нулевой, а взять туда убыль по стеку — смешать два источника
    (CLAUDE.md: никогда не выдумывать числа о деньгах).

    **Непроверенный вход молчит целиком** (`Verdict.not_checked` непуст). Сейчас
    в этом списке бывает ровно одно — шоудаун, решённый на доукомплектованных
    картах (`engine.validation._fabricated_showdown`): на скрине карта соперника
    бывает не прочитана, движок добирает её из остатка колоды и разыгрывает
    вскрытие ею. Там, где банк решался этим вскрытием, стек героя на конец руки
    — его исход, а не факт раздачи, и «Отдаёте Z ББ» напечатало бы проигрыш,
    которого могло не быть: пришпилено
    `test_an_unverified_showdown_leaves_the_outcome_unsaid`.
    Правило то же, каким `worker.pipeline._hand_zone` понижает зону до
    «предполагая»: вход, часть которого проверить было нечем, точным числом не
    подаётся. Проверка стоит по НЕПУСТОМУ списку, а не по имени пометки: новая
    пометка в `not_checked` — тоже причина промолчать, пока не разобрано, что
    именно она ставит под сомнение.

    Цена этой ширины названа прямо: пометку ставит и раздача, где выдуманное
    вскрытие до стека героя не дотянулось вовсе — он спасовал на префлопе, а
    мерились двое соперников (`_fabricated_showdown` ищет непрочитанного среди
    НЕ спасовавших, и герой в этот счёт не входит). Там исход его раздачи —
    честное анте, и блок всё равно промолчит. Молчание не выдумка, а доверие
    понижено ко всей руке целиком, не к одной её строке.
    """
    if en.verdict.not_checked:
        return None
    delta_bb = hero_stack_delta_bb(en)
    if f"{abs(delta_bb):.1f}" == "0.0":
        return None
    if delta_bb < 0:
        return f"Отдаёте {abs(delta_bb):.1f} ББ."
    hand = en.hand
    pot = sum(entry.amount for entry in hand.collected if entry.label == hand.hero_label)
    taken = bb(pot, hand.bb)
    return None if taken == "0.0" else f"Забираете {taken} ББ."


def hand_replay(
    en: EnrichedHand,
    *,
    stats: Mapping[str, PlayerStats] | None = None,
) -> HandReplay:
    """Реплей одной руки: шапка строкой, ход раздачи прозой одним абзацем.

    Завершает абзац исход раздачи — сколько фишек герой забрал или отдал
    (`_outcome_line`). Цены решения в блоке нет: она живёт строкой ниже, в
    разборе точки.

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
        flow[0] = first_step.model_copy(update={"text": _sentence_start(first_step.text)})
        spans.extend(flow)
        spans.append(ReplaySpan(text="."))
        pot_before = en.report.pot_by_street.get(street, pot_before)
    flush_quiet()

    showdown = _showdown_line(hand)
    if showdown is not None:
        spans.append(ReplaySpan(text=f" {showdown}."))
    outcome = _outcome_line(en)
    if outcome is not None:
        spans.append(ReplaySpan(text=f" {outcome}"))
    return HandReplay(spans=spans)
