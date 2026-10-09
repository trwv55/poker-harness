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
  живёт строкой ниже — в разборе точки (`presentation.hand_analysis_msgs`). Блок
  заканчивается ИСХОДОМ раздачи в фишках (`_outcome_line`). Судить решение блок
  не берётся вовсе, поэтому правило «против диапазона, а не против вскрытой
  карты» он не нарушает: он говорит только, что было. Точка решения героя при
  этом выделена (`spans`), то есть найти её в потоке можно без единого числа
  отсюда.
* Ни одного числа, которого нет в руке: пришпилено
  `test_the_replay_prints_no_number_the_hand_does_not_contain`. Одна величина
  приходит аргументом и потому этому запрету не противоречит — префлоп-частоты
  оппонента (`stats`): они посчитаны снаружи, реплей их только печатает.

**Форма — строка на улицу** (решение владельца 2026-10-07: абзац одной строкой
не читался). Сверху шапка раздачи — номер, уровень, блайнды, анте, сколько
игроков; под ней строка героя (позиция, карты, стек), затем по строке на улицу
с банком в скобках после борда, конечный банк, деление банка, если оно было, и
последней — вскрытие с исходом раздачи (исход отдельной строкой, если героя
на вскрытии не было)
(`test_the_block_has_the_owners_layout_line_by_line`,
`test_every_street_is_its_own_line`).

**Почему `HandReplay`, а не голая строка.** Спека требует выделить точку решения
героя жирным ПРЯМО в потоке действий, а разметка Телеграма — дело
`presentation`, не изложения (правило «`explanation` — покерное содержание,
`presentation` — вид в Телеграме»). Поэтому реплей возвращает последовательность
кусков с флагом выделения, а во что превратится выделение — решает тот, кто
собирает сообщение. `plain` — тот же текст без разметки, для тестов, отчётов и
любого канала, где выделять нечем.

**Масти — символом, никогда буквами** (спека §5.6: от скорости чтения
одномастности зависит оценка дро, и `Qs 8s` её не даёт). Селектора
эмодзи-презентации U+FE0F за символом нет (решение владельца 2026-10-07: «♥», а
не «♥️»); каким цветом клиент нарисует голый символ, решает его шрифт, и
проверить это отсюда нельзя. Тест пришпиливает то, что в нашей власти: букв
нет, селектора нет (`test_suits_are_symbols_without_the_emoji_selector`).
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

_SUIT_SYMBOL: dict[str, str] = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}

# Слово игрока для типа анте. Анте «с каждого» — обычное для MTT и в шапке не
# называется (решение владельца 2026-10-07: «анте 30 фишек»); тип, которого в
# словаре нет, печатается как есть в скобках — выдумать вместо него нечего, а
# спрятать нельзя: анте большого блайнда той же суммой — другая раздача.
_ANTE_TYPE_WORD: dict[str, str] = {"per_player": ""}

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
    """Первая буква строки — заглавная.

    Один способ на весь модуль: так поднимается и самостоятельная строка показа
    карт («Вы показали J♥»), и исход раздачи, вставший своей строкой
    («Отдаёте 2.6 ББ.») —
    `test_a_show_without_a_showdown_is_not_called_a_showdown`,
    `test_a_lost_hand_ends_with_the_chips_that_left_the_stack`. Ход улицы
    стоит после двоеточия и не поднимается: «Флоп … (банк 5.3): вы чек».
    """
    return text[:1].upper() + text[1:]


def _card(card: str) -> str:
    return f"{card[0]}{_SUIT_SYMBOL.get(card[1], card[1])}"


def _cards(cards: list[str]) -> str:
    """Карманные карты подряд, без пробелов: `J♥9♥` — так одномастность видна разом."""
    return "".join(_card(card) for card in cards)


def _board(cards: list[str]) -> str:
    """Борд — через пробел (`6♠ J♦ Q♦`): это три отдельные карты, а не рука."""
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


def _stack_bb(value_chips: int, big_blind: int) -> str:
    """Стек оппонента в ББ: тот же `bb`, но ровное число без «.0» — «100», не
    «100.0» (форма владельца 2026-10-07: «UTG+1 (100ББ)»;
    `test_an_opponent_who_stays_in_carries_his_starting_stack`)."""
    text = bb(value_chips, big_blind)
    return text.removesuffix(".0")


def _opponent_mark(
    hand: CanonicalHand, label: str, stats: Mapping[str, PlayerStats] | None
) -> str:
    """` (100ББ)` или ` (100ББ · P5: VPIP 25%, PFR 18%, раздач 40)`.

    Стек — стартовый, до раздачи (`PlayerState.stack`), и стоит всегда: в
    скобку попадает только тот, кто остался в раздаче отдельным шагом.

    Частоты — только при ненулевой выборке: `vpip_pct`/`pfr_pct` возвращают
    `None` при `hands == 0`, и «VPIP 0%» по нулю раздач никто не измерял
    (`test_an_opponent_without_a_sample_carries_only_his_stack`). Знаменатель
    у них общий, поэтому либо обе, либо ни одной. Число раздач печатается
    рядом: доля без знаменателя не говорит, чего она стоит. Что выборка не
    содержит САМУ разбираемую руку, обеспечивает вызывающий
    (`worker.pipeline._tournament_stats`).
    """
    player = next((p for p in hand.players if p.label == label), None)
    stack = "" if player is None else f"{_stack_bb(player.stack, hand.bb)}ББ"
    row = None if stats is None else stats.get(label)
    freq = ""
    if row is not None and row.vpip_pct is not None and row.pfr_pct is not None:
        freq = (
            f"{label}: VPIP {_half_up(row.vpip_pct)}%, PFR {_half_up(row.pfr_pct)}%, "
            f"раздач {row.hands}"
        )
    inside = " · ".join(part for part in (stack, freq) if part)
    return f" ({inside})" if inside else ""


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

    `mark` — готовая скобка со стеком и, если есть выборка, частотами оппонента
    (`_opponent_mark`) либо пустая строка; КОМУ и когда она достаётся, решает `_street_flow`:
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

    Здесь же решается, кому достанется скобка со стеком (и частотами): её
    получает ПЕРВЫЙ ход оппонента, ставший отдельным шагом (`marked` копит уже
    помеченных, `test_an_opponent_carries_its_label_and_both_frequencies_once`).
    Правило стоит рядом со слипанием не случайно — оно от него и зависит:
    слипшийся пас отдельным шагом не становится и скобки не несёт, у пасующего
    сказать нечего (`test_a_folding_opponent_gets_no_label`). У героя скобки
    нет — к нему обращаются «вы».

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
            mark = _opponent_mark(hand, action.label, stats)
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

    **У героя показ — не пометка, а фраза: «вы показали J♥»** (решение владельца
    2026-09-12). Скобка третьего лица у обращения во втором («вы J♥ (игрок
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
    """Исход раздачи в фишках, в ББ её уровня: «отдаёте 13.6 ББ» строчными и без
    точки — куда его поставить (за вскрытием через ` · ` или своей строкой),
    решает `hand_replay`.

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
        return f"отдаёте {abs(delta_bb):.1f} ББ"
    hand = en.hand
    pot = sum(entry.amount for entry in hand.collected if entry.label == hand.hero_label)
    taken = bb(pot, hand.bb)
    return None if taken == "0.0" else f"забираете {taken} ББ"


def _header(hand: CanonicalHand) -> str:
    """Шапка раздачи двумя строками: номер, уровень, блайнды, анте; число игроков.

    Блайнды без слова «фишек», анте — с ним и без «(с каждого)» (форма
    владельца 2026-10-07, `test_the_block_has_the_owners_layout_line_by_line`).
    """
    if hand.ante:
        kind = _ANTE_TYPE_WORD.get(hand.ante_type, hand.ante_type)
        ante = f"{hand.ante} фишек" + (f" ({kind})" if kind else "")
    else:
        ante = "нет"
    return (
        f"Рука {hand.hand_no} · уровень {hand.level} · "
        f"блайнды {hand.sb}/{hand.bb} · анте {ante}\n"
        f"Игроков в раздаче: {len(hand.players)}."
    )


def _side_pots_line(en: EnrichedHand) -> str | None:
    """«Банк делится на части: …» — или ничего, если банк не делился.

    Движок кладёт в `side_pots` ВСЕ поты PokerKit, включая главный, поэтому
    один элемент означает неделёный банк и печатать его нечем
    (`test_the_hand_with_one_pot_says_nothing_about_side_pots`). Претенденты
    названы так же, как во всём блоке: позицией, герой — «вы»
    (`test_a_hand_with_two_pots_names_each_part_and_who_claims_it`).
    """
    hand = en.hand
    pots = en.report.side_pots
    if len(pots) <= 1:
        return None

    def who(label: str) -> str:
        return "вы" if label == hand.hero_label else _position(hand, label)

    parts = " · ".join(
        f"{bb(pot.amount, hand.bb)} ББ (претендуют: {', '.join(who(lb) for lb in pot.eligible)})"
        for pot in pots
    )
    return f"Банк делится на части: {parts}."


def hand_replay(
    en: EnrichedHand,
    *,
    stats: Mapping[str, PlayerStats] | None = None,
) -> HandReplay:
    """Блок «Что было» одной руки: шапка, строка героя, по строке на улицу, итог.

    Итог — конечный банк (`EngineReport.final_pot`), деление банка, если оно
    было, и последней строкой вскрытие с исходом раздачи через ` · `
    («Вскрытие: … · отдаёте 13.6 ББ») — если герой на вскрытии был; иначе, как и
    без вскрытия, исход стоит своей строкой
    (`test_a_lost_hand_ends_with_the_chips_that_left_the_stack`,
    `test_a_showdown_without_the_hero_does_not_carry_his_outcome`). Цены решения
    в блоке нет: она живёт ниже, в разборе точки.

    `stats` — статистика мест по ключу `PlayerState.label` (тот же ключ, что у
    `analysis.player_stats.player_stats_by_label`); её добывает вызывающий, и
    правило зависимостей не нарушается — `explanation` не знает `memory`.
    Оппонент, ставший отдельным шагом, при первом ходе получает скобку со
    стартовым стеком, а при ненулевой выборке — и с меткой и парой
    префлоп-частот (`_opponent_mark`).

    Улица без ходов — прогон борда после олл-ина: печатается только борд, своей
    строкой (`test_a_run_out_street_prints_only_its_board`). Чек — действие
    движка и печатается как ход, «чек-чек» здесь не выдумывается.
    """
    hand = en.hand
    hero = _hero(hand)
    spans: list[ReplaySpan] = [ReplaySpan(text=f"{_header(hand)}\n\n")]

    hero_cards = hand.dealt.get(hand.hero_label, [])
    cards_part = f", {_cards(hero_cards)}" if hero_cards else ""
    spans.append(
        ReplaySpan(text=f"Вы на {hero.position}{cards_part}, {bb(hero.stack, hand.bb)} ББ.")
    )

    decisions = _hero_decision_indices(en)
    marked: set[str] = set()
    pot_before = _dead_before_deal(hand)

    for street in Street:
        actions = _street_actions(hand, street)
        board = hand.boards.get(street, [])
        if not actions:
            if street is not Street.PREFLOP and board:
                spans.append(ReplaySpan(text=f"\n{_STREET_TITLE[street]} {_board(board)}"))
            continue
        if street is Street.PREFLOP:
            head = f"{_STREET_TITLE[street]}: "
        else:
            head = (
                f"{_STREET_TITLE[street]} {_board(board)} "
                f"(банк {bb(pot_before, hand.bb)}): "
            )
        spans.append(ReplaySpan(text=f"\n{head}"))
        spans.extend(_street_flow(hand, actions, decisions, stats, marked))
        pot_before = en.report.pot_by_street.get(street, pot_before)

    spans.append(ReplaySpan(text=f"\nКонечный банк: {bb(en.report.final_pot, hand.bb)} ББ."))
    side_pots = _side_pots_line(en)
    if side_pots is not None:
        spans.append(ReplaySpan(text=f"\n{side_pots}"))

    showdown = _showdown_line(hand)
    outcome = _outcome_line(en)
    # Исход клеится к вскрытию, только если на нём был сам герой: иначе «Вскрытие:
    # SB … vs BB … · отдаёте 0.1 ББ» читалось бы как проигрыш на вскрытии, а это
    # анте спасовавшего (`test_a_showdown_without_the_hero_does_not_carry_his_outcome`).
    hero_at_showdown = went_to_showdown(hand, hand.hero_label)
    glued = showdown is not None and outcome is not None and hero_at_showdown
    if showdown is not None:
        spans.append(ReplaySpan(text=f"\n{showdown}" + (f" · {outcome}" if glued else "")))
    if outcome is not None and not glued:
        spans.append(ReplaySpan(text=f"\n{_sentence_start(outcome)}."))
    return HandReplay(spans=spans)
