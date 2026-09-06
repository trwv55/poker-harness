"""Реплей руки по улицам — скелет раздачи, собранный кодом (спека §5.6).

Третий выход изложения рядом с текстом вердикта (LLM) и матрицей диапазонов
(код). **Ноль токенов и ноль новых расчётов**: всё, что здесь печатается,
`EnrichedHand` уже содержит — действия по улицам в порядке хода, банк каждой
улицы, стеки, точки решения героя. Модуль только форматирует.

**Зачем он есть.** Разбор без хода руки нечитаем: читатель не может ни
восстановить раздачу, ни проверить вывод. И следствие для второго выхода: раз
ход руки показан кодом, промпт вердикта не пересказывает раздачу — класс ошибок
«модель переврала ход руки» становится негде совершить.

**Границы, которые модуль себе ставит.**

* Комбинации на вскрытии не называются («пара валетов») и эквити не печатается:
  и то и другое — оценка руки, то есть расчёт, а расчёт живёт в `analysis`.
  Печатаются только карты, которые игроки действительно показали
  (`test_showdown_line_shows_the_cards_that_were_actually_shown`).
* Вердиктов и цен в bb здесь нет: код показывает, ЧТО было, а чего с этим не
  так — соседние строки разбора (`presentation.deep_dive_msg`) и текст модели.
  Точка решения героя при этом выделена (`spans`), то есть найти её в потоке
  можно без единого числа отсюда.
* Ни одного числа, которого нет в руке: пришпилено
  `test_the_replay_prints_no_number_the_hand_does_not_contain`.

**Почему `HandReplay`, а не голая строка.** Спека требует выделить точку решения
героя жирным ПРЯМО в потоке действий, а разметка Телеграма — дело
`presentation`, не изложения (правило «`explanation` — покерное содержание,
`presentation` — вид в Телеграме»). Поэтому реплей возвращает последовательность
кусков с флагом выделения, а во что превратится выделение — решает тот, кто
собирает сообщение. `plain` — тот же текст без разметки, для тестов, отчётов и
любого канала, где выделять нечем.

**Масти — символ и цвет, никогда буквы.** Цвет в тексте Телеграма даёт только
эмодзи-презентация, поэтому за каждым символом масти стоит селектор U+FE0F:
♠️♣️ тёмные, ♥️♦️ красные (спека §5.6: от скорости чтения одномастности зависит
оценка дро, и `Qs 8s` её не даёт).
"""

from __future__ import annotations

from pydantic import BaseModel

from harness.contracts import (
    ActionKind,
    CanonicalAction,
    CanonicalHand,
    EnrichedHand,
    Street,
)

__all__ = ["HandReplay", "ReplaySpan", "hand_replay"]

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
    Street.PREFLOP: "ПРЕФЛОП",
    Street.FLOP: "ФЛОП",
    Street.TURN: "ТЁРН",
    Street.RIVER: "РИВЕР",
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

# Названия повторных рейзов на префлопе по порядковому номеру рейза улицы.
# Это счёт уже записанных действий, а не новая величина.
_RERAISE_WORD: dict[int, str] = {2: "3-бет", 3: "4-бет", 4: "5-бет"}

# Суммы, которые показываются: у ставки видно, СКОЛЬКО поставлено, у колла и
# паса показывать нечего — сумма колла равна названной до него (спека §5.6:
# «реплей это скелет раздачи, а не протокол»).
_ACTIONS_WITH_AMOUNT = frozenset({ActionKind.BET, ActionKind.RAISE})

# «Улица изменила банк существенно» (спека §5.6 — только тогда итоговый банк
# получает отдельную строку): банк вырос вдвое или больше. Порог, а не любое
# изменение, потому что строка стоит места: банк, подросший на один колл, читатель
# и так видит в следующем заголовке.
_MATERIAL_POT_GROWTH = 2


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


def _chips(amount: int) -> str:
    return f"{amount:,}".replace(",", _THIN)


def _bb(value: float) -> str:
    return f"{value:.1f}bb"


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


def _action_word(action: CanonicalAction, raise_ordinal: int) -> str:
    """Слово действия: рейзы на префлопе получают порядковое имя (3-бет, 4-бет)."""
    if action.is_all_in and action.kind in (ActionKind.BET, ActionKind.RAISE, ActionKind.CALL):
        return "олл-ин"
    if action.kind is ActionKind.RAISE and action.street is Street.PREFLOP:
        return _RERAISE_WORD.get(raise_ordinal, "рейз")
    return _ACTION_WORD[action.kind]


def _action_text(hand: CanonicalHand, action: CanonicalAction, raise_ordinal: int) -> str:
    """Один ход: кто (позицией, не ником), что сделал и — у ставок — на сколько.

    Размер в bb приписывается только действиям героя: спека разрешает bb «только
    у ключевых сумм», а ключевая сумма разбора — та, которую поставил он.
    """
    word = _action_word(action, raise_ordinal)
    who = "Hero" if action.label == hand.hero_label else _position(hand, action.label)
    show_amount = action.is_all_in or action.kind in _ACTIONS_WITH_AMOUNT
    if not show_amount:
        return f"{who} {word}"
    amount = _chips(action.committed_after)
    if action.label != hand.hero_label:
        return f"{who} {word} {amount}"
    depth = _bb(action.committed_after / hand.bb)
    return f"{who} {word} {amount} ({depth})"


def _street_flow(
    hand: CanonicalHand,
    actions: list[tuple[int, CanonicalAction]],
    hero_decisions: set[int],
) -> list[ReplaySpan]:
    """Поток действий улицы: шаги через `→`, слипшиеся фолды, выделенный герой.

    Подряд идущие пасы сливаются в один шаг (`UTG/HJ фолд`) — они одинаковы по
    смыслу и занимают место, которого у сообщения нет. Пас героя в слипание не
    попадает: его решение обязано остаться видимым отдельно.
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
            _action_text(hand, action, raises_so_far),
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

    Единственное число реплея не из отчёта движка — и взято оно не как оценка, а
    как сумма записанных постов: то, что лежит в банке ДО первого действия,
    движок отдельной величиной не публикует (`pot_by_street` — итог улицы).
    """
    return sum(post.amount for post in hand.posts)


def _last_street_with_actions(hand: CanonicalHand) -> Street:
    """Последняя улица, на которой кто-то ходил.

    Нужна ровно для одного: итоговый банк печатается отдельной строкой ТОЛЬКО
    после неё. На любой более ранней улице то же число уже стоит в заголовке
    следующей (`… · банк N`), и вторая его копия была бы протоколом, а не
    скелетом (`test_the_final_pot_is_printed_once_not_twice`).
    """
    streets = [action.street for action in hand.actions]
    return streets[-1] if streets else Street.PREFLOP


def _showdown_line(hand: CanonicalHand) -> str | None:
    """Строка вскрытия: кто что показал. Комбинации не называются (см. докстринг)."""
    if not hand.showdowns:
        return None
    shown = [
        f"{'Hero' if entry.label == hand.hero_label else _position(hand, entry.label)} "
        f"{_cards(entry.cards)}"
        for entry in hand.showdowns
        if entry.cards
    ]
    return f"Вскрытие: {' vs '.join(shown)}" if shown else None


def hand_replay(en: EnrichedHand) -> HandReplay:
    """Реплей одной руки: шапка в две строки, улицы, вскрытие.

    Улицы без действий вовсе схлопываются в одну строку с соседними такими же
    (`ТЁРН 7♥️ · РИВЕР A♥️`) — на них нечего разбирать, а место они занимают.
    Улица с действиями получает свою строку: борд, банк на входе, поток ходов.
    """
    hand = en.hand
    hero = _hero(hand)
    spans: list[ReplaySpan] = []

    def line(text: str) -> None:
        spans.append(ReplaySpan(text=text))

    def newline() -> None:
        spans.append(ReplaySpan(text="\n"))

    ante_total = sum(post.amount for post in hand.posts if post.kind.value == "ante")
    ante_part = f" анте {_chips(ante_total)}" if ante_total else ""
    line(
        f"{hand.tournament_id} · ур. {hand.level} · "
        f"{_chips(hand.sb)}/{_chips(hand.bb)}{ante_part}"
    )
    newline()
    hero_cards = hand.dealt.get(hand.hero_label, [])
    cards_part = f" {_cards(hero_cards)}" if hero_cards else ""
    line(
        f"Hero {hero.position}{cards_part} · {_chips(hero.stack)} ({_bb(hero.stack_bb)})"
    )

    decisions = _hero_decision_indices(en)
    pot_before = _dead_before_deal(hand)
    last_active = _last_street_with_actions(hand)
    quiet: list[str] = []
    for street in Street:
        actions = _street_actions(hand, street)
        board = hand.boards.get(street, [])
        title = _STREET_TITLE[street]
        head = f"{title} {_board(board)}" if board else title
        if not actions:
            if street is not Street.PREFLOP and not board:
                continue  # улицы не было вовсе
            quiet.append(head)
            continue
        if quiet:
            newline()
            line(" · ".join(quiet))
            quiet.clear()
        newline()
        newline()
        line(f"{head} · банк {_chips(pot_before)}")
        newline()
        spans.extend(_street_flow(hand, actions, decisions))
        pot_after = en.report.pot_by_street.get(street, pot_before)
        if street is last_active and pot_after >= pot_before * _MATERIAL_POT_GROWTH:
            newline()
            line(f"банк {_chips(pot_after)}")
        pot_before = pot_after

    if quiet:
        newline()
        line(" · ".join(quiet))

    showdown = _showdown_line(hand)
    if showdown is not None:
        newline()
        newline()
        line(showdown)

    return HandReplay(spans=spans)
