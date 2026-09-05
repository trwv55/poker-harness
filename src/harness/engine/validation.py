"""Политика пригодности руки: вердикт по рапорту движка.

Валидатор читает готовый `EngineReport` и решает, годна ли рука для разбора.
Данные он **никогда не правит**: придумать «правильную» сумму значило бы
сгаллюцинировать про деньги игрока. Расхождение он только называет.

Как именно называет — решает провенанс. Скриншот — это гипотеза: vision мог
ошибиться, поэтому спорные поля возвращаются игроку вопросом (`escalate`).
Hand history — это факт, записанный самим румом: расхождение с ней означает
баг нашего парсера, а не ошибку игрока, поэтому рука уходит в лог
разработчику (`reject`), и игрока мы ни о чём не спрашиваем.
"""

from __future__ import annotations

from harness.contracts import (
    CanonicalHand,
    EngineReport,
    PlayerState,
    PostKind,
    Provenance,
    Street,
    ValidationStatus,
    Verdict,
)

_FIELD_STACKS = "stacks"
_FIELD_ACTIONS = "actions"
_FIELD_CARDS = "cards"
_FIELD_BUTTON = "button"

_QUESTIONS: dict[str, str] = {
    _FIELD_ACTIONS: "Ставки и повышения в раздаче распознаны верно?",
    _FIELD_CARDS: "Карты героя и карты на столе распознаны верно?",
    _FIELD_BUTTON: "Фишка дилера стоит на том игроке?",
}


def _cards_in_play(hand: CanonicalHand) -> list[str]:
    """Все карты руки: по одному экземпляру на игрока плюс борд.

    Карты игрока собираются в объединение — источник называет их дважды (раздача
    и вскрытие), и это не дубль. Дубль — это одна карта у разных игроков или у
    игрока и на столе.
    """
    by_label: dict[str, list[str]] = {}
    for label, cards in hand.dealt.items():
        by_label.setdefault(label, []).extend(cards)
    for entry in hand.showdowns:
        by_label.setdefault(entry.label, []).extend(entry.cards)

    cards = [card for street_cards in hand.boards.values() for card in street_cards]
    for player_cards in by_label.values():
        cards += dict.fromkeys(player_cards)
    return cards


def _has_duplicate_cards(hand: CanonicalHand) -> bool:
    cards = _cards_in_play(hand)
    return len(cards) != len(set(cards))


def forced_blind(hand: CanonicalHand, player: PlayerState, ante: int) -> int:
    """Вынужденная ставка игрока по его позиции, урезанная остатком стека.

    В хедз-апе малый блайнд ставит кнопка — позиции `SB` там просто нет.

    Публичная: этой же формулой аналитическое ядро восстанавливает посты в точке
    решения (`harness.analysis.classifier`). Считать блайнды двумя разными
    формулами нельзя — расхождение уехало бы прямо в цену решения.
    """
    heads_up = len(hand.players) == 2
    if player.position == "BB":
        blind = hand.bb
    elif player.position == "SB" or (heads_up and player.position == "BTN"):
        blind = hand.sb
    else:
        return 0
    return max(min(blind, player.stack - ante), 0)


def _blind_mismatch(hand: CanonicalHand) -> str | None:
    """Сверить блайнды, выведенные из кнопки, с теми, что записал источник.

    Два независимых факта об одной рассадке: позиции нормалайзер считает от места
    кнопки, а `posts` говорят, кто малый и большой блайнд поставил в реальности.
    Разошлись — кнопка прочитана не на том месте, и дальше неверна вся раскладка
    позиций, от которой зависят и диапазоны, и вердикт.

    На hand history проверка дублирует движок: чужая кнопка ломает порядок хода,
    и реплей упирается в «ход за не тем игроком» (измерено: 636 подмен кнопки на
    318 руках — 636 отказов). Ценность здесь в том, что причина названа, а не
    выведена из каскада денежных расхождений.

    **На скриншоте она единственная.** Руку в момент решения проиграть нельзя —
    истории улиц нет, упереться в порядок хода не во что. Остаются ровно два
    видимых на столе факта: где лежит фишка дилера и какие ставки стоят перед
    игроками. Их сверка и есть проверка — тот же приём, что банк как контрольная
    сумма.

    Молчит, когда сверять нечего: источник не записал блайнды (та же оговорка,
    что у сверки выплат — выдумывать расхождение из отсутствия данных нельзя).
    Отсюда требование к схеме извлечения: на живом столе блайнды нужны
    **по игрокам**, а не только уровень из шапки, иначе проверке нечего сравнивать.
    """
    actual: dict[PostKind, str] = {}
    for post in hand.posts:
        if post.kind in (PostKind.SMALL_BLIND, PostKind.BIG_BLIND):
            actual.setdefault(post.kind, post.label)  # первый пост своего вида
    if not actual:
        return None

    heads_up = len(hand.players) == 2
    expected: dict[PostKind, str] = {}
    for player in hand.players:
        # В хедз-апе малый блайнд ставит кнопка — позиции SB там нет.
        if player.position == "BB":
            expected[PostKind.BIG_BLIND] = player.label
        elif player.position == "SB" or (heads_up and player.position == "BTN"):
            expected[PostKind.SMALL_BLIND] = player.label

    wrong = [
        f"{kind.value}: по кнопке {expected.get(kind, '—')}, источник пишет {label}"
        for kind, label in sorted(actual.items())
        if expected.get(kind) != label
    ]
    return "blind mismatch: " + "; ".join(wrong) if wrong else None


def _source_stacks_end(hand: CanonicalHand) -> dict[str, int]:
    """Стеки на конец руки, посчитанные напрямую по строкам источника.

    Пересчёт независим от движка: из стартового стека вычитается всё вложенное
    (анте, вынужденная ставка, итоговые коммиты по улицам), обратно добавляются
    возвращённая непоколленная ставка и выигрыш. Оговорка про силу этой улики —
    в докстринге `validate`.
    """
    commits: dict[tuple[Street, str], int] = {}
    for action in hand.actions:
        commits[(action.street, action.label)] = action.committed_after  # последний — итоговый

    ends: dict[str, int] = {}
    for player in hand.players:
        ante = min(hand.ante, player.stack)
        # Игрок без записанных действий на префлопе всё равно поставил блайнд.
        contributed = ante + commits.get(
            (Street.PREFLOP, player.label), forced_blind(hand, player, ante)
        )
        for street in (Street.FLOP, Street.TURN, Street.RIVER):
            contributed += commits.get((street, player.label), 0)
        ends[player.label] = player.stack - contributed

    for entry in hand.uncalled:
        ends[entry.label] = ends.get(entry.label, 0) + entry.amount
    for entry in hand.collected:
        ends[entry.label] = ends.get(entry.label, 0) + entry.amount
    return ends


def _question_for(field: str, hand: CanonicalHand) -> str:
    """Короткий человеческий вопрос по полю — без покерного жаргона."""
    if field == _FIELD_STACKS:
        hero = next((p for p in hand.players if p.label == hand.hero_label), None)
        if hero is not None:
            return f"Стек героя {hero.stack_bb:.1f}bb — верно?"
        return "Стеки игроков распознаны верно?"
    return _QUESTIONS.get(field, f"Поле «{field}» распознано верно?")


def validate(hand: CanonicalHand, report: EngineReport) -> Verdict:
    """Вынести вердикт по руке: `pass` / `escalate` / `reject`.

    Сверка выигрышей (`payout mismatch`) намеренно считает стеки по строкам
    источника заново, а не спрашивает движок: правильный по размеру банк,
    уехавший не тому игроку, все остальные проверки проходят насквозь. Улика
    эта сильная, но не абсолютная — пересчёт независим от движка, но **не** от
    парсера: если парсер прочитал суммы неверно, обе стороны ошибутся
    одинаково и сверка промолчит.
    """
    reasons: list[str] = []
    fields: list[str] = []

    # Первой: неверная кнопка проявляется каскадом денежных расхождений ниже,
    # но причина у него одна, и назвать её должна отдельная проверка.
    blinds = _blind_mismatch(hand)
    if blinds is not None:
        reasons.append(blinds)
        fields.append(_FIELD_BUTTON)
    if report.illegal_actions:
        reasons.append(f"illegal: {report.illegal_actions}")
        fields.append(_FIELD_ACTIONS)
    if hand.summary is not None and report.final_pot != hand.summary.total_pot:
        reasons.append(
            f"pot mismatch: engine={report.final_pot} summary={hand.summary.total_pot}"
        )
        fields += [_FIELD_STACKS, _FIELD_ACTIONS]
    if _has_duplicate_cards(hand):
        reasons.append("duplicate cards")
        fields.append(_FIELD_CARDS)
    # Только если источник вообще пишет выплаты: у скриншота строк `collected`
    # нет, и выдумывать по их отсутствию расхождение нельзя.
    if hand.collected and report.stacks_end != _source_stacks_end(hand):
        reasons.append("payout mismatch: stacks_end vs collected/uncalled")
        fields.append(_FIELD_STACKS)

    # Реплей играет руку до рейка: PokerKit раздаёт банк целиком, ничего не
    # удерживая. Поэтому фишки обязаны сойтись ровно, без слагаемого за рейк —
    # добавить его значило бы отклонять любую руку с ненулевым рейком. Сам рейк
    # ловится сверкой `final_pot` с `Total pot`: рум пишет обе суммы до удержания.
    start = sum(p.stack for p in hand.players)
    if sum(report.stacks_end.values()) != start:
        reasons.append(
            f"chip conservation violated: start={start} end={sum(report.stacks_end.values())}"
        )
        fields.append(_FIELD_STACKS)

    if not reasons:
        return Verdict(status=ValidationStatus.PASS)
    if hand.provenance == Provenance.SCREENSHOT:
        asked = sorted(set(fields))
        return Verdict(
            status=ValidationStatus.ESCALATE,
            fields=asked,
            questions=[_question_for(field, hand) for field in asked],
            reasons=reasons,
        )
    # Hand history — факт рума: чинить надо парсер, а не данные и не игрока.
    return Verdict(status=ValidationStatus.REJECT, reasons=reasons)
