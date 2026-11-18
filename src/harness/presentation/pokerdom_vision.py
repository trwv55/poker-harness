"""Complete local transcript, independent of analysis and model prose."""
from harness.contracts.pokerdom_vision import (
    PokerdomEvent,
    PokerdomPlayer,
    PokerdomReading,
    PokerdomResult,
)
from harness.contracts.raw import Street
from harness.presentation.messages import Msg


def pokerdom_report(result: PokerdomResult) -> str:
    reading = result.reading
    lines = [f"Чтение: {result.read_status}; реплей: {result.replay_status}; эталон: {result.truth_status}"]
    positions = ({p.label: p.position for p in result.enriched.hand.players}
                 if result.enriched else {})
    labels = ({p.seat: p.label for p in result.raw.seats} if result.raw else {})
    for i, p in enumerate(reading.players, 1):
        position = positions.get(labels.get(i, ""), "позиция не определена")
        row_id = p.row_id if isinstance(p, PokerdomPlayer) else f"P{i}"
        lines.append(f"{row_id} {p.nickname} · {position} · старт до взносов: "
                     f"{p.stack_shown if p.stack_shown is not None else 'неизвестен'} · "
                     + (" ".join(p.cards) if p.cards else "карты неизвестны"))
    actions = ({int(a.raw_line.split()[0][1:]): a for a in result.raw.actions}
               if result.raw else {})
    for street in Street:
        events = sorted((e for e in reading.events if e.street == street), key=lambda e: e.number)
        if not events:
            continue
        lines.append(f"{street.value} · борд: {' '.join(reading.board.get(street, [])) or 'нет'}")
        for e in events:
            amount = f" {e.amount}" if e.amount is not None else ""
            detail = " (возврат)" if e.token == "UB" else ""
            if e.token == "R" and e.number in actions:
                detail = f" → рейз до {actions[e.number].to_amount} (доплата{amount})"
            actor = e.player if isinstance(e, PokerdomEvent) else e.player_nickname
            lines.append(f"#{e.number} {actor} {e.token}{amount}{detail}")
        pot = reading.pot_after_street.get(street)
        lines.append(f"Банк после улицы, после возврата (прочитано): {pot if pot is not None else 'неизвестен'}")
    if result.enriched and result.replay_status == "passed" and result.outcome_confirmed:
        lines.append("Рассчитанные выплаты: " + ", ".join(
            f"{label} {amount}" for label, amount in result.calculated_payouts.items()))
        lines.append(f"Рассчитанный итоговый банк: {result.enriched.report.final_pot}")
        lines.append("Рассчитанные конечные стеки: " + ", ".join(
            f"{label} {stack}" for label, stack in result.enriched.report.stacks_end.items()))
    elif result.enriched:
        lines.append("Исход и конечные стеки не подтверждены.")
        if result.replay_status == "passed" and result.outcome_basis == "visible_cards":
            lines.append("Условный расчёт по видимым картам; вскрытие не подтверждено:")
            lines.append("Выплаты: " + ", ".join(
                f"{label} {amount}" for label, amount in result.calculated_payouts.items()))
            lines.append("Конечные стеки при этих картах: " + ", ".join(
                f"{label} {stack}" for label, stack in result.enriched.report.stacks_end.items()))
    if not isinstance(reading, PokerdomReading):
        lines.extend(f"Примечание: {note}" for note in reading.notes)
        lines.extend(f"Неопределённость {u.field}: {u.reason}" for u in reading.uncertainties)
    for check in result.checks:
        lines.append(f"Сверка {check.name}: {check.status}; прочитано={check.observed}; "
                     f"рассчитано={check.calculated}; {check.reason}")
    for comparison in result.amount_comparisons:
        if not comparison.matched:
            lines.append(f"Эталон #{comparison.number}: {comparison.actual} / {comparison.expected}; "
                         f"отклонение {comparison.absolute_error}; относительное {comparison.relative_error}; "
                         f"в допуске={comparison.within_tolerance}")
    lines.extend(f"Ошибка: {error}" for error in result.errors)
    lines.extend(f"Расхождение эталона: {error}" for error in result.truth_errors)
    return "\n".join(lines) + "\n"


def _chips(value: int) -> str:
    return f"{value:,}".replace(",", "\u2009")


def _cards(values: list[str]) -> str:
    suits = {"h": "♥", "d": "♦", "c": "♣", "s": "♠"}
    return "".join(c[:-1] + suits.get(c[-1:], c[-1:]) for c in values)


_STREET_NAMES = dict(zip(Street, ("Префлоп", "Флоп", "Тёрн", "Ривер"), strict=True))


def _friendly_problem(result: PokerdomResult) -> str:
    """Localize the blocking category; raw diagnostics remain in JSON/CLI."""
    if result.read_status == "not_hand":
        return "Не удалось распознать полную таблицу истории раздачи. Пришли таблицу с игроками, номерами действий и банками по улицам."
    if not result.reading.events:
        return "Не удалось прочитать последовательность действий. Пришли полную таблицу с номерами всех действий."
    for check in result.checks:
        if check.status == "failed" and check.name.startswith("pot_after."):
            street = check.name.removeprefix("pot_after.")
            title = next((name for value, name in _STREET_NAMES.items() if value.value == street), "Улица")
            if not isinstance(check.observed, int) or not isinstance(check.calculated, int):
                return "Банк не удалось сверить. Проверь банк и пришли полный читаемый скрин."
            return f"{title}: банк на скрине {_chips(check.observed)} не сошёлся с действиями ({_chips(check.calculated)}). Проверь эту улицу и пришли полный читаемый скрин."
    errors = " ".join(result.errors).lower()
    if "nickname" in errors or "hero" in errors:
        return "Не удалось однозначно определить игрока по нику. Проверь ник: /pdnick ТВОЙ_НИК, затем пришли скрин заново."
    if "card" in errors or "board" in errors:
        return "Не удалось прочитать карты без пропусков или противоречий. Пришли полный скрин, на котором видны карты и масти."
    if "банк" in errors or "pot" in errors:
        return "Не удалось прочитать банк сыгранной улицы. Пришли полный скрин с банками после каждой сыгранной улицы."
    if "stack" in errors:
        return "Не удалось однозначно прочитать стартовые стеки. Пришли скрин с читаемыми числами под никами."
    return "Не удалось полностью прочитать и проверить последовательность действий. Пришли полную читаемую таблицу без обрезки, лучше файлом без сжатия."


def _friendly_story(result: PokerdomResult) -> str:
    from decimal import ROUND_HALF_UP, Decimal

    reading, raw, enriched = result.reading, result.raw, result.enriched
    if result.read_status != "ok" or result.replay_status != "passed" or raw is None or enriched is None:
        return _friendly_problem(result)
    played = {e.street for e in reading.events}
    if any(reading.pot_after_street.get(street) is None for street in played):
        return "Не удалось прочитать банк сыгранной улицы. Пришли полный скрин с банками после каждой сыгранной улицы."
    players = {seat.label: player for seat, player in zip(raw.seats, reading.players, strict=True)}
    positions = {p.label: p.position for p in enriched.hand.players}
    if len(players) == 2:
        for post in raw.posts:
            if post.kind.value == "small_blind":
                positions[post.label] = "SB"  # heads-up button also posts SB
    hero = enriched.hand.hero_label
    hero_player = players[hero]
    # Participants who continued voluntarily, not every row that folded preflop.
    opponents = list(dict.fromkeys(a.label for a in raw.actions if a.label != hero and
                                   (a.kind.value != "fold" or a.street != Street.PREFLOP)))
    if not opponents:
        opponents = [a.label for a in raw.actions if a.label != hero and a.kind.value == "fold"][-1:]
    opponents = [seat.label for seat in raw.seats if seat.label in opponents]
    def actor(label: str) -> str:
        return "ты" if label == hero else positions[label]
    def hand(label: str) -> str:
        value = _cards(players[label].cards)
        return f"{actor(label)} с {value}" if value else f"{actor(label)} с неизвестными картами"
    def stack(label: str) -> str:
        value = players[label].stack_shown
        assert value is not None
        bb = str((Decimal(value) / raw.bb).quantize(Decimal(".01"), rounding=ROUND_HALF_UP)).replace(".", ",")
        return f"{_chips(value)} ({bb} ББ)"
    hero_cards = _cards(hero_player.cards)
    title = f"Рука с {hero_cards}." if hero_cards else "Рука с неизвестными картами."
    versus = " и ".join(hand(label) for label in opponents)
    header = f"Ты на {positions[hero]} с {hero_cards or 'неизвестными картами'}"
    if versus:
        header += " против " + versus
    paragraphs = [title + "\n" + header + ".\n" +
                  f"Игроков: {len(players)}. Блайнды {_chips(raw.sb)}/{_chips(raw.bb)}, анте {_chips(raw.ante)} с каждого.\n" +
                  f"Начальные стеки до взносов: у тебя {stack(hero)}" +
                  "".join(f", у {positions[label]} {stack(label)}" for label in opponents) + "."]
    actions = {int(a.raw_line.split()[0][1:]): a for a in raw.actions}
    for street in Street:
        if street not in played:
            # State only absence of action, not invented later cards/pots.
            paragraphs.append(f"{_STREET_NAMES[street]}: не сыгран.")
            continue
        moves = []
        for e in sorted((e for e in reading.events if e.street == street), key=lambda e: e.number):
            if e.token in {"Ante", "SB", "BB"}:
                continue  # already stated in the header
            action = actions.get(e.number)
            if e.token == "UB":
                key = e.player if isinstance(e, PokerdomEvent) else e.player_nickname
                label = next(seat.label for seat, player in zip(raw.seats, reading.players, strict=True)
                             if (player.row_id if isinstance(player, PokerdomPlayer) else player.nickname) == key)
                assert e.amount is not None
                moves.append(f"{'тебе' if label == hero else positions[label]} возвращается {_chips(e.amount)}")
                continue
            assert action is not None
            if e.token in {"R", "B", "C"}:
                assert e.amount is not None
            who = actor(action.label)
            if e.token == "R":
                assert action.to_amount is not None
                move = f"{'повышаешь' if action.label == hero else 'повышает'} до {_chips(action.to_amount)}"
            elif e.token == "B":
                assert e.amount is not None
                move = f"{'ставишь' if action.label == hero else 'ставит'} {_chips(e.amount)}"
            elif e.token == "C":
                assert e.amount is not None
                move = f"{'доплачиваешь' if action.label == hero else 'доплачивает'} {_chips(e.amount)}"
            elif e.token == "Check":
                move = "чек"
            else:
                move = "пасуешь" if action.label == hero else "пасует"
            moves.append(f"{who} {move}")
        board = _cards(reading.board.get(street, []))
        heading = _STREET_NAMES[street] + (f" {board}" if board else "")
        pot = reading.pot_after_street[street]
        assert pot is not None
        paragraphs.append(f"{heading}: " + ", ".join(moves) + f".\nБанк {_chips(pot)}.")
    final_pot = _chips(enriched.report.final_pot)
    if result.outcome_confirmed or result.outcome_basis == "visible_cards":
        end = enriched.report.stacks_end[hero]
        assert hero_player.stack_shown is not None
        net = end - hero_player.stack_shown
        signed = ("+" if net > 0 else "−" if net < 0 else "") + _chips(abs(net))
        prefix = "Результат по расчёту" if result.outcome_confirmed else "По видимым картам, условный результат"
        paragraphs.append(f"{prefix}: банк {final_pot}, твой чистый результат {signed}, конечный стек {_chips(end)}." +
                          ("" if result.outcome_confirmed else " Явное вскрытие не подтверждено."))
    else:
        paragraphs.append(f"Результат: банк {final_pot}. Победитель и конечный стек не определены: не хватает известных карт для расчёта.")
    return "\n\n".join(paragraphs)


def pokerdom_messages(result: PokerdomResult) -> list[Msg]:
    """Friendly TG projection only; detailed CLI/JSON diagnostics are unchanged."""
    text = _friendly_story(result)
    # Plain text prevents tag injection; UTF-16 chunks never drop source characters.
    chunks = [""]
    for part in text.splitlines(keepends=True):
        for character in part:
            if len((chunks[-1] + character).encode("utf-16-le")) // 2 > 3500:
                chunks.append("")
            chunks[-1] += character
    return [Msg(text=chunk) for chunk in chunks if chunk]
