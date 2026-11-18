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


def pokerdom_messages(result: PokerdomResult) -> list[Msg]:
    """Telegram plain text: source observations and code-derived outcome kept apart."""
    from decimal import ROUND_HALF_UP, Decimal


    def cards(values: list[str]) -> str:
        suits = {'h': '♥', 'd': '♦', 'c': '♣', 's': '♠'}
        return ' '.join(c[:-1] + suits.get(c[-1:], c[-1:]) for c in values) or 'карты неизвестны'

    reading = result.reading
    bb = next((e.amount for e in reading.events if e.token == 'BB' and e.amount), None)
    sb = next((e.amount for e in reading.events if e.token == 'SB'), None)
    antes = [e.amount for e in reading.events if e.token == 'Ante']
    ante = str(antes[0]) + ' с каждого' if len(antes) == len(reading.players) and len(set(antes)) == 1 else 'не определено'
    lines = ['Таблица истории · проверка чтения',
             'Проверка пройдена по прочитанным данным.' if result.replay_status == 'passed'
             else 'Чтение не прошло полную проверку; ниже только транскрипция.',
             f'Игроков: {len(reading.players)}. Блайнды {sb if sb is not None else "?"}/{bb if bb else "?"}, анте {ante}. Фишки.']
    positions = {p.label: p.position for p in result.enriched.hand.players} if result.enriched else {}
    labels = {p.seat: p.label for p in result.raw.seats} if result.raw else {}
    names = {}
    for i, player in enumerate(reading.players, 1):
        label = labels.get(i, '')
        name = 'Вы' if label == 'Hero' else (player.nickname or f'строка {i}')
        names[label] = name
        stack = str(player.stack_shown) if player.stack_shown is not None else 'неизвестен'
        if bb and player.stack_shown is not None:
            stack += f' ({(Decimal(player.stack_shown) / bb).quantize(Decimal(".01"), rounding=ROUND_HALF_UP)}BB)'
        lines.append(f'{name} · {positions.get(label, "позиция не определена")} · {cards(player.cards)} · старт до взносов {stack}')
    actions = {int(a.raw_line.split()[0][1:]): a for a in result.raw.actions} if result.raw else {}
    words = {'Ante': 'анте', 'SB': 'малый блайнд', 'BB': 'большой блайнд', 'Fold': 'фолд',
             'Check': 'чек', 'C': 'колл, доплата', 'B': 'ставка', 'R': 'рейз, доплата', 'UB': 'возврат непокрытой ставки'}
    streets = dict(zip(Street, ('Префлоп', 'Флоп', 'Тёрн', 'Ривер'), strict=True))
    for street in Street:
        events = sorted((e for e in reading.events if e.street == street), key=lambda e: e.number)
        if not events:
            continue
        board = reading.board.get(street, [])
        lines.append(f'\n{streets[street]} · борд этой улицы: {cards(board) if board else "не указан"}')
        for e in events:
            actor = e.player if isinstance(e, PokerdomEvent) else e.player_nickname
            text = f'#{e.number} {actor or "неизвестный игрок"}: {words.get(e.token, e.token)}'
            if e.amount is not None:
                text += f' {e.amount}'
            if e.token == 'R' and e.number in actions:
                text += f' → до {actions[e.number].to_amount} (переведено кодом)'
            lines.append(text)
        pot = reading.pot_after_street.get(street)
        lines.append(f'Банк после улицы и возвратов (прочитан): {pot if pot is not None else "не указан; не сверен"}')
    if result.enriched and result.replay_status == 'passed':
        if result.outcome_confirmed or result.outcome_basis == 'visible_cards':
            lines.append('\nРасчёт кодом по пасам/явному вскрытию:' if result.outcome_confirmed else
                         '\nУсловный расчёт по видимым картам; явное вскрытие не подтверждено:')
            for label, amount in result.calculated_payouts.items():
                lines.append(f'Выплата {names.get(label, label)}: {amount}')
            for label, stack in result.enriched.report.stacks_end.items():
                lines.append(f'Конечный стек {names.get(label, label)}: {stack}')
        else:
            lines.append('\nИсход не подтверждён; победитель не определён.')
    lines.extend(f'Не проверено: {c.name} — {c.reason}' for c in result.checks if c.status == 'not_checked')
    lines.extend(f'Расхождение: {c.name}; прочитано {c.observed}, рассчитано {c.calculated}'
                 for c in result.checks if c.status == 'failed')
    lines.extend(f'Ошибка: {e}' for e in result.errors)
    if not isinstance(reading, PokerdomReading):
        lines.extend(f'Неопределённость {u.field}: {u.reason}' for u in reading.uncertainties)
        lines.extend(f'Примечание: {n}' for n in reading.notes)
    if result.read_status == 'not_hand':
        lines.append('Нужна полная таблица истории завершённой раздачи поддерживаемого вида, не live-стол или другой экран.')
    elif result.replay_status != 'passed':
        lines.append('Проверьте названные поля и пришлите полную читаемую таблицу файлом. Если ник неверен, сначала /pdnick ВАШ_ТОЧНЫЙ_НИК.')
    lines.append('Рука не записана в игровую историю. Это чтение и проверка, без стратегической оценки. Успешный реплей не доказывает точность чтения изображения.')
    # Plain text needs no HTML escaping, including hostile nickname/tag content.
    # UTF-16 units bound Telegram's limit even for astral characters; no data lost.
    chunks = ['']
    for character in '\n'.join(lines):
        if len((chunks[-1] + character).encode('utf-16-le')) // 2 > 3500:
            chunks.append('')
        chunks[-1] += character
    return [Msg(text=chunk) for chunk in chunks if chunk]
