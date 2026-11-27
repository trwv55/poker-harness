"""Complete local transcript, independent of analysis and model prose."""
from dataclasses import dataclass

from harness.contracts.pokerdom_vision import (
    PokerdomEvent,
    PokerdomPlayer,
    PokerdomReading,
    PokerdomReadingV2,
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


def _amount(value: int, bb: int, *, signed: bool = False) -> str:
    """Display conversion only; preserve exact chips and round BB half up to .01."""
    from decimal import ROUND_HALF_UP, Decimal, localcontext

    from harness.presentation.messages import _plural_form

    if bb <= 0:
        raise ValueError("positive big blind required")
    sign = "−" if value < 0 else "+" if signed and value > 0 else ""
    with localcontext() as context:
        context.prec = max(28, len(str(abs(value))) + len(str(bb)) + 4)
        converted = (Decimal(abs(value)) / bb).quantize(Decimal(".01"), rounding=ROUND_HALF_UP)
    unit = _plural_form(value, "фишка", "фишки", "фишек")
    return f"{sign}{_chips(abs(value))} {unit} ({sign}{str(converted).replace('.', ',')} ББ)"


def _cards(values: list[str]) -> str:
    suits = {"h": "♥", "d": "♦", "c": "♣", "s": "♠"}
    return "".join(c[:-1] + suits.get(c[-1:], c[-1:]) for c in values)


_STREET_NAMES = dict(zip(Street, ("Префлоп", "Флоп", "Тёрн", "Ривер"), strict=True))


@dataclass(frozen=True)
class VisionFailure:
    stage: str
    code: str
    job_id: int
    error_type: str = "PokerdomResult"


_FAILURE_STAGES = {
    "intake": "приём изображения", "provider": "запрос к модели", "reading": "чтение таблицы",
    "replay": "проверка раздачи", "persistence": "сохранение результата",
    "analysis": "расчёт разбора", "delivery": "отправка ответа", "unknown": "обработка задачи",
}
_FAILURE_CODES = {
    "provider_unavailable": "служба распознавания недоступна; попробуйте позже",
    "schema_rejected": "ответ модели не прошёл проверку структуры",
    "provider_not_configured": "служба распознавания не настроена",
    "timeout": "операция не уложилась в отведённое время; попробуйте позже",
    "source_unavailable": "изображение или сохранённые данные недоступны",
    "persistence_failed": "не удалось сохранить проверенный результат",
    "analysis_failed": "рука сохранена, но расчёт разбора недоступен; попробуйте позже",
    "internal_error": "внутренняя ошибка обработки",
    "worker_interrupted": "обработка прервалась до завершения",
    "delivery_failed": "не удалось отправить ответ; попробуйте позже",
    "format_not_supported": "формат изображения не поддерживается",
    "missing_total_pot": "не видна строка Total Pot",
    "reading_incomplete": "чтение таблицы неполное",
    "replay_failed": "последовательность действий не прошла проверку",
    "bank_mismatch": "банк на скрине не совпал с расчётом действий",
}
_FAILURE_TYPES = {
    "PokerdomResult", "LLMProviderError", "LLMSchemaError", "LLMNotConfigured",
    "TimeoutError", "StorageError", "PersistenceError", "DeliveryError", "AnalysisError",
    "InternalError", "WorkerInterrupted",
}


def vision_failure_message(failure: VisionFailure, *, detail: str | None = None) -> Msg:
    stage = _FAILURE_STAGES.get(failure.stage, _FAILURE_STAGES["unknown"])
    code = failure.code if failure.code in _FAILURE_CODES else "internal_error"
    error_type = failure.error_type if failure.error_type in _FAILURE_TYPES else "InternalError"
    reason = (detail or _FAILURE_CODES[code]).rstrip(". ")
    text = (f"Этап: {stage}. Причина: {reason}. Тип: {error_type}. "
            f"Код: {code}. Код обращения: {failure.job_id}.")
    return Msg(text=text[:900], parse_mode=None)


def _friendly_problem(result: PokerdomResult) -> str:
    """Localize the blocking category; raw diagnostics remain in JSON/CLI."""
    if result.read_status == "not_hand":
        observation = getattr(result.reading, "format_observation", None)
        if observation is not None:
            if observation.kind == "live_table":
                return "Похоже, это игровой стол, а не история раздачи. Откройте историю и пришлите полный скрин таблицы."
            if observation.kind == "other_hh":
                return "Этот формат истории пока не поддерживается. Пришлите скрин таблицы с игроками, улицами, номерами действий и Total Pot."
            if not observation.end_street_pots and observation.player_rows and observation.street_columns and observation.numbered_actions and observation.known_tokens and observation.board_area:
                return "На скрине не видна строка Total Pot. Пришлите полный скрин с нижней частью таблицы."
            if not observation.end_street_pots:
                return "Не видна часть таблицы истории. Пришлите полный скрин с игроками, действиями и Total Pot."
            if not observation.player_rows:
                return "Не видны строки игроков. Пришлите полный скрин таблицы истории."
            if not observation.numbered_actions or not observation.street_columns:
                return "Не видны улицы или номера действий. Пришлите полный скрин таблицы истории."
        return "Не удалось определить формат скрина. Пришлите полный скрин таблицы истории с игроками, улицами и Total Pot."
    if not result.reading.events:
        return "Не удалось прочитать последовательность действий. Пришли полную таблицу с номерами всех действий."
    for check in result.checks:
        if check.status == "failed" and check.name.startswith("pot_after."):
            street = check.name.removeprefix("pot_after.")
            title = next((name for value, name in _STREET_NAMES.items() if value.value == street), "Улица")
            if (not isinstance(check.observed, int) or not isinstance(check.calculated, int)
                    or max(len(str(abs(check.observed))), len(str(abs(check.calculated)))) > 15):
                return "Банк не удалось сверить. Проверь банк и пришли полный читаемый скрин."
            return f"{title}: банк на скрине {_chips(check.observed)} не сошёлся с действиями ({_chips(check.calculated)}). Проверь эту улицу и пришли полный читаемый скрин."
    errors = " ".join(result.errors).lower()
    if any(term in errors for term in ("nickname", "hero", "player nickname")):
        return "Не удалось однозначно найти ваш ник на скрине. Проверьте его командой /pdnick актуальный_ник и пришлите скрин заново."
    if "card" in errors or "board" in errors:
        return "Не удалось прочитать карты без пропусков или противоречий. Пришли полный скрин, на котором видны карты и масти."
    if "банк" in errors or "pot" in errors:
        return "Не удалось прочитать банк сыгранной улицы. Пришли полный скрин с банками после каждой сыгранной улицы."
    if "stack" in errors:
        return "Не удалось однозначно прочитать стартовые стеки. Пришли скрин с читаемыми числами под никами."
    if isinstance(result.reading, PokerdomReadingV2) and result.reading.uncertainties:
        field = result.reading.uncertainties[0].field.lower()
        if "pot" in field:
            return "Не удалось прочитать банк сыгранной улицы. Пришлите полный скрин с банками после каждой сыгранной улицы."
        if "event" in field or "action" in field:
            return "Не удалось прочитать одно из действий. Пришлите полный скрин, чтобы вся таблица действий была видна."
        if "nick" in field or "player" in field:
            return "Не удалось прочитать ник игрока. Проверьте /pdnick и пришлите полный скрин таблицы."
        if "board" in field or "card" in field:
            return "Не удалось прочитать карты на скрине. Пришлите полный скрин, где видны борд и масти."
    if any(term in errors for term in ("terminal", "response", "closed", "all-in", "allin")):
        return "Не удалось восстановить завершение последовательности действий. Пришлите полный скрин, включая последнее действие."
    return "Не удалось полностью проверить раздачу. Пришлите полный читаемый скрин таблицы истории."


def _friendly_story(result: PokerdomResult) -> str:
    reading, raw, enriched = result.reading, result.raw, result.enriched
    if result.read_status != "ok" or result.replay_status != "passed" or raw is None or enriched is None:
        return _friendly_problem(result)
    if not isinstance(raw.bb, int) or raw.bb <= 0:
        return "Не удалось определить большой блайнд. Пришли полный читаемый скрин с блайндами."
    played = {e.street for e in reading.events} | {
        street for street, cards in reading.board.items() if cards}
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
        return _amount(value, raw.bb)
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
                moves.append(f"{'тебе' if label == hero else positions[label]} возвращается {_amount(e.amount, raw.bb)}")
                continue
            assert action is not None
            if e.token in {"R", "B", "C"}:
                assert e.amount is not None
            who = actor(action.label)
            if e.token == "R":
                assert action.to_amount is not None
                move = f"{'повышаешь' if action.label == hero else 'повышает'} до {_amount(action.to_amount, raw.bb)}"
            elif e.token == "B":
                assert e.amount is not None
                move = f"{'ставишь' if action.label == hero else 'ставит'} {_amount(e.amount, raw.bb)}"
            elif e.token == "C":
                assert e.amount is not None
                move = f"{'доплачиваешь' if action.label == hero else 'доплачивает'} {_amount(e.amount, raw.bb)}"
            elif e.token == "Check":
                move = "чек"
            else:
                move = "пасуешь" if action.label == hero else "пасует"
            moves.append(f"{who} {move}")
        board = _cards(reading.board.get(street, []))
        heading = _STREET_NAMES[street] + (f" {board}" if board else "")
        pot = reading.pot_after_street[street]
        assert pot is not None
        paragraphs.append(f"{heading}: " + (", ".join(moves) or "без дальнейших ставок") + f".\nБанк {_amount(pot, raw.bb)}.")
    final_pot = _amount(enriched.report.final_pot, raw.bb)
    if result.outcome_confirmed or result.outcome_basis == "visible_cards":
        end = enriched.report.stacks_end[hero]
        assert hero_player.stack_shown is not None
        net = end - hero_player.stack_shown
        signed = _amount(net, raw.bb, signed=True)
        prefix = "Результат по расчёту" if result.outcome_confirmed else "По видимым картам, условный результат"
        paragraphs.append(f"{prefix}: банк {final_pot}, твой чистый результат {signed}, конечный стек {_amount(end, raw.bb)}." +
                          ("" if result.outcome_confirmed else " Явное вскрытие не подтверждено."))
    else:
        paragraphs.append(f"Результат: банк {final_pot}. Победитель и конечный стек не определены: не хватает известных карт для расчёта.")
    return "\n\n".join(paragraphs)


def pokerdom_messages(result: PokerdomResult, *, job_id: int | None = None) -> list[Msg]:
    """Friendly TG projection only; detailed CLI/JSON diagnostics are unchanged."""
    text = _friendly_story(result)
    if (result.read_status != "ok" or result.replay_status != "passed") and job_id is not None:
        stage = "replay" if result.replay_status == "failed" else "reading"
        code = "replay_failed" if stage == "replay" else "reading_incomplete"
        if result.read_status == "not_hand":
            observation = getattr(result.reading, "format_observation", None)
            if observation is not None and observation.kind in {"live_table", "other_hh"}:
                code = "format_not_supported"
            elif (observation is not None and not observation.end_street_pots
                    and observation.player_rows and observation.street_columns
                    and observation.numbered_actions and observation.known_tokens
                    and observation.board_area):
                code = "missing_total_pot"
            else:
                code = "format_not_supported"
        elif any(check.status == "failed" and check.name.startswith("pot_after.") for check in result.checks):
            code = "bank_mismatch"
        return [vision_failure_message(VisionFailure(stage=stage, code=code, job_id=job_id), detail=text)]
    # Keep each chips+BB pair intact across Telegram messages. Preserve all text.
    import re
    tokens = re.findall(r"[+−]?[\d\u2009]+ фиш(?:ка|ки|ек) \([+−]?\d+,\d{2} ББ\)|\s+|\S+", text)
    chunks = [""]
    for token in tokens:
        if len((chunks[-1] + token).encode("utf-16-le")) // 2 > 3500:
            chunks.append("")
        chunks[-1] += token
    return [Msg(text=chunk) for chunk in chunks if chunk]
