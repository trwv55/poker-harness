"""Парсер hand history формата GG (GGPoker/GGNetwork).

Единственное место в системе, которое знает синтаксис исходного текста.
Наружу отдаёт `RawHand` — буквальное, неинтерпретированное представление
руки: суммы действий — доплаты (как в источнике), позиции не вычислены,
банк не пересчитан. Всё, что парсер не смог распознать, попадает в
`unknown_lines` — это диагностика полноты покрытия формата, а не карман
для мусора.
"""

from __future__ import annotations

import re
from datetime import datetime

from harness.contracts import (
    ActionKind,
    Collected,
    Post,
    PostKind,
    Provenance,
    RawAction,
    RawHand,
    SeatInfo,
    ShowdownEntry,
    Street,
    SummaryInfo,
    Uncalled,
)

RE_HEADER = re.compile(
    r"^Poker Hand #(?P<no>\S+): Tournament #(?P<tid>\d+), (?P<name>.+) Hold'em No Limit - "
    r"Level(?P<lvl>\d+)\((?P<sb>[\d,]+)/(?P<bb>[\d,]+)\((?P<ante>[\d,]+)\)\) - "
    r"(?P<ts>\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})$"
)
RE_TABLE = re.compile(r"^Table '(?P<t>[^']+)' (?P<mx>\d+)-max Seat #(?P<btn>\d+) is the button$")
RE_SEAT = re.compile(r"^Seat (?P<s>\d+): (?P<l>\S+) \((?P<st>[\d,]+) in chips\)$")
RE_ANTE = re.compile(r"^(?P<l>\S+): posts the ante (?P<a>[\d,]+)$")
RE_BLIND = re.compile(r"^(?P<l>\S+): posts (?P<k>small|big) blind (?P<a>[\d,]+)$")
RE_DEALT = re.compile(r"^Dealt to (?P<l>\S+)(?: \[(?P<c>[^\]]+)\])?\s*$")
RE_ACTION = re.compile(
    r"^(?P<l>\S+): (?P<v>folds|checks|calls (?P<ca>[\d,]+)|bets (?P<ba>[\d,]+)|"
    r"raises (?P<ra>[\d,]+) to (?P<rt>[\d,]+))(?P<ai> and is all-in)?$"
)
RE_UNCALLED = re.compile(r"^Uncalled bet \((?P<a>[\d,]+)\) returned to (?P<l>\S+)$")
RE_SHOWS = re.compile(r"^(?P<l>\S+): shows \[(?P<c>[^\]]+)\]")
RE_COLLECTED = re.compile(r"^(?P<l>\S+) collected (?P<a>[\d,]+) from pot$")
RE_STREET = re.compile(
    r"^\*\*\* (?P<s>HOLE CARDS|FLOP|TURN|RIVER|SHOWDOWN|SUMMARY) \*\*\*"
    r"(?: \[(?P<b>[^\]]+)\])?(?: \[(?P<b2>[^\]]+)\])?$"
)
RE_TOTAL = re.compile(
    r"^Total pot (?P<tp>[\d,]+) \| Rake (?P<r>[\d,]+) \| Jackpot (?P<j>[\d,]+) \| "
    r"Bingo (?P<bi>[\d,]+) \| Fortune (?P<f>[\d,]+) \| Tax (?P<tx>[\d,]+)$"
)
RE_BOARD = re.compile(r"^Board \[(?P<b>[^\]]+)\]$")

# Строки "Seat N: ..." внутри *** SUMMARY *** — результаты по местам (folded
# before Flop / showed [..] and won (..) / collected (..) и т.п.). Отдельного
# разбора в v1 не требуют — сохраняются как есть в summary.seat_lines.
RE_SUMMARY_SEAT_LINE = re.compile(r"^Seat \d+: .+$")

_STREET_BY_MARKER: dict[str, Street] = {
    "HOLE CARDS": Street.PREFLOP,
    "FLOP": Street.FLOP,
    "TURN": Street.TURN,
    "RIVER": Street.RIVER,
}

_ACTION_KIND_BY_VERB = {
    "folds": ActionKind.FOLD,
    "checks": ActionKind.CHECK,
}


def _num(raw: str) -> int:
    """`85,440` -> `85440`."""
    return int(raw.replace(",", ""))


def parse_hand(block: str, source_ref: str) -> RawHand:
    """Разобрать один блок текста (одна рука) в `RawHand`."""
    table_name = ""
    max_seats = 0
    button_seat = 0
    seats: list[SeatInfo] = []
    posts: list[Post] = []
    dealt: dict[str, list[str]] = {}
    actions: list[RawAction] = []
    boards: dict[Street, list[str]] = {}
    uncalled: list[Uncalled] = []
    showdowns: list[ShowdownEntry] = []
    collected: list[Collected] = []
    unknown_lines: list[str] = []

    summary_total: re.Match[str] | None = None
    summary_board: list[str] = []
    summary_seat_lines: list[str] = []

    header: re.Match[str] | None = None
    current_street: Street = Street.PREFLOP
    in_summary = False

    for line in block.split("\n"):
        if line.strip() == "":
            continue  # пустая строка — разделитель, не unknown

        if in_summary and RE_SUMMARY_SEAT_LINE.match(line):
            summary_seat_lines.append(line)
            continue

        if m := RE_HEADER.match(line):
            header = m
            continue
        if m := RE_TABLE.match(line):
            table_name = m["t"]
            max_seats = int(m["mx"])
            button_seat = int(m["btn"])
            continue
        if m := RE_SEAT.match(line):
            seats.append(SeatInfo(seat=int(m["s"]), label=m["l"], stack=_num(m["st"])))
            continue
        if m := RE_ANTE.match(line):
            posts.append(Post(label=m["l"], kind=PostKind.ANTE, amount=_num(m["a"])))
            continue
        if m := RE_BLIND.match(line):
            kind = PostKind.SMALL_BLIND if m["k"] == "small" else PostKind.BIG_BLIND
            posts.append(Post(label=m["l"], kind=kind, amount=_num(m["a"])))
            continue
        if m := RE_DEALT.match(line):
            dealt[m["l"]] = m["c"].split() if m["c"] else []
            continue
        if m := RE_ACTION.match(line):
            verb = m["v"]
            if verb in _ACTION_KIND_BY_VERB:
                kind = _ACTION_KIND_BY_VERB[verb]
                amount = None
                to_amount = None
            elif m["ca"] is not None:
                kind, amount, to_amount = ActionKind.CALL, _num(m["ca"]), None
            elif m["ba"] is not None:
                kind, amount, to_amount = ActionKind.BET, _num(m["ba"]), None
            else:
                kind = ActionKind.RAISE
                amount = _num(m["ra"])
                to_amount = _num(m["rt"])
            actions.append(
                RawAction(
                    street=current_street,
                    label=m["l"],
                    kind=kind,
                    amount=amount,
                    to_amount=to_amount,
                    is_all_in=m["ai"] is not None,
                    raw_line=line,
                )
            )
            continue
        if m := RE_UNCALLED.match(line):
            uncalled.append(Uncalled(label=m["l"], amount=_num(m["a"])))
            continue
        if m := RE_SHOWS.match(line):
            showdowns.append(ShowdownEntry(label=m["l"], cards=m["c"].split()))
            continue
        if m := RE_COLLECTED.match(line):
            collected.append(Collected(label=m["l"], amount=_num(m["a"])))
            continue
        if m := RE_STREET.match(line):
            marker = m["s"]
            if marker == "SUMMARY":
                in_summary = True
            elif marker == "SHOWDOWN":
                pass  # не несёт своей улицы/доски
            else:
                current_street = _STREET_BY_MARKER[marker]
                if marker == "FLOP":
                    if m["b"]:
                        boards[Street.FLOP] = m["b"].split()
                else:  # TURN / RIVER: новая карта — последняя скобка
                    new_card = m["b2"] or m["b"]
                    if new_card:
                        boards[current_street] = new_card.split()
            continue
        if m := RE_TOTAL.match(line):
            summary_total = m
            continue
        if m := RE_BOARD.match(line):
            summary_board = m["b"].split()
            continue

        unknown_lines.append(line)

    if header is None:
        raise ValueError(f"не найдена строка заголовка Poker Hand # в блоке ({source_ref})")

    summary = None
    if summary_total is not None:
        m = summary_total
        summary = SummaryInfo(
            total_pot=_num(m["tp"]),
            rake=_num(m["r"]),
            jackpot=_num(m["j"]),
            bingo=_num(m["bi"]),
            fortune=_num(m["f"]),
            tax=_num(m["tx"]),
            board=summary_board,
            seat_lines=summary_seat_lines,
        )

    return RawHand(
        provenance=Provenance.HAND_HISTORY,
        source_ref=source_ref,
        hand_no=header["no"],
        tournament_id=header["tid"],
        tournament_name=header["name"],
        level=int(header["lvl"]),
        sb=_num(header["sb"]),
        bb=_num(header["bb"]),
        ante=_num(header["ante"]),
        # источник не несёт таймзоны — RawHand.timestamp намеренно naive
        timestamp=datetime.strptime(header["ts"], "%Y/%m/%d %H:%M:%S"),  # noqa: DTZ007
        table_name=table_name,
        max_seats=max_seats,
        button_seat=button_seat,
        seats=seats,
        posts=posts,
        dealt=dealt,
        actions=actions,
        boards=boards,
        uncalled=uncalled,
        showdowns=showdowns,
        collected=collected,
        summary=summary,
        unknown_lines=unknown_lines,
    )


def parse_file(text: str, source_ref: str) -> list[RawHand]:
    """Разобрать файл hand history: несколько рук, разделённых `Poker Hand #`.

    Файл GG хранит руки в обратном хронологическом порядке — на выходе они
    отсортированы по `timestamp` (при равенстве — по `hand_no`).
    """
    blocks = re.split(r"(?=^Poker Hand #)", text, flags=re.MULTILINE)
    hands = [parse_hand(block, source_ref) for block in blocks if block.strip()]
    hands.sort(key=lambda h: (h.timestamp, h.hand_no))
    return hands
