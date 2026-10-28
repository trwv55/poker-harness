"""Солверный ориентир для диапазона соперника на флопе (спека 2026-10-24, §6).

**Данные — библиотека выгрузок владельца**, закрытый каталог `DEFAULT_SOLVER_DIR` (в
`.gitignore` и в pre-push хуке, в образ попадает с `src`). Файл — одна выгрузка в формате
`solver_flop.json`: ключ спота (`seats`, `position` — отвечающий на ставку, `vs_position`
— ставящий, `depth_bb`, `ante_type`), `street`, `board`, `pot_bb`, размер ставки
(`bet_bb`, а без него — последнее «bet <x>bb» в `flop_line`), доли действий `call`,
`raise`, `fold`, `allin` записью чартов, `source`. Каталога нет или он пуст — ориентира
нет (`test_an_empty_library_gives_no_guide`); файл не читается — `SolverFileError`.

**Лукап — по ключу, а не по смыслу** (`nearest`): точный ключ спота и тип борда
(`texture`) и корзина размера ставки (`size_bucket`); нет — та же текстура в соседней
корзине; нет и её — ориентира нет. Из нескольких подходящих — ближайшая по размеру
ставки, при равенстве — первая по имени файла
(`test_the_guide_takes_the_exact_bucket_then_the_next_one_and_nothing_else`).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from harness.analysis.charts import ChartMissing, NotationError, parse_range
from harness.analysis.open_chart import defend_table
from harness.analysis.tools.hand_class import RANK_VALUE
from harness.contracts import ActionKind, CanonicalHand, Range, Street

__all__ = [
    "DEFAULT_SOLVER_DIR",
    "SolverDump",
    "SolverFileError",
    "SolverKey",
    "hand_bet",
    "load_library",
    "nearest",
    "size_bucket",
    "texture",
]

DEFAULT_SOLVER_DIR = Path(__file__).parent / "charts" / "solver"

ActionName = Literal["call", "raise", "fold", "allin"]
_ACTIONS: tuple[ActionName, ...] = ("call", "raise", "fold", "allin")
_BET_RE = re.compile(r"bet\s+(\d+(?:\.\d+)?)\s*bb", re.IGNORECASE)
_CARD_RE = re.compile(r"[2-9TJQKA][shdc]")
_FLOP = 3
_HEADS_UP = 2


class SolverFileError(ValueError):
    """Выгрузка библиотеки не прочиталась — причина с именем файла."""


class _RawDump(BaseModel):
    seats: int
    position: str
    vs_position: str
    depth_bb: float
    ante_type: str
    street: str
    board: str
    pot_bb: float
    bet_bb: float | None = None
    flop_line: str = ""
    call: str = ""
    raise_: str = ""
    fold: str = ""
    allin: str = ""
    source: str = ""

    model_config = {"populate_by_name": True}


@dataclass(frozen=True)
class SolverKey:
    seats: int
    position: str
    vs_position: str
    depth_bb: float
    ante_type: str


@dataclass(frozen=True)
class SolverDump:
    """Выгрузка: ключ спота, борд флопа, ставка в долях банка, доли действий."""

    name: str
    key: SolverKey
    board: tuple[str, ...]
    bet_share: float
    actions: dict[ActionName, Range]
    source: str


def _parse(path: Path) -> SolverDump:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "raise" in data:
            data = {**data, "raise_": data.pop("raise")}
        raw = _RawDump.model_validate(data)
    except (OSError, ValueError, ValidationError) as exc:
        raise SolverFileError(f"{path.name}: {exc}") from exc
    board = tuple(_CARD_RE.findall(raw.board))
    if raw.street != "flop" or len(board) != _FLOP or "".join(board) != raw.board.replace(" ", ""):
        raise SolverFileError(f"{path.name}: нужна выгрузка флопа с бордом из трёх карт")
    bet = raw.bet_bb
    if bet is None:
        found = _BET_RE.findall(raw.flop_line)
        if not found:
            raise SolverFileError(f"{path.name}: размера ставки нет ни в bet_bb, ни в flop_line")
        bet = float(found[-1])
    if raw.pot_bb <= 0 or bet <= 0:
        raise SolverFileError(f"{path.name}: банк и ставка должны быть больше нуля")
    texts = {"call": raw.call, "raise": raw.raise_, "fold": raw.fold, "allin": raw.allin}
    actions: dict[ActionName, Range] = {}
    for action in _ACTIONS:
        text = texts[action].strip()
        if not text:
            continue
        try:
            actions[action] = parse_range(text)
        except (NotationError, ValueError) as exc:
            raise SolverFileError(f"{path.name}: {action} не разобран: {exc}") from exc
    key = SolverKey(raw.seats, raw.position, raw.vs_position, raw.depth_bb, raw.ante_type)
    return SolverDump(path.name, key, board, bet / raw.pot_bb, actions, raw.source)


def load_library(directory: Path | None = None) -> list[SolverDump]:
    """Все выгрузки каталога (`*.json`, по имени файла); каталога нет — пусто."""
    root = directory or DEFAULT_SOLVER_DIR
    if not root.is_dir():
        return []
    return [_parse(path) for path in sorted(root.glob("*.json"))]


def texture(board: Sequence[str]) -> tuple[str, str, str, str]:
    """Тип флопа — ВРЕМЕННЫЙ классификатор v1 (спека 2026-10-24, §6, §11 п. 2: свой список
    даст владелец; замена — эта одна функция): спаренность, масти, связность, старшинство
    (`test_the_texture_v1_reads_pairing_suits_connectedness_and_height`).

    Связность — по различным рангам (туз — и младший): все три в окне стрита из пяти
    рангов — «связный», хотя бы два — «средний», иначе «сухой». Старшинство — по старшей
    карте: туз или король — «высокий», дама…девятка — «средний», ниже — «низкий»."""
    ranks = {card[0] for card in board}
    paired = "спаренный" if len(ranks) < len(board) else "неспаренный"
    suits = len({card[1] for card in board})
    suited = {1: "одна масть", 2: "две масти"}.get(suits, "радуга")
    values = {RANK_VALUE[rank] for rank in ranks}
    if RANK_VALUE["A"] in values:
        values |= {1}

    ordered = sorted(values)
    together = max(len([v for v in ordered if low <= v <= low + 4]) for low in ordered)
    if len(ranks) == _FLOP and together == _FLOP:
        connected = "связный"
    elif together >= 2:
        connected = "средний"
    else:
        connected = "сухой"
    top = max(RANK_VALUE[rank] for rank in ranks)
    if top >= RANK_VALUE["K"]:
        height = "высокий"
    elif top >= RANK_VALUE["9"]:
        height = "средний"
    else:
        height = "низкий"
    return paired, suited, connected, height


def size_bucket(bet_share: float) -> int:
    """Корзина размера ставки: 0 — меньше 40% банка, 1 — от 40% до 80% включительно,
    2 — больше 80% (`test_the_size_buckets_split_at_forty_and_eighty_percent`)."""
    if bet_share < 0.4:
        return 0
    if bet_share <= 0.8:
        return 1
    return 2


def nearest(
    library: Sequence[SolverDump], key: SolverKey, board: Sequence[str], bet_share: float
) -> SolverDump | None:
    """Выгрузка-ориентир: точный ключ и текстура, корзина — та же, иначе соседняя."""
    kind = texture(board)
    bucket = size_bucket(bet_share)
    same = [d for d in library if d.key == key and texture(d.board) == kind]
    for wanted in (lambda b: b == bucket, lambda b: abs(b - bucket) == 1):
        found = [d for d in same if wanted(size_bucket(d.bet_share))]
        if found:
            return min(found, key=lambda d: (abs(d.bet_share - bet_share), d.name))
    return None


@dataclass(frozen=True)
class HandBet:
    """Ставка флопа в руке, на которую отвечает соперник: ключ лукапа, борд, доля банка."""

    key: SolverKey
    board: tuple[str, ...]
    bet_share: float


def hand_bet(
    hand: CanonicalHand, opponent_label: str, pot_before_flop: int, depth_bb: float
) -> HandBet | None:
    """Первая ставка флопа — не соперника, и он на неё отвечает: ключ лукапа (позиции —
    по правилу неполного стола, как у чарта защиты; глубина — глубина чарта), борд и
    ставка в долях банка до флопа. Ставки на флопе нет, ставил сам соперник, позиции
    вне раскладки или на его ответе в банке больше двух игроков — `None`
    (`test_the_hand_bet_is_the_first_flop_bet_the_opponent_faces`,
    `test_a_multiway_reply_has_no_guide`)."""
    flop = [a for a in hand.actions if a.street is Street.FLOP]
    bets = [i for i, a in enumerate(flop) if a.kind is ActionKind.BET]
    if not bets or pot_before_flop <= 0:
        return None
    bet = flop[bets[0]]
    replies = [i for i, a in enumerate(flop) if i > bets[0] and a.label == opponent_label]
    if bet.label == opponent_label or not replies:
        return None
    reply = hand.actions.index(flop[replies[0]])
    folded = {a.label for a in hand.actions[:reply] if a.kind is ActionKind.FOLD}
    if len([p for p in hand.players if p.label not in folded]) > _HEADS_UP:
        return None
    position = {p.label: p.position for p in hand.players}
    try:
        seats, mine, theirs = defend_table(
            len(hand.players), position[opponent_label], position[bet.label]
        )
    except ChartMissing:
        return None
    key = SolverKey(seats, mine, theirs, depth_bb, hand.ante_type)
    board = tuple(hand.boards.get(Street.FLOP, []))
    if len(board) != _FLOP:
        return None
    return HandBet(key, board, bet.committed_after / pot_before_flop)
