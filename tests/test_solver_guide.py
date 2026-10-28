"""Солверный ориентир (спека 2026-10-24, §6): загрузчик, текстура v1, корзины, лукап.
Выгрузки синтетические, пишутся в tmp; ни одной реальной выгрузки тест не читает."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.analysis.charts import parse_range
from harness.analysis.solver_guide import (
    SolverFileError,
    SolverKey,
    hand_bet,
    load_library,
    nearest,
    size_bucket,
    texture,
)
from harness.contracts import Street
from tests.test_opponent_range import BTN_OPENS_BB_CALLS, _hand
from tests.test_postflop_line import F, P, _bet, _call, _check, _fold, _raise

_KEY = {"seats": 8, "position": "BB", "vs_position": "BTN", "depth_bb": 25,
        "ante_type": "per_player"}


def write_dump(directory: Path, name: str, **over: object) -> Path:
    """Синтетическая выгрузка формата `solver_flop.json`."""
    dump: dict[str, object] = {
        **_KEY, "street": "flop", "board": "Qh7c2d", "pot_bb": 5.0,
        "flop_line": "BB check, BTN bet 1bb", "call": "QQ, AQs:0.5, 77",
        "raise": "AA", "fold": "32o", "source": "синтетика",
    }
    dump.update(over)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(dump, ensure_ascii=False), encoding="utf-8")
    return path


def test_an_empty_library_gives_no_guide(tmp_path: Path):
    assert load_library(tmp_path / "нет такого каталога") == []
    (tmp_path / "solver").mkdir()
    assert load_library(tmp_path / "solver") == []


def test_the_loader_reads_the_dump_and_the_bet_from_the_line(tmp_path: Path):
    write_dump(tmp_path, "a.json")
    write_dump(tmp_path, "b.json", bet_bb=4.0, board="As Kd 9c")
    first, second = load_library(tmp_path)
    assert first.key == SolverKey(8, "BB", "BTN", 25, "per_player")
    assert (first.board, first.bet_share) == (("Qh", "7c", "2d"), 0.2)
    assert first.actions["call"] == parse_range("QQ, AQs:0.5, 77")
    assert set(first.actions) == {"call", "raise", "fold"}
    assert (second.board, second.bet_share) == (("As", "Kd", "9c"), 0.8)


@pytest.mark.parametrize(
    "over",
    [{"street": "turn"}, {"board": "Qh7c"}, {"flop_line": "BB check"}, {"call": "QQ+++"}],
    ids=["turn", "two-cards", "no-bet", "bad-range"],
)
def test_a_dump_that_cannot_be_read_is_an_error_with_the_file_name(tmp_path: Path, over):
    write_dump(tmp_path, "bad.json", **over)
    with pytest.raises(SolverFileError, match=r"^bad\.json: "):
        load_library(tmp_path)


@pytest.mark.parametrize(
    ("board", "expected"),
    [
        (["Qc", "7d", "2s"], ("неспаренный", "радуга", "сухой", "средний")),
        (["Jh", "Th", "9c"], ("неспаренный", "две масти", "связный", "средний")),
        (["Ks", "8s", "3s"], ("неспаренный", "одна масть", "сухой", "высокий")),
        (["8h", "8d", "Js"], ("спаренный", "радуга", "средний", "средний")),
        (["Ah", "4d", "2c"], ("неспаренный", "радуга", "связный", "высокий")),
        (["7h", "5d", "2c"], ("неспаренный", "радуга", "средний", "низкий")),
    ],
)
def test_the_texture_v1_reads_pairing_suits_connectedness_and_height(board, expected):
    assert texture(board) == expected


@pytest.mark.parametrize(
    ("share", "bucket"), [(0.2, 0), (0.399, 0), (0.4, 1), (0.8, 1), (0.801, 2), (1.5, 2)],
)
def test_the_size_buckets_split_at_forty_and_eighty_percent(share: float, bucket: int):
    assert size_bucket(share) == bucket


def test_the_guide_takes_the_exact_bucket_then_the_next_one_and_nothing_else(tmp_path: Path):
    """Точный ключ и текстура: та же корзина (ближайшая ставка, при равенстве — первая по
    имени), иначе соседняя; другая текстура, ключ или корзина через одну — не ориентир."""
    write_dump(tmp_path, "a-small.json", bet_bb=1.5)  # 30%
    write_dump(tmp_path, "b-small.json", bet_bb=1.0)  # 20%
    write_dump(tmp_path, "bb-small-twin.json", bet_bb=1.0)  # 20%, равная b — позже по имени
    write_dump(tmp_path, "c-medium.json", bet_bb=3.0)  # 60%
    write_dump(tmp_path, "d-wet.json", board="Jh Th 9c", bet_bb=1.0)
    write_dump(tmp_path, "e-co.json", vs_position="CO", bet_bb=1.0)
    library = load_library(tmp_path)
    key = SolverKey(8, "BB", "BTN", 25, "per_player")
    dry = ["Qs", "7d", "2h"]

    def name(found) -> str | None:
        return None if found is None else found.name

    assert name(nearest(library, key, dry, 0.24)) == "b-small.json"
    assert name(nearest(library, key, dry, 0.29)) == "a-small.json"
    assert name(nearest(library, key, dry, 0.7)) == "c-medium.json"
    assert name(nearest(library, key, dry, 0.9)) == "c-medium.json"
    only_small = [d for d in library if d.name != "c-medium.json"]
    assert nearest(only_small, key, dry, 0.9) is None
    assert nearest(library, key, ["Ks", "7d", "2h"], 0.2) is None
    assert nearest(library, SolverKey(8, "BB", "BTN", 40, "per_player"), dry, 0.2) is None


def test_the_hand_bet_is_the_first_flop_bet_the_opponent_faces():
    """С-бет героя с BTN, BB отвечает: ключ — как у чарта защиты (6 мест → 8-max), борд
    флопа, ставка в долях банка до флопа; сам ставивший соперник — не ориентир."""
    hand = _hand(BTN_OPENS_BB_CALLS)
    bet = hand_bet(hand, "P6", 550, 25.0)
    assert bet is not None
    assert bet.key == SolverKey(8, "BB", "BTN", 25.0, hand.ante_type)
    assert bet.board == tuple(hand.boards[Street.FLOP])
    assert bet.bet_share == pytest.approx(160 / 550)
    assert hand_bet(hand, "Hero", 550, 25.0) is None
    donk = _hand([
        *BTN_OPENS_BB_CALLS[:6], _bet(F, "P6", 160), _raise(F, "Hero", 500), _call(F, "P6", 340),
    ])
    assert hand_bet(donk, "P6", 550, 25.0) is None


def test_a_multiway_reply_has_no_guide():
    """Три игрока на ответе BB — ориентира нет; сбросил третий до ответа — хедз-ап."""
    pre = [
        _fold(P, "P1"), _fold(P, "P2"), _fold(P, "P3"), _raise(P, "Hero", 250),
        _call(P, "P5", 200), _call(P, "P6", 150), _check(F, "P5"), _check(F, "P6"),
        _bet(F, "Hero", 240),
    ]
    multiway = _hand([*pre, _call(F, "P5", 240), _call(F, "P6", 240)])
    assert hand_bet(multiway, "P6", 750, 25.0) is None
    heads_up = _hand([*pre, _fold(F, "P5"), _call(F, "P6", 240)])
    assert hand_bet(heads_up, "P6", 750, 25.0) is not None
