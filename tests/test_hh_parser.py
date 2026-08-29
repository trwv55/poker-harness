from harness.contracts import ActionKind, PostKind, Street
from harness.parsers.hh_parser import parse_file, parse_hand
from tests.conftest import FIXTURE_DAILY

SAMPLE = """Poker Hand #TM6316081388: Tournament #306148954, Daily Classic $4 Hold'em No Limit - Level23(3,000/6,000(750)) - 2026/08/20 22:22:36
Table '8' 8-max Seat #3 is the button
Seat 1: c30a7c9e (85,440 in chips)
Seat 2: bb4aa4e0 (25,109 in chips)
Seat 3: c3986130 (222,896 in chips)
Seat 4: Hero (3,891 in chips)
Seat 5: fcc9bf19 (415,055 in chips)
Seat 7: 5553a2cd (70,471 in chips)
Seat 8: 95b4992 (151,005 in chips)
95b4992: posts the ante 750
c3986130: posts the ante 750
fcc9bf19: posts the ante 750
bb4aa4e0: posts the ante 750
5553a2cd: posts the ante 750
c30a7c9e: posts the ante 750
Hero: posts the ante 750
Hero: posts small blind 3,000
fcc9bf19: posts big blind 6,000
*** HOLE CARDS ***
Dealt to c30a7c9e 
Dealt to bb4aa4e0 
Dealt to c3986130 
Dealt to Hero [3c Kc]
Dealt to fcc9bf19 
Dealt to 5553a2cd 
Dealt to 95b4992 
5553a2cd: raises 63,000 to 69,000
95b4992: folds
c30a7c9e: folds
bb4aa4e0: folds
c3986130: folds
Hero: calls 141 and is all-in
fcc9bf19: folds
Uncalled bet (63,000) returned to 5553a2cd
5553a2cd: shows [Js Ah]
Hero: shows [3c Kc]
*** FLOP *** [Kd Td 3s]
*** TURN *** [Kd Td 3s] [Qd]
*** RIVER *** [Kd Td 3s Qd] [2d]
*** SHOWDOWN ***
5553a2cd collected 14,673 from pot
5553a2cd collected 5,718 from pot
*** SUMMARY ***
Total pot 20,391 | Rake 0 | Jackpot 0 | Bingo 0 | Fortune 0 | Tax 0
Board [Kd Td 3s Qd 2d]
Seat 1: c30a7c9e folded before Flop
Seat 2: bb4aa4e0 folded before Flop
Seat 3: c3986130 (button) folded before Flop
Seat 4: Hero (small blind) showed [3c Kc] and lost with two pair, Kings and Threes
Seat 5: fcc9bf19 (big blind) folded before Flop
Seat 7: 5553a2cd showed [Js Ah] and won (20,391) with a straight, Ace to Ten
Seat 8: 95b4992 folded before Flop
"""

def test_parse_sample_hand():
    h = parse_hand(SAMPLE, source_ref="daily-classic-146.txt")
    assert h.hand_no == "TM6316081388" and h.tournament_id == "306148954"
    assert (h.level, h.sb, h.bb, h.ante) == (23, 3000, 6000, 750)
    assert h.max_seats == 8 and h.button_seat == 3
    assert len(h.seats) == 7 and h.seats[3].label == "Hero" and h.seats[3].stack == 3891
    assert sum(1 for p in h.posts if p.kind == PostKind.ANTE) == 7
    assert h.dealt["Hero"] == ["3c", "Kc"] and h.dealt["c30a7c9e"] == []
    raise_a = h.actions[0]
    assert (raise_a.kind, raise_a.amount, raise_a.to_amount) == (ActionKind.RAISE, 63000, 69000)
    call_a = next(a for a in h.actions if a.label == "Hero")
    assert (call_a.kind, call_a.amount, call_a.is_all_in) == (ActionKind.CALL, 141, True)
    assert h.boards[Street.FLOP] == ["Kd", "Td", "3s"] and h.boards[Street.RIVER] == ["2d"]
    assert [c.amount for c in h.collected] == [14673, 5718]
    assert h.uncalled[0].amount == 63000
    assert h.summary is not None  # сужение для pyright (SummaryInfo | None)
    assert h.summary.total_pot == 20391 and h.summary.tax == 0
    assert h.showdowns[0].cards == ["Js", "Ah"]
    assert h.unknown_lines == []

def test_parse_file_sorted_and_counted():
    raws = parse_file(FIXTURE_DAILY.read_text(encoding="utf-8"), source_ref="daily")
    assert len(raws) == 146
    ts = [r.timestamp for r in raws]
    assert ts == sorted(ts)          # файл GG — в обратном порядке, парсер сортирует
