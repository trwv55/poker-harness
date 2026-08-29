import pytest

from tests.conftest import FIXTURE_DAILY, FIXTURE_PKO


def _pipeline(path):
    from harness.engine import enrich
    from harness.normalizer import normalize
    from harness.parsers.hh_parser import parse_file
    raws = parse_file(path.read_text(encoding="utf-8"), source_ref=path.name)
    return raws, [enrich(normalize(r)) for r in raws]

@pytest.mark.parametrize("path,expected_hands", [(FIXTURE_DAILY, 146), (FIXTURE_PKO, 172)])
def test_grid(path, expected_hands):
    raws, enriched = _pipeline(path)
    assert len(raws) == expected_hands
    # руки отсортированы хронологически (в файле GG — обратный порядок)
    ts = [r.timestamp for r in raws]
    assert ts == sorted(ts)
    # формат покрыт полностью: ни одной нераспознанной строки
    assert all(r.unknown_lines == [] for r in raws), \
        [line for r in raws for line in r.unknown_lines][:5]
    for en in enriched:
        # каждая рука проходит валидатор без эскалаций (HH = факт)
        assert en.verdict.status == "pass", (en.hand.hand_no, en.verdict)
        assert en.hand.summary is not None          # сужение для pyright (tests в include)
        # банк, пересчитанный движком, сходится с SUMMARY (rake в фикстурах = 0)
        assert en.report.final_pot == en.hand.summary.total_pot, en.hand.hand_no
        assert en.hand.summary.rake == 0
        # стеки на конец руки неотрицательны, сумма стеков сохранилась
        assert all(v >= 0 for v in en.report.stacks_end.values())
        start = sum(p.stack for p in en.hand.players)
        end = sum(en.report.stacks_end.values())
        assert start == end      # rake 0: фишки не исчезают
