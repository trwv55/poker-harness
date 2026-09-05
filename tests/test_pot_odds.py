from harness.analysis.tools.pot_odds import required_equity


def test_required_equity():
    assert abs(required_equity(50, 100) - 50 / 150) < 1e-9  # пот 100 (ставка внутри), колл 50
    assert abs(required_equity(141, 20250) - 141 / 20391) < 1e-9  # рука из фикстуры
