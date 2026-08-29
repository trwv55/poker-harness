from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent.parent / "fixtures" / "hh"
FIXTURE_DAILY = FIXTURES / "daily-classic-146.txt"
FIXTURE_PKO = FIXTURES / "pko-bounty-172.txt"

# Реальные hand history — приватные данные игрока и в публичный репозиторий не попадают.
# Тесты, которым они нужны, пропускаются с явной причиной: без файлов регрессионная
# сетка на 318 руках НЕ выполняется, и зелёный прогон без них не означает, что конвейер
# проверен. Порядок сборки данных — в README.
FIXTURES_PRESENT = FIXTURE_DAILY.exists() and FIXTURE_PKO.exists()
requires_fixtures = pytest.mark.skipif(
    not FIXTURES_PRESENT,
    reason="нет приватных HH-фикстур в fixtures/hh/ — гейт на 318 руках не выполнен",
)
