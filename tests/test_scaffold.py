import harness
from tests.conftest import FIXTURE_DAILY, FIXTURE_PKO, requires_fixtures


def test_package_importable():
    assert harness is not None


@requires_fixtures
def test_fixtures_present():
    assert FIXTURE_DAILY.stat().st_size > 100_000
    assert FIXTURE_PKO.stat().st_size > 100_000


def test_stdlib_platform_not_shadowed():
    import platform

    assert "harness" not in (platform.__file__ or "")
