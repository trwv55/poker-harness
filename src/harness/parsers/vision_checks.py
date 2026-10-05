"""Контрольные суммы прочитанного экрана — четыре штуки, все из реестра.

Проверить скриншот нечем извне: у него нет ни `Total pot`, ни строк `collected`,
ни порядка хода, который движок обязан воспроизвести (реестр D1). Всё, чем он
проверяется, — избыточность внутри самого экрана: одно и то же напечатано на нём
дважды, и два прочтения обязаны сойтись.

| проверка | два независимых прочтения |
|---|---|
| `pot` | показанный банк против суммы видимых вкладов (реестр D2) |
| `button` | фишка дилера против рассадки, восстановленной по логу (реестр D3) |
| `cards` | карты у места против карт в колонке улицы («отрисованы дважды») |
| `equity` | напечатанный процент против посчитанного нашим эквити (оракул) |
| `hero` | ник из профиля против ников, прочитанных на экране |

Провал любой — сигнал каскада (повтор на дорогой модели), а не тихая правка:
подогнать число значило бы соврать про деньги игрока (CLAUDE.md).

**Общая слабость всех пяти названа в реестре (D4) и никуда не делась:**
контрольная сумма ловит расхождение двух прочтений, но не случай, когда оба
неверны согласованно. Насколько велик этот угол, покажет только размеченный
датасет — проверок для этого недостаточно по построению.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from harness.contracts import VisionCheck

__all__ = [
    "CHECK_BUTTON",
    "CHECK_EQUITY",
    "CHECK_HERO",
    "CHECK_POT",
    "CHECK_SEATS",
    "EQUITY_TOLERANCE_PP",
    "POT_TOLERANCE_BB",
    "equity_check",
    "match_hero",
    "pot_check",
    "seats_check",
    "within_tolerance",
]

CHECK_POT = "pot"
CHECK_BUTTON = "button"
CHECK_EQUITY = "equity"
CHECK_HERO = "hero"
CHECK_SEATS = "seats"

# Допуск сверки банка — в больших блайндах. Экспорт печатает стеки и суммы с
# двумя знаками и ОБРЕЗАЕТ, а не округляет (сверено с текстом рума на двух
# руках фикстуры: 87 589 фишек при bb 5 000 напечатаны как 17.51, а не 17.52).
# На восьмерых накопленная обрезка не превышает 0.08 ББ, поэтому допуск взят с
# запасом; расхождение из-за пропущенного анте — величины блайнда и больше, то
# есть отличается от обрезки на порядок.
POT_TOLERANCE_BB = 0.1

# Допуск сверки эквити — в процентных пунктах. Порог стоит между расхождением на
# ВЕРНО прочитанных картах и расхождением на неверно прочитанной масти (числа —
# реестр, «Эквити с экрана GG»), а не у нуля: наш расчёт — Монте-Карло на
# 200 000 раздач (`equity_hand_vs_hand`), его собственный разброс порядка
# 0.1 п.п., и порог 0.1 п.п. загорался бы от собственного шума. Эта проверка —
# третья по силе: первой идёт сверка карт у места с картами в логе, она работает
# без вскрытия и без напечатанных процентов.
EQUITY_TOLERANCE_PP = 1.0


def within_tolerance(delta: float, tolerance: float) -> bool:
    """Расхождение, равное допуску, проходит.

    Обе сверки вычитают одну прочитанную с экрана десятичную величину из другой,
    а в двоичной плавающей точке такая разность бывает чуть больше своего
    десятичного значения — голое `delta <= tolerance` отвергает тогда ровно
    граничный случай, который допуск обязан пропускать. Запас здесь
    относительный (`math.isclose`), а не приписанный к допуску слагаемым:
    он не зависит от того, каким числом записан сам допуск, и не расширяет его
    до следующего печатаемого знака.

    Закреплено тестами `test_a_pot_off_by_exactly_the_tolerance_still_passes` и
    `test_an_equity_off_by_exactly_the_tolerance_still_passes`; что расхождение
    заведомо больше допуска по-прежнему не проходит — тестами
    `test_a_pot_off_by_more_than_the_tolerance_still_fails` и
    `test_an_equity_off_by_more_than_the_tolerance_still_fails`.
    """
    return delta <= tolerance or math.isclose(delta, tolerance)


def pot_check(pot_shown_bb: float | None, contributions_bb: float) -> VisionCheck:
    """Показанный банк против суммы видимых вкладов (реестр D2).

    Именно этой суммой было доказано пропущенное обеими моделями анте: банк не
    сходился ровно на пул анте. Проверка молчит, когда банка на экране нет —
    выдумывать расхождение из отсутствия данных нельзя (тот же принцип, что у
    сверки выплат в валидаторе).
    """
    if pot_shown_bb is None:
        return VisionCheck(name=CHECK_POT, passed=True, detail="банк на экране не показан")
    delta = abs(pot_shown_bb - contributions_bb)
    return VisionCheck(
        name=CHECK_POT,
        passed=within_tolerance(delta, POT_TOLERANCE_BB),
        detail=(
            f"банк на экране {pot_shown_bb:.2f} ББ, сумма видимых вкладов "
            f"{contributions_bb:.2f} ББ, расхождение {delta:.2f} ББ"
        ),
        options=[f"{pot_shown_bb:.2f}", f"{contributions_bb:.2f}"],
    )


def equity_check(
    hands: Sequence[tuple[list[str], float | None]], board: list[str]
) -> VisionCheck:
    """Напечатанный GG процент против посчитанного нами (реестр, «Эквити — оракул»).

    Единственная проверка, которая проверяет именно КАРТЫ, а не суммы, и
    единственная, где второе прочтение не с экрана, а из нашего расчёта. Считает
    тем же модулем, что и ядро (`equity_multiway`): две реализации эквити
    разошлись бы молча, и тогда проверка стала бы измерять разницу между ними, а
    не ошибку чтения.

    На вход идут ВСЕ участники олл-ина, чью долю подписал экран, а не пара:
    посчитанная вдвоём доля трёхстороннего олл-ина расходится с напечатанной на
    десяток процентных единиц, и проверка объявляла бы ошибкой чтения свою
    собственную неполноту. Доля без процента (`None`) участвует в расчёте, но не
    сверяется — её карты влияют на чужие доли.

    Молчит, когда сверять нечего: процент напечатан только на экспортах с
    олл-ином, и не на каждом.

    **Импорт эквити — внутри функции, и это не стиль.** Через
    `harness.analysis` в процесс затягивается весь расчётный стек (`eval7`,
    `pokerkit`), а `harness.parsers.vision_adapter` импортирует процесс БОТА
    ради подстановки ответа игрока в руку. Модульный импорт вернул бы в образ
    бота ровно ту зависимость, которую из него уже однажды выносили
    (`test_bot_image_does_not_import_calculation_stack`). Считает эквити воркер,
    и грузит его тоже он.
    """
    from harness.analysis.tools.equity import equity_multiway

    known = [(cards, shown) for cards, shown in hands if len(cards) == 2]
    printed = [shown for _, shown in known if shown is not None]
    if len(known) < 2 or not printed:
        return VisionCheck(
            name=CHECK_EQUITY, passed=True, detail="эквити на экране не напечатано"
        )

    cards_flat = [card for pair, _ in known for card in pair]
    if len(set(cards_flat) | set(board)) != len(cards_flat) + len(board):
        return VisionCheck(
            name=CHECK_EQUITY,
            passed=False,
            detail=f"карта названа дважды: {[pair for pair, _ in known]} на борде {board}",
            options=[f"{printed[0]:.2f}", "—"],
        )

    computed = equity_multiway([(pair[0], pair[1]) for pair, _ in known], board)
    worst = max(
        (
            (abs(shown - 100.0 * value), shown, 100.0 * value)
            for (_, shown), value in zip(known, computed, strict=True)
            if shown is not None
        ),
        key=lambda item: item[0],
    )
    delta, shown_pct, computed_pct = worst
    return VisionCheck(
        name=CHECK_EQUITY,
        passed=within_tolerance(delta, EQUITY_TOLERANCE_PP),
        detail=(
            f"на экране {shown_pct:.2f}%, по прочитанным картам {computed_pct:.2f}%, "
            f"расхождение {delta:.2f} п.п."
        ),
        options=[f"{shown_pct:.2f}", f"{computed_pct:.2f}"],
    )


def seats_check(players: int, max_seats: int | None) -> VisionCheck:
    """Игроков не больше, чем мест за столом, — если размер стола прочитан.

    Однострочная арифметика, ловящая лишнего участника раньше всех денежных
    сверок: за восьмиместным столом девятого игрока не бывает.
    """
    if not max_seats:
        return VisionCheck(name=CHECK_SEATS, passed=True, detail="размер стола не прочитан")
    return VisionCheck(
        name=CHECK_SEATS,
        passed=players <= max_seats,
        detail=f"игроков {players}, мест за столом {max_seats}",
    )


def _normalized(nickname: str) -> str:
    """Ник без хвоста обрезки и регистра: экран режет длинные ники многоточием.

    Многоточие бывает и одним символом `…`, и тремя точками — модель пишет как
    видит, а обрезаются оба вида одинаково.
    """
    trimmed = nickname.strip().rstrip(".…").strip()
    return trimmed.casefold()


def match_hero(profile_nickname: str, seen: list[str]) -> tuple[str | None, VisionCheck]:
    """Найти героя КОДОМ по нику из профиля — сопоставление по префиксу.

    Решение реестра 2026-09-05 («Герой определяется кодом по нику из профиля»), и
    оно закрывает дыру, которую не ловит ни одна контрольная сумма: банк, кнопка
    и эквити от того, кого назвали героем, не зависят вовсе. Цена ошибки
    максимальная — чужие решения, предъявленные игроку как его собственные.

    Префикс, а не равенство: экран обрезает длинные ники многоточием
    (`длинный_ник_иг..`). Совпадение считается, когда один из ников — начало
    другого, в любую сторону: обрезан бывает экранный, а сокращён — записанный
    в профиль.

    Ровно одно совпадение — герой найден. Ноль или больше одного — эскалация, а
    не догадка: продукт разбирает решения героя, и ошибиться тут дороже, чем
    переспросить.

    **Ник в промпт не передаётся** (реестр, «Почему ник НЕ передаётся»): модель
    читает ники, не зная, кто из них герой. Подсказка превратила бы проверку в
    повтор подсказки, и отличить прочитанное от подсказанного стало бы нечем.
    """
    profile = _normalized(profile_nickname)
    matches = [
        nickname
        for nickname in seen
        if (norm := _normalized(nickname))
        and (profile.startswith(norm) or norm.startswith(profile))
    ]
    if len(matches) == 1:
        return matches[0], VisionCheck(
            name=CHECK_HERO, passed=True, detail=f"герой опознан по нику из профиля: {matches[0]!r}"
        )
    detail = (
        "ник из профиля не совпал ни с одним прочитанным"
        if not matches
        else f"ник из профиля совпал с несколькими: {matches}"
    )
    return None, VisionCheck(name=CHECK_HERO, passed=False, detail=detail, options=list(seen))
