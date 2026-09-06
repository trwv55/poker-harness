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

from harness.analysis.tools.equity import equity_hand_vs_hand
from harness.contracts import VisionCheck

__all__ = [
    "CHECK_BUTTON",
    "CHECK_CARDS",
    "CHECK_EQUITY",
    "CHECK_HERO",
    "CHECK_POT",
    "EQUITY_TOLERANCE_PP",
    "POT_TOLERANCE_BB",
    "button_check",
    "cards_check",
    "equity_check",
    "match_hero",
    "pot_check",
]

CHECK_POT = "pot"
CHECK_BUTTON = "button"
CHECK_CARDS = "cards"
CHECK_EQUITY = "equity"
CHECK_HERO = "hero"

# Допуск сверки банка — в больших блайндах. Экспорт печатает стеки и суммы с
# двумя знаками и ОБРЕЗАЕТ, а не округляет (сверено с текстом рума на двух
# руках фикстуры: 87 589 фишек при bb 5 000 напечатаны как 17.51, а не 17.52).
# На восьмерых накопленная обрезка не превышает 0.08 ББ, поэтому допуск взят с
# запасом; расхождение из-за пропущенного анте — величины блайнда и больше, то
# есть отличается от обрезки на порядок.
POT_TOLERANCE_BB = 0.1

# Допуск сверки эквити — в процентных пунктах. Измерено на трёх экранах реестра:
# при верно прочитанных картах расхождение с GG было 0.03, 0.03 и 0.14 п.п., при
# неверно прочитанной МАСТИ — около 2 п.п. Порог стоит между этими двумя
# величинами, а не у нуля: наш расчёт — Монте-Карло на 200 000 раздач
# (`equity_hand_vs_hand`), его собственный разброс порядка 0.1 п.п., и порог
# 0.1 п.п. загорался бы от собственного шума. Эта проверка — третья по силе:
# первой идёт сверка карт у места с картами в логе, она работает без вскрытия и
# без напечатанных процентов.
EQUITY_TOLERANCE_PP = 1.0


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
        passed=delta <= POT_TOLERANCE_BB,
        detail=(
            f"банк на экране {pot_shown_bb:.2f} ББ, сумма видимых вкладов "
            f"{contributions_bb:.2f} ББ, расхождение {delta:.2f} ББ"
        ),
        options=[f"{pot_shown_bb:.2f}", f"{contributions_bb:.2f}"],
    )


def button_check(marked: str | None, derived: str | None) -> VisionCheck:
    """Фишка дилера против рассадки, восстановленной из порядка хода (реестр D3).

    Два прочтения одной рассадки: кружок с буквой `D` за столом и последний
    ходивший до блайндов в логе префлопа. Ошибка в кнопке сдвигает раскладку
    позиций целиком и меняет вердикт, не меняя ни одного числа на экране (A3),
    поэтому она вынесена в отдельную проверку, а не выводится из каскада
    денежных расхождений.

    Молчит, когда сверять нечего: на живом столе лога нет, и второго прочтения
    не существует — там кнопку проверяет валидатор против блайндов.
    """
    if marked is None or derived is None:
        return VisionCheck(
            name=CHECK_BUTTON, passed=True, detail="второго прочтения рассадки на экране нет"
        )
    return VisionCheck(
        name=CHECK_BUTTON,
        passed=marked == derived,
        detail=f"фишка дилера у {marked!r}, по порядку хода кнопка у {derived!r}",
        options=[marked, derived],
    )


def cards_check(at_seat: dict[str, list[str]], in_log: dict[str, list[str]]) -> VisionCheck:
    """Карты у места против карт в колонке улицы — сильнейшая из проверок карт.

    Измерено (реестр, «Карты отрисованы дважды»): на полном экране Sonnet прочёл
    у места `5♦` из-под баннера WIN и подставил то же значение в поле лога; на
    вырезке одной только колонки лога тот же Sonnet прочёл `A♠ 5♠` верно. Модель
    читает чистый рендер безошибочно, а перекрытую баннером карту — нет.

    Работает без вскрытия и без напечатанных процентов, то есть на любой руке с
    олл-ином, а не только там, где GG показал эквити, — поэтому идёт первой.
    Сравниваются только те игроки, у кого прочитаны ОБА места.
    """
    disagreements = [
        f"{label}: у места {at_seat[label]}, в логе {in_log[label]}"
        for label in sorted(set(at_seat) & set(in_log))
        if sorted(at_seat[label]) != sorted(in_log[label])
    ]
    if not disagreements:
        return VisionCheck(name=CHECK_CARDS, passed=True, detail="карты обоих мест совпали")
    first = disagreements[0].split(": ", 1)[0]
    return VisionCheck(
        name=CHECK_CARDS,
        passed=False,
        detail="; ".join(disagreements),
        options=[" ".join(at_seat[first]), " ".join(in_log[first])],
    )


def equity_check(
    shown_pct: float | None, hero: list[str], villain: list[str], board: list[str]
) -> VisionCheck:
    """Напечатанный GG процент против посчитанного нами (реестр, «Эквити — оракул»).

    Единственная проверка, которая проверяет именно КАРТЫ, а не суммы, и
    единственная, где второе прочтение не с экрана, а из нашего расчёта. Считает
    тот же `equity_hand_vs_hand`, что и ядро: две реализации эквити разошлись бы
    молча, и тогда проверка стала бы измерять разницу между ними, а не ошибку
    чтения.

    Молчит, когда сверять нечего: процент напечатан только на экспортах с
    олл-ином, и не на каждом.
    """
    if shown_pct is None or len(hero) != 2 or len(villain) != 2:
        return VisionCheck(
            name=CHECK_EQUITY, passed=True, detail="эквити на экране не напечатано"
        )
    if len(set(hero) | set(villain) | set(board)) != len(hero) + len(villain) + len(board):
        return VisionCheck(
            name=CHECK_EQUITY,
            passed=False,
            detail=f"карта названа дважды: {hero} против {villain} на борде {board}",
            options=[f"{shown_pct:.2f}", "—"],
        )
    computed_pct = 100.0 * equity_hand_vs_hand((hero[0], hero[1]), (villain[0], villain[1]), board)
    delta = abs(shown_pct - computed_pct)
    return VisionCheck(
        name=CHECK_EQUITY,
        passed=delta <= EQUITY_TOLERANCE_PP,
        detail=(
            f"на экране {shown_pct:.2f}%, по прочитанным картам {computed_pct:.2f}%, "
            f"расхождение {delta:.2f} п.п."
        ),
        options=[f"{shown_pct:.2f}", f"{computed_pct:.2f}"],
    )


def _normalized(nickname: str) -> str:
    """Ник без хвоста обрезки и регистра: экран режет длинные ники многоточием."""
    trimmed = nickname.strip().rstrip(".").strip()
    return trimmed.casefold()


def match_hero(profile_nickname: str, seen: list[str]) -> tuple[str | None, VisionCheck]:
    """Найти героя КОДОМ по нику из профиля — сопоставление по префиксу.

    Решение реестра 2026-09-05, и оно закрывает дыру, которую не ловит ни одна
    контрольная сумма: измерено, что Sonnet перепутал героя с победителем на
    трёх экранах из трёх, а банк, кнопка и эквити от того, кого назвали героем,
    не зависят вовсе. Цена ошибки максимальная — чужие решения, предъявленные
    игроку как его собственные.

    Префикс, а не равенство: экран обрезает длинные ники многоточием
    (`FictionalNick46..`). Совпадение считается, когда один из ников — начало
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
