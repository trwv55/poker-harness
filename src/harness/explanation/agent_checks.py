"""Проверки ответа агента и сводка разбора для его контекста (спека агента №1, §4, §6).

Одна реализация на прод (`harness.agent`) и evals (спека №2): вторая разошлась
бы с первой на первой же правке. Модуль чистый — ни pydantic-ai, ни базы:
разрешённые числа, метки и признак вызова `leaks` приходят аргументами.

**Допущение — с меткой: механика узкая, остальное — судья.** В v1 диапазоны не
считает ни один расчёт (они — спека №3). Код ловит два вида предложений без
«(допущение)», даже с «если» и с разрешёнными числами: со словом диапазона
(«диапазон», «рендж», «держ…» — метка по умолчанию,
`test_a_range_word_needs_the_mark_by_default`) и с явным утверждением о сопернике
— слово о нём плюс слово из короткого списка рук и действий
(`test_a_claim_about_what_the_opponent_holds_or_does_needs_the_mark`). Списки не
полны и не будут: другие формулировки о сопернике ловит судья (спека №2,
принцип владельца 2026-10-11). Метки не требуют: действие, за которым стоит
измеренная частота из расчёта в разговоре (`frequency_numbers`), или порог и доля
защиты из расчёта (`calc_numbers`), но только как арифметика порога: в придаточном
с «если/когда», с «при» перед долей со словом порога («при 24% фолдов»; «при такой
ставке» — не условие) или с «достаточно/нужно/должен» и инфинитивом; не доля банка,
не «поставите N%», без руки между действием и долей
(`test_calc_backed_statements_and_meta_speech_need_no_mark`,
`test_a_bet_size_or_a_hand_in_the_window_backs_nothing`,
`test_a_number_not_from_a_calculation_backs_nothing`,
`test_only_a_measured_frequency_backs_an_opponent_action`,
`test_a_bare_preposition_pri_is_not_a_condition`); узкая метаречь («про
его диапазон это мало говорит»); пересказ руки — глагол прошедшего времени,
придаточное с «вы» без соперника подлежащим, «вы держите» о своей руке
(`test_hand_narration_is_not_an_opponent_claim`). Прошедшее время не освобождает,
если в предложении «бы» или наречие повторяемости («всегда», «обычно», «часто»,
«редко», «раньше», «никогда»…) — это прогноз или обобщение
(`test_the_subjunctive_is_a_prediction_not_a_fact_of_the_hand`,
`test_a_habit_in_the_past_is_a_claim_not_a_fact_of_the_hand`); «вы» не освобождает
придаточное, где подлежащее — соперник, с запятой или без
(`test_a_clause_with_an_opponent_subject_is_checked_despite_you`). «Фолд», «сброс»,
«защита» существительным — не действие только после предлога (на/по/от/при/без/
для, через одно прилагательное) или в части о прибыльности ставки и никогда при
«бы», будущем («будет», «сделает») или наречии повторяемости в предложении;
иначе существительное — действие, как «рейз»
(`test_a_fold_or_defence_noun_after_a_preposition_is_not_an_opponent_action`,
`test_a_fold_or_defence_noun_outside_a_preposition_is_an_action`,
`test_the_subjunctive_future_or_habit_cancels_the_noun_exemption`,
`test_a_fold_noun_in_a_profit_clause_is_not_an_action`); часть о прибыльности
ставки («он окупается против любой пары») — о ставке, рука в ней не утверждение
о сопернике, а глагол соперника — прогноз
(`test_a_hand_word_in_a_profit_clause_is_about_the_bet`,
`test_an_opponent_verb_in_a_profit_clause_is_still_a_prediction`). В предложении о
прибыльности форма «если <подлежащее> сбрасывает чаще/реже N%» с порогом из
расчёта — арифметика порога и через запятые, если между ними только вставки («по
расчёту», «судя по выходу»); подлежащее-герой, глагол или другое придаточное
между условием и действием опору разрывают
(`test_threshold_math_after_a_profit_word_is_exempt_across_commas`,
`test_live_threshold_math_about_the_bet_passes`,
`test_a_hero_subject_or_another_clause_breaks_the_threshold_condition`).

**«В плюсе», «в минусе», «окупается» — вывод только по порогу.** Голый вывод
допустим, только когда в разговоре есть `frequency_vs_threshold` с выводом выше
или ниже порога (`test_a_profit_conclusion_needs_a_decided_threshold`), и вывод
«окупается / в плюсе (и) без фолдов», когда ядро дало его в сводке
(`test_a_semibluff_that_pays_without_folds_may_be_retold_from_the_summary`). Арифметика
допустима всегда: в одном из придаточных условие («если/от/когда/чтобы» или «при»
перед долей со словом порога) и доля, выданная расчётом (не доля банка), а
свободного придаточного-вывода рядом нет
(`test_profit_arithmetic_with_a_calc_share_is_allowed`,
`test_a_bare_profit_conclusion_with_some_share_still_needs_a_threshold`,
`test_a_threshold_in_the_main_clause_is_not_a_condition`,
`test_a_profit_conclusion_with_a_bare_pri_needs_a_threshold`).

**Отрицание термина — не термин.** «Без флеш-дро», «нет флеш-дро», «не
флеш-дро» метки не требуют (`test_a_negated_term_is_not_the_term`).

**Граница, а не гарантия.** Допущение, термины и лучшая линия проверяются по
явным оборотам; что сказано другими словами, ловит судья (спека №2).

**Обозначения карт — не числа.** `A5s`, `T9o`, `KQ`, карта с мастью, ранги борда
вырезаются ДО сверки чисел. Чисто цифровая пара (`22`…`99`) от суммы шаблоном
не отличима (тех-спека §5.7); решение — она всегда число, даже если герой
держит эту пару (`test_a_bare_digit_pair_is_always_a_number`): выдуманная
сумма, совпавшая с парой героя, иначе прошла бы без проверки. Карту модель
пишет мастями или словами («пара двоек»); голая «22» вместо карты уходит на
переписывание — принятая цена. Цифры борда без разделителя («258») — сумма, а
не борд (`test_digits_without_separators_are_not_a_board`).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from harness.contracts import (
    AnalysisResult,
    DrawKind,
    EnrichedHand,
    PostflopLineDetail,
    Purpose,
    Street,
    class_of,
    postflop_line_detail,
)
from harness.explanation.digest import NumberBook
from harness.explanation.faithfulness import error_words_in, unsupported_numbers
from harness.explanation.hand_replay import cards_text, hand_replay
from harness.explanation.verdict_text import verdict_digest

__all__ = [
    "ACTION_LABEL",
    "ASSUMPTION_MARK",
    "DRAW_LABEL",
    "HAND_LABEL",
    "STAT_LABEL",
    "TERM_LABELS",
    "AgentContext",
    "agent_context",
    "check_agent_answer",
    "strip_card_tokens",
]

ASSUMPTION_MARK = "(допущение)"

_RANK = r"(?:10|[2-9TJQKA])"
_EDGE_L = r"(?<!\w)(?<!\d[.,])"
_EDGE_R = r"(?![\w%]|[.,]\d)"
_SUITED_CARD_RE = re.compile(_RANK + r"[♠♥♦♣]️?")
_CLASS_RE = re.compile(_EDGE_L + r"([2-9TJQKA])([2-9TJQKA])([so])?" + _EDGE_R)
_DIGITS_ONLY_RE = re.compile(r"\d+")
# Без разделителя ранги борда слипаются, только если рядом буква («KQ7»): три
# цифры подряд («258») — сумма, а не борд 2-5-8.
_BOARD_JOIN = r"(?:[-–]|(?<=[TJQKA])|(?=[TJQKA]))"
_BOARD_RE = re.compile(_EDGE_L + _RANK + r"(?:" + _BOARD_JOIN + _RANK + r"){2,4}" + _EDGE_R)
# Точка внутри числа («6.6») предложение не кончает.
_SENTENCE_RE = re.compile(r"(?:[^.!?\n]|\.(?=\d))+[.!?]?")
# Допущение (решения владельца 2026-10-11 и 2026-10-13, правки контроллера по
# ревью). Механика узкая, остальное — судья (спека №2): метку требует предложение
# со словом диапазона и придаточное с явным утверждением о сопернике (слово о нём
# плюс рука или действие из короткого списка). До поиска вырезается то, что
# держится на расчёте (`_strip_backed`), и метаречь («про его диапазон это мало
# говорит»). Пересказ руки — не утверждение: глагол прошедшего времени (без «бы» и
# наречий повторяемости в предложении), придаточное с «вы» без соперника
# подлежащим (`_opponent_clause`).
_W = r"(?<![а-яёa-z])"
_WE = r"(?![а-яё])"
_POSITION = r"(?<![A-Za-z\d.])(?<![\d.]\s)(?:BB|SB|BTN|CO|HJ|LJ|MP|UTG(?:\+\d)?)(?![A-Za-z])"
_OPPONENT_RE = re.compile(
    _W + r"(?:он|его|ему|него|нему|ним|нём)" + _WE
    + r"|соперник|оппонент|противник|регуляр"
    + r"|" + _W + r"пол(?:е|я|ю|ем)" + _WE
    + r"|так(?:ие|их|им|ой|ого)\s+игрок"
    + r"|" + _POSITION,
    re.IGNORECASE,
)
_ACTION = r"сброс|сбрас|сда[её]т|сдаст|сдаю|сдава|фолд|сфолд|защит|защищ|колл|заколл|рейз|блеф"
_ACTION_RE = re.compile(_ACTION, re.IGNORECASE)
_HAND = (
    r"диапазон|рендж|держ|бродве|капн|кэпн|капирован"
    + r"|" + _W + r"пар(?:а|ы|у|ой|е|ам|ами|ах)?" + _WE
    # Рука словом-классом: `Jx`, `AK`, `A5s`, `98s` — с буквой; голые цифры («33»
    # из «33.3%») — не рука.
    + r"|(?<![\w.])(?-i:[2-9TJQKA]x|[TJQKA][2-9TJQKA][so]?|[2-9][TJQKA][so]?|[2-9]{2}[so])"
    r"(?![\w])"
)
_HAND_RE = re.compile(_HAND, re.IGNORECASE)
# Формы глагола: прошедшее время — факт руки; не прошедшее (настоящее, будущее,
# инфинитив) — прогноз. Остальное в списке действий — существительные («рейз»,
# «коллом», «фолдах»).
_PAST_END = r"(?:[аяиыеуоё]л|[аяиыеуо]л[аио]|[аяиыеу]л(?:ся|ась|ось|ись)|ш[её]л)"
_PAST_WORD_RE = re.compile(r"[а-яё]{2,}" + _PAST_END + _WE, re.IGNORECASE)
_PAST_END_RE = re.compile(_PAST_END + r"$", re.IGNORECASE)
_VERB_END_RE = re.compile(
    r"(?:ть|ться|ти|[еиё]т|[уюая]т|[еиё]тся|[уюая]тся|[еиё]шь|[еи]м|[еи]те|ст)$", re.IGNORECASE
)
_INFINITIVE_END_RE = re.compile(r"(?:ть|ться|ти)$", re.IGNORECASE)
_WORD_RE = re.compile(r"[а-яё-]+", re.IGNORECASE)
_HERO_SUBJECT_RE = re.compile(_W + r"(?:вы|у\s+вас)" + _WE, re.IGNORECASE)
# Соперник подлежащим (именительный падеж): придаточное с ним не освобождается «вы»
# («Если вы поставите он сбросит»). «Его рейз» в придаточном о герое — не подлежащее.
_OPPONENT_SUBJECT_RE = re.compile(
    _W + r"(?:он|они)" + _WE
    + r"|" + _W + r"(?:соперник|оппонент|противник|регуляр)(?:и|ы)?" + _WE
    + r"|" + _W + r"поле" + _WE
    + r"|так(?:ие|ой)\s+игрок"
    + r"|" + _POSITION,
    re.IGNORECASE,
)
# Фолд, сброс и защита существительным — не действие соперника только после
# предлога («на одних фолдах», «порог по фолдам») или в части о прибыльности ставки,
# и никогда при «бы», будущем или наречии повторяемости в предложении (раунды 5–6).
# Иначе существительное — действие, как «рейз» («он сделает фолд»).
_FOLD_NOUN_RE = re.compile(
    r"(?:фолд|сброс)(?:а|у|ом|е|ы|ов|ам|ами|ах)?|защит(?:а|ы|е|у|ой|ам|ами|ах)?",
    re.IGNORECASE,
)
# Предлог прямо перед существительным или через одно прилагательное.
_ADJECTIVE = r"[а-яё]+(?:ых|их|ым|им|ыми|ими|ой|ей|ом|ем|ую|юю|ая|яя|ое|ее|ые|ие|ий|ый)"
_PREPOSITION_BEFORE_RE = re.compile(
    _W + r"(?:на|по|от|при|без|для)\s+(?:" + _ADJECTIVE + r"\s+)?$", re.IGNORECASE
)
# Будущее время — узко, по формам из решения: «будет…», «сделает…».
_FUTURE_RE = re.compile(
    _W + r"(?:буд(?:у|ешь|ет|ем|ете|ут)|сдела(?:ю|ешь|ет|ем|ете|ют))" + _WE, re.IGNORECASE
)
# Части придаточного: время глагола освобождает существительное («рейз») только в
# своей части («вы поставили и он сделает рейз» — прогноз).
_PART_SPLIT_RE = re.compile(r"\s(?:и|а|но)\s", re.IGNORECASE)
# Прошедшее время — не факт руки, если в предложении «бы» (сослагательное — прогноз;
# «чтобы» не в счёт) или наречие повторяемости (обобщение, а не ход руки).
_HYPOTHETICAL_RE = re.compile(
    _W + r"(?:бы|всегда|обычно|часто|редко|раньше|постоянно|регулярно|никогда"
    r"|как\s+правило)" + _WE,
    re.IGNORECASE,
)
_HERO_HOLDS_RE = re.compile(_W + r"вы\s+(?:[^\s,.;:!?]+\s+)?держ[а-яё]*", re.IGNORECASE)
_CLAUSE_SPLIT_RE = re.compile(r"[,;:—–()]")
# «При» — условие только перед долей со словом порога («при 24% фолдов», «при 19.4%
# эквити»); «при такой ставке» — предлог, не условие.
_PRI_SHARE = (
    r"при\s+\d+(?:[.,]\d+)?\s*%\s+(?:его\s+)?(?:фолд|сброс|защит|эквити|диапазон|рендж)"
    r"[а-яё]*"
)
_CONDITION_RE = re.compile(_W + r"(?:(?:если|когда)" + _WE + r"|" + _PRI_SHARE + r")", re.IGNORECASE)
# Вставка между запятыми, которую условие пропускает («по расчёту», «судя по
# выходу»); остальное между запятыми — другое придаточное.
_INSERT_RE = re.compile(
    r"\s*(?:по|судя|как|согласно|видимо|похоже|вероятно|кажется)" + _WE, re.IGNORECASE
)
_MODAL_RE = re.compile(
    _W + r"(?:достаточно|нужно|надо|должн[а-яё]*|требуется|необходимо|можно)" + _WE,
    re.IGNORECASE,
)
# Опора на расчёт (правки контроллера, раунды 2–3): действие, за которым через пять
# слов или меньше стоит доля или «N из M». Опора — измеренная частота из расчёта
# (`frequency_numbers`: доля и «N из M» частоты или измеренной величины
# порога); порог или доля защиты (`calc_numbers`) опорой служат только арифметике
# порога — в придаточном с «если/когда», «при N% фолдов/эквити» или с
# «достаточно/нужно/должен» и инфинитивом. Не опора: доля банка («33.3% банка»), окно со ставкой героя глаголом
# («если вы поставите 25%») и окно с рукой или диапазоном («с любой парой в 60% раз»).
_NUMBER = r"\d+(?:[.,]\d+)?"
_CALC_NUMBER = r"(?P<pct>" + _NUMBER + r")\s*%|(?P<num>\d+)\s+из\s+(?P<den>\d+)"
_TAIL = r"(?P<tail>\s+(?:его\s+)?(?:диапазон|рендж|фолд|эквити|случа|раз)[а-яё]*)?"
_BACKED_RE = re.compile(
    r"(?:" + _ACTION + r")[^\s.,;:!?]*(?P<window>(?:[\s,]+[^\s.;:!?]+){0,5}?)[\s,]+(?:"
    + _CALC_NUMBER + r")" + _TAIL + r"(?:\s*\((?:" + _CALC_NUMBER.replace("?P<", "?P<p_")
    + r")\))?",
    re.IGNORECASE,
)
_SHARE_OF_RANGE_RE = re.compile(
    r"(?P<pct>" + _NUMBER + r")\s*%\s+(?:его\s+)?(?:диапазон|рендж|фолд)[а-яё]*", re.IGNORECASE
)
_POT_AFTER_RE = re.compile(r"\s*банк", re.IGNORECASE)
_MORE_LESS_RE = re.compile(_W + r"(?:чаще|реже)" + _WE, re.IGNORECASE)
# Ставка героя глаголом («если вы поставите 25%»); «продолженную ставку» — имя
# стата, не ставка.
_BET_WORD_RE = re.compile(r"постав|став(?:ит|ишь|лю|им|ят)", re.IGNORECASE)
_RANGE_WORD_RE = re.compile(r"диапазон|рендж|держ", re.IGNORECASE)
_META_RANGE_RE = re.compile(
    r"(?:про|о|об)\s+(?:его\s+|их\s+)?(?:диапазон|рендж)[а-яё]*(?:\s+[^\s.,;:!?]+){0,2}?"
    r"\s+(?:мало|ничего\s+не|почти\s+ничего\s+не)\s+говорит"
    r"|(?:мало|ничего\s+не|почти\s+ничего\s+не)\s+говорит\s+(?:про|о|об)\s+"
    r"(?:его\s+|их\s+)?(?:диапазон|рендж)[а-яё]*"
    r"|расч[её]т[а-яё]*\s+диапазон[а-яё]*",
    re.IGNORECASE,
)


def _in(text: str, numbers: frozenset[float]) -> bool:
    return round(float(text.replace(",", ".")), 1) in {round(n, 1) for n in numbers}


def _numbers_of(match: re.Match[str]) -> list[str]:
    if match.group("pct") is not None:
        return [match.group("pct")]
    return [match.group("num"), match.group("den")]


def _clause_start(text: str, position: int) -> int:
    starts = [m.end() for m in _CLAUSE_SPLIT_RE.finditer(text, 0, position)]
    return starts[-1] if starts else 0


def _word_at(text: str, position: int) -> str:
    match = _WORD_RE.match(text, position)
    return match.group(0) if match else ""


def _verb_like(word: str) -> bool:
    return bool(_VERB_END_RE.search(word) or _PAST_END_RE.search(word))


def _condition_holds(segment: str) -> bool:
    """Между словом условия и действием только подлежащее защищающегося и вставки:
    подлежащее-герой («если вы поставите»), глагол или другое придаточное разрывают
    опору (`test_a_hero_subject_or_another_clause_breaks_the_threshold_condition`)."""
    if _HERO_SUBJECT_RE.search(segment) or re.search(r"[;:—–()]", segment):
        return False
    if any(_verb_like(word) for word in _WORD_RE.findall(segment)):
        return False
    pieces = segment.split(",")
    if len(pieces) == 1:
        return True
    first, *middle, last = pieces
    if not all(_INSERT_RE.match(piece) for piece in middle):
        return False
    return not (_WORD_RE.search(first) and _WORD_RE.search(last))


def _backed(
    match: re.Match[str],
    text: str,
    calc_numbers: frozenset[float],
    frequency_numbers: frozenset[float],
) -> bool:
    """Действие держится на расчёте: измеренной частотой — всегда; порогом или долей
    защиты — только как арифметика порога (условие или модальное слово с
    инфинитивом в том же придаточном; голое «при» — не условие,
    `test_a_bare_preposition_pri_is_not_a_condition`). Между условием и действием —
    только подлежащее защищающегося (`_condition_holds`). В предложении о
    прибыльности с «чаще/реже» перед долей условие ищется во всём предложении до
    действия, через запятые — только вставки
    (`test_threshold_math_after_a_profit_word_is_exempt_across_commas`,
    `test_across_commas_only_the_more_or_less_than_threshold_form_is_exempt`)."""
    if _BET_WORD_RE.search(match.group("window")) or _HAND_RE.search(match.group("window")):
        return False
    if (
        match.group("pct") is not None
        and not match.group("tail")
        and _POT_AFTER_RE.match(text, text.index("%", match.end("pct")) + 1)
    ):
        return False
    numbers = _numbers_of(match)
    if all(_in(n, frequency_numbers) for n in numbers):
        return True
    if not all(_in(n, calc_numbers) for n in numbers):
        return False
    clause = text[_clause_start(text, match.start()) : match.end()]
    # «X окупается…, если <подлежащее> сбрасывает чаще N%» — условие ищется во всём
    # предложении до действия, через запятые со вставками («по расчёту»); без слова
    # прибыльности или без «чаще/реже» перед долей — только в своём придаточном
    # (раунды 5–6).
    across = bool(_PROFIT_RE.search(text)) and bool(_MORE_LESS_RE.search(match.group("window")))
    head_start = 0 if across else _clause_start(text, match.start())
    head = text[head_start : match.start()]
    conditions = list(_CONDITION_RE.finditer(head))
    if conditions and _condition_holds(head[conditions[-1].end() :]):
        return True
    infinitive = bool(_INFINITIVE_END_RE.search(_word_at(text, match.start())))
    return infinitive and bool(_MODAL_RE.search(clause))


def _strip_backed(
    sentence: str, calc_numbers: frozenset[float], frequency_numbers: frozenset[float]
) -> str:
    """Предложение без того, что держится на расчёте, и без метаречи."""
    rest = _META_RANGE_RE.sub(" ", sentence)
    changed = True
    while changed:
        changed = False
        for anchor in _ACTION_RE.finditer(rest):
            match = _BACKED_RE.match(rest, anchor.start())
            if match is None or not _backed(match, rest, calc_numbers, frequency_numbers):
                continue
            rest = rest[: match.start()] + " " + rest[match.end():]
            changed = True
            break
    return _SHARE_OF_RANGE_RE.sub(
        lambda m: " " if _in(m.group("pct"), calc_numbers) else m.group(0), rest
    )


def _opponent_clause(clause: str, hypothetical: bool, noun_locked: bool) -> bool:
    """Придаточное утверждает, что соперник держит или сделает: слово о нём и рука
    или действие в форме прогноза. Придаточное с «вы» без соперника подлежащим — о
    герое (`test_a_clause_with_an_opponent_subject_is_checked_despite_you`). Глагол
    прошедшего времени — факт руки, существительное («рейз», «коллом») в одной части
    с ним (части делят «и», «а», «но») — тоже
    (`test_hand_narration_is_not_an_opponent_claim`,
    `test_a_noun_action_after_and_is_not_covered_by_a_past_verb_before_it`); при
    `hypothetical` («бы» или наречие повторяемости в предложении) прошедшее время
    ничего не освобождает (`test_the_subjunctive_is_a_prediction_not_a_fact_of_the_hand`,
    `test_a_habit_in_the_past_is_a_claim_not_a_fact_of_the_hand`). «Фолд», «сброс»,
    «защита» существительным — не действие только после предлога или в части о
    прибыльности и не при `noun_locked` («бы», будущее, наречие повторяемости;
    `test_a_fold_or_defence_noun_after_a_preposition_is_not_an_opponent_action`,
    `test_a_fold_or_defence_noun_outside_a_preposition_is_an_action`,
    `test_the_subjunctive_future_or_habit_cancels_the_noun_exemption`); рука в части о
    прибыльности ставки — не утверждение о сопернике
    (`test_a_hand_word_in_a_profit_clause_is_about_the_bet`)."""
    if not _OPPONENT_RE.search(clause):
        return False
    if _HERO_SUBJECT_RE.search(clause) and not _OPPONENT_SUBJECT_RE.search(clause):
        return False
    parts = _PART_SPLIT_RE.split(clause)
    # Часть о прибыльности ставки («он окупается против любой пары», «выше него она
    # окупается») — о ставке, а не о сопернике: «он» там — ставка, и рука рядом не
    # утверждение о нём (раунд 5). Глагол соперника в ней — по-прежнему прогноз.
    claim = " ".join(part for part in parts if not _PROFIT_RE.search(part))
    if _OPPONENT_RE.search(claim) and _HAND_RE.search(claim):
        return True
    for part in parts:
        narrated = not hypothetical and bool(_PAST_WORD_RE.search(part))
        for action in _ACTION_RE.finditer(part):
            # Слово целиком — от начала совпадения назад до границы.
            start = action.start()
            while start > 0 and part[start - 1].isalpha():
                start -= 1
            word = _word_at(part, start)
            if (
                _FOLD_NOUN_RE.fullmatch(word)
                and not noun_locked
                and (_PREPOSITION_BEFORE_RE.search(part[:start]) or _PROFIT_RE.search(part))
            ):
                continue
            if _PAST_END_RE.search(word) and not hypothetical:
                continue
            if _VERB_END_RE.search(word) or not narrated:
                return True
    return False


def _opponent_claim(
    sentence: str, calc_numbers: frozenset[float], frequency_numbers: frozenset[float]
) -> bool:
    """Предложение требует «(допущение)»: после опоры на расчёт и метаречи в нём
    осталось слово диапазона («вы держите» — о герое и не в счёт;
    `test_a_range_word_needs_the_mark_by_default`) или придаточное с явным
    утверждением о сопернике
    (`test_a_claim_about_what_the_opponent_holds_or_does_needs_the_mark`).
    Остальное — судья (спека №2)."""
    rest = _strip_backed(sentence, calc_numbers, frequency_numbers)
    if _RANGE_WORD_RE.search(_HERO_HOLDS_RE.sub(" ", rest)):
        return True
    hypothetical = bool(_HYPOTHETICAL_RE.search(rest))
    noun_locked = hypothetical or bool(_FUTURE_RE.search(rest))
    return any(
        _opponent_clause(clause, hypothetical, noun_locked)
        for clause in _CLAUSE_SPLIT_RE.split(rest)
    )


# Лучшая линия на флопе и тёрне (решение владельца 2026-10-13). «Стоило» убрано:
# это слово вопроса игрока («стоило ставить?»), а не вердикт.
_BEST_LINE_RE = re.compile(
    r"лучш[а-яё]*\s+лини|лучше было|было лучше|лучше бы|правильн[а-яё]*\s+было"
    r"|правильн[а-яё]*\s+лини|следовало|надо было|нужно было"
    r"|(?<![A-Za-z])GTO(?![A-Za-z])|(?<![А-Яа-яЁё])ГТО(?![А-Яа-яЁё])",
    re.IGNORECASE,
)
# Вывод о прибыльности: голый — только по решённому порогу в разговоре.
_PROFIT_RE = re.compile(_W + r"в\s+(?:плюсе|минусе)(?![а-яё])|окуп[а-яё]*", re.IGNORECASE)
_PROFIT_CONDITION_RE = re.compile(
    _W + r"(?:(?:если|от|когда|чтобы)" + _WE + r"|" + _PRI_SHARE + r")", re.IGNORECASE
)
_CLAUSE_CONDITION_RE = re.compile(
    r"\s*(?:(?:если|когда|чтобы)" + _WE + r"|" + _PRI_SHARE + r")", re.IGNORECASE
)
_PCT_RE = re.compile(r"(?P<pct>" + _NUMBER + r")\s*%")
# «Окупается / в плюсе (и) без фолдов» — признак ядра `semibluff_free`, без «не» перед.
_WITHOUT_FOLDS_RE = re.compile(
    r"(?<!не\s)" + _W + r"(?:окуп[а-яё]*|в\s+плюсе" + _WE + r")\s+(?:и\s+)?без\s+фолд[а-яё]*",
    re.IGNORECASE,
)


def _profit_without_folds(sentence: str) -> bool:
    """Каждый вывод о прибыльности в предложении — «окупается / в плюсе (и) без
    фолдов» (`test_the_without_folds_flag_frees_only_that_conclusion`)."""
    free = [m.start() for m in _WITHOUT_FOLDS_RE.finditer(sentence)]
    return bool(free) and all(m.start() in free for m in _PROFIT_RE.finditer(sentence))


def _calc_share_in(clause: str, calc_numbers: frozenset[float]) -> bool:
    return any(
        _in(m.group("pct"), calc_numbers) and not _POT_AFTER_RE.match(clause, m.end())
        for m in _PCT_RE.finditer(clause)
    )


def _profit_arithmetic(sentence: str, calc_numbers: frozenset[float]) -> bool:
    """«Окупается от 19.4% эквити» — арифметика, а не вывод, по придаточным: в
    одном из них условие («если/от/когда/чтобы», «при» только перед долей со словом
    порога) и доля из выхода расчёта (не доля банка), а свободных придаточных — без
    доли расчёта и не начатых условием — нет («…, тут так и есть» — вывод;
    `test_a_bare_profit_conclusion_with_some_share_still_needs_a_threshold`,
    `test_profit_arithmetic_with_a_calc_share_is_allowed`,
    `test_a_profit_conclusion_with_a_bare_pri_needs_a_threshold`)."""
    clauses = [c for c in _CLAUSE_SPLIT_RE.split(sentence) if c.strip(" .!?")]
    conditional = [
        c for c in clauses
        if _PROFIT_CONDITION_RE.search(c) and _calc_share_in(c, calc_numbers)
    ]
    purpose = [
        c for c in clauses
        if re.match(r"\s*чтобы" + _WE, c, re.IGNORECASE) and _PROFIT_RE.search(c)
    ]
    if not conditional and not (purpose and any(_calc_share_in(c, calc_numbers) for c in clauses)):
        return False
    return all(
        _PROFIT_RE.search(c)
        or _calc_share_in(c, calc_numbers)
        or _CLAUSE_CONDITION_RE.match(c)
        for c in clauses
    )


# Отрицание перед термином: «без флеш-дро», «нет …», «не …», «ни …».
_NEGATION_BEFORE_RE = re.compile(_W + r"(?:без|нет|не|ни)\s+$", re.IGNORECASE)
_FLOP_TURN_RE = re.compile(r"флоп|т[её]рн", re.IGNORECASE)
_LEAK_RE = re.compile(r"(?<![а-яё])лик(?:и|ов|ам|ами|ах|а|ом|у|е)?(?![а-яё])", re.IGNORECASE)

# Метки — с пространством имён: `HandCategory.FLUSH` и `DrawKind.FLUSH` — одна
# строка `"flush"`, и без префикса «флеш-дро» прошёл бы на готовом флеше.
# `hand:` — `HandCategory`, `draw:` — `DrawKind`, `action:` — `ActionTag`
# (контекст разбора, `agent_context`); `stat:` — `FrequencyStat` (выход расчёта,
# метку собирает вызывающий).
HAND_LABEL = "hand:"
DRAW_LABEL = "draw:"
ACTION_LABEL = "action:"
STAT_LABEL = "stat:"

# Термин → метки, при которых он допустим.
TERM_LABELS: dict[str, tuple[re.Pattern[str], frozenset[str]]] = {
    "c-бет": (
        re.compile(r"[cс]-?бет|контбет", re.IGNORECASE),
        frozenset({"action:cbet", "action:delayed_cbet", "stat:cbet_flop", "stat:fold_to_cbet"}),
    ),
    "донк": (re.compile(r"(?<![а-яё])донк", re.IGNORECASE), frozenset({"action:donk"})),
    "проба": (
        re.compile(r"(?<![а-яё])проб(?:а|у|ой|ы|е)(?![а-яё])", re.IGNORECASE),
        frozenset({"action:probe"}),
    ),
    "баррель": (
        re.compile(r"баррел", re.IGNORECASE),
        frozenset({"action:barrel", "stat:barrel_turn", "stat:barrel_river"}),
    ),
    "чек-рейз": (re.compile(r"чек-?рейз", re.IGNORECASE), frozenset({"action:check_raise"})),
    "сет": (
        re.compile(r"(?<![а-яё])сет(?:а|ом|у|ы)?(?![а-яё])", re.IGNORECASE),
        frozenset({"hand:set"}),
    ),
    "оверпара": (re.compile(r"оверпар", re.IGNORECASE), frozenset({"hand:overpair"})),
    "топ-пара": (
        re.compile(r"топ-?пар", re.IGNORECASE),
        frozenset({"hand:top_pair_strong_kicker", "hand:top_pair_weak_kicker"}),
    ),
    "гатшот": (
        re.compile(r"гатшот", re.IGNORECASE),
        frozenset({"draw:gutshot", "draw:double_gutshot"}),
    ),
    "флеш-дро": (re.compile(r"флеш-?дро", re.IGNORECASE), frozenset({"draw:flush"})),
}


_STREET_NAME: dict[Street, str] = {
    Street.PREFLOP: "префлоп",
    Street.FLOP: "флоп",
    Street.TURN: "тёрн",
    Street.RIVER: "ривер",
}


_DRAW_WORD: dict[DrawKind, str] = {
    DrawKind.FLUSH: "флеш",
    DrawKind.OPEN_ENDED: "двусторонний стрит",
    DrawKind.DOUBLE_GUTSHOT: "двойной гатшот",
    DrawKind.GUTSHOT: "гатшот",
}
_BACKDOOR_WORD = {"flush": "флеш", "straight": "стрит"}

# Слова назначения — те же, что в разборе (`presentation.messages._PURPOSE_WORD`);
# копия, потому что `explanation` не импортирует `presentation`
# (`test_the_purpose_words_are_the_words_of_the_hand_breakdown`).
_PURPOSE_WORD: dict[Purpose, str] = {
    Purpose.VALUE: "вэлью",
    Purpose.MEDIUM_HAND: "ставка со средней рукой",
    Purpose.SEMIBLUFF: "полублеф",
    Purpose.BLUFF: "блеф",
    Purpose.CALL_STRONG: "колл с сильной рукой",
    Purpose.BLUFF_CATCH: "ловля блефа",
    Purpose.CALL_WITH_DRAW: "колл с дро",
}


def _outs_word(count: int) -> str:
    if count % 10 == 1 and count % 100 != 11:
        return "аут"
    if count % 10 in (2, 3, 4) and count % 100 not in (12, 13, 14):
        return "аута"
    return "аутов"


def _approx(share: float, book: NumberBook) -> str:
    """Константа владельца (бэкдоры, оверкарты) — с «≈», как в разборе."""
    return f"≈{book.pct(100.0 * share, trim=True)}%"


def _hero_equity_line(line: PostflopLineDetail, street: Street, book: NumberBook) -> str | None:
    """Дро, бэкдоры и оверкарты героя из посчитанного разбора — одной строкой, числа
    через книгу (`test_the_context_carries_the_hero_draws_backdoors_and_overcards`).
    Нет ни одного — строки нет."""
    parts: list[str] = []
    draw = line.draw
    if draw is not None:
        kinds = " + ".join(_DRAW_WORD[kind] for kind in draw.kinds)
        outs = len(draw.outs)
        chance = f"собрать на следующей улице {book.pct(100.0 * draw.hit_next, trim=True)}%"
        if draw.hit_by_river is not None:
            chance += f", к риверу {book.pct(100.0 * draw.hit_by_river, trim=True)}%"
        parts.append(f"дро — {kinds}, {book.count(outs)} {_outs_word(outs)}, {chance}")
    if line.backdoors:
        named = ", ".join(
            f"{_BACKDOOR_WORD[b.kind]} {_approx(b.approx, book)}" for b in line.backdoors
        )
        parts.append(f"бэкдор — {named}")
    over = line.overcards
    if over is not None:
        chance = (
            f"{_approx(over.approx_by_river, book)} к риверу"
            if over.approx_by_river is not None
            else f"{_approx(over.approx_next, book)} на следующей улице"
        )
        parts.append(f"оверкарты — {cards_text(over.cards)}, {chance}, если пара будет лучшей")
    if not parts:
        return None
    return f"Рука героя, {_STREET_NAME[street]}: {'; '.join(parts)}."


def _purpose_line(
    line: PostflopLineDetail, street: Street, book: NumberBook, thresholds: NumberBook
) -> str | None:
    """Назначение и порог фолдов героя в точке — из ядра (`postflop_line`, §4.5–§4.6).
    Числа порога пишутся в обе книги: они числа сводки и опора арифметики порога
    (`AgentContext.threshold_numbers`;
    `test_the_context_names_the_purpose_and_the_fold_threshold_of_a_hero_bet`)."""
    played = line.line
    if played is None or played.purpose is None:
        return None
    text = f"Назначение, {_STREET_NAME[street]}: {_PURPOSE_WORD[played.purpose]}"
    fold = line.fold_threshold

    def pct(share: float) -> str:
        thresholds.pct(100.0 * share, trim=True)
        return book.pct(100.0 * share, trim=True)

    if fold is not None:
        if fold.semibluff_free:
            text += (
                f"; порог фолдов как у чистого блефа {pct(fold.bluff)}%, "
                "с учётом аутов ставка окупается и без фолдов"
            )
        elif fold.semibluff is not None:
            text += (
                f"; порог фолдов как у чистого блефа {pct(fold.bluff)}%, "
                f"с учётом аутов {pct(fold.semibluff)}%"
            )
        else:
            text += f"; ставка окупается, если соперник сбрасывает чаще {pct(fold.bluff)}%"
    return text + "."


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Сводка последнего разбора для агента: текст, разрешённые числа, метки
    классификатора, класс руки героя и ранги борда (для вырезания карт).
    `threshold_numbers` — числа порога фолдов из ядра, напечатанные в сводке:
    опора арифметики порога
    (`test_a_threshold_from_the_summary_backs_profit_arithmetic`).
    `profit_without_folds` — хоть у одной точки ядро дало `semibluff_free`: вывод
    «окупается без фолдов» — факт сводки
    (`test_a_semibluff_that_pays_without_folds_may_be_retold_from_the_summary`)."""

    text: str
    allowed: frozenset[float]
    labels: frozenset[str]
    hero_class: str | None
    board_ranks: tuple[str, ...]
    threshold_numbers: frozenset[float] = frozenset()
    profit_without_folds: bool = False


def agent_context(result: AnalysisResult, enriched: EnrichedHand) -> AgentContext:
    """Сводка разбора: ход руки (реплей, код), судимые точки (выжимка вердикта),
    метки классификатора всех точек и дро, бэкдоры, оверкарты героя из них,
    назначение ставки героя и порог фолдов
    (`test_the_context_names_the_purpose_and_the_fold_threshold_of_a_hero_bet`).
    Разрешены числа реплея без обозначений
    карт: ранг карты героя или борда числом не становится
    (`test_a_card_rank_of_the_hand_is_not_an_allowed_number`). Идентификаторы в
    заголовке точки выжимки не разрешены, как в выжимке вердикта
    (`test_the_context_registers_every_number_it_prints`). Без судимых точек
    выжимки вердикта нет — только номер раздачи и это словами, без служебного
    «судимых точек 0» (`test_an_unjudged_hand_prints_no_service_count`)."""
    hand = enriched.hand
    cards = hand.dealt.get(hand.hero_label, [])
    hero_class = class_of(cards[0], cards[1]) if len(cards) == 2 else None
    # Порядок объявления `Street`, как у реплея: `sorted` по строке дал бы
    # «flop, river, turn».
    board = tuple(card[0] for street in Street for card in hand.boards.get(street, []))
    book = NumberBook()
    replay = hand_replay(enriched).plain
    # Регистрируется копия без карт; модели показывается реплей как есть.
    book.prose(strip_card_tokens(replay, hero_class=hero_class, board_ranks=board))
    if result.ranked:
        verdict = verdict_digest(result)
        verdict_text, verdict_allowed = verdict.text, verdict.allowed
    else:
        verdict_text = (
            f"Раздача {book.token(result.hand_no)}.\nРешений с вердиктом в этой раздаче нет."
        )
        verdict_allowed = frozenset()
    labels: set[str] = set()
    label_lines: list[str] = []
    equity_lines: list[str] = []
    thresholds = NumberBook()
    purpose_lines: list[str] = []
    without_folds = False
    for point in result.points:
        line = postflop_line_detail(point)
        if line is None:
            continue
        found = []
        if line.hand is not None:
            found.append(HAND_LABEL + line.hand.category.value)
        if line.draw is not None:
            found.extend(DRAW_LABEL + kind.value for kind in line.draw.kinds)
        if line.line is not None:
            found.append(ACTION_LABEL + line.line.action.value)
        labels.update(found)
        label_lines.append(
            f"Метки классификатора, {_STREET_NAME[point.street]}: {', '.join(found)}."
        )
        equity = _hero_equity_line(line, point.street, book)
        if equity is not None:
            equity_lines.append(equity)
        purpose = _purpose_line(line, point.street, book, thresholds)
        without_folds |= line.fold_threshold is not None and line.fold_threshold.semibluff_free
        if purpose is not None:
            purpose_lines.append(purpose)
    # Две точки одной улицы с одной рукой дали бы одну строку дважды.
    equity_text = "\n".join(dict.fromkeys(equity_lines))
    text = "\n\n".join(
        part
        for part in (
            replay, verdict_text, "\n".join(label_lines), "\n".join(purpose_lines), equity_text
        )
        if part
    )
    return AgentContext(
        text=text,
        allowed=book.allowed | verdict_allowed,
        labels=frozenset(labels),
        hero_class=hero_class,
        board_ranks=board,
        threshold_numbers=thresholds.allowed,
        profit_without_folds=without_folds,
    )


def strip_card_tokens(
    text: str, *, hero_class: str | None = None, board_ranks: Sequence[str] = ()
) -> str:
    """Текст без обозначений карт — то, что идёт в сверку чисел.

    Вырезается только однозначное: карта с мастью, класс с буквой (`A5s`,
    `T9o`, `KQ`, `98s`), ранги борда через дефис. Две цифры без буквы и масти
    (`22`) остаются числом всегда. `hero_class` сохранён ради интерфейса
    (`AgentContext.hero_class`, `check_agent_answer`) и на вырезание не влияет.
    """
    text = _SUITED_CARD_RE.sub(" ", text)
    board = Counter(board_ranks)

    def board_sub(match: re.Match[str]) -> str:
        ranks = ["T" if rank == "10" else rank for rank in re.findall(_RANK, match.group(0))]
        has_letter = any(rank in "TJQKA" for rank in ranks)
        if (board and not (Counter(ranks) - board)) or (has_letter and not board):
            return " "
        return match.group(0)

    def class_sub(match: re.Match[str]) -> str:
        return match.group(0) if _DIGITS_ONLY_RE.fullmatch(match.group(0)) else " "

    return _CLASS_RE.sub(class_sub, _BOARD_RE.sub(board_sub, text))


def _says_term(pattern: re.Pattern[str], text: str) -> bool:
    """Термин сказан хотя бы раз без отрицания перед ним."""
    return any(
        not _NEGATION_BEFORE_RE.search(text[: match.start()]) for match in pattern.finditer(text)
    )


def check_agent_answer(
    answer: str,
    *,
    allowed: frozenset[float],
    labels: frozenset[str],
    leaks_called: bool,
    hero_class: str | None = None,
    board_ranks: Sequence[str] = (),
    threshold_decided: bool = False,
    profit_without_folds: bool = False,
    calc_numbers: frozenset[float],
    frequency_numbers: frozenset[float],
) -> list[str]:
    """Нарушения ответа словами; пустой список — ответ прошёл (спека §6).

    `threshold_decided` — в разговоре есть `frequency_vs_threshold` с выводом выше
    или ниже порога: только тогда голый вывод «в плюсе» / «в минусе» допустим.
    `profit_without_folds` — `AgentContext.profit_without_folds`: тогда допустим и
    вывод «окупается / в плюсе (и) без фолдов», если других выводов в предложении нет.
    `calc_numbers` — числа из выходов расчётов разговора и числа порога сводки
    (`threshold_numbers` — их добавляет вызывающий), без прочих чисел сводки и
    слов игрока: опора арифметики порога и «окупается от N%». `frequency_numbers` —
    из них измеренные частоты (доля и «N из M» частоты и измеренной величины
    порога): только они служат опорой действию соперника вне арифметики порога.
    """
    text = answer.strip()
    if not text:
        return ["пустой ответ"]

    def cleaned(part: str) -> str:
        return strip_card_tokens(part, hero_class=hero_class, board_ranks=board_ranks)

    violations: list[str] = []
    invented = unsupported_numbers(cleaned(text), allowed)
    if invented:
        violations.append(f"числа не из расчёта и не из разбора — {invented}")
    for sentence in _SENTENCE_RE.findall(text):
        quoted = sentence.strip()
        if (
            _opponent_claim(sentence, calc_numbers, frequency_numbers)
            and ASSUMPTION_MARK not in sentence
        ):
            violations.append(
                f"утверждение о том, что соперник держит или сделает, без метки "
                f"{ASSUMPTION_MARK} — «{quoted}»"
            )
        if _BEST_LINE_RE.search(sentence) and _FLOP_TURN_RE.search(sentence):
            violations.append(f"лучшая линия на флопе или тёрне — «{quoted}»")
        if (
            _PROFIT_RE.search(sentence)
            and not _profit_arithmetic(sentence, calc_numbers)
            and not threshold_decided
            and not (profit_without_folds and _profit_without_folds(sentence))
        ):
            violations.append(
                "«в плюсе», «в минусе» или «окупается» выводом без расчёта "
                f"frequency_vs_threshold с выводом против порога — «{quoted}»"
            )
    for term, (pattern, needed) in TERM_LABELS.items():
        if _says_term(pattern, text) and not (needed & labels):
            violations.append(f"термин «{term}» без такой метки классификатора")
    if _LEAK_RE.search(text) and not leaks_called:
        violations.append("слово «лик» без расчёта leaks")
    errors = error_words_in(text)
    if errors:
        violations.append(f"решение названо ошибкой — {errors}")
    return violations
