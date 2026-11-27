"""Recognize explicitly typed range notation, never infer a range from a player name."""
import re

from harness.analysis.charts import parse_range, to_notation

_TOKEN = re.compile(r'(?<!\w)([AKQJT98765432]{2}[so]?(?:\+|-[AKQJT98765432]{2}[so]?)?(?::(?:0(?:\.\d+)?|1(?:\.0+)?))?)(?!\w)',re.IGNORECASE)
_MARKER = re.compile(r'диапазон\w*|рендж\w*|\brange\b|\bпротив\b',re.IGNORECASE)


def canonical_literal(text: str) -> str:
    """Existing grammar; bare unpaired AK is shorthand for AKs, AKo."""
    parts=[]
    for token in re.split(r'[,\s]+',text.strip()):
        if not token:
            continue
        if re.fullmatch(r'[AKQJT98765432]{2}',token,re.IGNORECASE) and token[0].upper()!=token[1].upper():
            parts.extend([token.upper()+'s',token.upper()+'o'])
        else:
            parts.append(token)
    return to_notation(parse_range(','.join(parts)))


def user_literal_ranges(texts: list[str]) -> frozenset[str]:
    found=set()
    for text in texts:
        if re.search(r'по\s+чарту|суж\w*\s+чарт', text, re.IGNORECASE):
            continue
        marker=_MARKER.search(text)
        if marker is None:
            continue
        tokens=_TOKEN.findall(text[marker.end():])
        if not tokens:
            continue
        try:
            found.add(canonical_literal(','.join(tokens)))
        except ValueError:
            continue
    return frozenset(found)
