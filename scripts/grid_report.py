"""Визуальный отчёт по регрессионной сетке: те же проверки, что в тесте, но глазами.

Тест отвечает «да/нет» на весь файл разом. Отчёт показывает результат по каждой
руке: где сошёлся банк, где сохранились фишки, сколько строк формата осталось
нераспознанными. Полезен, когда сетка КРАСНАЯ — видно, какие руки и по какому
именно утверждению упали, вместо первого assert'а.

Проверки повторяют tests/test_regression_grid.py и намеренно продублированы:
отчёт — инструмент чтения, а не источник истины. Гейт — тест.

Вывод содержит реальные руки игрока и поэтому пишется в reports/ (gitignore).

    uv run python scripts/grid_report.py [--open]
"""

from __future__ import annotations

import html
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from harness.contracts import EnrichedHand, RawHand
from harness.engine import enrich
from harness.normalizer import normalize
from harness.parsers.hh_parser import parse_file

ROOT = Path(__file__).parent.parent
FIXTURES = [
    ("Daily Classic", ROOT / "fixtures" / "hh" / "daily-classic-146.txt", 146),
    ("Bounty Hunters (PKO)", ROOT / "fixtures" / "hh" / "pko-bounty-172.txt", 172),
]
OUT = ROOT / "reports" / "grid.html"

CHECKS = [
    ("формат", "ни одной нераспознанной строки парсером"),
    ("валидатор", "verdict.status == pass"),
    ("банк", "банк движка == total_pot из SUMMARY"),
    ("фишки", "сумма стеков до == после (рейк 0)"),
    ("стеки", "ни один стек на конец руки не отрицателен"),
]


@dataclass
class Row:
    n: int
    hand_no: str
    ts: datetime
    level: int
    blinds: str
    players: int
    unknown: int
    status: str
    pot_engine: int
    pot_summary: int | None
    chips_start: int
    chips_end: int
    min_stack: int
    hero_points: int
    ok: dict[str, bool] = field(default_factory=dict)

    @property
    def all_ok(self) -> bool:
        return all(self.ok.values())


def build_rows(raws: list[RawHand], ens: list[EnrichedHand]) -> list[Row]:
    rows = []
    for i, (raw, en) in enumerate(zip(raws, ens, strict=True), start=1):
        s = en.hand.summary
        start = sum(p.stack for p in en.hand.players)
        end = sum(en.report.stacks_end.values())
        r = Row(
            n=i,
            hand_no=en.hand.hand_no,
            ts=en.hand.timestamp,
            level=en.hand.level,
            blinds=f"{en.hand.sb:,}/{en.hand.bb:,}"
            + (f" a{en.hand.ante:,}" if en.hand.ante else ""),
            players=len(en.hand.players),
            unknown=len(raw.unknown_lines),
            status=en.verdict.status.value,
            pot_engine=en.report.final_pot,
            pot_summary=s.total_pot if s else None,
            chips_start=start,
            chips_end=end,
            min_stack=min(en.report.stacks_end.values(), default=0),
            hero_points=len(en.report.decision_points),
        )
        r.ok = {
            "формат": r.unknown == 0,
            "валидатор": r.status == "pass",
            "банк": r.pot_summary is not None and r.pot_engine == r.pot_summary,
            "фишки": start == end,
            "стеки": r.min_stack >= 0,
        }
        rows.append(r)
    return rows


def chrono_ok(raws: list[RawHand]) -> bool:
    ts = [r.timestamp for r in raws]
    return ts == sorted(ts)


CSS = """
:root{--bg:#f7f7f5;--fg:#1c1b19;--muted:#6b6862;--line:#dedbd4;--card:#fff;
--ok:#1a7f4b;--ok-bg:#e6f4ec;--bad:#b3261e;--bad-bg:#fce8e6;--accent:#8a6a3a}
@media(prefers-color-scheme:dark){:root{--bg:#17181a;--fg:#e9e7e3;--muted:#9b978f;
--line:#2e3033;--card:#1e2022;--ok:#4ec98a;--ok-bg:#163325;--bad:#ff8a80;--bad-bg:#3a1b18}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:32px 20px 80px}
h1{font-size:22px;margin:0 0 4px;letter-spacing:-.01em}
.sub{color:var(--muted);margin:0 0 28px}
.cards{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:28px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:12px 16px;min-width:120px}
.card .v{font-size:22px;font-weight:600;font-variant-numeric:tabular-nums}
.card .k{color:var(--muted);font-size:12px;margin-top:2px}
.card.good .v{color:var(--ok)}
h2{font-size:15px;margin:32px 0 10px;font-weight:600}
h2 .n{color:var(--muted);font-weight:400}
.grid{display:flex;flex-wrap:wrap;gap:3px;margin:10px 0 18px}
.sq{width:14px;height:14px;border-radius:3px;background:var(--ok);opacity:.85}
.sq.bad{background:var(--bad);opacity:1}
.sq:hover{outline:2px solid var(--fg);outline-offset:1px}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{padding:6px 10px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}
th{position:sticky;top:0;background:var(--card);text-align:right;font-weight:600;
color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
th:first-child,td:first-child,th.l,td.l{text-align:left}
tbody tr:hover{background:color-mix(in srgb,var(--fg) 4%,transparent)}
td.num{font-variant-numeric:tabular-nums}
.chk{color:var(--ok);font-weight:600}
.chk.bad{color:var(--bad)}
tr.bad td{background:var(--bad-bg)}
.legend{margin:14px 0 0;color:var(--muted);font-size:12.5px}
.legend li{margin:2px 0}
.note{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--accent);
border-radius:8px;padding:14px 18px;margin-top:34px}
.note b{font-weight:600}
code{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;
background:color-mix(in srgb,var(--fg) 7%,transparent);padding:1px 5px;border-radius:4px}
"""


def render(sections: list[tuple[str, list[Row], bool, int, int]]) -> str:
    total = sum(len(rs) for _, rs, _, _, _ in sections)
    bad = sum(1 for _, rs, _, _, _ in sections for r in rs if not r.all_ok)
    hero_pts = sum(r.hero_points for _, rs, _, _, _ in sections for r in rs)
    parts: list[str] = []
    parts.append(f"""<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Регрессионная сетка — {total} рук</title><style>{CSS}</style><div class="wrap">
<h1>Регрессионная сетка: {total} реальных рук</h1>
<p class="sub">Гейт этапа B. Один прогон накрывает парсер → нормалайзер → движок → валидатор.
Сформировано {datetime.now(UTC).astimezone():%Y-%m-%d %H:%M} · источник истины — <code>tests/test_regression_grid.py</code></p>
<div class="cards">
<div class="card good"><div class="v">{total - bad}/{total}</div><div class="k">рук без замечаний</div></div>
<div class="card{' good' if bad == 0 else ''}"><div class="v">{bad}</div><div class="k">расхождений</div></div>
<div class="card"><div class="v">{len(CHECKS)}</div><div class="k">проверки на руку</div></div>
<div class="card"><div class="v">{hero_pts}</div><div class="k">точек решения Hero</div></div>
</div>""")

    for name, rows, chrono, expected, got in sections:
        squares = "".join(
            f'<div class="sq{"" if r.all_ok else " bad"}" title="#{r.n} · рука {html.escape(r.hand_no)} · '
            f'банк {r.pot_engine:,}"></div>'
            for r in rows
        )
        cnt_ok = "chk" if expected == got else "chk bad"
        chr_ok = "chk" if chrono else "chk bad"
        parts.append(f"""<h2>{html.escape(name)} <span class="n">— {got} рук
(ожидалось {expected}: <span class="{cnt_ok}">{"✓" if expected == got else "✗"}</span>) ·
хронологический порядок: <span class="{chr_ok}">{"✓" if chrono else "✗"}</span></span></h2>
<div class="grid">{squares}</div><div class="scroll"><table>
<thead><tr><th class="l">#</th><th class="l">рука</th><th class="l">время</th><th>ур.</th>
<th class="l">блайнды</th><th>игроков</th>""")
        parts.append("".join(f"<th>{c}</th>" for c, _ in CHECKS))
        parts.append("""<th>банк движка</th><th>SUMMARY</th><th>Δ</th>
<th>фишки до→после</th><th>мин. стек</th><th>точек Hero</th></tr></thead><tbody>""")
        for r in rows:
            d = r.pot_engine - (r.pot_summary or 0)
            checks = "".join(
                f'<td class="chk{"" if r.ok[c] else " bad"}">{"✓" if r.ok[c] else "✗"}</td>'
                for c, _ in CHECKS
            )
            parts.append(
                f'<tr class="{"" if r.all_ok else "bad"}">'
                f'<td class="l num">{r.n}</td><td class="l">{html.escape(r.hand_no)}</td>'
                f'<td class="l num">{r.ts:%d.%m %H:%M}</td><td class="num">{r.level}</td>'
                f'<td class="l num">{r.blinds}</td><td class="num">{r.players}</td>{checks}'
                f'<td class="num">{r.pot_engine:,}</td>'
                f'<td class="num">{r.pot_summary:,}</td><td class="num">{d:+,}</td>'
                f'<td class="num">{r.chips_start:,} → {r.chips_end:,}</td>'
                f'<td class="num">{r.min_stack:,}</td><td class="num">{r.hero_points}</td></tr>'
            )
        parts.append("</tbody></table></div>")

    parts.append('<ul class="legend">')
    parts.extend(f"<li><b>{c}</b> — {d}</li>" for c, d in CHECKS)
    parts.append("""</ul>
<div class="note"><b>Чего эта сетка не проверяет.</b> Ни одного вердикта: это гейт входного
тракта — рука прочитана, проиграна по правилам, деньги сошлись. Правильность разбора
проверяют другие тесты. Сохранение фишек значимо только потому, что рейк в этих турнирах
ноль. И это два конкретных турнира, а не формат GG вообще.</div>
</div></html>""")
    return "".join(parts)


def main() -> int:
    sections = []
    for name, path, expected in FIXTURES:
        if not path.exists():
            print(f"нет фикстуры {path} — отчёт неполон", file=sys.stderr)
            return 1
        raws = parse_file(path.read_text(encoding="utf-8"), source_ref=path.name)
        ens = [enrich(normalize(r)) for r in raws]
        sections.append((name, build_rows(raws, ens), chrono_ok(raws), expected, len(raws)))

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(render(sections), encoding="utf-8")
    bad = sum(1 for _, rs, _, _, _ in sections for r in rs if not r.all_ok)
    total = sum(len(rs) for _, rs, _, _, _ in sections)
    print(f"{OUT}  —  {total - bad}/{total} рук без замечаний")
    if "--open" in sys.argv:
        subprocess.run(["open", str(OUT)], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
