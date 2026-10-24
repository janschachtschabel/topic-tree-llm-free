"""Kostengrafiken aus data/kosten.json erzeugen (PNG, wie in docs/kosten/).

    uv run --with matplotlib python scripts/grafiken_kosten.py data/kosten.json docs/kosten
"""
import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import blended_transform_factory as blend

D = json.load(open(sys.argv[1], encoding="utf-8")); OUT = sys.argv[2]
M = ["gpt-5.6-luna", "gpt-6-luna"]; MC = {"gpt-5.6-luna": "#2a78d6", "gpt-6-luna": "#eb6834"}
C1, C2, C3 = "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, MUTED, LINE, BG = "#14181a", "#4c5256", "#80868a", "#dfe2dc", "#fcfcfb"
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10.5, "axes.edgecolor": LINE,
                     "xtick.color": MUTED, "ytick.color": INK2, "figure.facecolor": BG, "axes.facecolor": BG, "text.parse_math": False})


def de(v):
    s = f"{v:,.3f}" if v < 1 else f"{v:,.2f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".") + " $"


def by(a, k, v): return next(x for x in a if x[k] == v)


def komp(st, m):
    k = by(D["komp"], "struct", st); return (k["low"][m] + k["high"][m]) / 2


def overview(st):
    u1 = by(D["uc1"], "struct", st); rev = by(D["rev"], "struct", st); asg = by(D["assign"], "struct", st); u2 = D["uc2"][1]
    u3 = lambda p: next(x for x in D["uc3"] if x["struct"] == st and x["label"].startswith(p))
    a, b = u3("nur Metadaten"), u3("Metadaten + Volltext mittel")
    spec = [
        ("Themenbaum aus einem Thema", lambda m: u1["basis_ist"][m], lambda m: 0),
        ("Themenbaum aus einem Thema + 1 Überarbeitung", lambda m: u1["basis_ist"][m] + rev["basis"]["B"][m], lambda m: 0),
        ("Themenbaum aus einem Thema + 100 eigene Inhalte", lambda m: u1["basis_ist"][m], lambda m: u2["basis"][m] + asg["mit"][m]),
        ("Themenbaum aus 100 Inhalten – nur Metadaten", lambda m: a["mit"][m], lambda m: a["meta"][m]),
        ("Themenbaum aus 100 Inhalten – mit Volltext (mit Cache)", lambda m: b["mit"][m], lambda m: b["meta"][m]),
        ("Themenbaum aus 100 Inhalten – mit Volltext (ohne Cache)", lambda m: b["ohne"][m], lambda m: b["meta"][m]),
    ]
    return [dict(name=n, vals={m: (t(m), me(m), komp(st, m)) for m in M}) for n, t, me in spec]


def plot_overview(st, fn):
    rows = overview(st); nodes = by(D["komp"], "struct", st)["nodes"]
    step, bh = 1.5, 0.36
    W, H = 15, 1.6 + len(rows) * 1.25 + 0.9
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.13, 0.9 / H, 0.52, (len(rows) * 1.25) / H])
    tx = fig.add_axes([0.70, 0.9 / H, 0.29, (len(rows) * 1.25) / H]); tx.axis("off")
    ylim = (-0.45, step * len(rows) - 0.25)
    pos, heads = [], []
    for i, r in enumerate(rows[::-1]):
        y = i * step
        for j, m in enumerate(M[::-1]):
            pos.append((y + j * 0.42, r, m))
        heads.append((y + 0.42 + 0.4, r["name"]))
    maxv = 0
    for yy, r, m in pos:
        x = 0
        for v, c in zip(r["vals"][m], (C1, C2, C3)):
            if v > 0:
                ax.barh(yy, v, left=x, height=bh, color=c, edgecolor=BG, linewidth=1.5); x += v
        maxv = max(maxv, x)
        ax.text(x + 0.12, yy, de(x), va="center", fontsize=10, color=INK, fontweight="semibold")
        ax.text(-0.012, yy, m, va="center", ha="right", fontsize=9.5, color=MC[m], fontweight="semibold", transform=blend(ax.transAxes, ax.transData))
        t, me, k = r["vals"][m]
        for xx, v in ((0.0, t), (0.33, me), (0.67, k)):
            tx.text(xx, yy, "–" if v == 0 else de(v), va="center", fontsize=9.5, color=INK if v else MUTED, family="Consolas")
    for yy, name in heads:
        ax.text(-0.2, yy, name, va="center", ha="left", fontsize=11.5, color=INK, fontweight="semibold", transform=blend(ax.transAxes, ax.transData), clip_on=False)
    for a_ in (ax, tx): a_.set_ylim(*ylim)
    tx.set_xlim(0, 1)
    for xx, h in ((0.0, "Themenbaum"), (0.33, "Inhaltsmetadaten"), (0.67, "Kompendium")):
        tx.text(xx, ylim[1] + 0.05, h, fontsize=9.5, color=INK2, fontweight="semibold", va="bottom")
    ax.set_xlim(0, maxv * 1.2); ax.set_yticks([])
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=LINE, lw=0.8); ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f} $"))
    fig.text(0.02, 1 - 0.5 / H, f"Kosten je Use Case – Themenbaum {st} ({nodes} Knoten)", fontsize=17, fontweight="bold", color=INK)
    fig.text(0.02, 1 - 0.85 / H, "US-Dollar pro Lauf, aufgeteilt nach Themenbaum-Erzeugung, Inhaltsmetadaten und Kompendium-Generierung", fontsize=11, color=INK2)
    hs = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (C1, C2, C3)]
    fig.legend(hs, ["Themenbaum-Erzeugung", "Inhaltsmetadaten (inkl. Zuordnung)", "Kompendium-Generierung"], loc="lower left",
               bbox_to_anchor=(0.015, 1 - 1.45 / H), ncol=3, frameon=False, fontsize=10)
    fig.text(0.02, 0.12 / H,
             "Annahmen: 60 Wörter je Beschreibung, kein Reasoning · 100 Inhalte à ca. 5 Tsd. Tokens Volltext · Kompendium 70 Tsd. Tokens je Knoten (Mittel aus 60–80 Tsd.)\n"
             "Überarbeitung = nur betroffener Zweig neu · „Aus 100 Inhalten“ mit umgebauten Prompts und Prompt-Caching (30 min)",
             ha="left", fontsize=8.5, color=MUTED)
    fig.savefig(f"{OUT}/{fn}", dpi=170); plt.close(fig)


def grouped(fn, title, sub, rows, whisk=True, note=None, labelw=0.27):
    n = len(rows); H = 1.3 + 0.62 * n + 0.8
    fig = plt.figure(figsize=(13, H))
    ax = fig.add_axes([labelw, 0.8 / H, 0.95 - labelw, (0.62 * n) / H])
    mx = max(max(v.get("hi", v["v"]) for v in r["vals"]) for r in rows)
    for i, r in enumerate(rows[::-1]):
        for j, v in enumerate(r["vals"][::-1]):
            yy = i + j * 0.32 - 0.16
            ax.barh(yy, v["v"], height=0.28, color=v["c"], lw=0)
            end, lab = v["v"], de(v["v"])
            if whisk and v.get("hi") and v["hi"] > v["v"]:
                ax.plot([v["v"], v["hi"]], [yy, yy], color=MUTED, lw=1.3)
                ax.plot([v["hi"]] * 2, [yy - 0.09, yy + 0.09], color=MUTED, lw=1.3)
                end, lab = v["hi"], de(v["v"]) + " bis " + de(v["hi"])
            ax.text(end + mx * 0.01, yy, lab, va="center", fontsize=9, color=INK2, family="Consolas")
    ax.set_yticks(range(n)); ax.set_yticklabels([r["label"] for r in rows[::-1]], fontsize=10.5)
    ax.set_ylim(-0.6, n - 0.4); ax.set_xlim(0, mx * 1.32)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0); ax.grid(axis="x", color=LINE, lw=0.8); ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: (f"{v:.2f}" if mx < 2 else f"{v:.1f}".rstrip("0").rstrip(".")).replace(".", ",") + " $"))
    fig.text(0.02, 1 - 0.45 / H, title, fontsize=15, fontweight="bold", color=INK)
    fig.text(0.02, 1 - 0.78 / H, sub, fontsize=10.5, color=INK2)
    hs = [plt.Rectangle((0, 0), 1, 1, color=MC[m]) for m in M]; labs = list(M)
    if whisk:
        hs.append(plt.Line2D([0], [0], color=MUTED, lw=1.3)); labs.append(note or "Spanne bis Fall „hoch“")
    fig.legend(hs, labs, loc="lower left", bbox_to_anchor=(0.015, 0.0), ncol=len(labs), frameon=False, fontsize=10)
    fig.savefig(f"{OUT}/{fn}", dpi=170); plt.close(fig)


def V(v, hi, m):
    d = {"v": v, "c": MC[m]}
    if hi is not None: d["hi"] = hi
    return d


plot_overview("12×5×5", "00_uebersicht_usecases_12x5x5.png")
plot_overview("12×10×5", "00b_uebersicht_usecases_12x10x5.png")

rows = [{"label": f"{u['struct']} · {l}", "vals": [V(u["basis_" + c][m], u["hoch_" + c][m], m) for m in M]}
        for u in D["uc1"] for c, l in (("ist", "Ist-Code"), ("fix", "Prompt-Fix"))]
grouped("01_themenbaum_aus_thema.png", "Themenbaum aus einem Thema",
        "Kosten pro Baum · Balken = Basis (60 Wörter, kein Reasoning), Linie = hoch (100 Wörter + Reasoning)", rows)

names = {"A": "ganzer Baum neu", "B": "nur Zweig neu", "Bc": "nur Zweig + Cache (Ø 5 Wünsche)"}
rows = [{"label": f"{r['struct']} · {names[v]}", "vals": [V(r["basis"][v][m], r["hoch"][v][m], m) for m in M]}
        for r in D["rev"] for v in ("A", "B", "Bc")]
grouped("02_ueberarbeitungswunsch.png", "Zusätzlicher Überarbeitungswunsch",
        "Bestehender Baum + Änderungswunsch erneut ins Modell · Kosten pro Wunsch", rows, labelw=0.31)

rows = [{"label": "Metadaten · " + u["label"], "vals": [V(u["basis"][m], u["hoch"][m], m) for m in M]} for u in D["uc2"]]
rows += [{"label": f"Zuordnung zu {a['struct']} · {c} Cache", "vals": [V(a[c][m], None, m) for m in M]}
         for a in D["assign"] for c in ("ohne", "mit")]
grouped("03_100_eigene_inhalte.png", "Themenbaum aus einem Thema + 100 eigene Inhalte",
        "LLM-Kosten für Titel, Beschreibung, Keywords der 100 Inhalte und optionale Zuordnung zum Baum", rows,
        note="Spanne mit Reasoning", labelw=0.33)

for st, fn in (("12×5×5", "04_baum_aus_100_inhalten_12x5x5.png"), ("12×10×5", "04b_baum_aus_100_inhalten_12x10x5.png")):
    rows = [{"label": f"{u['label'].replace('Metadaten + ', '').split(' (')[0]} · {c} Cache", "vals": [V(u[c][m], None, m) for m in M]}
            for u in D["uc3"] if u["struct"] == st for c in ("ohne", "mit")]
    grouped(fn, f"Themenbaum aus 100 Inhalten ({st})",
            "Korpus aus 100 Inhalten in jedem Generator-Call · ohne vs. mit Prompt-Caching (ohne Metadaten-Erzeugung)",
            rows, whisk=False, labelw=0.3)

rows = [{"label": f"{k['struct']} · {k['nodes']} Knoten", "vals": [V(k["low"][m], k["high"][m], m) for m in M]} for k in D["komp"]]
grouped("05_kompendium.png", "Kompendium-Text je Knoten", "60–80 Tsd. Tokens und 30–40 s pro Knoten · Kosten für den ganzen Baum",
        rows, note="Spanne 60 → 80 Tsd. Tokens/Knoten")
print("done")
