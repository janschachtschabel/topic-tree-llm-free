"""Erzeugt die Grafiken zu den Verfahren und Messergebnissen in docs/img/.

    uv run --with matplotlib python scripts/grafiken_verfahren.py

Die Messwerte stammen aus den Läufen mit data/beispiel_inhalte.json (siehe docs/verfahren.md).
"""

from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)

INK, INK2, MUTED, LINE, BG = "#16201d", "#4a5753", "#7d8a86", "#dbe2df", "#fcfcfb"
ACCENT, ACCENT_SOFT = "#1f6b5c", "#e2efeb"
SRC = {"GND": "#6a4fc8", "Wikipedia": "#2a78d6", "Wikidata": "#b8561c", "Hybrid": "#1f8a5c"}
FONT = "Segoe UI, Helvetica, Arial, sans-serif"


# ---------------------------------------------------------------- Ablaufdiagramme (SVG)
class Svg:
    def __init__(self, w, h, title):
        self.w, self.h, self.parts = w, h, []
        self.parts.append(f'<rect x="0" y="0" width="{w}" height="{h}" rx="14" fill="{BG}" stroke="{LINE}"/>')
        self.text(24, 38, title, 20, INK, "700")

    def text(self, x, y, s, size=13, color=INK, weight="400", anchor="start"):
        self.parts.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" fill="{color}" '
                          f'font-weight="{weight}" text-anchor="{anchor}">{escape(s)}</text>')

    def box(self, x, y, w, h, lines, fill="#ffffff", stroke=LINE, head=None, head_color=INK):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>')
        ty = y + 22
        if head:
            self.text(x + 12, ty, head, 13.5, head_color, "700")
            ty += 19
        for ln in lines:
            self.text(x + 12, ty, ln, 12, INK2)
            ty += 16

    def arrow(self, x1, y1, x2, y2, color=MUTED):
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.6" marker-end="url(#a)"/>')

    def save(self, name):
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}">'
               f'<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
               f'<path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker></defs>' + "".join(self.parts) + "</svg>")
        (OUT / name).write_text(svg, "utf-8")


def diagram_topic():
    s = Svg(1180, 700, "Thema → Themenbaum ohne LLM: drei Quellen und ein Hybrid")
    s.text(24, 62, "Jede Quelle liefert Struktur, Namen und Beschreibungen aus vorhandenem Wissen. Kein Text wird generiert.", 13, INK2)
    s.box(24, 300, 150, 90, ["z. B. „Physik“,", "„Photosynthese“"], fill=ACCENT_SOFT, stroke=ACCENT, head="Thema")
    lanes = [
        ("GND", 100, ["Schlagwort suchen", "(lobid-gnd)"],
         ["Unterbegriffe + verwandte", "Begriffe; bei Fächern:", "GND-Sachgruppen"],
         ["Ebene 2/3: zentrale Begriffe", "und deren Unterbegriffe"]),
        ("Wikipedia", 245, ["Artikel zum Thema", "(Titel oder Suche)"],
         ["Abschnitte des Artikels", "(ohne Literatur, Weblinks)"],
         ["Unterabschnitte oder", "Gliederung des Hauptartikels"]),
        ("Wikidata", 390, ["Objekt zum Thema", "(Wikidata-Suche)"],
         ["„Teil von“, „Unterklasse von“,", "„hat Teil“ (SPARQL)"],
         ["sortiert nach Zahl der", "Sprachversionen"]),
        ("Hybrid", 535, ["Wikipedia-Gliederung ohne", "allgemeine Abschnitte"],
         ["+ passende Wikidata-Begriffe;", "Mehrheitsentscheid: was", "mehrere Quellen bestätigen"],
         ["Gliederung des Artikels je", "Knoten; GND-ID je Knoten"]),
    ]
    for name, y, a, b, c in lanes:
        col = SRC[name]
        s.arrow(176, 345, 230, y + 45, col)
        s.box(232, y, 210, 92, a, stroke=col, head=name, head_color=col)
        s.arrow(444, y + 46, 474, y + 46)
        s.box(476, y, 240, 92, b, head="Ebene 1")
        s.arrow(718, y + 46, 748, y + 46)
        s.box(750, y, 230, 92, c, head="Ebene 2 und 3")
        s.arrow(982, y + 46, 1010, 345)
    s.box(1012, 275, 150, 140, ["Titel, Kurztitel,", "Beschreibung (aus", "Wikipedia/GND),", "Schlagwörter, Link"],
          fill=ACCENT_SOFT, stroke=ACCENT, head="Themenbaum")
    s.text(24, 670, "Beschreibungen: Wikipedia-Einleitungen (bei der GND über GND → Wikidata → Wikipedia). "
           "Ausgabe im JSON-Format des topic-tree-generator.", 12, MUTED)
    s.save("verfahren-thema.svg")


def diagram_contents():
    s = Svg(1180, 600, "Inhaltsmenge → Themenbaum ohne LLM")
    s.text(24, 62, "Embeddings sind Zahlenvektoren für Bedeutung, kein Textgenerator. Die Struktur entsteht aus den Inhalten oder aus einem fertigen Gerüst.", 13, INK2)
    s.box(24, 250, 160, 100, ["Titel + Beschreibung", "(z. B. 100 Inhalte)"], fill=ACCENT_SOFT, stroke=ACCENT, head="Inhalte")
    s.arrow(186, 300, 214, 300)
    s.box(216, 240, 170, 120, ["Volltext (Transformer)", "oder Substantive in", "Grundform + Komposita", "(model2vec)"], head="Text vorbereiten")
    s.arrow(388, 300, 416, 300)
    s.box(418, 250, 180, 100, ["mpnet, MiniLM", "(Transformer, lokal)", "oder model2vec (statisch)"], head="Embeddings")
    # Zweig A
    s.arrow(600, 290, 640, 160)
    s.box(642, 95, 230, 120, ["agglomerativ (Ward):", "Ebene 1 = k Gruppen,", "große Gruppen erneut", "geteilt"], stroke=ACCENT, head="A · Gruppen bilden", head_color=ACCENT)
    s.arrow(874, 155, 902, 155)
    s.box(904, 80, 250, 150, ["Begriff aus den Texten", "GND-Schlagwort (Hybrid)", "Wikipedia-Artikel (Hybrid)", "Knoten einer Taxonomie,", "z. B. Wikidata (Hybrid)"],
          head="Gruppen benennen")
    # Zweig B
    s.arrow(600, 310, 640, 440)
    s.box(642, 385, 230, 120, ["Baum zum Thema aus", "Wikipedia, Wikidata oder", "GND (wie „Thema → Baum“)"], stroke=SRC["Wikidata"],
          head="B · Gerüst einsortieren", head_color=SRC["Wikidata"])
    s.arrow(874, 445, 902, 445)
    s.box(904, 385, 250, 120, ["Inhalt → ähnlichster Knoten", "(Kosinus); unter der", "Schwelle: nicht zugeordnet;", "leere Äste kappen"], head="Einordnen")
    s.text(24, 560, "Bewertung mit Beispieldaten: Reinheit, ARI und NMI gegenüber Soll-Kategorien; Namensnähe = Ähnlichkeit "
           "zwischen Gruppenname und Soll-Kategorie.", 12, MUTED)
    s.save("verfahren-inhalte.svg")


# ---------------------------------------------------------------- Ergebnisdiagramme (PNG)
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10.5, "axes.edgecolor": LINE,
                     "xtick.color": MUTED, "ytick.color": INK2, "figure.facecolor": BG, "axes.facecolor": BG,
                     "text.parse_math": False})


def de(v, d=2):
    return f"{v:.{d}f}".replace(".", ",")


def barh_two(fn, title, sub, labels, a, b, la, lb, ca, cb, xmax, fmt):
    n = len(labels)
    fig, ax = plt.subplots(figsize=(11, 1.4 + 0.55 * n))
    fig.subplots_adjust(left=0.36, right=0.95, top=1 - 0.95 / (1.4 + 0.55 * n), bottom=0.55 / (1.4 + 0.55 * n) + 0.05)
    for i, (x, y) in enumerate(zip(a[::-1], b[::-1])):
        if x is not None:
            ax.barh(i + 0.18, x, height=0.34, color=ca)
            ax.text(x + xmax * 0.01, i + 0.18, fmt(x), va="center", fontsize=9, color=INK2)
        if y is not None:
            ax.barh(i - 0.18, y, height=0.34, color=cb)
            ax.text(y + xmax * 0.01, i - 0.18, fmt(y), va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels[::-1])
    ax.set_xlim(0, xmax)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=LINE)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt(v)))
    fig.text(0.02, 1 - 0.35 / (1.4 + 0.55 * n), title, fontsize=14, fontweight="bold", color=INK)
    fig.text(0.02, 1 - 0.62 / (1.4 + 0.55 * n), sub, fontsize=10, color=INK2)
    hs = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (ca, cb)]
    fig.legend(hs, [la, lb], loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=2, frameon=False)
    fig.savefig(OUT / fn, dpi=160)
    plt.close(fig)


def chart_clustering():
    labels = ["TF-IDF (gemeinsame Wörter)", "MiniLM, Volltext", "mpnet, Volltext", "model2vec edu, Volltext",
              "model2vec edu, nur Substantive", "model2vec edu, Grundformen + Komposita",
              "Gerüst Wikidata „Physik“ (mpnet)", "Gerüst Wikipedia „Physik“ (MiniLM)"]
    purity = [None, 0.57, 0.64, 0.48, 0.59, 0.64, 0.48, 0.32]
    ari = [0.03, 0.30, 0.39, 0.20, 0.35, 0.41, 0.18, 0.10]
    barh_two("ergebnisse-inhalte.png", "Inhalte → Baum: Wie gut treffen die Gruppen die Soll-Kategorien?",
             "100 Beispielinhalte, 10 Soll-Kategorien, 10 Hauptgruppen · Reinheit = Anteil in der überwiegenden Kategorie, ARI: 0 zufällig, 1 identisch",
             labels, purity, ari, "Reinheit", "ARI", "#1baf7a", "#2a78d6", 0.8, lambda v: de(v))


def chart_naming():
    labels = ["Begriff aus den Texten", "GND-Schlagwort (Hybrid)", "Wikipedia-Artikel (Hybrid)",
              "Taxonomie Wikidata (Hybrid)", "Taxonomie Wikipedia (Hybrid)"]
    mpnet = [0.61, 0.57, 0.55, 0.61, 0.44]
    m2v = [0.61, 0.58, 0.58, 0.65, 0.40]
    barh_two("ergebnisse-benennung.png", "Benennung der Gruppen: Namensnähe zur Soll-Kategorie",
             "Mittlere Ähnlichkeit (MiniLM) zwischen Name der Hauptgruppe und ihrer überwiegenden Soll-Kategorie · grobes Hilfsmaß",
             labels, mpnet, m2v, "Gruppen mit mpnet", "Gruppen mit model2vec edu", "#2a78d6", "#eb6834", 0.8, lambda v: de(v))


def chart_topic():
    themes = ["Physik", "Photosynthese", "Klimawandel"]
    nodes = {"GND": [225, 1, 5], "Wikipedia": [74, 43, 165], "Wikidata": [199, 6, 5], "Hybrid": [190, 69, 118]}
    fig, ax = plt.subplots(figsize=(11, 4.4))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.78, bottom=0.18)
    w = 0.2
    for j, (src, vals) in enumerate(nodes.items()):
        xs = [i + (j - 1.5) * w for i in range(len(themes))]
        ax.bar(xs, vals, width=w - 0.02, color=SRC[src], label=src)
        for x, v in zip(xs, vals):
            ax.text(x, v + 4, str(v), ha="center", fontsize=9.5, color=INK2)
    ax.set_xticks(range(len(themes)))
    ax.set_xticklabels(themes, fontsize=11)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color=LINE)
    ax.set_axisbelow(True)
    ax.set_ylabel("Knoten (max. 732 bei 12×10×5)", color=INK2)
    fig.text(0.02, 0.93, "Thema → Baum: Wie viele Knoten liefert jede Quelle?", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.02, 0.86, "Vorgabe 12 × 10 × 5 · Hybrid = Wikipedia + Wikidata + GND mit Mehrheitsentscheid, ohne allgemeine Abschnitte",
             fontsize=10, color=INK2)
    fig.legend(loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=4, frameon=False)
    fig.savefig(OUT / "ergebnisse-thema.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    diagram_topic()
    diagram_contents()
    chart_clustering()
    chart_naming()
    chart_topic()
    print("Grafiken geschrieben nach", OUT)
