"""Test: Thema -> Baum als Hybrid aus Wikipedia, Wikidata und GND (mit/ohne Mehrheitsentscheid) gegen reines Wikipedia.

    uv run python scripts/test_hybrid_thema.py .

Braucht WIKIMEDIA_CONTACT in der .env. Für alle Varianten wird gleich gemessen: Füllgrad je Ebene, Anteil
allgemeiner Abschnitte, Beschreibungen, Wikidata-Zuordnung, GND-ID und Bestätigung durch mindestens zwei Quellen
(Wikipedia: aus Wikipedia oder im Elternabschnitt verlinkt; Wikidata: Beziehung zum Elternobjekt; GND: Eintrag).
"""

import asyncio
import sys

sys.path.insert(0, sys.argv[1])
from themenbaum import hybrid as H  # noqa: E402
from themenbaum import wikipedia as WP  # noqa: E402
from themenbaum.common import Fetcher  # noqa: E402

THEMES = ["Physik", "Photosynthese", "Klimawandel", "Bruchrechnung", "Demokratie", "Optik", "Mittelalter"]
N1, N2, N3 = 12, 5, 5


async def evaluate(F, root):
    nodes, edges = [], []

    def walk(n, lvl, parent):
        for c in n["children"]:
            nodes.append((c, lvl, parent))
            walk(c, lvl + 1, c)
    walk(root, 1, root)
    allp = [root] + [n for n, _, _ in nodes]
    need = [n for n in allp if not n.get("_qid")]
    tm = await H.qids_for_titles(F, [n.get("_main") or n["title"] for n in need] + [n["title"] for n in need])
    for n in need:
        n["_qid"] = tm.get(n.get("_main") or "") or tm.get(n["title"])
    conf = await H.confirmed_pairs(F, [(c["_qid"], p["_qid"]) for c, _, p in nodes if c.get("_qid") and p.get("_qid")])
    gnd = await H.gnd_ids(F, [c.get("_qid") for c, _, _ in nodes])
    per = [0, 0, 0]
    stats = dict(generic=0, desc=0, qid=0, gnd=0, multi=0, n=len(nodes))
    for c, lvl, p in nodes:
        per[lvl - 1] += 1
        links = {H.norm(x) for x in (p.get("_links") or [])}
        src = set()
        if not c.get("_wdlink") or H.norm(c.get("_main") or "") in links or H.norm(c["title"]) in links:
            src.add("WP")
        if c.get("_wdlink") or (c.get("_qid") and p.get("_qid") and (c["_qid"], p["_qid"]) in conf):
            src.add("WD")
        if gnd.get(c.get("_qid")):
            src.add("GND")
        stats["generic"] += H.generic(c["title"])
        stats["desc"] += bool(c["description"])
        stats["qid"] += bool(c.get("_qid"))
        stats["gnd"] += bool(gnd.get(c.get("_qid")))
        stats["multi"] += len(src) >= 2
    return per, stats


async def main():
    variants = [("Wikipedia", lambda F, t: WP.build(F, t, None, N1, N2, N3, True)),
                ("Hybrid ohne Mehrheit", lambda F, t: H.build(F, t, None, N1, N2, N3, True, vote=False)),
                ("Hybrid (Standard)", lambda F, t: H.build(F, t, None, N1, N2, N3, True, vote=True)),
                ("Hybrid, Mehrheit E1", lambda F, t: H.build(F, t, None, N1, N2, N3, True, vote="l1"))]
    for theme in THEMES:
        print(f"\n=== {theme}  (Ziel {N1} / {N1*N2} / {N1*N2*N3})")
        for name, fn in variants:
            F = Fetcher()
            root, _ = await fn(F, theme)
            per, s = await evaluate(F, root)
            pct = lambda k: round(100 * s[k] / max(1, s["n"]))
            print(f"  {name:21s} Knoten {per[0]:2d}/{per[1]:3d}/{per[2]:3d} | allgemein {pct('generic'):2d} % | Beschreibung {pct('desc'):3d} % | "
                  f"Wikidata {pct('qid'):3d} % | GND {pct('gnd'):3d} % | ≥2 Quellen {pct('multi'):3d} %")
            print(f"  {'':21s} Ebene 1: {', '.join(c['title'] for c in root['children'])}")
            if root["children"]:
                c0 = max(root["children"], key=lambda c: len(c["children"]))
                print(f"  {'':21s} z. B. {c0['title']} → {', '.join(k['title'] for k in c0['children'])}")


asyncio.run(main())
