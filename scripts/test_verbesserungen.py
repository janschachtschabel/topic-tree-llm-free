"""Test der Verbesserungen für model2vec (nicht Teil der App).

    uv run --with simplemma python scripts/test_verbesserungen.py . m2v-edu

Vergleicht auf data/beispiel_inhalte.json: Füllwörter automatisch, Grundformen, Komposita, gelenktes Clustering
mit einem Wikidata-Gerüst und Oberbegriffe aus Wikidata. Ergebnisse: docs/verfahren.md, Abschnitt 3.
Braucht WIKIMEDIA_CONTACT in der .env (für Wikidata).
"""
import asyncio
import json
import re
import sys
from collections import Counter

import numpy as np
import simplemma
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, sys.argv[1])
from themenbaum import contents as C  # noqa: E402
from themenbaum import wikidata as W  # noqa: E402
from themenbaum.common import Fetcher, sentences  # noqa: E402

DATA = json.load(open(sys.argv[1] + "/data/beispiel_inhalte.json", encoding="utf-8"))
GOLD = [d["gold"] for d in DATA]
MODEL = sys.argv[2] if len(sys.argv) > 2 else "m2v-edu"
K = 10


def score(labels):
    labels = list(labels)
    pur = sum(Counter(g for g, p in zip(GOLD, labels) if p == c).most_common(1)[0][1] for c in set(labels)) / len(labels)
    return dict(n=len(set(labels)), purity=round(pur, 2), ari=round(adjusted_rand_score(GOLD, labels), 2),
                nmi=round(normalized_mutual_info_score(GOLD, labels), 2))


def ward(E, k=K):
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(E)


# ---------------------------------------------------------------- Text: Substantive, Füllwörter, Grundformen
def nouns(text):
    return [w for w in re.findall(r"[\w-]+", text) if w[:1].isupper() and w.lower() not in C.STOP and len(w) > 2]


RAW = [nouns(C.doc_text(d)) for d in DATA]


def lemma(w):
    return simplemma.lemmatize(w, lang="de")


def build_texts(dyn_df=None, lemmatize=False, split=False):
    docs = [[lemma(w) for w in ws] for ws in RAW] if lemmatize else [list(ws) for ws in RAW]
    if split:  # einfache Komposita-Regel: bekanntes Wort am Ende zusätzlich aufnehmen
        vocab = {w for ws in docs for w in ws}
        heads = sorted((v for v in vocab if len(v) >= 4), key=len, reverse=True)
        cache = {}

        def parts(w):
            if w not in cache:
                cache[w] = [w]
                for h in heads:
                    if len(h) < len(w) - 2 and w.lower().endswith(h.lower()):
                        cache[w] = [w, h]
                        break
            return cache[w]
        docs = [[p for w in ws for p in parts(w)] for ws in docs]
    removed = []
    if dyn_df:
        df = Counter(w.lower() for ws in docs for w in set(ws))
        common = {w for w, c in df.items() if c / len(docs) > dyn_df}
        removed = sorted(common, key=lambda w: -df[w])
        docs = [[w for w in ws if w.lower() not in common] for ws in docs]
    return [" ".join(ws) or "leer" for ws in docs], removed


async def main():
    print(f"Modell: {MODEL}, {len(DATA)} Inhalte, Soll-Kategorien: {len(set(GOLD))}, Ward k={K}\n")
    print("== A) Text-Vorverarbeitung (Ward, 10 Gruppen)")
    variants = [("Basis: nur Substantive", {}),
                ("+ Füllwörter automatisch (>30 %)", dict(dyn_df=0.30)),
                ("+ Füllwörter automatisch (>20 %)", dict(dyn_df=0.20)),
                ("+ Füllwörter automatisch (>12 %)", dict(dyn_df=0.12)),
                ("+ Grundformen", dict(lemmatize=True)),
                ("+ Grundformen + Komposita", dict(lemmatize=True, split=True)),
                ("+ Grundformen + Füllwörter (>20 %)", dict(lemmatize=True, dyn_df=0.20)),
                ("+ Grundformen + Komposita + Füllwörter (>20 %)", dict(lemmatize=True, split=True, dyn_df=0.20)),
                ("+ Grundformen + Komposita + Füllwörter (>12 %)", dict(lemmatize=True, split=True, dyn_df=0.12))]
    best = None
    for name, kw in variants:
        texts, removed = build_texts(**kw)
        E = await C.embed(texts, MODEL)
        s = score(ward(E))
        extra = f"  entfernt: {', '.join(removed[:8])}{' …' if len(removed) > 8 else ''}" if removed else ""
        print(f"  {name:48s} Reinheit {s['purity']:.2f}  ARI {s['ari']:.2f}  NMI {s['nmi']:.2f}{extra}")
        if best is None or s["ari"] > best[1]["ari"]:
            best = (name, s, kw, E, texts)
    print(f"  → beste Vorverarbeitung: {best[0]}")
    Ebest, kwbest = best[3], best[2]

    # ---------------------------------------------------------------- B) Gelenktes Clustering mit Wikidata-Gerüst
    print("\n== B) Gelenktes Clustering mit Wikidata-Gerüst „Physik“ als Startpunkten (beste Vorverarbeitung)")
    F = Fetcher()
    tree, _ = await W.build(F, "Physik", None, 12, 10, 0, True)
    l1 = tree["children"]
    seed_texts = [f"{n['title']}. {sentences(n['description'], 15, 40)}" for n in l1]
    prep_seed = lambda t: " ".join(lemma(w) for w in nouns(t)) if kwbest.get("lemmatize") else " ".join(nouns(t))
    S = await C.embed([prep_seed(t) or t for t in seed_texts], MODEL)
    print("  Startpunkte:", ", ".join(n["title"] for n in l1))

    def seeded(E, S, alpha, iters=25):
        Cn = S.copy()
        for _ in range(iters):
            lab = np.argmax(E @ Cn.T, axis=1)
            new = []
            for j in range(len(S)):
                m = E[lab == j]
                c = alpha * S[j] + (1 - alpha) * m.mean(axis=0) if len(m) else S[j]
                new.append(c / np.linalg.norm(c))
            Cn = np.array(new)
        lab = np.argmax(E @ Cn.T, axis=1)
        return lab, (E @ Cn.T).max(axis=1)

    for alpha in (1.0, 0.7, 0.4, 0.1):
        lab, sim = seeded(Ebest, S, alpha)
        s = score(lab)
        tag = "fest (= Einsortieren)" if alpha == 1.0 else f"Gerüst-Gewicht {alpha}"
        print(f"  {tag:30s} Gruppen {s['n']:2d}  Reinheit {s['purity']:.2f}  ARI {s['ari']:.2f}  NMI {s['nmi']:.2f}")

    # Variante mit Restgruppen: schlecht passende Inhalte bilden eigene, neue Gruppen
    for alpha, q in ((0.4, 0.25), (0.4, 0.35)):
        lab, sim = seeded(Ebest, S, alpha)
        thr = np.quantile(sim, q)
        rest = np.where(sim < thr)[0]
        lab = lab.copy()
        if len(rest) >= 6:
            k_rest, _ = C.choose_k(Ebest[rest], 2, min(5, len(rest) // 3))
            sub = AgglomerativeClustering(n_clusters=k_rest, linkage="ward").fit_predict(Ebest[rest])
            lab[rest] = 100 + sub
        s = score(lab)
        used = [l1[j]["title"] for j in sorted(set(lab) - {x for x in set(lab) if x >= 100})]
        print(f"  Gewicht {alpha} + Restgruppen ({int(q*100)} % schwächste) Gruppen {s['n']:2d}  Reinheit {s['purity']:.2f}  ARI {s['ari']:.2f}  NMI {s['nmi']:.2f}")
    print("  belegte Gerüst-Knoten (Gewicht 0.4):", ", ".join(used))
    tree2, _ = await W.build(F, "Physik", None, 12, 10, 0, True)
    alln = [n for a in tree2["children"] for n in [a] + a["children"]]
    S_all = await C.embed([prep_seed(f"{n['title']}. {sentences(n['description'], 15, 40)}") or n["title"] for n in alln], MODEL)
    near = np.argmax(Ebest @ S_all.T, axis=1)
    top = [j for j, _ in Counter(near).most_common(K)]
    for alpha in (0.7, 0.4, 0.1):
        lab, _ = seeded(Ebest, S_all[top], alpha)
        s = score(lab)
        print(f"  Startpunkte aus Daten (10 von {len(alln)}), Gewicht {alpha}: Gruppen {s['n']:2d}  Reinheit {s['purity']:.2f}  ARI {s['ari']:.2f}  NMI {s['nmi']:.2f}")
    print("  gewählte Startpunkte:", ", ".join(alln[j]["title"] for j in top))
    domain = {}
    for n in alln:
        domain[n["id"]] = n["title"]

    # ---------------------------------------------------------------- C) Oberbegriffe aus Wikidata für Ward-Gruppen
    print("\n== C) Namen der Ward-Gruppen: Begriff aus Texten vs. Oberbegriff aus Wikidata")
    lab = ward(Ebest)
    groups = [list(np.where(lab == c)[0]) for c in range(K)]
    texts_full = [C.doc_text(d) for d in DATA]
    cmap = C.casing_map(texts_full)
    labels_k = await C.label_groups(F, texts_full, groups, Ebest, "korpus", MODEL, cmap)
    GENERIC = {"Q35120", "Q151885", "Q336", "Q413", "Q11862829", "Q2995644", "Q1207505", "Q16722960", "Q7184903",
               "Q24229398", "Q937228", "Q17537576", "Q1047113", "Q58778", "Q4026292", "Q16889133", "Q309314",
               "Q1371849", "Q488383", "Q99527517", "Q107715", "Q28813620", "Q131569", "Q223557", "Q27948"}

    async def wd_item(term):
        r = await F.get_json(W.API, {"action": "wbsearchentities", "search": term, "language": "de", "uselang": "de",
                                     "type": "item", "limit": 1, "format": "json"})
        hits = (r or {}).get("search", [])
        return hits[0]["id"] if hits else None

    ent = {}

    async def entities(ids):
        need = [i for i in dict.fromkeys(ids) if i not in ent]
        for k in range(0, len(need), 40):
            r = await F.get_json(W.API, {"action": "wbgetentities", "ids": "|".join(need[k:k + 40]), "props": "claims|labels|sitelinks",
                                         "languages": "de", "sitefilter": "dewiki", "format": "json"})
            for q, e in (r or {}).get("entities", {}).items():
                par = [c["mainsnak"]["datavalue"]["value"]["id"] for p in ("P279", "P361")
                       for c in e.get("claims", {}).get(p, []) if c["mainsnak"].get("datavalue")]
                ent[q] = dict(label=e.get("labels", {}).get("de", {}).get("value"), wiki=bool(e.get("sitelinks", {}).get("dewiki")), parents=par[:6])

    async def ancestors(qid, depth=3):
        found, frontier = {qid: 0}, [qid]
        for d in range(1, depth + 1):
            await entities(frontier)
            nxt = [p for q in frontier for p in ent.get(q, {}).get("parents", []) if p not in found]
            for p in nxt:
                found[p] = d
            frontier = list(dict.fromkeys(nxt))
        await entities(list(found))
        return found

    async def lca(terms):
        ids = [i for i in await asyncio.gather(*(wd_item(t) for t in terms)) if i]
        if len(ids) < 2:
            return None, ids, None
        anc_sets = [await ancestors(i) for i in ids]
        cover = Counter()
        depth_sum = Counter()
        for a in anc_sets:
            for q, d in a.items():
                cover[q] += 1
                depth_sum[q] += d
        cands = [(cover[q], -depth_sum[q], ent[q]["label"]) for q in cover
                 if q not in GENERIC and ent.get(q, {}).get("label") and ent[q]["wiki"] and cover[q] >= 2]
        cands.sort(reverse=True)
        dom = [(cover[q], -depth_sum[q], domain[q]) for q in cover if q in domain]
        dom.sort(reverse=True)
        return (cands[0] if cands else None), ids, (dom[0] if dom else None)

    rows = []
    for g, lk in zip(groups, labels_k):
        terms = [t for t in C.frequent_nouns(texts_full, g, cmap, 8)]
        best_lca, ids, dom = await lca(terms)
        gold = Counter(GOLD[i] for i in g).most_common(1)[0]
        name = best_lca[2] if best_lca else lk["title"]
        dname = dom[2] if dom else lk["title"]
        rows.append((lk["title"], name, gold[0], dname))
        print(f"  {len(g):3d} | Soll: {gold[0][:24]:24s} | Text: {lk['title']:16s} | Wikidata frei: {name:26s} | im Gerüst: {dname:26s} ({dom[0] if dom else 0} Begr.)")
    a = await C.embed([r[0] for r in rows], "minilm")
    b = await C.embed([r[1] for r in rows], "minilm")
    g = await C.embed([r[2] for r in rows], "minilm")
    dd = await C.embed([r[3] for r in rows], "minilm")
    print(f"  Namensnähe: Begriff aus Texten {float(np.mean(np.sum(a * g, 1))):.2f} · Oberbegriff Wikidata frei {float(np.mean(np.sum(b * g, 1))):.2f} · Oberbegriff im Gerüst {float(np.mean(np.sum(dd * g, 1))):.2f}")
    print(f"\n  API-Aufrufe: {F.calls}, aus Cache: {F.cache_hits}")


asyncio.run(main())
