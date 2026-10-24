"""Test: UMAP + HDBSCAN statt Ward (nicht Teil der App).

    uv run --with umap-learn python scripts/test_umap_hdbscan.py .

Vergleicht auf data/beispiel_inhalte.json mit model2vec (Grundformen + Komposita) und mpnet (Volltext):
Ward (bisher), HDBSCAN allein, UMAP + Ward, UMAP + HDBSCAN. UMAP ist zufallsabhängig, daher je Einstellung
5 Startwerte (Mittelwert und Spanne). HDBSCAN markiert schwer zuordenbare Inhalte als Rauschen; sie zählen hier
als eigene Gruppe "Sonstiges", ihr Anteil wird getrennt ausgewiesen. Ergebnisse: docs/ideen.md.
"""

import asyncio
import json
import sys
import warnings
from collections import Counter

import numpy as np
from sklearn.cluster import HDBSCAN, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1])
from themenbaum import contents as C  # noqa: E402

DATA = json.load(open(sys.argv[1] + "/data/beispiel_inhalte.json", encoding="utf-8"))
GOLD = [d["gold"] for d in DATA]
SEEDS = [0, 1, 2, 3, 4]


def score(labels):
    labels = list(labels)
    pur = sum(Counter(g for g, p in zip(GOLD, labels) if p == c).most_common(1)[0][1] for c in set(labels)) / len(labels)
    noise = sum(1 for x in labels if x == -1) / len(labels)
    return dict(n=len(set(x for x in labels if x != -1)), purity=pur, ari=adjusted_rand_score(GOLD, labels),
                nmi=normalized_mutual_info_score(GOLD, labels), noise=noise)


def umap_reduce(E, n_neighbors, n_components, seed):
    import umap
    return umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=0.0, metric="cosine",
                     random_state=seed).fit_transform(E)


def hdb(X, min_size, min_samples, method):
    return HDBSCAN(min_cluster_size=min_size, min_samples=min_samples, cluster_selection_method=method).fit_predict(X)


def line(name, scores):
    a = np.array([[s["purity"], s["ari"], s["nmi"], s["n"], s["noise"]] for s in scores])
    m, lo, hi = a.mean(0), a.min(0), a.max(0)
    rng = f" (ARI {lo[1]:.2f}–{hi[1]:.2f})" if len(scores) > 1 else ""
    print(f"  {name:58s} Gruppen {m[3]:4.1f}  Rauschen {m[4]*100:3.0f} %  Reinheit {m[0]:.2f}  ARI {m[1]:.2f}  NMI {m[2]:.2f}{rng}")
    return m[1]


async def main():
    texts = [C.doc_text(d) for d in DATA]
    for model, prep in (("m2v-edu", "lemma"), ("mpnet", "full")):
        E = await C.embed(C.prep_texts(texts, prep), model)
        print(f"\n== {model} ({prep})")
        line("Ward, 10 Gruppen (bisher)", [score(AgglomerativeClustering(n_clusters=10, linkage="ward").fit_predict(E))])
        for ms in (3, 4, 5):
            line(f"HDBSCAN allein, min. Gruppengröße {ms}", [score(hdb(E, ms, 1, "eom"))])
        best = []
        for nn, nc in ((10, 5), (15, 5), (15, 10)):
            red = {s: umap_reduce(E, nn, nc, s) for s in SEEDS}
            line(f"UMAP ({nn} Nachbarn, {nc} Dim.) + Ward 10", [score(AgglomerativeClustering(n_clusters=10, linkage="ward").fit_predict(red[s])) for s in SEEDS])
            for ms, msamp, meth in ((4, 1, "eom"), (5, 2, "eom"), (6, 2, "eom"), (4, 1, "leaf"), (6, 3, "leaf")):
                name = f"UMAP ({nn}, {nc}) + HDBSCAN (Größe {ms}, Kern {msamp}, {meth})"
                ari = line(name, [score(hdb(red[s], ms, msamp, meth)) for s in SEEDS])
                best.append((ari, name))
        best.sort(reverse=True)
        print(f"  → beste UMAP+HDBSCAN-Einstellung: {best[0][1]} (Ø ARI {best[0][0]:.2f})")


asyncio.run(main())
