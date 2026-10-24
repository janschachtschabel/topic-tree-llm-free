"""Themenbaum aus einer unstrukturierten Inhaltsmenge – ohne generatives LLM.

Verfahren:
- cluster:  Inhalte zu Gruppen bündeln (Embeddings oder TF-IDF, hierarchisch-agglomerativ),
            Gruppen benennen über Schlagwörter (c-TF-IDF), einen Begriff aus den Texten
            oder ein passendes GND-Schlagwort.
- scaffold: Inhalte per Embedding-Ähnlichkeit in ein vorhandenes Gerüst einsortieren
            (Baum aus Wikipedia, Wikidata oder GND zu einem Thema) und leere Äste kappen.
"""

import asyncio
import re
import threading
import time
from collections import Counter

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from . import gnd as src_gnd
from . import wikidata as src_wikidata
from . import wikipedia as src_wikipedia
from .common import MODELS_DIR, Fetcher, require_wikimedia, sentences

SCAFFOLD_SOURCES = {"wikipedia": src_wikipedia, "wikidata": src_wikidata, "gnd": src_gnd}
# Statische Embeddings über model2vec (sehr schnell, kein Transformer zur Laufzeit).
M2V_MODELS = {"m2v-edu": "JanSchachtschabel/m2v-gte-256-edu"}
MODELS = {
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "mpnet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
}
TOKEN = r"(?u)\b[^\W\d_][\w-]{2,}\b"

# Allgemeine deutsche Stoppwörter plus Wörter, die in fast jeder Materialbeschreibung stehen
# und sonst Gruppen nach Materialart statt nach Thema bilden würden.
STOP = set("""
aber alle allem allen aller alles als also am an ander andere anderem anderen anderer anderes anders auch auf aus
bei beim bin bis bist da dabei dadurch dafür dagegen daher damit dann daran darauf daraus darin darum darüber das
dass dasselbe davon dazu dem den denen denn der deren des deshalb dessen die dies diese diesem diesen dieser dieses
doch dort du durch ein eine einem einen einer eines einige einigen einmal er es etwa etwas euch euer für gegen
geht gibt hat hatte haben hier hin hinter ich ihm ihn ihnen ihr ihre ihrem ihren ihrer ihres im in indem ins ist
ja je jede jedem jeden jeder jedes jetzt kann kein keine können könnte machen man manche mehr mein mit muss nach
neben nicht nichts noch nun nur ob oder ohne sehr sein seine seinem seinen seiner sich sie sind so solche soll
sollen sondern sowie über um und uns unser unter viel viele vom von vor wann war waren warum was weil weiter
welche welchem welchen welcher welches wenn wer werden wie wieder will wir wird wo wobei wurde wurden zu zum zur
zwar zwischen sowohl besonders bereits dabei jeweils dafür hierbei zudem außerdem anschließend eigene eigenen
schülerinnen schüler schülern lernende lernenden lehrkräfte lehrkraft lehrer lehrerinnen unterricht unterrichts
material materialien arbeitsblatt arbeitsblätter video videos simulation simulationen quiz lernpfad podcast
präsentation unterrichtsentwurf erklärtext experimentieranleitung sekundarstufe primarstufe klasse klassen
klassenstufe jahrgangsstufe beruflichen bildung eignet geeignet eignen einsatz einsetzen eingesetzt inhalt inhalte
thema themen themas lernen lernt erarbeiten erarbeitet vertiefen vertieft ideal dient bietet bieten zeigt zeigen
erklärt erklären werden wurde stunde stunden doppelstunde aufgaben aufgabe fragen beantworten anhand mithilfe
grundlagen grundlegende verschiedene verschiedenen wichtige wichtigen typische einfache einfachen konkrete
konkreten alltag alltags beispiele beispielen beispiel ermöglicht ermöglichen gemeinsam selbst selbstständig
hinweise hinweisen text texte textes kinder kindern sachunterricht doppelstunden physik physikalische physikalischen
physikalischer experiment experimente experimenten versuch versuche versuchen simulieren simuliert simulation
experimentieren lesen berechnen vergleichen beobachten untersuchen durchführen durchgeführt antwort antworten blatt
abschluss abschließend schritt schritte schritten schrittweise gruppen gruppenarbeit partnerarbeit auszubildende
auszubildenden grundschule oberstufe kurs kurze kurzen kurzes leicht anschaulich verständlich interaktiv interaktive
interaktiven digital digitale digitalen minuten drei zwei vier ersten zweiten dritten unterscheidet unterscheiden
kommt kommen bleibt bleiben angezeigt anzeigen lernpfades schritten erkunden erkundet begriffe begriff fachbegriffe
zusammenhang zusammenhänge rolle frage hilfe eignet mögliche möglichen moderne modernen kenntnisse vorwissen
gelernte gelernten erstellen erstellt bearbeiten bearbeitet lösung lösungen variante varianten teil teile neue neuen
hausaufgabe hausaufgaben unterrichtsstunde unterrichtsstunden anleitung anleitungen szenarien szenario rückmeldung
mittelstufe umgang unterrichtseinheit hörfragen rechenbeispiele rechenbeispiel projekt podcastfolge auswertung
weitere weiteren lernziel lernziele kompetenzen leitfragen leitfrage abschnitt abschnitte ende anfang einstieg
folge folgen betrachten betrachtet arbeitet arbeiten erfahren erfahrung ergebnisse ergebnis daten nutzen nutzt
auswertungsbogen auswertungsbögen beobachtungsbogen beobachtungsbögen aussagen aussage vorbereitung arbeitsauftrag
""".split())

_models: dict = {}
_model_lock = threading.Lock()
_emb_cache: dict = {}


def _model(name: str):
    with _model_lock:
        if name not in _models:
            if name in M2V_MODELS:
                from model2vec import StaticModel
                _models[name] = StaticModel.from_pretrained(M2V_MODELS[name])
            else:
                from fastembed import TextEmbedding
                _models[name] = TextEmbedding(MODELS[name], cache_dir=str(MODELS_DIR))
        return _models[name]


def _embed_sync(texts: list[str], model: str) -> np.ndarray:
    missing = [t for t in dict.fromkeys(texts) if (model, t) not in _emb_cache]
    if missing:
        m = _model(model)
        vectors = m.encode(missing) if model in M2V_MODELS else m.embed(missing)
        for t, v in zip(missing, vectors):
            _emb_cache[(model, t)] = np.asarray(v, dtype=np.float32)
    m = np.vstack([_emb_cache[(model, t)] for t in texts])
    return m / np.linalg.norm(m, axis=1, keepdims=True).clip(min=1e-9)


async def embed(texts: list[str], model: str) -> np.ndarray:
    return await asyncio.to_thread(_embed_sync, texts, model)


def doc_text(c: dict) -> str:
    return f"{c['title']}. {c['title']}. {c.get('description', '')}"


def prep_text(text: str, mode: str) -> str:
    """Eingabe für die Embeddings vorbereiten. Statische Modelle (model2vec) profitieren davon,
    nur die Substantive zu sehen; Transformer-Modelle arbeiten mit dem vollen Text besser."""
    if mode == "nostop":
        return " ".join(w for w in re.findall(r"[\w-]+", text) if w.lower() not in STOP and len(w) > 2)
    if mode in ("nouns", "lemma"):
        return " ".join(w for w in re.findall(r"[\w-]+", text) if w[:1].isupper() and w.lower() not in STOP and len(w) > 2)
    return text


def prep_texts(texts: list[str], mode: str) -> list[str]:
    """Wie prep_text, für mehrere Texte zugleich. Modus "lemma": Substantive in Grundform (simplemma), und
    zusammengesetzte Wörter bekommen ihr Grundwort dazu, wenn es im Bestand vorkommt
    (z. B. "Spannungsquellen" -> "Spannungsquelle Quelle"). Getestet: bei model2vec ARI 0,35 -> 0,41."""
    if mode != "lemma":
        return [prep_text(t, mode) for t in texts]
    import simplemma
    lemma_cache: dict[str, str] = {}

    def lemma(w):
        if w not in lemma_cache:
            lemma_cache[w] = simplemma.lemmatize(w, lang="de")
        return lemma_cache[w]

    docs = [[lemma(w) for w in prep_text(t, "nouns").split()] for t in texts]
    vocab = {w.lower(): w for ws in docs for w in ws}
    split_cache: dict[str, list[str]] = {}

    def parts(w):  # längstes bekanntes Grundwort am Wortende (mind. 4 Zeichen, Bestimmungswort mind. 3)
        if w not in split_cache:
            split_cache[w] = [w]
            for i in range(3, len(w) - 3):
                head = vocab.get(w[i:].lower())
                if head:
                    split_cache[w] = [w, head]
                    break
        return split_cache[w]

    return [" ".join(p for w in ws for p in parts(w)) for ws in docs]


def resolve_prep(cfg: dict) -> str:
    mode = cfg.get("text_prep", "auto")
    if mode == "auto":  # gemessen: Grundformen + Komposita helfen beim Clustern mit model2vec, nicht beim Einsortieren
        if cfg.get("model", "mpnet") not in M2V_MODELS:
            return "full"
        return "nouns" if cfg.get("method") == "scaffold" else "lemma"
    return mode


def casing_map(texts: list[str]) -> dict[str, str]:
    """Kleinschreibung -> häufigste Originalschreibweise (für lesbare Begriffe)."""
    forms: dict[str, Counter] = {}
    for t in texts:
        for w in re.findall(r"[^\W\d_][\w-]{2,}", t):
            forms.setdefault(w.lower(), Counter())[w] += 1
    return {k: v.most_common(1)[0][0] for k, v in forms.items()}


def display(term: str, cmap: dict) -> str:
    return " ".join(cmap.get(p, p) for p in term.split())


def tfidf_features(texts: list[str]) -> np.ndarray:
    vec = TfidfVectorizer(token_pattern=TOKEN, stop_words=list(STOP), ngram_range=(1, 2), min_df=2, max_df=0.4, sublinear_tf=True)
    X = vec.fit_transform(texts)
    k = max(2, min(60, X.shape[1] - 1, X.shape[0] - 1))
    Z = TruncatedSVD(n_components=k, random_state=0).fit_transform(X)
    return Z / np.linalg.norm(Z, axis=1, keepdims=True).clip(min=1e-9)


def ctfidf_terms(texts: list[str], groups: list[list[int]], n: int = 15) -> list[list[str]]:
    """Begriffe, die eine Gruppe von ihren Geschwistergruppen unterscheiden (c-TF-IDF wie BERTopic)."""
    cv = CountVectorizer(token_pattern=TOKEN, stop_words=list(STOP), ngram_range=(1, 2), min_df=2)
    try:
        X = cv.fit_transform(texts)
    except ValueError:
        return [[] for _ in groups]
    M = np.vstack([np.asarray(X[g].sum(axis=0)).ravel() for g in groups]).astype(float)
    tf = M / M.sum(axis=1, keepdims=True).clip(min=1)
    avg = M.sum() / max(1, len(groups))
    W = tf * np.log(1 + avg / M.sum(axis=0).clip(min=1))
    vocab = cv.get_feature_names_out()
    out = []
    for i in range(len(groups)):
        terms = []
        for j in np.argsort(-W[i]):
            if W[i, j] <= 0 or len(terms) >= n:
                break
            t = vocab[j]
            if any(t in s or s in t for s in terms):
                continue
            terms.append(t)
        out.append(terms)
    return out


def coverage(texts: list[str], group: list[int], term: str) -> float:
    """Anteil der Inhalte einer Gruppe, in denen der Begriff vorkommt (allgemeine Begriffe gewinnen)."""
    pat = re.compile(r"\b" + re.escape(term.lower()) + r"\b")
    return sum(1 for i in group if pat.search(texts[i].lower())) / max(1, len(group))


def choose_k(X: np.ndarray, lo: int, hi: int) -> tuple[int, float]:
    best = (lo, -1.0)
    for k in range(lo, hi + 1):
        if k >= len(X):
            break
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        s = silhouette_score(X, labels, metric="cosine")
        if s > best[1]:
            best = (k, s)
    return best


def split(X: np.ndarray, idx: list[int], k: int | None, lo: int, hi: int) -> tuple[list[list[int]], float | None]:
    sub = X[idx]
    sil = None
    if k is None:
        k, sil = choose_k(sub, lo, hi)
    k = max(1, min(k, len(idx)))
    if k == 1:
        return [idx], None
    labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(sub)
    if sil is None and len(set(labels)) > 1:
        sil = float(silhouette_score(sub, labels, metric="cosine"))
    groups = [[idx[i] for i in range(len(idx)) if labels[i] == c] for c in range(k)]
    return sorted(groups, key=len, reverse=True), sil


def gold_summary(items: list[int], contents: list[dict]) -> dict | None:
    golds = [contents[i].get("gold") for i in items if contents[i].get("gold")]
    if not golds:
        return None
    top, cnt = Counter(golds).most_common(1)[0]
    return {"top": top, "count": cnt, "total": len(items), "mix": Counter(golds).most_common()}


def metrics(pred: list, contents: list[dict]) -> dict | None:
    gold = [c.get("gold") for c in contents]
    if not all(gold):
        return None
    purity = sum(Counter(g for g, p in zip(gold, pred) if p == c).most_common(1)[0][1] for c in set(pred)) / len(pred)
    return {"purity": round(purity, 3), "ari": round(adjusted_rand_score(gold, pred), 3),
            "nmi": round(normalized_mutual_info_score(gold, pred), 3), "gold_classes": len(set(gold))}


async def gnd_label(F: Fetcher, terms: list[str], centroid: np.ndarray, model: str, weights: dict) -> tuple[str, str] | None:
    """GND-Schlagwort, dessen Name genau einem Gruppenbegriff entspricht (ohne Zusatz wie „<Motiv>“ bevorzugt)."""
    terms = sorted((t for t in terms if t[:1].isupper()), key=lambda t: -weights.get(t, 0))[:12]
    results = await asyncio.gather(*(src_gnd.search(F, f'preferredName:"{re.sub(r"[^\w\s-]", " ", t)}"', 10) for t in terms))
    found: dict[str, tuple[str, str]] = {}
    for t, r in zip(terms, results):
        hits = [m for m in r.get("member", []) if re.sub(r"\s*<.*?>", "", m["preferredName"]).lower() == t.lower()
                and "<Motiv>" not in m["preferredName"]]
        hits.sort(key=lambda m: "<" in m["preferredName"])
        if hits:
            found[t] = (src_gnd.clean_title(hits[0]["preferredName"], ""), src_gnd.full(src_gnd.gid(hits[0])))
    if not found:
        return None
    cand = list(found)
    sims = (await embed(cand, model)) @ centroid + COVERAGE_WEIGHT * np.array([weights.get(t, 0) for t in cand])
    return [found[cand[i]] for i in np.argsort(-sims)]


COVERAGE_WEIGHT = 0.6


def frequent_nouns(texts: list[str], group: list[int], cmap: dict, n: int = 10) -> list[str]:
    """Großgeschriebene Wörter, die in vielen Inhalten der Gruppe vorkommen (Kandidaten für Oberbegriffe)."""
    df = Counter()
    for i in group:
        df.update({w.lower() for w in re.findall(r"[^\W\d_][\w-]{2,}", texts[i]) if w.lower() not in STOP})
    return [cmap.get(w, w) for w, _ in df.most_common(60) if cmap.get(w, w)[:1].isupper()][:n]


def base_title(t: str) -> str:
    return re.sub(r"\s*\(.*\)$", "", t)


async def wiki_label(F, cand: dict, covf, centroid: np.ndarray, model: str, prep: str) -> list[tuple]:
    """Hybrid: Wikipedia-Artikel zu den Gruppenbegriffen suchen, den passendsten zum Gruppenzentrum wählen."""
    terms = sorted(cand, key=lambda t: -cand[t])[:6]
    results = await asyncio.gather(*(F.get_json(src_wikipedia.API, {
        "action": "query", "list": "search", "srsearch": t, "srlimit": 3, "srnamespace": 0,
        "format": "json", "formatversion": 2}) for t in terms))
    titles = []
    for r in results:
        for h in (r or {}).get("query", {}).get("search", []):
            t = h["title"]
            if t not in titles and not t.startswith("Liste ") and "Begriffsklärung" not in t:
                titles.append(t)
    if not titles:
        return []
    ex = await src_wikipedia.extracts(F, titles)
    titles = [t for t in titles if ex.get(t) and "steht für" not in ex[t][:200]]
    if not titles:
        return []
    raw = [f"{t}. {sentences(ex[t], 10, 30)}" for t in titles]
    reps = [p or t for p, t in zip(prep_texts(raw, prep), titles)]
    score = (await embed(reps, model)) @ centroid + COVERAGE_WEIGHT * np.array([covf(base_title(t)) for t in titles])

    def off_topic(t):  # Klammerzusatz, der in der Gruppe nie vorkommt, z. B. „Weg-Zeit-Diagramm (Linienbaustelle)“
        q = re.search(r"\(([^)]+)\)$", t)
        return 0.3 if q and covf(q.group(1)) == 0 and q.group(1) not in ("Physik", "Begriff") else 0.0

    score = score - np.array([off_topic(t) for t in titles])
    return [(titles[i], src_wikipedia.wiki_url(titles[i]), sentences(ex[titles[i]])) for i in np.argsort(-score)]


async def load_taxonomy(F, cfg: dict, model: str, prep: str) -> dict:
    """Hybrid: Baum aus Wikipedia/Wikidata/GND zum Thema als Vokabular für die Gruppennamen."""
    sc = cfg.get("taxonomy", {})
    module = SCAFFOLD_SOURCES[sc.get("source", "wikipedia")]
    tree, _ = await module.build(F, sc.get("theme", "Physik"), sc.get("entry") or None,
                                 int(sc.get("n1", 12)), int(sc.get("n2", 10)), int(sc.get("n3", 5)), True)
    nodes = []

    def walk(n, parent, depth):
        for ch in n["children"]:
            nodes.append({"node": ch, "parent": parent, "depth": depth})
            walk(ch, len(nodes) - 1, depth + 1)

    walk(tree, None, 1)
    if not nodes:
        raise ValueError("Die Taxonomie ist leer. Bitte ein anderes Thema oder eine andere Quelle wählen.")
    raw = [f"{x['node']['title']}. {sentences(x['node']['description'], 15, 40)}" for x in nodes]
    reps = [p or x["node"]["title"] for p, x in zip(prep_texts(raw, prep), nodes)]
    return {"tree": tree, "nodes": nodes, "E": await embed(reps, model), "source": sc.get("source", "wikipedia")}


def tax_descendants(tax: dict, idx: int) -> list[int]:
    out = []
    for j, x in enumerate(tax["nodes"]):
        p = x["parent"]
        while p is not None:
            if p == idx:
                out.append(j)
                break
            p = tax["nodes"][p]["parent"]
    return out


async def label_groups(F, texts, groups, E, mode, model, cmap, avoid: set | None = None, prep: str = "full",
                       tax: dict | None = None, parent_tax: int | None = None) -> list[dict]:
    """Benennt Geschwistergruppen; Namen aus `avoid` (z. B. der Elterngruppe) und doppelte werden übersprungen."""
    used = {a.lower() for a in (avoid or set())}
    terms = ctfidf_terms(texts, groups)
    out = []
    for g, ts in zip(groups, terms):
        shown = [display(t, cmap) for t in ts]
        out.append({"title": " · ".join(shown[:3]) or "Sonstiges", "keywords": shown[:8], "url": "", "kind": "Schlagwörter (c-TF-IDF)"})
    if mode == "keywords" or E is None:
        return out
    cents = [E[g].mean(axis=0) for g in groups]
    cents = [c / max(np.linalg.norm(c), 1e-9) for c in cents]
    # Kandidaten: unterscheidende Begriffe (c-TF-IDF) plus häufige Substantive der Gruppe; nur Substantive.
    cands = []
    for g, ts in zip(groups, terms):
        pool = list(dict.fromkeys([display(t, cmap) for t in ts[:15]] + frequent_nouns(texts, g, cmap)))
        pool = [x for x in pool if all(w[:1].isupper() for w in x.split())]
        cands.append({x: coverage(texts, g, x) for x in pool})
    if mode == "korpus":
        for o, cand, c in zip(out, cands, cents):
            if not cand:
                continue
            names = list(cand)
            score = (await embed(names, model)) @ c + COVERAGE_WEIGHT * np.array([cand[x] for x in names])
            for i in np.argsort(-score):
                if names[i].lower() not in used:
                    used.add(names[i].lower())
                    o["title"], o["kind"] = names[i], "Begriff aus den Texten"
                    break
    elif mode == "gnd":
        found = await asyncio.gather(*(gnd_label(F, list(cand), c, model, cand) for cand, c in zip(cands, cents)))
        for o, ranked in zip(out, found):
            pick = next((f for f in ranked or [] if f[0].lower() not in used), None)
            if pick:
                used.add(pick[0].lower())
                o["title"], o["url"], o["kind"] = pick[0], pick[1], "GND-Schlagwort"
            else:
                o["kind"] = "kein passendes GND-Schlagwort, Schlagwörter"
    elif mode == "wikipedia":
        require_wikimedia()
        found = await asyncio.gather(*(wiki_label(F, cand, (lambda t, g=g: coverage(texts, g, t)), c, model, prep)
                                       for cand, c, g in zip(cands, cents, groups)))
        for o, ranked in zip(out, found):
            pick = next((f for f in ranked if f[0].lower() not in used), None)
            if pick:
                used.add(pick[0].lower())
                o["title"], o["url"], o["desc"], o["kind"] = pick[0], pick[1], pick[2], "Wikipedia-Artikel"
            else:
                o["kind"] = "kein passender Wikipedia-Artikel, Schlagwörter"
    elif mode == "taxonomie" and tax:
        # Hauptgruppen nur auf die oberen zwei Ebenen der Taxonomie abbilden (sonst zu spezielle Namen),
        # Untergruppen bevorzugt auf Knoten unterhalb des Knotens ihrer Elterngruppe.
        if parent_tax is None:
            pool = [j for j, x in enumerate(tax["nodes"]) if x["depth"] <= 2]
        else:
            pool = tax_descendants(tax, parent_tax)
            if len(pool) < 2:
                pool = [j for j, x in enumerate(tax["nodes"]) if x["depth"] <= 3]
        for o, c in zip(out, cents):
            score = tax["E"][pool] @ c
            if parent_tax is None:  # allgemeinere Knoten (Ebene 1) leicht bevorzugen
                score = score + np.array([0.08 if tax["nodes"][j]["depth"] == 1 else 0.0 for j in pool])
            for i in np.argsort(-score):
                n = tax["nodes"][pool[i]]["node"]
                if n["title"].lower() not in used:
                    used.add(n["title"].lower())
                    o.update(title=n["title"], url=n["url"], desc=n["description"], _tax=pool[i],
                             kind=f"Knoten der Taxonomie ({n['kind']}) · Ähnlichkeit {float(score[i]):.2f}")
                    break
    return out


def medoid_titles(idx: list[int], X: np.ndarray, contents: list[dict], n: int = 3) -> str:
    c = X[idx].mean(axis=0)
    order = sorted(idx, key=lambda i: -float(X[i] @ c))[:n]
    return "Typische Inhalte: " + ", ".join(f"„{contents[i]['title']}“" for i in order) + "."


MODEL_INFO = {
    "mpnet": "paraphrase-multilingual-mpnet-base-v2 (Transformer, 768 Dimensionen, lokal über fastembed/ONNX)",
    "minilm": "paraphrase-multilingual-MiniLM-L12-v2 (Transformer, 384 Dimensionen, lokal über fastembed/ONNX)",
    "m2v-edu": "JanSchachtschabel/m2v-gte-256-edu (statische Wortvektoren über model2vec, 256 Dimensionen, ohne Transformer)",
}
PREP_INFO = {"full": "Volltext", "nostop": "ohne Füllwörter", "nouns": "nur Substantive (großgeschriebene Wörter ohne Füllwörter)",
             "lemma": "Substantive in Grundform, zusammengesetzte Wörter zusätzlich mit ihrem Grundwort (z. B. Spannungsquelle + Quelle)"}
LABEL_INFO = {
    "keywords": "Schlagwortliste: die 3 Begriffe, die die Gruppe am stärksten von ihren Geschwistergruppen unterscheiden (c-TF-IDF wie bei BERTopic).",
    "korpus": "Begriff aus den Texten: Kandidaten sind unterscheidende Begriffe (c-TF-IDF) und häufige Substantive der Gruppe; gewählt wird der Begriff, der dem Gruppenzentrum am nächsten ist und in vielen Inhalten der Gruppe vorkommt.",
    "gnd": "GND-Schlagwort (Hybrid): Gruppenbegriffe werden in der GND (lobid) nachgeschlagen; nur Schlagwörter mit genau diesem Namen zählen, gewählt wird das passendste zum Gruppenzentrum.",
    "wikipedia": "Wikipedia-Artikel (Hybrid): Zu den häufigsten Gruppenbegriffen werden Wikipedia-Artikel gesucht; gewählt wird der Artikel, dessen Titel und Einleitung dem Gruppenzentrum am nächsten sind. Beschreibung = Artikeleinleitung.",
    "taxonomie": "In Taxonomie einordnen (Hybrid): Ein Baum zum Thema (Wikipedia/Wikidata/GND) dient als Vokabular; jede Gruppe bekommt den Namen des ähnlichsten Knotens, Untergruppen bevorzugt einen Knoten unterhalb des Knotens ihrer Elterngruppe.",
}


def cluster_pipeline(cfg: dict, prep: str, k_used: int, auto: bool, tax: dict | None) -> list[str]:
    feats = cfg.get("features", "embedding")
    steps = [f"Text je Inhalt: Titel (doppelt gewichtet) + Beschreibung; für die Embeddings: {PREP_INFO.get(prep, prep)}."]
    if feats == "embedding":
        steps.append(f"Embeddings: {MODEL_INFO.get(cfg.get('model', 'mpnet'))}. Kein generatives LLM.")
    else:
        steps.append("TF-IDF über Wörter und Wortpaare (deutsche und Plattform-Füllwörter entfernt), verdichtet per LSA auf bis zu 60 Dimensionen.")
    steps.append(f"Hierarchisches Clustering (agglomerativ, Ward-Verfahren): Ebene 1 = {k_used} Gruppen "
                 f"({'per Silhouette-Wert gewählt' if auto else 'fest vorgegeben'}); Gruppen ab {cfg.get('split_min', 8)} Inhalten "
                 f"werden erneut hierarchisch in 2–{cfg.get('k2max', 4)} Untergruppen geteilt (Anzahl per Silhouette).")
    steps.append("Benennung: " + LABEL_INFO.get(cfg.get("labels", "keywords"), ""))
    if tax:
        steps.append(f"Taxonomie: {len(tax['nodes'])} Knoten aus {tax['source']} zum Thema „{tax['tree']['title']}“.")
    steps.append("Gruppenbeschreibung: " + ("Text des gewählten Artikels/Knotens plus " if cfg.get("labels") in ("wikipedia", "taxonomie") else "")
                 + "die 3 Inhalte, die dem Gruppenzentrum am nächsten liegen.")
    return steps


async def build_cluster(F, contents, cfg) -> tuple[dict, dict]:
    texts = [doc_text(c) for c in contents]
    model = cfg.get("model", "mpnet")
    mode = cfg.get("labels", "keywords")
    need_emb = cfg.get("features") == "embedding" or mode in ("korpus", "gnd", "wikipedia", "taxonomie")
    prep = resolve_prep(cfg)
    E = await embed(prep_texts(texts, prep), model) if need_emb else None
    X = E if cfg.get("features") == "embedding" else tfidf_features(texts)
    cmap = casing_map([c["title"] + " " + c.get("description", "") for c in contents])
    tax = await load_taxonomy(F, cfg, model, prep) if mode == "taxonomie" else None
    n = len(contents)

    k1 = cfg.get("k1")
    top_groups, sil = split(X, list(range(n)), k1, 4, min(14, max(4, n // 4)))
    labels1 = await label_groups(F, texts, top_groups, E, mode, model, cmap, prep=prep, tax=tax)
    split_min, k2max = int(cfg.get("split_min", 8)), int(cfg.get("k2max", 4))

    def describe(lab, g):
        return (lab.pop("desc", "") + " " if lab.get("desc") else "") + medoid_titles(g, X, contents)

    root = {"title": "Inhaltsmenge", "children": []}
    pred = [0] * n
    level2 = 0
    for gi, (g, lab) in enumerate(zip(top_groups, labels1)):
        for i in g:
            pred[i] = gi
        parent_tax = lab.pop("_tax", None)
        node = {**lab, "description": describe(lab, g), "size": len(g), "gold": gold_summary(g, contents),
                "children": [], "items": []}
        node.pop("desc", None)
        if len(g) >= split_min and k2max >= 2:
            subs, _ = split(X, g, None, 2, max(2, min(k2max, len(g) // 3)))
            if len(subs) > 1:
                sub_labels = await label_groups(F, texts, subs, E, mode, model, cmap, {node["title"]},
                                                prep=prep, tax=tax, parent_tax=parent_tax)
                for s, sl in zip(subs, sub_labels):
                    sl.pop("_tax", None)
                    child = {**sl, "description": describe(sl, s), "size": len(s),
                             "gold": gold_summary(s, contents), "children": [], "items": s}
                    child.pop("desc", None)
                    node["children"].append(child)
                    level2 += 1
            else:
                node["items"] = g
        else:
            node["items"] = g
        root["children"].append(node)
    stats = {"text_prep": prep, "levels": [len(top_groups), level2], "silhouette": round(sil, 3) if sil is not None else None,
             "unassigned": 0, "metrics": metrics(pred, contents),
             "pipeline": cluster_pipeline(cfg, prep, len(top_groups), k1 is None, tax)}
    if tax:
        stats.update(scaffold_entry=tax["tree"]["title"], scaffold_url=tax["tree"]["url"])
    return root, stats


async def build_scaffold(F, contents, cfg) -> tuple[dict, dict]:
    sc = cfg.get("scaffold", {})
    module = SCAFFOLD_SOURCES[sc.get("source", "wikipedia")]
    tree, _ = await module.build(F, sc.get("theme", "Physik"), sc.get("entry") or None,
                                 int(sc.get("n1", 12)), int(sc.get("n2", 10)), int(sc.get("n3", 5)), True)
    model = cfg.get("model", "mpnet")
    flat = []  # (node, level, l1_index)

    def walk(n, lvl, l1):
        for i, ch in enumerate(n["children"]):
            top = i if lvl == 1 else l1
            flat.append((ch, lvl, top))
            walk(ch, lvl + 1, top)

    walk(tree, 1, None)
    if not flat:
        raise ValueError("Das Gerüst ist leer. Bitte ein anderes Thema oder eine andere Quelle wählen.")
    node_texts = [f"{n['title']}. {sentences(n['description'], 15, 40)}" for n, _, _ in flat]
    prep = resolve_prep(cfg)
    prepared = prep_texts([doc_text(c) for c in contents] + node_texts, prep)  # gemeinsamer Wortbestand für Komposita
    E_items = await embed(prepared[:len(contents)], model)
    E_nodes = await embed([p or x for p, x in zip(prepared[len(contents):], node_texts)], model)
    S = E_items @ E_nodes.T
    best, score = S.argmax(axis=1), S.max(axis=1)
    thr = float(cfg.get("threshold", 0.3))

    for n, _, _ in flat:
        n["items"], n["size"] = [], 0
    unassigned, pred = [], []
    for i, (b, s) in enumerate(zip(best, score)):
        if s < thr:
            unassigned.append(i)
            pred.append(-1)
        else:
            flat[b][0]["items"].append(i)
            pred.append(flat[b][2])

    def prune(n):
        kids = [prune(ch) for ch in n["children"]]
        kids = [k for k in kids if k]
        size = len(n.get("items", [])) + sum(k["size"] for k in kids)
        if size == 0:
            return None
        all_items = list(n.get("items", [])) + [i for k in kids for i in k["_all"]]
        return {"title": n["title"], "description": n["description"], "keywords": n["keywords"], "url": n["url"],
                "kind": n["kind"], "size": size, "gold": gold_summary(all_items, contents), "children": kids,
                "items": n.get("items", []), "_all": all_items}

    kids = [k for k in (prune(ch) for ch in tree["children"]) if k]
    if unassigned:
        kids.append({"title": "Nicht zugeordnet", "description": f"Ähnlichkeit zu keinem Knoten über {thr:.2f}.",
                     "keywords": [], "url": "", "kind": "Rest", "size": len(unassigned),
                     "gold": gold_summary(unassigned, contents), "children": [], "items": unassigned, "_all": unassigned})
    root = {"title": tree["title"], "url": tree["url"], "children": kids}
    stats = {"text_prep": prep, "levels": [len(kids), sum(len(k["children"]) for k in kids)], "silhouette": None,
             "unassigned": len(unassigned), "scaffold_nodes": len(flat),
             "used_nodes": sum(1 for n, _, _ in flat if n["items"]), "metrics": metrics(pred, contents),
             "scaffold_entry": tree["title"], "scaffold_url": tree["url"],
             "pipeline": [
                 f"Gerüst: Baum aus {sc.get('source', 'wikipedia')} zum Thema „{tree['title']}“ mit {len(flat)} Knoten "
                 f"(dieselbe Erzeugung wie auf der Seite „Thema → Baum“).",
                 f"Text je Inhalt: Titel + Beschreibung ({PREP_INFO.get(prep, prep)}); je Knoten: Titel + erste Sätze der Beschreibung.",
                 f"Embeddings: {MODEL_INFO.get(model)}. Kein generatives LLM.",
                 f"Einordnung: Jeder Inhalt kommt an den Knoten mit der höchsten Kosinus-Ähnlichkeit; unter {thr:.2f} bleibt er „nicht zugeordnet“.",
                 "Leere Äste werden gekappt; Namen und Beschreibungen stammen aus dem Gerüst.",
             ]}
    return root, stats


async def name_fit(root: dict) -> float | None:
    """Wie nah liegen die Namen der Hauptgruppen an ihrer überwiegenden Soll-Kategorie? (Kosinus, MiniLM)"""
    pairs = [(n["title"], n["gold"]["top"]) for n in root["children"] if n.get("gold") and n["title"] != "Nicht zugeordnet"]
    if not pairs:
        return None
    a = await embed([p[0] for p in pairs], "minilm")
    b = await embed([p[1] for p in pairs], "minilm")
    return round(float(np.mean(np.sum(a * b, axis=1))), 3)


async def build(contents: list[dict], cfg: dict) -> dict:
    contents = [c for c in contents if (c.get("title") or "").strip()]
    if len(contents) < 8:
        raise ValueError("Bitte mindestens 8 Inhalte mit Titel angeben.")
    for i, c in enumerate(contents):
        c.setdefault("id", f"c{i + 1}")
    F = Fetcher()
    t0 = time.perf_counter()
    if cfg.get("method") == "scaffold":
        root, stats = await build_scaffold(F, contents, cfg)
    else:
        root, stats = await build_cluster(F, contents, cfg)

    def strip(n):
        n.pop("_all", None)
        for ch in n.get("children", []):
            strip(ch)

    strip(root)
    seconds = round(time.perf_counter() - t0, 1)
    if stats.get("metrics"):
        stats["metrics"]["name_fit"] = await name_fit(root)
    stats.update({"seconds": seconds, "api_calls": F.calls, "items": len(contents)})
    return {"config": cfg, "stats": stats, "tree": root,
            "contents": [{"id": c["id"], "title": c["title"], "resourceType": c.get("resourceType", ""),
                          "level": c.get("level", ""), "gold": c.get("gold", ""),
                          "description": c.get("description", "")} for c in contents]}
