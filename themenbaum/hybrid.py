"""Quelle Hybrid: Wikipedia-Gliederung als Gerüst, ergänzt um Wikidata und GND (Test).

Ablauf:
1. Wikipedia-Baum mit doppelt so vielen Kandidaten je Ebene wie gewünscht.
2. Allgemeine Abschnitte ("Geschichte", "Rezeption", "Überblick" …) entfernen.
3. Jeden Knoten seinem Wikidata-Objekt zuordnen (Hauptartikel oder gleichnamiger Artikel).
4. Fehlende Unterthemen mit Wikidata-Unterbegriffen ("Teil von", "Unterklasse von", "hat Teil") auffüllen.
5. GND-ID aus Wikidata (P227) an jeden Knoten hängen.
Mit vote=True (Mehrheitsentscheid) werden Wikipedia- und Wikidata-Kandidaten gemeinsam nach Bestätigung sortiert:
Wikipedia (im Elternartikel/-abschnitt verlinkt) + Wikidata (Beziehung zum Elternobjekt) + GND (Eintrag vorhanden).
"""

import asyncio
import re

from .common import node, require_wikimedia, sentences
from . import wikidata as wd
from . import wikipedia as wp

GENERIC = re.compile(
    r"^(geschichte|forschungsgeschichte|begriffsgeschichte|wortgeschichte|etymologie|wortherkunft|begriff|"
    r"bezeichnung|namensgebung|definition|überblick|allgemeines|einleitung|rezeption|kritik|trivia|sonstiges|"
    r"weitere aspekte|kulturelle bedeutung|bedeutung|in der kultur|in der populärkultur|rezeption in .*|"
    r"geschichte .*|.* in der gesellschaft|beziehung(en)? zu .*|grenzen .*|siehe auch|ausblick|"
    r"(die )?erforschung .*|.*forschungsgeschichte|weiterführendes|grenzen|bekannte .*|.*persönlichkeiten|"
    r"wissenschaftliche zeitschriften.*|zeitschriften|organisationen|institutionen)$", re.I)
# Zeitabschnitte als Überschrift ("1960er Jahre", "Anfang des 20. Jahrhunderts", "Antike") – außer bei Geschichtsthemen
TEMPORAL = re.compile(
    r"(^\d{3,4}(er)?\b|\b\d{3,4}er jahre|\b\d{1,2}\. jahrhundert|jahrhunderts?\b|^(antike|mittelalter|frühe neuzeit|neuzeit|"
    r"moderne|gegenwart|zeitgeschichte|vorgeschichte|frühgeschichte)\b|mittelalterliche|^(seit|bis|ab|nach|vor) \d)", re.I)
HISTORICAL_THEME = re.compile(r"geschichte|epoche|zeitalter|antike|mittelalter|neuzeit|krieg|revolution|reich\b|dynastie|ära\b", re.I)
RELEVANCE = 0.38  # MiniLM-Ähnlichkeit Kandidat (Titel + Einleitung) zu Thema; gemessen an Beispielen, siehe docs/ideen.md
COUNTRY_SUFFIX = re.compile(r"\sin\s(?:[A-ZÄÖÜ]|der |den |dem )")  # z. B. "Klimawandel in Südkorea"


def generic(title: str) -> bool:
    return bool(GENERIC.match(title.strip()))


def temporal(title: str) -> bool:
    return bool(TEMPORAL.search(title.strip()))


async def qids_for_titles(F, titles) -> dict[str, str]:
    """Wikipedia-Titel -> Wikidata-QID (über pageprops, Weiterleitungen aufgelöst)."""
    titles = [t for t in dict.fromkeys(titles) if t]
    out: dict[str, str] = {}

    async def chunk(ts):
        d = await F.get_json(wp.API, {"action": "query", "prop": "pageprops", "ppprop": "wikibase_item",
                                      "titles": "|".join(ts), "redirects": 1, "format": "json", "formatversion": 2})
        q = (d or {}).get("query", {})
        mapping = {x["from"]: x["to"] for x in q.get("normalized", []) + q.get("redirects", [])}
        found = {p["title"]: p.get("pageprops", {}).get("wikibase_item") for p in q.get("pages", [])}
        for t in ts:
            r = t
            for _ in range(3):
                r = mapping.get(r, r)
            if found.get(r):
                out[t] = found[r]

    await asyncio.gather(*(chunk(titles[i:i + 50]) for i in range(0, len(titles), 50)))
    return out


async def gnd_ids(F, qids) -> dict[str, str]:
    qids = [q for q in dict.fromkeys(qids) if q]
    out: dict[str, str] = {}

    async def chunk(ids):
        d = await F.get_json(wd.API, {"action": "wbgetentities", "ids": "|".join(ids), "props": "claims", "format": "json"})
        for q, e in (d or {}).get("entities", {}).items():
            claims = e.get("claims", {}).get("P227", [])
            if claims and claims[0]["mainsnak"].get("datavalue"):
                out[q] = claims[0]["mainsnak"]["datavalue"]["value"]

    await asyncio.gather(*(chunk(qids[i:i + 50]) for i in range(0, len(qids), 50)))
    return out


async def confirmed_pairs(F, pairs) -> set[tuple[str, str]]:
    """Welche (Kind, Eltern)-Paare bestätigt Wikidata (direkt oder über eine Zwischenstufe)?"""
    pairs = [p for p in dict.fromkeys(pairs) if p[0] and p[1] and p[0] != p[1]]
    out: set[tuple[str, str]] = set()

    async def chunk(ps):
        values = " ".join(f"(wd:{c} wd:{p})" for c, p in ps)
        q = f"""SELECT DISTINCT ?c ?p WHERE {{ VALUES (?c ?p) {{ {values} }}
          {{ ?c (wdt:P361|wdt:P279) ?p }} UNION {{ ?p wdt:P527 ?c }}
          UNION {{ ?c (wdt:P361|wdt:P279)/(wdt:P361|wdt:P279) ?p }} }}"""
        d = await F.get_json(wd.SPARQL, {"query": q, "format": "json"})
        for b in (d or {}).get("results", {}).get("bindings", []):
            out.add((b["c"]["value"].rsplit("/", 1)[1], b["p"]["value"].rsplit("/", 1)[1]))

    await asyncio.gather(*(chunk(pairs[i:i + 60]) for i in range(0, len(pairs), 60)))
    return out


def norm(t: str) -> str:
    return re.sub(r"\s*\(.*?\)", "", t).strip().lower()


async def build(F, theme: str, entry: str | None, n1: int, n2: int, n3: int, enrich: bool, vote: bool | str = True,
                clean: bool = True):
    """vote: True = Mehrheitsentscheid auf allen Ebenen (Standard, im Test die beste Variante), "l1" = nur auf
    Ebene 1, False = Wikipedia-Reihenfolge, Wikidata nur zum Auffüllen. Vergleich: scripts/test_hybrid_thema.py."""
    require_wikimedia()
    root, alts = await wp.build(F, theme, entry, min(30, n1 * 2), min(20, n2 * 2), min(20, n3 * 2), enrich)
    targets = {1: n1, 2: n2, 3: n3}

    # 1) Allgemeine Abschnitte und (außer bei Geschichtsthemen) Zeitabschnitte entfernen
    top_temporal = sum(temporal(c["title"]) for c in root["children"]) >= max(2, len(root["children"]) / 2)
    historical = bool(HISTORICAL_THEME.search(root["title"])) or top_temporal

    def drop(c):
        return clean and (generic(c["title"]) or (not historical and temporal(c["title"])))

    def prune(n):
        n["children"] = [c for c in n["children"] if not drop(c)]
        for c in n["children"]:
            prune(c)
    prune(root)

    # 2) Wikidata-Objekte zuordnen: über den Hauptartikel, sonst über einen gleichnamigen Artikel, den der
    #    Elternabschnitt verlinkt (sonst landen Titel wie "Methodik" beim allgemeinen Begriff statt beim Fachbegriff).
    all_nodes = []

    def walk(n, lvl, parent):
        all_nodes.append((n, lvl, parent))
        for c in n["children"]:
            walk(c, lvl + 1, n)
    walk(root, 0, None)
    title_map = await qids_for_titles(F, [n["_main"] for n, _, _ in all_nodes if n.get("_main")] + [n["title"] for n, _, _ in all_nodes])
    for n, _, parent in all_nodes:
        linked = parent is not None and norm(n["title"]) in {norm(x) for x in (parent.get("_links") or [])}
        n["_qid"] = title_map.get(n.get("_main") or "") or (title_map.get(n["title"]) if linked or parent is None else None)
        n["_src"] = {"Wikipedia"}

    # Relevanzfilter für Wikidata-Kandidaten: Titel + Einleitung gegen das Thema (MiniLM, Schwelle gemessen)
    from . import contents as emb
    root_vec = (await emb.embed([f"{root['title']}. {root['description']}"], "minilm"))[0]

    async def relevant(cands):
        if not cands:
            return []
        ex = await wp.extracts(F, [c["_main"] for c in cands])
        for c in cands:
            c["description"] = sentences(ex.get(c["_main"], ""))
        vecs = await emb.embed([f"{c['title']}. {sentences(ex.get(c['_main'], ''), 25, 50)}" for c in cands], "minilm")
        sims = vecs @ root_vec
        for c, s in zip(cands, sims):
            c["_rel"] = float(s)
        return [c for c in cands if c["_rel"] >= RELEVANCE]

    # 3) + 4) Ebene für Ebene auffüllen; Wikidata-Unterbegriffe je Ebene abfragen (auch für neu ergänzte Knoten)
    def wd_node(c):
        k = node(c["label"], "", c["aliases"][:5] + [c["article"]], wd.qid_url(c["id"]), "Wikidata-Unterbegriff", c["id"])
        k.update(_qid=c["id"], _main=c["article"], _links=[], _src={"Wikidata"}, _wdlink=True, children=[])
        return k

    async def wp_children(title, want):
        """Abschnitte des Wikipedia-Artikels eines Wikidata-Knotens als dessen Unterthemen."""
        if not title or norm(title) == norm(root["title"]):
            return [], []
        art = await wp.fetch_article(F, title)
        if not art:
            return [], []
        _, secs = wp.split_sections(art[1], art[0])
        kids = []
        for s in secs:
            if drop(s):
                continue
            main = s.get("main")
            k = node(s["title"], sentences(wp.plain(s["text"])), ([main] if main else []) + wp.links(s["text"]),
                     wp.wiki_url(main) if main else wp.wiki_url(art[0], s["title"]),
                     f"Abschnitt mit Hauptartikel „{main}“" if main else f"Abschnitt in „{art[0]}“", s["title"])
            k.update(_main=main, _links=wp.links(s["text"], 300), _src={"Wikipedia"}, children=[])
            kids.append(k)
            if len(kids) >= 2 * want:
                break
        return kids, wp.links(art[1], 1000)

    level_nodes = [root]
    used = {norm(root["title"]), root.get("_qid")}
    for lvl in range(0, 3):
        want = targets[lvl + 1]
        if want <= 0 or not level_nodes:
            for n in level_nodes:
                n["children"] = []
            break
        wd_children = await wd.children(F, [n["_qid"] for n in level_nodes if n.get("_qid")])
        confirmed = set()
        use_vote = vote is True or (vote == "l1" and lvl == 0)
        if use_vote:
            pairs = [(c["_qid"], n["_qid"]) for n in level_nodes for c in n["children"] if c.get("_qid") and n.get("_qid")]
            confirmed = await confirmed_pairs(F, pairs)
        nxt = []
        per_parent = {}
        for n in level_nodes:  # Kandidaten sammeln (je Elternknoten bis zu dreimal so viele wie gebraucht)
            have = [c for c in n["children"] if norm(c["title"]) not in used and (not c.get("_qid") or c["_qid"] not in used)]
            seen = {norm(c["title"]) for c in have} | {c.get("_qid") for c in have if c.get("_qid")} | used
            cands = []
            for c in wd_children.get(n.get("_qid"), []):
                if c["id"] in seen or norm(c["label"]) in seen or COUNTRY_SUFFIX.search(c["label"]) or c["label"].startswith("Liste "):
                    continue
                seen.add(c["id"])
                cands.append(wd_node(c))
                if len(cands) >= 3 * want:
                    break
            per_parent[id(n)] = (have, cands)
        kept = {id(c) for c in await relevant([c for _, cs in per_parent.values() for c in cs])}
        for n in level_nodes:
            have, cands = per_parent[id(n)]
            cands = [c for c in cands if id(c) in kept]
            if use_vote:
                links = {norm(x) for x in (n.get("_links") or [])}
                for c in have:  # Wikipedia-Kinder: bestätigt Wikidata die Beziehung zum Elternobjekt?
                    if c.get("_qid") and n.get("_qid") and (c["_qid"], n["_qid"]) in confirmed:
                        c["_src"].add("Wikidata")
                for c in cands:  # Wikidata-Kandidaten: im Wikipedia-Elternabschnitt verlinkt?
                    if norm(c["_main"]) in links or norm(c["title"]) in links:
                        c["_src"].add("Wikipedia")
                pool = have + cands
                pool.sort(key=lambda c: -len(c["_src"]))  # stabil: bei Gleichstand bleibt die Wikipedia-Reihenfolge
                chosen = pool[:want]
            else:
                chosen = have[:want] + cands[:max(0, want - len(have[:want]))]
            for c in chosen:
                used.add(norm(c["title"]))
                if c.get("_qid"):
                    used.add(c["_qid"])
            n["children"] = chosen
            nxt.extend(chosen)
        level_nodes = nxt
        # Ergänzte Wikidata-Knoten auf Ebene 1 und 2 bekommen die Gliederung ihres Wikipedia-Artikels als Unterthemen
        if lvl < 2 and targets[lvl + 2] > 0:
            wd_new = [c for c in nxt if c.get("_wdlink") and not c["children"]]
            results = await asyncio.gather(*(wp_children(c["_main"], targets[lvl + 2]) for c in wd_new))
            fresh = []
            for c, (kids, links) in zip(wd_new, results):
                c["children"], c["_links"] = kids, links
                fresh += [(k, c) for k in kids]
            if fresh:
                tm = await qids_for_titles(F, [k["_main"] for k, _ in fresh if k.get("_main")] + [k["title"] for k, _ in fresh])
                for k, par in fresh:
                    linked = norm(k["title"]) in {norm(x) for x in par["_links"]}
                    k["_qid"] = tm.get(k.get("_main") or "") or (tm.get(k["title"]) if linked else None)
    def cut(n, lvl):
        if lvl >= 3:
            n["children"] = []
        for c in n["children"]:
            cut(c, lvl + 1)
    cut(root, 0)

    # 5) GND-IDs, Beschreibungen für neue Knoten, Herkunft
    final = []

    def collect(n):
        for c in n["children"]:
            final.append(c)
            collect(c)
    collect(root)
    gnd = await gnd_ids(F, [c.get("_qid") for c in final])
    if enrich:  # Beschreibungen der Wikidata-Knoten sind meist schon vom Relevanzfilter gesetzt
        ex = await wp.extracts(F, [c["_main"] for c in final if c.get("_wdlink") and not c["description"]])
        for c in final:
            if c.get("_wdlink") and not c["description"]:
                c["description"] = sentences(ex.get(c["_main"], ""))
    for c in final:
        g = gnd.get(c.get("_qid"))
        if g:
            c["_src"].add("GND")
            c["keywords"] = (c["keywords"] + [f"GND {g}"])[:8]
        order = [s for s in ("Wikipedia", "Wikidata", "GND") if s in c["_src"]]
        c["kind"] = f"{' + '.join(order)}" + (f" · GND {g}" if g else "") + (f" · {c['kind']}" if "Wikipedia" in c["_src"] and not c.get("_wdlink") else "")
        c["_gnd"] = g
    mode = {"l1": " (Mehrheitsentscheid auf Ebene 1)", True: " (Mehrheitsentscheid auf allen Ebenen)"}.get(vote, "")
    root["kind"] = f"Hybrid{mode}: Wikipedia-Artikel „{root['title']}“"
    return root, alts


async def build_vote_all(F, theme, entry, n1, n2, n3, enrich):
    return await build(F, theme, entry, n1, n2, n3, enrich, vote=True)


async def build_plain(F, theme, entry, n1, n2, n3, enrich):
    return await build(F, theme, entry, n1, n2, n3, enrich, vote=False)
