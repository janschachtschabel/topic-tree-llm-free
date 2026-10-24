"""Quelle Wikidata: „Teil von“ (P361), „Unterklasse von“ (P279), „hat Teil“ (P527), sortiert nach Sitelinks."""

import asyncio
from urllib.parse import unquote

from .common import NotFound, node, require_wikimedia, sentences
from .wikipedia import extracts, wiki_url

API = "https://www.wikidata.org/w/api.php"
SPARQL = "https://query.wikidata.org/sparql"


def qid_url(q: str) -> str:
    return f"https://www.wikidata.org/wiki/{q}"


async def candidates(F, theme: str) -> list[dict]:
    d = await F.get_json(API, {"action": "wbsearchentities", "search": theme, "language": "de", "uselang": "de",
                               "type": "item", "limit": 7, "format": "json"})
    return [{"id": x["id"], "label": x.get("label", x["id"]), "hint": x.get("description", "")}
            for x in (d or {}).get("search", [])]


async def dewiki_titles(F, qids) -> dict[str, str]:
    """QID -> Titel des deutschen Wikipedia-Artikels (50 je Aufruf)."""
    qids = list(dict.fromkeys(qids))
    out: dict[str, str] = {}

    async def chunk(ids):
        d = await F.get_json(API, {"action": "wbgetentities", "ids": "|".join(ids), "props": "sitelinks",
                                   "sitefilter": "dewiki", "format": "json"})
        for q, e in (d or {}).get("entities", {}).items():
            t = e.get("sitelinks", {}).get("dewiki", {}).get("title")
            if t:
                out[q] = t

    await asyncio.gather(*(chunk(qids[i:i + 50]) for i in range(0, len(qids), 50)))
    return out


async def children(F, parents: list[str]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}

    async def chunk(ps):
        values = " ".join("wd:" + p for p in ps)
        q = f"""SELECT ?p ?x ?l ?sl ?art (GROUP_CONCAT(DISTINCT ?a; separator="|") AS ?alias) WHERE {{
          VALUES ?p {{ {values} }}
          {{ ?x wdt:P361 ?p }} UNION {{ ?x wdt:P279 ?p }} UNION {{ ?p wdt:P527 ?x }}
          ?x rdfs:label ?l . FILTER(lang(?l) = "de")
          ?x wikibase:sitelinks ?sl .
          ?art schema:about ?x ; schema:isPartOf <https://de.wikipedia.org/> .
          OPTIONAL {{ ?x skos:altLabel ?a . FILTER(lang(?a) = "de") }}
        }} GROUP BY ?p ?x ?l ?sl ?art"""
        d = await F.get_json(SPARQL, {"query": q, "format": "json"})
        for b in (d or {}).get("results", {}).get("bindings", []):
            p = b["p"]["value"].rsplit("/", 1)[1]
            out.setdefault(p, []).append({
                "id": b["x"]["value"].rsplit("/", 1)[1],
                "label": b["l"]["value"],
                "sitelinks": int(b["sl"]["value"]),
                "article": unquote(b["art"]["value"].rsplit("/wiki/", 1)[1]).replace("_", " "),
                "aliases": [a for a in b.get("alias", {}).get("value", "").split("|") if a],
            })

    await asyncio.gather(*(chunk(parents[i:i + 40]) for i in range(0, len(parents), 40)))
    for v in out.values():
        v.sort(key=lambda c: -c["sitelinks"])
    return out


async def build(F, theme: str, entry: str | None, n1: int, n2: int, n3: int, enrich: bool):
    require_wikimedia()
    alts = await candidates(F, theme)
    qid = entry or (alts[0]["id"] if alts else None)
    if not qid:
        raise NotFound(f"Kein Wikidata-Objekt zu „{theme}“ gefunden.")
    d = await F.get_json(API, {"action": "wbgetentities", "ids": qid, "props": "labels|descriptions|sitelinks",
                               "languages": "de", "sitefilter": "dewiki", "format": "json"})
    ent = (d or {}).get("entities", {}).get(qid)
    if not ent or "missing" in ent:
        raise NotFound(f"Wikidata-Objekt {qid} nicht gefunden.")
    label = ent.get("labels", {}).get("de", {}).get("value", qid)
    root_article = ent.get("sitelinks", {}).get("dewiki", {}).get("title")

    seen = {qid}

    def pick(cands, n):
        out = []
        for c in cands:
            if c["id"] in seen:
                continue
            seen.add(c["id"])
            out.append(c)
            if len(out) >= n:
                break
        return out

    l1 = pick((await children(F, [qid])).get(qid, []), n1)
    ch2 = await children(F, [c["id"] for c in l1]) if n2 and l1 else {}
    for c in l1:
        c["kids"] = pick(ch2.get(c["id"], []), n2)
    l2 = [k for c in l1 for k in c["kids"]]
    ch3 = await children(F, [c["id"] for c in l2]) if n3 and l2 else {}
    for c in l2:
        c["kids"] = pick(ch3.get(c["id"], []), n3)
        for k in c["kids"]:
            k["kids"] = []

    everything = l1 + l2 + [k for c in l2 for k in c["kids"]]
    ex = await extracts(F, [root_article] + [c["article"] for c in everything]) if enrich else {}

    def make(c):
        n = node(c["label"], sentences(ex.get(c["article"], "")), c["aliases"][:5] + [c["article"]],
                 qid_url(c["id"]), f"{c['sitelinks']} Sprachversionen", c["id"])
        n["children"] = [make(k) for k in c.get("kids", [])]
        return n

    root_desc = sentences(ex.get(root_article or "", "")) or ent.get("descriptions", {}).get("de", {}).get("value", "")
    root = node(label, root_desc, [], qid_url(qid), "Wikidata-Objekt", qid)
    root["children"] = [make(c) for c in l1]
    return root, alts
