"""Quelle GND (über lobid-gnd).

Zwei Wege, je nachdem, was die GND zum Einstieg hergibt:
- Hierarchie: Unterbegriffe (broaderTerm*) des Schlagworts, bei Bedarf ergänzt um verwandte Begriffe.
- Sachgruppen: Ist das Schlagwort eine Fachdisziplin mit kaum Unterbegriffen (z. B. „Physik“),
  bilden die GND-Sachgruppen des Fachs (21.1, 21.2, …) Ebene 1. Ebene 2 sind die Begriffe, auf die
  innerhalb der Sachgruppe die meisten anderen als Oberbegriff verweisen, Ebene 3 deren Unterbegriffe.
"""

import asyncio
import re
from collections import Counter, defaultdict

from .common import CONTACT, NotFound, node, sentences
from .wikidata import dewiki_titles
from .wikipedia import extracts

SEARCH = "https://lobid.org/gnd/search"
RECORD = "https://lobid.org/gnd/{}.json"
SC_TTL = "https://d-nb.info/standards/vocab/gnd/gnd-sc.ttl"
SC_BASE = "https://d-nb.info/standards/vocab/gnd/gnd-sc#"
TYPE_FILTER = "type:SubjectHeadingSensoStricto"
GROUP_CAP = 1500


def gid(m: dict) -> str:
    return m["id"].rsplit("/", 1)[1]


def full(i: str) -> str:
    return "https://d-nb.info/gnd/" + i


def qid(m: dict) -> str | None:
    for s in m.get("sameAs", []):
        hit = re.search(r"wikidata\.org/entity/(Q\d+)", s.get("id", ""))
        if hit:
            return hit.group(1)
    return None


def narrower_query(i: str) -> str:
    u = full(i)
    return f'broaderTermGeneric.id:"{u}" OR broaderTermPartitive.id:"{u}" OR broaderTermInstantial.id:"{u}"'


async def search(F, q: str, size: int = 100, frm: int = 0) -> dict:
    return await F.get_json(SEARCH, {"q": q, "filter": TYPE_FILTER, "format": "json", "size": size, "from": frm}) or {}


async def record(F, i: str) -> dict | None:
    return await F.get_json(RECORD.format(i))


async def count_narrower(F, i: str) -> int:
    return (await search(F, narrower_query(i), 1)).get("totalItems", 0)


async def candidates(F, theme: str) -> list[dict]:
    safe = re.sub(r"[^\w\s-]", " ", theme).strip()
    exact, free = await asyncio.gather(search(F, f'preferredName:"{safe}"', 8), search(F, safe, 8))
    members, ids = [], set()
    for m in exact.get("member", []) + free.get("member", []):
        if gid(m) not in ids:
            ids.add(gid(m))
            members.append(m)
    members.sort(key=lambda m: m["preferredName"].lower() != theme.lower())
    return [{"id": gid(m), "label": m["preferredName"],
             "hint": ", ".join(c.get("label", "") for c in m.get("gndSubjectCategory", []))} for m in members]


def clean_title(name: str, root_label: str) -> str:
    hit = re.match(r"^(.*?)\s*<(.+)>$", name)
    if not hit:
        return name
    base, qual = hit.groups()
    return base if qual.lower() == root_label.lower() else f"{base} ({qual})"


def definition(m: dict) -> str:
    d = m.get("definition")
    if isinstance(d, list):
        d = " ".join(x if isinstance(x, str) else x.get("label", "") for x in d)
    return d or ""


async def expand(F, i: str, n: int, seen: set, rank: bool = False, rec: dict | None = None) -> list[dict]:
    """Unterbegriffe von i; sind es zu wenige, werden verwandte Begriffe ergänzt."""
    if n <= 0:
        return []
    cands = [m for m in (await search(F, narrower_query(i))).get("member", []) if gid(m) not in seen]
    kind = {gid(m): "Unterbegriff" for m in cands}
    if len(cands) < n:
        rec = rec or await record(F, i) or {}
        rel_ids = [r["id"].rsplit("/", 1)[1] for r in rec.get("relatedTerm", [])][:15]
        rel_recs, rev = await asyncio.gather(
            asyncio.gather(*(record(F, r) for r in rel_ids)),
            search(F, f'relatedTerm.id:"{full(i)}"', 40),
        )
        for m in [r for r in rel_recs if r] + rev.get("member", []):
            if gid(m) not in seen and gid(m) not in kind:
                cands.append(m)
                kind[gid(m)] = "verwandter Begriff"
    if rank and len(cands) > n:
        top = cands[:40]
        counts = await asyncio.gather(*(count_narrower(F, gid(m)) for m in top))
        size = dict(zip((gid(m) for m in top), counts))
        cands = top
    else:
        size = {}
    cands.sort(key=lambda m: (kind[gid(m)] != "Unterbegriff", -size.get(gid(m), 0), qid(m) is None, len(m["preferredName"])))
    out = []
    for m in cands:
        if gid(m) in seen:
            continue
        seen.add(gid(m))
        m["_kind"] = kind[gid(m)]
        out.append(m)
        if len(out) >= n:
            break
    return out


async def subject_groups(F) -> dict[str, str]:
    """GND-Sachgruppen: Notation -> deutscher Name."""
    text = await F.get_text(SC_TTL) or ""
    out = {}
    for block in text.split("\n\n"):
        mn = re.search(r"gnd-sc#([^>\s]+)>\s*\n\s*a skos:Concept", block)
        ml = re.search(r"skos:prefLabel(.*?)(?:\s;\s*\n|\s\.\s*$)", block, re.S)
        if mn and ml:
            de = re.search(r'"([^"]+)"@de', ml.group(1))
            if de:
                out[mn.group(1)] = de.group(1)
    return out


async def group_members(F, notation: str) -> list[dict]:
    q = f'gndSubjectCategory.id:"{SC_BASE}{notation}"'
    first = await search(F, q, 100, 0)
    total = min(first.get("totalItems", 0), GROUP_CAP)
    pages = await asyncio.gather(*(search(F, q, 100, f) for f in range(100, total, 100)))
    return first.get("member", []) + [m for p in pages for m in p.get("member", [])]


def notation_key(n: str):
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", n)]


async def build(F, theme: str, entry: str | None, n1: int, n2: int, n3: int, enrich: bool):
    enrich = enrich and bool(CONTACT)  # Wikipedia-Beschreibungen nur mit Wikimedia-Kontakt
    alts = await candidates(F, theme)
    i = entry or (alts[0]["id"] if alts else None)
    if not i:
        raise NotFound(f"Kein GND-Schlagwort zu „{theme}“ gefunden.")
    rec = await record(F, i)
    if not rec:
        raise NotFound(f"GND-Datensatz {i} nicht gefunden.")
    root_label = rec["preferredName"]
    seen = {i}
    cats = rec.get("gndSubjectCategory", [])
    discipline = (await count_narrower(F, i)) < max(3, n1 // 3) and any("Allgemein" in c.get("label", "") for c in cats)
    mode = "Sachgruppen" if discipline else "Hierarchie"

    if discipline:
        vocab = await subject_groups(F)
        main = cats[0]["id"].split("#")[1].split(".")[0]
        # Sachgruppen mit „p“ am Ende sind Personengruppen („Personen zu Physik“) und gehören nicht in den Baum.
        groups = sorted([k for k in vocab if k.split(".")[0] == main and "." in k and not k.endswith("p")],
                        key=notation_key)[:n1]
        all_members = await asyncio.gather(*(group_members(F, g) for g in groups))
        l1, fill_tasks = [], []
        for g, members in zip(groups, all_members):
            by_id = {gid(m): m for m in members}
            hub, kids = Counter(), defaultdict(list)
            for m in members:
                for b in m.get("broaderTermGeneric", []) + m.get("broaderTermPartitive", []):
                    b_id = b["id"].rsplit("/", 1)[1]
                    if b_id in by_id:
                        hub[b_id] += 1
                        kids[b_id].append(gid(m))
            ranked = [by_id[k] for k, _ in hub.most_common()]
            ranked += sorted((m for m in members if gid(m) not in hub and qid(m)), key=lambda m: len(m["preferredName"]))
            l2 = []
            for m in ranked:
                if len(l2) >= n2:
                    break
                if gid(m) not in seen:
                    seen.add(gid(m))
                    m["_kind"] = f"zentraler Begriff der Sachgruppe ({hub.get(gid(m), 0)} Unterbegriffe dort)"
                    l2.append(m)
            for h in l2:
                h["kids"] = []
                for c in sorted(kids.get(gid(h), []), key=lambda c: -hub.get(c, 0)):
                    if len(h["kids"]) >= n3:
                        break
                    if c not in seen:
                        seen.add(c)
                        by_id[c]["_kind"] = "Unterbegriff"
                        h["kids"].append(by_id[c])
            l1.append({"id": "sc#" + g, "preferredName": vocab[g], "_kind": f"GND-Sachgruppe {g}", "_group": g,
                       "kids": l2, "_size": len(members)})
            fill_tasks += [h for h in l2 if len(h["kids"]) < n3]
        extra = await asyncio.gather(*(expand(F, gid(h), n3 - len(h["kids"]), seen) for h in fill_tasks))
        for h, more in zip(fill_tasks, extra):
            h["kids"] += more
    else:
        l1 = await expand(F, i, n1, seen, rank=True, rec=rec)
        l2s = await asyncio.gather(*(expand(F, gid(m), n2, seen) for m in l1))
        for m, kids in zip(l1, l2s):
            m["kids"] = kids
        l2 = [k for kids in l2s for k in kids]
        l3s = await asyncio.gather(*(expand(F, gid(m), n3, seen) for m in l2))
        for m, kids in zip(l2, l3s):
            m["kids"] = kids

    everything = []

    def collect(ms):
        for m in ms:
            everything.append(m)
            collect(m.get("kids", []))

    collect(l1)
    ex, titles = {}, {}
    if enrich:
        qids = [q for q in (qid(m) for m in everything + [rec]) if q]
        titles = await dewiki_titles(F, qids)
        ex = await extracts(F, list(titles.values()))

    def describe(m):
        d = definition(m)
        if not d and enrich and qid(m) in titles:
            d = ex.get(titles[qid(m)], "")
        return sentences(d)

    def make(m):
        if "_group" in m:
            n = node(m["preferredName"], "", [], SC_BASE + m["_group"], f"{m['_kind']} · {m['_size']} Schlagwörter", m["id"])
        else:
            kws = [re.sub(r"\s*<.*?>", "", v) for v in m.get("variantName", [])[:4]]
            kws += [r.get("label", "") for r in m.get("relatedTerm", [])[:3]]
            n = node(clean_title(m["preferredName"], root_label), describe(m), kws, full(gid(m)), m.get("_kind", ""), gid(m))
        n["children"] = [make(k) for k in m.get("kids", [])]
        return n

    root = node(root_label, describe(rec), [], full(i), f"GND-Schlagwort · Modus: {mode}", i)
    root["children"] = [make(m) for m in l1]
    return root, alts
