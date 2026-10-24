"""Quelle Wikipedia: Gliederung des Artikels zum Thema, tiefer über {{Hauptartikel}}-Verweise."""

import asyncio
import re
from urllib.parse import quote

from .common import NotFound, node, require_wikimedia, sentences

API = "https://de.wikipedia.org/w/api.php"
WIKI = "https://de.wikipedia.org/wiki/"

SKIP = {
    "siehe auch", "literatur", "weblinks", "einzelnachweise", "anmerkungen", "quellen", "fußnoten", "belege",
    "galerie", "filme", "film", "dokumentarfilme", "audio", "medien", "literatur und weblinks",
    "weblinks und literatur", "einzelnachweise und anmerkungen", "anmerkungen und einzelnachweise",
    "quellen und anmerkungen", "weiterführende literatur", "rezeption in film und literatur",
}
HEAD_RE = re.compile(r"^(={2,6})[ \t]*(.+?)[ \t]*\1[ \t]*$", re.M)
MAIN_RE = re.compile(r"\{\{\s*(?:Hauptartikel|Hauptseite)\s*\|([^{}]*)\}\}", re.I)
LINK_RE = re.compile(r"\[\[([^\]|#:]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")


def wiki_url(title: str, anchor: str | None = None) -> str:
    url = WIKI + quote(title.replace(" ", "_"))
    return url + ("#" + quote(anchor.replace(" ", "_")) if anchor else "")


def plain(wt: str) -> str:
    """Wikitext grob in Fließtext verwandeln (für Beschreibungen aus Abschnitten)."""
    t = re.sub(r"<!--.*?-->", "", wt, flags=re.S)
    t = re.sub(r"<ref[^>/]*/>", "", t)
    t = re.sub(r"<ref[^>]*>.*?</ref>", "", t, flags=re.S)
    t = re.sub(r"<(math|chem|score|gallery|syntaxhighlight)[^>]*>.*?</\1>", "", t, flags=re.S)
    prev = None
    while prev != t:  # verschachtelte Vorlagen von innen nach außen entfernen
        prev = t
        t = re.sub(r"\{\{[^{}]*\}\}", "", t)
    t = re.sub(r"\{\|.*?\|\}", "", t, flags=re.S)
    t = re.sub(r"\[\[(?:Datei|Bild|File|Image|Kategorie|Category):(?:[^\[\]]|\[\[[^\]]*\]\])*\]\]", "", t, flags=re.I)
    t = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", t)
    t = re.sub(r"\[\[([^\]]+)\]\]", r"\1", t)
    t = re.sub(r"\[https?://\S+\s([^\]]+)\]", r"\1", t)
    t = re.sub(r"\[https?://\S+\]", "", t)
    t = re.sub(r"^[*#:;|!].*$", "", t, flags=re.M)
    t = re.sub(r"'{2,}", "", t)
    t = re.sub(r"<[^>]+>", "", t)
    t = t.replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", t).strip()


def clean_heading(h: str) -> str:
    h = re.sub(r"\{\{[^{}]*\}\}", "", h)
    h = re.sub(r"<[^>]+>", "", h)
    h = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]", r"\1", h)
    return re.sub(r"'{2,}", "", h).strip()


def links(wt: str, n: int = 6) -> list[str]:
    out = []
    for m in LINK_RE.finditer(plain_links_only(wt)):
        t = m.group(1).strip()
        if t and not re.fullmatch(r"[\d\s.\-–]+(er)?|\d+\. Jahrhundert", t) and t not in out:
            out.append(t)
        if len(out) >= n:
            break
    return out


def plain_links_only(wt: str) -> str:
    t = re.sub(r"<ref[^>]*>.*?</ref>", "", wt, flags=re.S)
    prev = None
    while prev != t:
        prev = t
        t = re.sub(r"\{\{[^{}]*\}\}", "", t)
    return t


def main_articles(text: str) -> list[str]:
    out = []
    for m in MAIN_RE.finditer(text):
        for part in m.group(1).split("|"):
            part = part.strip()
            if part and "=" not in part:
                out.append(part)
    return out


def split_sections(wt: str, article: str):
    """Gibt (Einleitung, verschachtelte Abschnitte ohne Literatur/Weblinks/…) zurück."""
    heads = list(HEAD_RE.finditer(wt))
    intro = wt[: heads[0].start()] if heads else wt
    flat = []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(wt)
        text = wt[m.end():end]
        mains = main_articles(text)
        flat.append({"level": len(m.group(1)), "title": clean_heading(m.group(2)), "text": text,
                     "article": article, "mains": mains, "main": mains[0] if mains else None, "children": []})
    root = {"level": 0, "children": []}
    stack = [root]
    for s in flat:
        while stack[-1]["level"] >= s["level"]:
            stack.pop()
        stack[-1]["children"].append(s)
        stack.append(s)

    def prune(items):
        keep = [s for s in items if s["title"].lower() not in SKIP and s["title"]]
        for s in keep:
            s["children"] = prune(s["children"])
        return keep

    return intro, prune(root["children"])


async def fetch_article(F, title: str):
    d = await F.get_json(API, {"action": "parse", "page": title, "prop": "wikitext", "redirects": 1,
                               "format": "json", "formatversion": 2})
    p = (d or {}).get("parse")
    return (p["title"], p["wikitext"]) if p else None


async def extracts(F, titles) -> dict[str, str]:
    """Einleitungen (erste Sätze) für viele Artikel, 20 je Aufruf."""
    titles = list(dict.fromkeys(t for t in titles if t))
    out: dict[str, str] = {}

    async def chunk(ts):
        d = await F.get_json(API, {"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1,
                                   "exlimit": 20, "titles": "|".join(ts), "redirects": 1,
                                   "format": "json", "formatversion": 2})
        q = (d or {}).get("query", {})
        mapping = {x["from"]: x["to"] for x in q.get("normalized", []) + q.get("redirects", [])}
        by_title = {p["title"]: p.get("extract", "") for p in q.get("pages", [])}
        for t in ts:
            r = t
            for _ in range(3):
                r = mapping.get(r, r)
            if by_title.get(r):
                out[t] = by_title[r]

    await asyncio.gather(*(chunk(titles[i:i + 20]) for i in range(0, len(titles), 20)))
    return out


async def candidates(F, theme: str) -> list[dict]:
    exact, found = await asyncio.gather(
        F.get_json(API, {"action": "query", "titles": theme, "redirects": 1, "format": "json", "formatversion": 2}),
        F.get_json(API, {"action": "query", "list": "search", "srsearch": theme, "srlimit": 7, "srnamespace": 0,
                         "format": "json", "formatversion": 2}),
    )
    out = []
    for p in (exact or {}).get("query", {}).get("pages", []):
        if not p.get("missing") and "invalid" not in p:
            out.append({"id": p["title"], "label": p["title"], "hint": "Artikel mit genau diesem Titel"})
    for h in (found or {}).get("query", {}).get("search", []):
        if h["title"] not in [o["id"] for o in out]:
            out.append({"id": h["title"], "label": h["title"], "hint": plain(re.sub(r"<[^>]+>", "", h.get("snippet", "")))[:140]})
    return out


async def build(F, theme: str, entry: str | None, n1: int, n2: int, n3: int, enrich: bool):
    require_wikimedia()
    alts = await candidates(F, theme)
    title = entry or (alts[0]["id"] if alts else None)
    if not title:
        raise NotFound(f"Kein Wikipedia-Artikel zu „{theme}“ gefunden.")
    art = await fetch_article(F, title)
    if not art:
        raise NotFound(f"Wikipedia-Artikel „{title}“ nicht abrufbar.")
    atitle, wt = art
    visited = {atitle}
    intro, secs = split_sections(wt, atitle)

    async def expand(sec, n):
        """Unterabschnitte; wenn keine da sind, die Gliederung des Hauptartikels."""
        if n <= 0:
            return []
        if sec["children"]:
            return sec["children"][:n]
        for m in sec["mains"]:
            if m in visited:
                continue
            visited.add(m)
            sub = await fetch_article(F, m)
            if sub:
                visited.add(sub[0])
                sec["main"] = sub[0]
                return split_sections(sub[1], sub[0])[1][:n]
        return []

    l1 = secs[:n1]
    l2s = await asyncio.gather(*(expand(s, n2) for s in l1))
    for s, kids in zip(l1, l2s):
        s["kids"] = kids
    l2 = [k for kids in l2s for k in kids]
    l3s = await asyncio.gather(*(expand(s, n3) for s in l2))
    for s, kids in zip(l2, l3s):
        s["kids"] = kids
        for k in kids:
            k["kids"] = []

    everything = l1 + l2 + [k for kids in l3s for k in kids]
    ex = await extracts(F, [atitle] + [s["main"] for s in everything if s.get("main")])

    def make(s):
        desc = sentences(ex.get(s["main"], "")) if s.get("main") else ""
        if not desc:
            desc = sentences(plain(s["text"]))
        if s.get("main"):
            url, kind = wiki_url(s["main"]), f"Abschnitt mit Hauptartikel „{s['main']}“"
        else:
            url, kind = wiki_url(s["article"], s["title"]), f"Abschnitt in „{s['article']}“"
        kws = ([s["main"]] if s.get("main") else []) + links(s["text"])
        n = node(s["title"], desc, kws, url, kind, s["title"])
        n["children"] = [make(k) for k in s.get("kids", [])]
        return n

    root = node(atitle, sentences(ex.get(atitle, "")) or sentences(plain(intro)), [], wiki_url(atitle), "Artikel", atitle)
    root["children"] = [make(s) for s in l1]
    return root, alts
