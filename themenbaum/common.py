"""Gemeinsame Bausteine: HTTP mit Datei-Cache, Knotenmodell, Ausgabeformat wie topic-tree-generator."""

import asyncio
import hashlib
import json
import os
import re
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Liest eine lokale .env im Projektordner (nur für den Start ohne Docker); Umgebung hat Vorrang."""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text("utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

# Datenordner für Cache, Embedding-Modelle und Session-Schlüssel (im Container ein Volume).
DATA_DIR = Path(os.environ.get("DATA_DIR") or ROOT / ".data")
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = DATA_DIR / "models"
CACHE_TTL = 7 * 86400

# Wikimedia lehnt Anfragen ohne Kontaktangabe im User-Agent ab (https://w.wiki/4wJS).
CONTACT = os.environ.get("WIKIMEDIA_CONTACT", "").strip()
USER_AGENT = f"ThemenbaumOhneLLM/1.0 ({CONTACT or 'kein Kontakt konfiguriert'}) httpx/{httpx.__version__}"


class ConfigError(Exception):
    pass


def require_wikimedia() -> None:
    if not CONTACT:
        raise ConfigError("Für Wikipedia und Wikidata fehlt WIKIMEDIA_CONTACT (E-Mail oder URL) in der .env. "
                          "Wikimedia lehnt Anfragen ohne Kontaktangabe ab.")

# Ein Client und je Host ein Limit für parallele Anfragen (Rücksicht auf öffentliche APIs).
_client: httpx.AsyncClient | None = None
_host_limits = {"query.wikidata.org": 3}
_semaphores: dict[str, asyncio.Semaphore] = {}


class NotFound(Exception):
    pass


def client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=httpx.Timeout(45.0),
            follow_redirects=True,
        )
    return _client


class Fetcher:
    """Zählt API-Aufrufe je Anfrage und cacht Antworten eine Woche auf der Platte."""

    def __init__(self):
        self.calls = 0
        self.cache_hits = 0

    async def _get(self, url: str, params: dict | None, as_json: bool):
        key = hashlib.sha1((url + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()
        path = CACHE_DIR / f"{key}.{'json' if as_json else 'txt'}"
        if path.exists() and time.time() - path.stat().st_mtime < CACHE_TTL:
            self.cache_hits += 1
            text = path.read_text("utf-8")
            return json.loads(text) if as_json else text

        host = httpx.URL(url).host
        sem = _semaphores.setdefault(host, asyncio.Semaphore(_host_limits.get(host, 6)))
        async with sem:
            for attempt in range(4):
                try:
                    # Hängende Einzelanfragen früh abbrechen und wiederholen; SPARQL darf länger rechnen.
                    r = await client().get(url, params=params, timeout=60.0 if host == "query.wikidata.org" else 15.0)
                except httpx.TransportError:
                    if attempt == 3:
                        raise
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                if r.status_code == 429 or r.status_code >= 500:
                    await asyncio.sleep(float(r.headers.get("Retry-After", 1.5 * (attempt + 1))))
                    continue
                break
        self.calls += 1
        if r.status_code == 404:
            return None
        r.raise_for_status()
        CACHE_DIR.mkdir(exist_ok=True)
        path.write_text(r.text, "utf-8")
        return r.json() if as_json else r.text

    async def get_json(self, url: str, params: dict | None = None):
        return await self._get(url, params, True)

    async def get_text(self, url: str, params: dict | None = None):
        return await self._get(url, params, False)


def cap(title: str) -> str:
    """Titel beginnen immer mit einem Großbuchstaben (Wikidata schreibt z. B. "geometrische Optik")."""
    title = (title or "").strip()
    return title[:1].upper() + title[1:]


def node(title: str, description: str = "", keywords=(), url: str = "", kind: str = "", sid: str = "") -> dict:
    title = cap(title)
    kws = []
    for k in keywords:
        k = (k or "").strip()
        if k and k.lower() != title.lower() and k not in kws:
            kws.append(k)
    return {"title": title, "description": description or "", "keywords": kws[:8], "url": url, "kind": kind, "id": sid, "children": []}


def shorttitle(title: str, maxlen: int = 20) -> str:
    t = re.sub(r"\s+", " ", re.sub(r"[^\w ]", " ", title)).strip()
    if len(t) <= maxlen:
        return t
    out = ""
    for word in t.split():
        if len(f"{out} {word}".strip()) > maxlen:
            break
        out = f"{out} {word}".strip()
    return out or t[:maxlen]


def sentences(text: str, min_words: int = 25, max_words: int = 70) -> str:
    """Erste ganze Sätze bis etwa min_words Wörter, höchstens max_words."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ0-9„\"(])", text)
    out, words = [], 0
    for p in parts:
        w = len(p.split())
        if out and words + w > max_words:
            break
        out.append(p)
        words += w
        if words >= min_words:
            break
    res = " ".join(out)
    if len(res.split()) > max_words:
        res = " ".join(res.split()[:max_words]) + " …"
    return res


def to_collection(n: dict) -> dict:
    st = shorttitle(n["title"])
    c = {
        "title": n["title"],
        "shorttitle": st,
        "properties": {
            "cclom:general_keyword": n["keywords"] or [n["title"].lower()],
            "ccm:collectionshorttitle": [st],
            "ccm:educationalcontext": [],
            "ccm:educationalintendedenduserrole": ["http://w3id.org/openeduhub/vocabs/intendedEndUserRole/teacher"],
            "ccm:taxonid": [],
            "cm:description": [n["description"]] if n["description"] else [],
            "cm:title": [n["title"]],
        },
        "_source": {"kind": n["kind"], "url": n["url"], "id": n["id"]},
    }
    if n["children"]:
        c["subcollections"] = [to_collection(ch) for ch in n["children"]]
    return c


def stats(root: dict, fetcher: Fetcher, seconds: float, n1: int, n2: int, n3: int) -> dict:
    levels = {1: [], 2: [], 3: []}

    def walk(n, lvl):
        for ch in n["children"]:
            if lvl <= 3:
                levels[lvl].append(ch)
                walk(ch, lvl + 1)

    walk(root, 1)
    all_nodes = levels[1] + levels[2] + levels[3]
    with_desc = sum(1 for n in all_nodes if n["description"])
    return {
        "nodes": len(all_nodes),
        "per_level": [len(levels[1]), len(levels[2]), len(levels[3])],
        "max_per_level": [n1, n1 * n2, n1 * n2 * n3],
        "with_description": with_desc,
        "api_calls": fetcher.calls,
        "cache_hits": fetcher.cache_hits,
        "seconds": round(seconds, 1),
    }
