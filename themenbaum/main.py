"""Themenbaum ohne LLM: Themenbäume aus GND, Wikipedia, Wikidata oder einer Inhaltsmenge – ohne generatives LLM.

Start lokal:  uv run uvicorn themenbaum.main:app --port 8000   ->  http://localhost:8000
"""

import hmac
import json
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs

import httpx
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import auth, common
from . import contents
from . import gnd as src_gnd
from . import wikidata as src_wikidata
from . import wikipedia as src_wikipedia

SOURCES = {"gnd": src_gnd, "wikipedia": src_wikipedia, "wikidata": src_wikidata}
SOURCE_NAMES = {"gnd": "GND (lobid)", "wikipedia": "Wikipedia", "wikidata": "Wikidata"}

STATIC = Path(__file__).parent / "static"
VERSION = "1.0.0"

app = FastAPI(title="Themenbaum ohne LLM", version=VERSION, docs_url=None, redoc_url=None)
app.middleware("http")(auth.middleware)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/_ping", include_in_schema=False)
def ping():
    return {"status": "ok"}


@app.get("/login", include_in_schema=False)
def login_form(next: str = "/"):
    return auth.login_page(next)


@app.post("/login", include_in_schema=False)
async def login(request: Request):
    form = parse_qs((await request.body()).decode("utf-8", "replace"))
    given = (form.get("password") or [""])[0]
    target = auth.safe_next((form.get("next") or ["/"])[0])
    ip = auth.client_ip(request)
    if auth.too_many_attempts(ip):
        return auth.login_page(target, "Zu viele Fehlversuche. Bitte in einigen Minuten erneut versuchen.")
    if not auth.password() or not hmac.compare_digest(given.encode(), auth.password().encode()):
        auth.record_failure(ip)
        return auth.login_page(target, "Das Passwort stimmt nicht.")
    response = RedirectResponse(target, status_code=303)
    response.set_cookie(auth.COOKIE, auth.make_token(), max_age=auth.session_hours() * 3600,
                        httponly=True, secure=auth.cookie_secure(), samesite="lax")
    return response


@app.get("/logout", include_in_schema=False)
def logout():
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(auth.COOKIE)
    return response


@app.get("/kosten", include_in_schema=False)
def costs_page():
    return FileResponse(STATIC / "kosten.html")


def pipeline(source: str, root: dict, enrich: bool) -> list[str]:
    """Beschreibt in Schritten, wie der Baum dieser Quelle entstanden ist (für die Info-Anzeige)."""
    desc = ("Beschreibungen: GND-Definition, sonst Einleitung des verknüpften Wikipedia-Artikels (GND → Wikidata → Wikipedia)."
            if enrich else "Beschreibungen: nur GND-Definitionen (Wikipedia-Ergänzung ausgeschaltet).")
    if source == "gnd":
        steps = ["Einstieg: Suche nach dem Thema in der GND über lobid-gnd, nur Sachschlagwörter; genaue Namenstreffer zuerst."]
        if "Sachgruppen" in root["kind"]:
            steps += ["Modus Sachgruppen: Das Schlagwort ist ein Fach mit kaum Unterbegriffen, deshalb bilden die GND-Sachgruppen des Fachs Ebene 1 (z. B. 21.2 „Mechanik, Wärme, Akustik“).",
                      "Ebene 2: Begriffe der Sachgruppe, auf die dort die meisten anderen als Oberbegriff verweisen (bis 1.500 Schlagwörter je Gruppe ausgewertet).",
                      "Ebene 3: deren Unterbegriffe innerhalb der Sachgruppe, bei Bedarf ergänzt aus der ganzen GND."]
        else:
            steps += ["Modus Hierarchie: Ebene 1 bis 3 sind Unterbegriffe (Ober-/Unterbegriffs-Relationen der GND); gibt es zu wenige, werden verwandte Begriffe ergänzt.",
                      "Reihenfolge auf Ebene 1: Begriffe mit vielen eigenen Unterbegriffen zuerst."]
        return steps + [desc, "Kurztitel: regelbasiert gekürzt (max. 20 Zeichen). Kein generatives LLM."]
    if source == "wikipedia":
        return ["Einstieg: Wikipedia-Artikel mit genau diesem Titel, sonst bester Suchtreffer.",
                "Ebene 1: Abschnitte des Artikels (ohne Literatur, Weblinks, Einzelnachweise u. Ä.).",
                "Ebene 2 und 3: Unterabschnitte; hat ein Abschnitt keine, wird die Gliederung seines verlinkten Hauptartikels ({{Hauptartikel}}) verwendet.",
                "Beschreibungen: Einleitung des Hauptartikels, sonst die ersten Sätze des Abschnitts. Schlagwörter: Links im Abschnitt.",
                "Kurztitel: regelbasiert gekürzt (max. 20 Zeichen). Kein generatives LLM."]
    return ["Einstieg: Wikidata-Objekt zum Thema (Wikidata-Suche).",
            "Ebene 1 bis 3: Objekte, die „Teil von“ (P361) oder „Unterklasse von“ (P279) des Elternobjekts sind oder die es als „hat Teil“ (P527) nennt; nur mit deutschem Wikipedia-Artikel.",
            "Reihenfolge: nach Zahl der Sprachversionen (Maß für Bekanntheit); doppelte Objekte werden übersprungen.",
            ("Beschreibungen: Einleitung des deutschen Wikipedia-Artikels. " if enrich else "Beschreibungen ausgeschaltet. ")
            + "Schlagwörter: deutsche Alternativnamen aus Wikidata.",
            "Kurztitel: regelbasiert gekürzt (max. 20 Zeichen). Kein generatives LLM."]


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/inhalte", include_in_schema=False)
def contents_page():
    return FileResponse(STATIC / "inhalte.html")


@app.get("/api/contents/sample")
def contents_sample():
    return json.loads((common.ROOT / "data" / "beispiel_inhalte.json").read_text("utf-8"))


@app.post("/api/contents/tree")
async def contents_tree(body: dict = Body(...)):
    items = body.get("contents") or []
    if not isinstance(items, list):
        raise HTTPException(400, "„contents“ muss eine Liste sein.")
    try:
        return await contents.build([dict(c) for c in items if isinstance(c, dict)], body.get("config") or {})
    except (ValueError, common.ConfigError) as e:
        raise HTTPException(400, str(e))
    except common.NotFound as e:
        raise HTTPException(404, str(e))
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Eine Quelle hat nicht geantwortet: {e!r}")


@app.get("/api/tree")
async def tree(
    source: str,
    theme: str = Query(..., min_length=2),
    entry: str | None = None,
    n1: int = Query(12, ge=1, le=30),
    n2: int = Query(10, ge=0, le=20),
    n3: int = Query(5, ge=0, le=20),
    enrich: bool = True,
):
    module = SOURCES.get(source)
    if not module:
        raise HTTPException(404, f"Unbekannte Quelle „{source}“.")
    fetcher = common.Fetcher()
    t0 = time.perf_counter()
    try:
        root, alternatives = await module.build(fetcher, theme.strip(), entry or None, n1, n2, n3, enrich)
    except common.ConfigError as e:
        raise HTTPException(400, str(e))
    except common.NotFound as e:
        raise HTTPException(404, str(e))
    except httpx.HTTPError as e:
        raise HTTPException(502, f"{SOURCE_NAMES[source]} hat nicht geantwortet: {e!r}")
    return {
        "source": source,
        "source_name": SOURCE_NAMES[source],
        "entry": {"id": root["id"], "label": root["title"], "kind": root["kind"], "url": root["url"]},
        "alternatives": alternatives,
        "stats": common.stats(root, fetcher, time.perf_counter() - t0, n1, n2, n3),
        "pipeline": pipeline(source, root, enrich),
        "metadata": {
            "title": root["title"],
            "description": root["description"],
            "target_audience": "Lehrkräfte",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "version": VERSION,
            "author": f"Themenbaum ohne LLM ({SOURCE_NAMES[source]})",
        },
        "collection": [common.to_collection(c) for c in root["children"]],
    }
