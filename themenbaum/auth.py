"""Einfacher Passwortschutz für die Demo: Login-Seite, signiertes Sitzungs-Cookie, Bremse gegen Durchprobieren.

- APP_PASSWORD (Pflicht für den Zugang): ohne Passwort bleibt die App gesperrt.
- SESSION_SECRET (optional): sonst wird beim ersten Start ein zufälliger Schlüssel im Datenordner abgelegt.
- SESSION_HOURS (optional, Standard 168 = 7 Tage), COOKIE_SECURE (Standard true; lokal ohne HTTPS: false).
"""

import hashlib
import hmac
import os
import secrets
import time
from collections import defaultdict, deque
from html import escape
from urllib.parse import quote

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from .common import DATA_DIR

COOKIE = "themenbaum_session"
PUBLIC_PATHS = {"/login", "/logout", "/_ping", "/favicon.ico"}
MAX_ATTEMPTS, WINDOW_SECONDS = 8, 600


def password() -> str:
    return os.environ.get("APP_PASSWORD", "")


def session_hours() -> int:
    try:
        return max(1, int(os.environ.get("SESSION_HOURS", "168")))
    except ValueError:
        return 168


def cookie_secure() -> bool:
    return os.environ.get("COOKIE_SECURE", "true").strip().lower() not in ("0", "false", "no", "nein")


def _secret() -> bytes:
    if os.environ.get("SESSION_SECRET"):
        return os.environ["SESSION_SECRET"].encode()
    path = DATA_DIR / "session_secret"
    if not path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_hex(32), "utf-8")
    return path.read_text("utf-8").strip().encode()


SECRET = _secret()


def _sign(payload: str) -> str:
    # Das Passwort fließt in die Signatur ein: Ändert man APP_PASSWORD, sind alte Sitzungen ungültig.
    key = SECRET + hashlib.sha256(password().encode()).digest()
    return hmac.new(key, payload.encode(), hashlib.sha256).hexdigest()


def make_token() -> str:
    expires = str(int(time.time()) + session_hours() * 3600)
    return f"{expires}.{_sign(expires)}"


def valid_token(token: str | None) -> bool:
    if not token or "." not in token or not password():
        return False
    expires, sig = token.split(".", 1)
    return expires.isdigit() and int(expires) > time.time() and hmac.compare_digest(sig, _sign(expires))


_attempts: dict[str, deque] = defaultdict(deque)


def too_many_attempts(ip: str) -> bool:
    q = _attempts[ip]
    now = time.time()
    while q and now - q[0] > WINDOW_SECONDS:
        q.popleft()
    return len(q) >= MAX_ATTEMPTS


def record_failure(ip: str) -> None:
    _attempts[ip].append(time.time())


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unbekannt"


async def middleware(request: Request, call_next):
    if request.url.path in PUBLIC_PATHS or valid_token(request.cookies.get(COOKIE)):
        return await call_next(request)
    if request.url.path.startswith("/api/"):
        return JSONResponse({"detail": "Nicht angemeldet. Bitte die Seite neu laden und anmelden."}, status_code=401)
    target = request.url.path + (f"?{request.url.query}" if request.url.query else "")
    return RedirectResponse(f"/login?next={quote(target)}", status_code=303)


def safe_next(target: str | None) -> str:
    # Nur Pfade innerhalb der App; "//host" und "/\host" würden Browser als fremde Adresse lesen.
    ok = target and target.startswith("/") and not target.startswith("//") and "\\" not in target
    return target if ok else "/"


def login_page(next_url: str = "/", error: str = "") -> HTMLResponse:
    if not password():
        error = "Der Zugang ist noch nicht eingerichtet: APP_PASSWORD in der .env setzen und neu starten."
    html = LOGIN_HTML.replace("{{next}}", escape(safe_next(next_url))).replace(
        "{{error}}", f'<p class="err" role="alert">{escape(error)}</p>' if error else "")
    return HTMLResponse(html, status_code=401 if error else 200)


LOGIN_HTML = """<!doctype html>
<html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Anmelden – Themenbaum ohne LLM</title>
<style>
:root{--bg:#f4f6f5;--panel:#fdfefd;--ink:#16201d;--ink2:#4a5753;--line:#dbe2df;--accent:#1f6b5c;--err:#a3312a;--err-soft:#fbe6e4;color-scheme:light}
@media (prefers-color-scheme: dark){:root{--bg:#121615;--panel:#1a201e;--ink:#e9efec;--ink2:#bac6c1;--line:#2b3431;--accent:#6fc2ad;--err:#f08a80;--err-soft:#341c1a;color-scheme:dark}}
*{box-sizing:border-box}
body{margin:0;min-height:100vh;display:grid;place-items:center;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;padding:16px}
form{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:28px 28px 24px;width:min(380px,100%);display:grid;gap:14px}
h1{font-size:20px;margin:0}
p{margin:0;color:var(--ink2);font-size:14px}
label{display:grid;gap:4px;font-size:13px;color:var(--ink2)}
input{font:inherit;color:var(--ink);background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:9px 11px}
input:focus-visible,button:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
button{font:inherit;font-weight:600;cursor:pointer;border:0;border-radius:8px;padding:10px;background:var(--accent);color:var(--panel)}
.err{color:var(--err);background:var(--err-soft);border-radius:8px;padding:8px 10px}
</style></head>
<body><form method="post" action="/login">
<h1>Themenbaum ohne LLM</h1>
<p>Demo für Themenbäume aus GND, Wikipedia, Wikidata und Inhaltsmengen. Bitte mit dem Zugangspasswort anmelden.</p>
{{error}}
<label for="password">Passwort<input type="password" id="password" name="password" required autocomplete="current-password" autofocus></label>
<input type="hidden" name="next" value="{{next}}">
<button type="submit">Anmelden</button>
</form></body></html>"""
