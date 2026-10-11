"""Build NOVA as a static web app for any HTTPS host (GitHub Pages, Netlify, ...).

On iPad, open the hosted page in Safari, tap Share > Add to Home Screen, and NOVA opens full
screen in landscape like an app, works offline, and plays demo sessions with sound. A headset
session still needs ``nova app`` running on a computer.

    python -m nova.site [out_dir]      # default: _site
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from .live import MANIFEST, _asset, dashboard_html

ASSETS = ("icon-180.png", "icon-512.png")

# Cache the app shell so it opens offline from the Home Screen; refresh it in the background.
SERVICE_WORKER = """const CACHE = "nova-%(version)s";
const SHELL = ["./", "./index.html", "./manifest.webmanifest", "./icon-180.png", "./icon-512.png"];
self.addEventListener("install", e => { e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL))); self.skipWaiting(); });
self.addEventListener("activate", e => {
  e.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))));
  self.clients.claim();
});
self.addEventListener("fetch", e => {
  if (e.request.method !== "GET" || new URL(e.request.url).origin !== location.origin) return;
  e.respondWith(caches.open(CACHE).then(async c => {
    const hit = await c.match(e.request, { ignoreSearch: true });
    const net = fetch(e.request).then(r => { if (r.ok) c.put(e.request, r.clone()); return r; }).catch(() => hit);
    return hit || net;
  }));
});
"""

_REGISTER = ('<script>if ("serviceWorker" in navigator) addEventListener("load", () => '
             'navigator.serviceWorker.register("sw.js").catch(() => {}));</script></body>')


def site_manifest() -> dict:
    """The app manifest with paths relative to wherever the site is hosted (e.g. /Neuro/ on Pages)."""
    m = json.loads(json.dumps(MANIFEST))
    m["start_url"] = "./"
    m["scope"] = "./"
    for icon in m["icons"]:
        icon["src"] = icon["src"].lstrip("/")
    return m


def site_html() -> str:
    html = dashboard_html()
    html = html.replace('href="/manifest.webmanifest"', 'href="manifest.webmanifest"').replace('href="/icon-180.png"', 'href="icon-180.png"')
    head, _, tail = html.rpartition("</body>")
    return head + _REGISTER + tail


def build(out: str | Path = "_site") -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    html = site_html()
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / "manifest.webmanifest").write_text(json.dumps(site_manifest(), indent=2), encoding="utf-8")
    for name in ASSETS:
        (out / name).write_bytes(_asset(name))
    version = hashlib.sha1(html.encode("utf-8")).hexdigest()[:10]
    (out / "sw.js").write_text(SERVICE_WORKER % {"version": version}, encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")
    return out


if __name__ == "__main__":
    print(build(sys.argv[1] if len(sys.argv) > 1 else "_site"))
