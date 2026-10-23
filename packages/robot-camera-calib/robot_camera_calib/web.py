"""Live view of a running capture, served over HTTP (stdlib only).

Started by ``capture --web``. Shows each camera with the detected board, the
coverage grid that solve scores (>= 40% needed), saved-view counts and why the
current frame is not being saved.

Binds to 127.0.0.1 by default; view it through an SSH tunnel (``./sync.sh view``).
Use ``--web-host 0.0.0.0`` only on a trusted network: it serves live camera
frames with no authentication. If drawing or serving fails, the view disables
itself; the capture loop never sees an exception.
"""

from __future__ import annotations

import http.server
import json
import logging
import threading
import time
from typing import Any

import cv2
import numpy as np

from .common import COVERAGE_CELLS, coverage_grid

log = logging.getLogger("camera_calib.web")

PREVIEW_WIDTH = 640

STATE_COLOURS = {  # BGR
    "SAVED": (0, 200, 0),
    "same view - move the board somewhere new": (0, 200, 255),
    "board moving - hold still": (0, 140, 255),
    "robot moving": (0, 0, 255),
    "no board": (120, 120, 120),
}


class LiveView:
    """HTTP live view for the capture loop.

    Args:
        cams: Camera names, in display order.
        host: Bind address.
        port: Bind port.
    """

    def __init__(self, cams: list[str], host: str, port: int) -> None:
        self.cams = cams
        self._lock = threading.Lock()
        self._frames: dict[str, bytes] = {}
        self._status: dict = {"cams": {c: {} for c in cams}}
        self._saved_pts: dict[str, list[np.ndarray]] = {c: [] for c in cams}
        self._failed = False
        view = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                try:
                    view._serve(self)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *_a: Any) -> None:
                pass

        self._server = http.server.ThreadingHTTPServer((host, port), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        log.info("live view on http://%s:%d", host, port)

    def close(self) -> None:
        """Stop the HTTP server."""
        self._server.shutdown()
        self._server.server_close()

    def add_saved(self, cam: str, img_pts: np.ndarray) -> None:
        """Record the corners of a saved view (for the coverage overlay)."""
        if not self._failed:
            self._saved_pts[cam].append(np.asarray(img_pts, np.float32).reshape(-1, 2))

    def update(self, cam: str, gray: np.ndarray | None, state: str, img_pts: Any = None) -> None:
        """Redraw one camera tile. Never raises.

        Args:
            cam: Camera name.
            gray: Current grayscale frame, or None to keep the last tile.
            state: Capture state shown on the tile (a STATE_COLOURS key).
            img_pts: Corners detected in this frame, if any.
        """
        if self._failed or gray is None:
            return
        try:
            jpg, cov = self._draw(cam, gray, state, img_pts)
            with self._lock:
                self._frames[cam] = jpg
                self._status["cams"][cam] = {
                    "state": state, "coverage": cov, "views": len(self._saved_pts[cam]),
                    "t": time.time(),
                }  # fmt: skip
        except Exception as exc:  # noqa: BLE001 - diagnostics never break capture
            log.warning("live view disabled: %s", exc)
            self._failed = True

    def set_status(self, **kw: Any) -> None:
        """Merge fields into the status shown above the tiles."""
        with self._lock:
            self._status.update(kw)

    def _draw(self, cam: str, gray: np.ndarray, state: str, img_pts: Any) -> tuple[bytes, float]:
        h, w = gray.shape
        s = PREVIEW_WIDTH / w
        img = cv2.cvtColor(cv2.resize(gray, (PREVIEW_WIDTH, int(round(h * s)))), cv2.COLOR_GRAY2BGR)
        ph, pw = img.shape[:2]
        n = COVERAGE_CELLS

        hit = coverage_grid(self._saved_pts[cam], (w, h))
        tint = img.copy()
        for gy, gx in zip(*np.nonzero(hit), strict=True):
            p0 = (int(gx * pw / n), int(gy * ph / n))
            p1 = (int((gx + 1) * pw / n), int((gy + 1) * ph / n))
            cv2.rectangle(tint, p0, p1, (0, 160, 0), -1)
        img = cv2.addWeighted(tint, 0.25, img, 0.75, 0)
        for g in range(1, n):
            cv2.line(img, (int(g * pw / n), 0), (int(g * pw / n), ph), (90, 90, 90), 1)
            cv2.line(img, (0, int(g * ph / n)), (pw, int(g * ph / n)), (90, 90, 90), 1)
        for pts in self._saved_pts[cam]:
            for x, y in pts * s:
                cv2.circle(img, (int(x), int(y)), 1, (255, 160, 0), -1)

        colour = STATE_COLOURS.get(state, (255, 255, 255))
        if img_pts is not None:
            for x, y in np.asarray(img_pts).reshape(-1, 2) * s:
                cv2.circle(img, (int(x), int(y)), 3, colour, -1)
        cv2.rectangle(img, (0, 0), (pw - 1, ph - 1), colour, 4)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            raise RuntimeError("jpeg encode failed")
        return buf.tobytes(), float(hit.mean())

    def _serve(self, req: http.server.BaseHTTPRequestHandler) -> None:
        path = req.path.split("?")[0]
        if path == "/":
            body, ctype = _PAGE.replace("__CAMS__", json.dumps(self.cams)).encode(), "text/html"
        elif path == "/status.json":
            with self._lock:
                body = json.dumps(self._status).encode()
            ctype = "application/json"
        elif path.startswith("/frame/") and path.endswith(".jpg"):
            with self._lock:
                frame = self._frames.get(path[len("/frame/") : -len(".jpg")])
            if frame is None:
                req.send_error(404)
                return
            body, ctype = frame, "image/jpeg"
        else:
            req.send_error(404)
            return
        req.send_response(200)
        req.send_header("Content-Type", ctype)
        req.send_header("Cache-Control", "no-store")
        req.send_header("Content-Length", str(len(body)))
        req.end_headers()
        req.wfile.write(body)


_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Camera calibration live</title>
<style>
:root{--bg:#111;--fg:#eee;--muted:#999;--card:#1c1c1c;--ok:#2ecc40;--warn:#ffb000;--bad:#ff4136}
@media (prefers-color-scheme:light){:root{--bg:#f4f4f4;--fg:#111;--muted:#555;--card:#fff}}
body{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif}
h1{font-size:18px;margin:0 0 4px}.sub{color:var(--muted);margin-bottom:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
.card{background:var(--card);border-radius:8px;padding:10px}
.card img{width:100%;border-radius:4px;display:block;background:#000;min-height:120px}
.row{display:flex;justify-content:space-between;align-items:baseline;margin:6px 0 2px}
.name{font-weight:600}.state{font-weight:600}
.bar{height:8px;background:#444;border-radius:4px;overflow:hidden;margin-top:6px}
.bar>div{height:100%;background:var(--warn)}.bar>div.ok{background:var(--ok)}
.meta{color:var(--muted);font-size:12px;margin-top:4px}
</style></head><body>
<h1>Camera calibration &mdash; live</h1>
<div class="sub" id="sub">connecting&hellip;</div>
<div class="grid" id="grid"></div>
<p class="meta">Green cells = image areas already covered by saved photos
(need &ge; 40% per camera).
Border colour: green saved &middot; yellow same view &middot; orange board moving &middot;
red robot moving &middot; grey no board.</p>
<script>
const cams = __CAMS__;
const grid = document.getElementById('grid');
for (const c of cams) {
  grid.insertAdjacentHTML('beforeend', `<div class="card"><img id="img-${c}" alt="${c}">
  <div class="row"><span class="name">${c}</span><span class="state" id="st-${c}">-</span></div>
  <div class="bar"><div id="bar-${c}" style="width:0"></div></div>
  <div class="meta" id="meta-${c}"></div></div>`);
}
async function tick() {
  try {
    const s = await (await fetch('status.json?t=' + Date.now())).json();
    const c0 = s.counts || {};
    document.getElementById('sub').textContent =
      `${s.run || ''} · board ${s.board || ''} · saved ${s.saved ?? 0} samples · ` +
      Object.entries(c0).map(([k, v]) => `${k} ${v}`).join(' · ') + (s.done ? ' · FINISHED' : '');
    for (const c of cams) {
      const e = (s.cams || {})[c] || {};
      const cov = Math.round((e.coverage || 0) * 100);
      document.getElementById('st-' + c).textContent = e.state || '-';
      const bar = document.getElementById('bar-' + c);
      bar.style.width = cov + '%'; bar.className = cov >= 40 ? 'ok' : '';
      document.getElementById('meta-' + c).textContent = `views ${e.views ?? 0} · coverage ${cov}%`;
      document.getElementById('img-' + c).src = 'frame/' + c + '.jpg?t=' + Date.now();
    }
  } catch (err) {
    document.getElementById('sub').textContent = 'disconnected - is capture running?';
  }
}
setInterval(tick, 400); tick();
</script></body></html>"""
