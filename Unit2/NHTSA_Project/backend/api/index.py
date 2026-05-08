"""
Minimal Vercel Python function placeholder.

Vercel treats Python files inside the local ``api`` directory as serverless
functions when they expose a ``handler`` class derived from
``BaseHTTPRequestHandler`` or an ASGI/WSGI app. This file intentionally avoids
FastAPI because the current goal is only to give the backend team a deployable
starting point.
"""

import json
from http.server import BaseHTTPRequestHandler


PLACEHOLDER_RESPONSE = {
    "service": "nhtsa-project-backend",
    "status": "placeholder",
    "message": "Backend skeleton deployed successfully. Real API logic has not been implemented yet.",
}


class handler(BaseHTTPRequestHandler):
    """Serve a deterministic JSON response for the placeholder backend."""

    def _send_json(self, payload: dict[str, str]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._send_json(PLACEHOLDER_RESPONSE)
