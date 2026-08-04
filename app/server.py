#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hmac
import json
import mimetypes
import os
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from inference import APP_DIR, PredictionService


STATIC_DIR = APP_DIR / "static"
SERVICE: PredictionService | None = None


class AppHandler(BaseHTTPRequestHandler):
    server_version = "CoordinateStructureApp/2.0"

    def log_message(self, format_string: str, *args):
        print(f"[{self.log_date_time_string()}] {format_string % args}")

    def authorized(self) -> bool:
        expected_password = os.environ.get("APP_AUTH_PASSWORD", "")
        if not expected_password:
            return True
        expected_user = os.environ.get("APP_AUTH_USER", "eya")
        header = self.headers.get("Authorization", "")
        if not header.startswith("Basic "):
            return False
        try:
            decoded = base64.b64decode(header.removeprefix("Basic "), validate=True).decode("utf-8")
            supplied_user, supplied_password = decoded.split(":", 1)
        except (ValueError, UnicodeDecodeError):
            return False
        return hmac.compare_digest(supplied_user, expected_user) and hmac.compare_digest(
            supplied_password, expected_password
        )

    def require_authorization(self) -> bool:
        if self.authorized():
            return False
        self.send_response(HTTPStatus.UNAUTHORIZED)
        self.send_header("WWW-Authenticate", 'Basic realm="Conformational Coordinate Lab"')
        self.send_header("Content-Length", "0")
        self.end_headers()
        return True

    def send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/health" and self.require_authorization():
            return
        if parsed.path == "/":
            self.send_file(STATIC_DIR / "index.html")
            return
        if parsed.path == "/api/health":
            assert SERVICE is not None
            self.send_json(
                {
                    "status": "ready",
                    "device": str(SERVICE.device),
                    "systems": [entry["key"] for entry in SERVICE.catalog()],
                }
            )
            return
        if parsed.path == "/api/systems":
            assert SERVICE is not None
            self.send_json({"systems": SERVICE.catalog()})
            return
        if parsed.path == "/api/examples":
            assert SERVICE is not None
            query = parsed.query
            system_key = "her2_experimental"
            for item in query.split("&"):
                if item.startswith("system="):
                    system_key = unquote(item.split("=", 1)[1])
            self.send_json({"examples": SERVICE.system(system_key).example_catalog()})
            return
        if parsed.path.startswith("/static/"):
            relative = Path(unquote(parsed.path.removeprefix("/static/")))
            requested = (STATIC_DIR / relative).resolve()
            if STATIC_DIR.resolve() not in requested.parents:
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            self.send_file(requested)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.require_authorization():
            return
        if self.path != "/api/predict":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 35 * 1024 * 1024:
            self.send_json({"error": "Invalid request size."}, HTTPStatus.BAD_REQUEST)
            return
        try:
            payload = json.loads(self.rfile.read(length))
            assert SERVICE is not None
            system = SERVICE.system(str(payload.get("system", "her2_experimental")))
            method = str(payload.get("method", "direct"))
            if "example_slot" in payload:
                result = system.predict_example(
                    int(payload["example_slot"]),
                    method=method,
                    kind=str(payload.get("example_kind", "held_out")),
                )
            else:
                result = system.predict_upload(
                    str(payload["filename"]),
                    str(payload["content_base64"]),
                    method=method,
                )
            self.send_json(result)
        except (ValueError, KeyError, TypeError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except FileNotFoundError as error:
            self.send_json({"error": str(error)}, HTTPStatus.SERVICE_UNAVAILABLE)
        except Exception:
            traceback.print_exc()
            self.send_json({"error": "Inference failed. Inspect the server log."}, HTTPStatus.INTERNAL_SERVER_ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the three-system image-to-coordinate research app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    global SERVICE
    args = parse_args()
    SERVICE = PredictionService()
    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    print(f"Coordinate structure app ready at http://{args.host}:{args.port}")
    print(f"Inference device: {SERVICE.device}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
