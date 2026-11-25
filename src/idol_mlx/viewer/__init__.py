"""A local page for looking at .ply splats in the browser."""

from __future__ import annotations

import json
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files as package_files
from pathlib import Path
from urllib.parse import unquote


def serve(plys: list[Path], open_browser: bool = True) -> None:
    """Serve the viewer and the given .ply files on a free localhost port until Ctrl-C."""
    by_name = {p.name: p.resolve() for p in plys}
    page = package_files(__package__).joinpath("index.html").read_bytes()

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                return self._send(page, "text/html; charset=utf-8")
            if path == "/files.json":
                return self._send(json.dumps(list(by_name)).encode(), "application/json")
            if path.startswith("/splats/"):
                name = unquote(path[len("/splats/") :])
                if name in by_name:
                    return self._send(by_name[name].read_bytes(), "application/octet-stream")
            self.send_error(404)

        def _send(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.send_header("cache-control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler))
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Viewer at {url}  (Ctrl-C to stop)", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
