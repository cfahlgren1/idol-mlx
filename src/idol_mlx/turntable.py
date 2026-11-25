"""Render a spinning GIF or MP4 of a .ply splat, using a headless Chromium-based browser for the splat renderer."""

from __future__ import annotations

import base64
import io
import json
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files as package_files
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

BROWSERS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
]
BACKGROUND = "#0d1117"
RENDER_TIMEOUT_S = 600
OPAQUE = 0.5  # Gaussians at least this opaque decide the framing


class TurntableError(RuntimeError):
    pass


@dataclass(frozen=True)
class Framing:
    center: tuple[float, float, float]
    radius: float
    height: float


def find_browser() -> str:
    for path in BROWSERS:
        if Path(path).exists():
            return path
    for name in ("google-chrome", "chromium", "chromium-browser"):
        if found := shutil.which(name):
            return found
    raise TurntableError(
        "Turntables need Google Chrome (or Chromium, Brave or Edge) installed to render the splat."
    )


def check_turntable_tools(fmt: str) -> None:
    """Fail before any slow work when the browser (or ffmpeg, for MP4) isn't available."""
    if fmt not in ("gif", "mp4"):
        raise ValueError(f"Turntables are saved as .gif or .mp4, not .{fmt}")
    find_browser()
    if fmt == "mp4" and not shutil.which("ffmpeg"):
        raise TurntableError("MP4 turntables need ffmpeg (`brew install ffmpeg`). GIF works without it.")


def frame_splat(ply: bytes) -> Framing:
    """Aim at the middle of the opaque Gaussians and back off far enough to keep the whole object in view."""
    header_end = ply.index(b"end_header\n") + len(b"end_header\n")
    header = ply[:header_end].decode()
    properties = [line.split()[-1] for line in header.splitlines() if line.startswith("property")]
    rows = np.frombuffer(ply[header_end:], dtype="<f4").reshape(-1, len(properties))
    column = {name: i for i, name in enumerate(properties)}

    opacity = 1 / (1 + np.exp(-rows[:, column["opacity"]]))
    xyz = rows[opacity > OPAQUE][:, [column["x"], column["y"], column["z"]]]
    if len(xyz) == 0:
        xyz = rows[:, [column["x"], column["y"], column["z"]]]
    low, high = np.percentile(xyz, 1, axis=0), np.percentile(xyz, 99, axis=0)
    extent = float(np.max(high - low))
    center = (low + high) / 2
    return Framing(center=tuple(float(c) for c in center), radius=2.6 * extent, height=0.5 * extent)


def render_frames(
    ply: bytes,
    frames: int = 72,
    size: int = 640,
    on_frame: Callable[[int, int], None] | None = None,
) -> list[Image.Image]:
    framing = frame_splat(ply)
    config = {
        "center": framing.center,
        "radius": framing.radius,
        "height": framing.height,
        "frames": frames,
        "size": size,
        "background": BACKGROUND,
        "startAngle": 0.5,
    }
    page = package_files("idol_mlx.viewer").joinpath("turntable.html").read_bytes()
    captured: dict[int, Image.Image] = {}
    errors: list[str] = []
    done = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def do_GET(self) -> None:
            routes = {
                "/": (page, "text/html; charset=utf-8"),
                "/config.json": (json.dumps(config).encode(), "application/json"),
                "/splat.ply": (ply, "application/octet-stream"),
            }
            path = urlparse(self.path).path
            if path not in routes:
                self.send_error(404)
                return
            body, content_type = routes[path]
            self.send_response(200)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            url = urlparse(self.path)
            body = self.rfile.read(int(self.headers.get("content-length", 0)))
            if url.path == "/frame":
                index = int(parse_qs(url.query)["i"][0])
                data = base64.b64decode(body.decode().split(",", 1)[1])
                captured[index] = Image.open(io.BytesIO(data)).convert("RGB")
                if on_frame:
                    on_frame(len(captured), frames)
            elif url.path == "/error":
                errors.append(body.decode())
                done.set()
            elif url.path == "/done":
                done.set()
            self.send_response(204)
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/"

    with tempfile.TemporaryDirectory(prefix="idol-mlx-chrome-") as profile:
        browser = subprocess.Popen(
            [
                find_browser(),
                "--headless=new",
                f"--user-data-dir={profile}",
                "--no-first-run",
                "--disable-extensions",
                f"--window-size={size + 40},{size + 40}",
                url,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            finished = done.wait(RENDER_TIMEOUT_S)
        finally:
            browser.terminate()
            try:
                browser.wait(timeout=10)
            except subprocess.TimeoutExpired:
                browser.kill()
            server.shutdown()

    if errors:
        raise TurntableError(f"The renderer failed: {errors[0]}")
    if not finished or len(captured) != frames:
        raise TurntableError(f"Rendered {len(captured)} of {frames} frames before timing out")
    return [captured[i] for i in range(frames)]


def write_animation(images: list[Image.Image], out_path: Path, fps: int = 18) -> Path:
    suffix = out_path.suffix.lower()
    if suffix == ".gif":
        return _write_gif(images, out_path, fps)
    if suffix == ".mp4":
        return _write_mp4(images, out_path, fps)
    raise ValueError(f"Turntables are saved as .gif or .mp4, not {suffix or 'a file without an extension'}")


def _write_gif(images: list[Image.Image], out_path: Path, fps: int) -> Path:
    if shutil.which("ffmpeg"):
        return _write_gif_ffmpeg(images, out_path, fps)
    # Without ffmpeg, Pillow: one shared palette across all frames avoids colour flicker as the object turns.
    samples = images[:: max(1, len(images) // 12)]
    sheet = Image.new("RGB", (images[0].width, images[0].height * len(samples)))
    for row, image in enumerate(samples):
        sheet.paste(image, (0, row * image.height))
    palette = sheet.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    frames = [image.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG) for image in images]
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / fps),
        loop=0,
        optimize=True,
        disposal=1,
    )
    return out_path


def _write_gif_ffmpeg(images: list[Image.Image], out_path: Path, fps: int) -> Path:
    """ffmpeg's palettegen/paletteuse makes noticeably smaller GIFs than Pillow at the same quality."""
    graph = (
        "split[a][b];[a]palettegen=stats_mode=diff:max_colors=200[p];"
        "[b][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle"
    )
    return _run_ffmpeg(images, out_path, fps, ["-filter_complex", graph, "-loop", "0"])


def _write_mp4(images: list[Image.Image], out_path: Path, fps: int) -> Path:
    if not shutil.which("ffmpeg"):
        raise TurntableError("MP4 turntables need ffmpeg (`brew install ffmpeg`). GIF works without it.")
    return _run_ffmpeg(
        images,
        out_path,
        fps,
        ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart"],
    )


def _run_ffmpeg(images: list[Image.Image], out_path: Path, fps: int, output_args: list[str]) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    width, height = images[0].size
    command = [
        ffmpeg, "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        *output_args, str(out_path),
    ]  # fmt: skip
    raw = b"".join(image.tobytes() for image in images)
    result = subprocess.run(command, input=raw, capture_output=True, check=False)
    if result.returncode != 0:
        raise TurntableError(f"ffmpeg failed: {result.stderr.decode().strip()}")
    return out_path
