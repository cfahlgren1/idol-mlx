"""Generate splats through mlx-serve's /v1/3d/generations endpoint."""

from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from PIL import Image

from . import pack
from .server import DEFAULT_URL, Server, ensure_server, rescan
from .splat import Splat

MIN_STEPS = 1
MAX_STEPS = 200
MIN_GAUSSIANS = 32_768
MAX_GAUSSIANS = 1_048_576
REQUEST_TIMEOUT_S = 1800


@dataclass(frozen=True)
class Progress:
    stage: str  # e.g. "idol-denoise"
    step: int
    total: int


ProgressCallback = Callable[[Progress], None]


class GenerationError(RuntimeError):
    pass


def check_settings(*, steps: int, gaussians: int, mode: str) -> None:
    """Reject values mlx-serve would refuse, before any server or model work starts."""
    if not MIN_STEPS <= steps <= MAX_STEPS:
        raise ValueError(f"steps must be between {MIN_STEPS} and {MAX_STEPS}, got {steps}")
    if not MIN_GAUSSIANS <= gaussians <= MAX_GAUSSIANS:
        raise ValueError(
            f"gaussians must be between {MIN_GAUSSIANS:,} and {MAX_GAUSSIANS:,}, got {gaussians:,}"
        )
    if mode not in ("synthetic", "real"):
        raise ValueError(f"mode must be 'synthetic' or 'real', got {mode!r}")


class Idol:
    """Image to Gaussian splat with Idol Turbo on a local mlx-serve.

    >>> idol = Idol()
    >>> idol.generate("photo.jpg").save("photo.ply")
    """

    def __init__(self, url: str = DEFAULT_URL, keep_server: bool = False, original: bool = False):
        self.model = pack.BASE_MODEL_ID if original else pack.TURBO_MODEL_ID
        if original:
            pack.ensure_base_pack()
        else:
            pack.ensure_turbo_pack()
        self.server: Server = ensure_server(url, keep_running=keep_server)
        if not self.server.has_model(self.model):
            rescan(self.server.url)
        if not self.server.has_model(self.model):
            raise GenerationError(f"mlx-serve at {self.server.url} doesn't list the model {self.model}")

    def generate(
        self,
        image: str | Path | bytes | Image.Image,
        *,
        steps: int = 5,
        guidance: float = 0.0,
        gaussians: int = 262_144,
        seed: int = 42,
        mode: str = "synthetic",
        on_progress: ProgressCallback | None = None,
    ) -> Splat:
        check_settings(steps=steps, gaussians=gaussians, mode=mode)
        body = {
            "model": self.model,
            "image": base64.b64encode(_png_or_jpeg(image)).decode(),
            "steps": steps,
            "guidance_scale": guidance,
            "num_gaussians": gaussians,
            "seed": seed,
            "mode": mode,
            "geometry": "splat",
            "stream": True,
        }
        request = urllib.request.Request(
            f"{self.server.url}/v1/3d/generations",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json"},
        )
        try:
            response = urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S)
        except urllib.error.HTTPError as err:
            raise GenerationError(
                f"mlx-serve refused the request ({err.code}): {_error_message(err)}"
            ) from err

        with response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                event = json.loads(line[len("data:") :])
                kind = event.get("type")
                if kind == "progress" and on_progress:
                    on_progress(Progress(event["stage"], event["step"], event["total"]))
                elif kind == "complete":
                    if event.get("format") != "glb":
                        raise GenerationError(f"Expected a GLB from mlx-serve, got {event.get('format')!r}")
                    return Splat.from_glb(base64.b64decode(event["data"]))
                elif kind == "error":
                    raise GenerationError(f"mlx-serve failed: {event.get('message') or event}")
        raise GenerationError("mlx-serve closed the stream before sending a result")

    def close(self) -> None:
        self.server.stop()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _png_or_jpeg(image: str | Path | bytes | Image.Image) -> bytes:
    """mlx-serve decodes PNG and JPEG; anything else (WebP, HEIC, …) is converted to PNG first."""
    if isinstance(image, Image.Image):
        return _to_png(image)
    data = image if isinstance(image, bytes) else Path(image).read_bytes()
    if data[:8] == b"\x89PNG\r\n\x1a\n" or data[:3] == b"\xff\xd8\xff":
        return data
    return _to_png(Image.open(io.BytesIO(data)))


def _to_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGBA" if "A" in image.getbands() else "RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _error_message(err: urllib.error.HTTPError) -> str:
    try:
        return json.loads(err.read())["error"]["message"]
    except (ValueError, KeyError, TypeError):
        return str(err.reason)
