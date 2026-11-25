"""Turntable framing and GIF encoding, without a browser. Rendering itself is covered by the e2e tests."""

from pathlib import Path

import pytest
from PIL import Image

from idol_mlx import Splat
from idol_mlx.turntable import frame_splat, write_animation

FIXTURE = Path(__file__).parent / "fixtures" / "lamp.glb"


def test_framing_aims_at_the_object_and_backs_off():
    splat = Splat.from_glb(FIXTURE.read_bytes())
    framing = frame_splat(splat.to_ply())
    low, high = splat.positions.min(axis=0), splat.positions.max(axis=0)
    assert all(lo <= c <= hi for lo, c, hi in zip(low, framing.center, high, strict=True))
    assert framing.radius > float((high - low).max())  # the camera starts outside the object


def frames(count: int = 6) -> list[Image.Image]:
    return [Image.new("RGB", (64, 64), (20 * i, 120, 200 - 20 * i)) for i in range(count)]


def test_gif_without_ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("idol_mlx.turntable.shutil.which", lambda name: None)
    out = write_animation(frames(), tmp_path / "spin.gif", fps=10)
    gif = Image.open(out)
    assert gif.n_frames == 6
    assert gif.info["loop"] == 0
    assert gif.info["duration"] == 100


def test_rejects_unknown_formats(tmp_path: Path):
    with pytest.raises(ValueError, match=".gif or .mp4"):
        write_animation(frames(), tmp_path / "spin.webm")


def test_flat_frames_are_blank_but_dark_objects_are_not():
    from idol_mlx.turntable import BACKGROUND, _is_blank

    assert _is_blank(Image.new("RGB", (640, 640), (0, 0, 0)))  # the browser drew nothing
    assert _is_blank(Image.new("RGB", (640, 640), (60, 20, 18)))  # camera inside the splat
    dark_statue = Image.new("RGB", (640, 640), BACKGROUND)
    dark_statue.paste((40, 46, 36), (280, 160, 360, 520))  # dark bronze, close to the background colour
    assert not _is_blank(dark_statue)
