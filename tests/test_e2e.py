"""End-to-end tests against the real model. They need an Apple Silicon Mac with mlx-serve installed, take about a
minute, and download about 5 GB the first time. Run with: IDOL_MLX_E2E=1 uv run pytest tests/test_e2e.py"""

import os
import platform
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("IDOL_MLX_E2E") != "1" or platform.machine() != "arm64",
    reason="set IDOL_MLX_E2E=1 on an Apple Silicon Mac to run",
)

IMAGE = Path(__file__).parent / "fixtures" / "lamp.png"


def test_turbo_pack_quantizes_like_the_published_base_pack():
    """Our quantizer, run on Idol's untouched denoiser, must reproduce the base pack's 8-bit denoiser exactly.
    This pins the merge path to the same layout mlx-serve was built for."""
    import mlx.core as mx
    from huggingface_hub import hf_hub_download

    from idol_mlx import pack

    pack.ensure_base_pack()
    original = mx.load(hf_hub_download(pack.IDOL_REPO, "denoiser.safetensors"))
    shipped = mx.load(str(pack.BASE_DIR / "denoiser.safetensors"))
    ours = pack.quantize_blocks(original)

    assert ours.keys() == shipped.keys()
    different = [k for k in shipped if not mx.array_equal(ours[k], shipped[k]).item()]
    assert different == []


def test_published_turbo_denoiser_is_what_a_local_merge_builds(tmp_path: Path):
    """The merged denoiser idol-mlx downloads must hold exactly the tensors that merging Idol and the Turbo adapter
    produces. Compared tensor by tensor: mlx versions can lay out the same tensors differently in the file."""
    import mlx.core as mx

    from idol_mlx import pack

    pack.download_turbo_denoiser(tmp_path / "downloaded")
    pack.build_turbo_denoiser(tmp_path / "built.safetensors", log=lambda _: None)

    assert sorted(p.name for p in (tmp_path / "downloaded").iterdir()) == ["denoiser.safetensors"]
    published = mx.load(str(tmp_path / "downloaded" / "denoiser.safetensors"))
    built = mx.load(str(tmp_path / "built.safetensors"))
    assert published.keys() == built.keys()
    different = [
        k
        for k in published
        if published[k].dtype != built[k].dtype or not mx.array_equal(published[k], built[k]).item()
    ]
    assert different == []


def test_generates_a_splat(tmp_path: Path):
    from idol_mlx import Idol

    stages = []
    with Idol() as idol:
        splat = idol.generate(IMAGE, gaussians=32_768, on_progress=lambda p: stages.append(p.stage))

    assert len(splat) == 32_768
    assert "idol-denoise" in stages
    out = splat.save(tmp_path / "lamp.ply")
    assert out.stat().st_size > 32_768 * 17 * 4


def test_renders_a_turntable(tmp_path: Path):
    from idol_mlx import Splat
    from idol_mlx.turntable import render_frames, write_animation

    ply = Splat.from_glb((Path(__file__).parent / "fixtures" / "lamp.glb").read_bytes()).to_ply()
    images = render_frames(ply, frames=8, size=256)
    assert len(images) == 8
    # something other than the background was drawn
    assert max(images[0].getextrema(), key=lambda band: band[1])[1] > 60
    out = write_animation(images, tmp_path / "spin.gif")
    assert out.stat().st_size > 0
