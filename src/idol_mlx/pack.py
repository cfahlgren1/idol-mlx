"""The model packs mlx-serve loads: the base 8-bit Idol pack, and the Turbo pack built from it.

The Turbo pack is the base pack with one file replaced: Idol's denoiser with Idol Turbo's high-noise ("slow") LoRA
expert merged into its weights (W' = W + alpha/rank * B @ A), then quantized exactly the way the base pack was.
To mlx-serve it is plain Idol, so it runs Turbo without adapter support.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from .server import MODELS_DIR

BASE_REPO = "ddalcu/Idol-MLX-Serve-8bit"
IDOL_REPO = "speridlabs/idol"
TURBO_REPO = "speridlabs/idol-turbo"
TURBO_LORA_FILE = "idol_turbo_lora.safetensors"

BASE_MODEL_ID = "Idol-MLX-Serve-8bit"
TURBO_MODEL_ID = "Idol-Turbo"
BASE_DIR = MODELS_DIR / BASE_MODEL_ID
TURBO_DIR = MODELS_DIR / TURBO_MODEL_ID
MARKER = "idol-mlx.json"

LORA_RANK = 32
LORA_ALPHA = 128.0  # from idol-turbo's turbo.py
EXPERT = "slow"
GROUP_SIZE = 64
BITS = 8
BLOCK_PREFIXES = ("blocks.", "noise_refiner.", "context_refiner.")

Log = Callable[[str], None]


def turbo_ready() -> bool:
    return (TURBO_DIR / MARKER).exists() and (TURBO_DIR / "denoiser.safetensors").exists()


def base_ready() -> bool:
    return (BASE_DIR / "config.json").exists() and (BASE_DIR / "denoiser.safetensors").exists()


def ensure_base_pack(log: Log = print) -> Path:
    if base_ready():
        return BASE_DIR
    log(f"Downloading the base Idol pack ({BASE_REPO}, about 3 GB, once)")
    snapshot_download(BASE_REPO, local_dir=BASE_DIR)
    return BASE_DIR


def ensure_turbo_pack(log: Log = print) -> Path:
    """Build ~/.mlx-serve/models/Idol-Turbo if it isn't there yet. Safe to interrupt and rerun."""
    if turbo_ready():
        return TURBO_DIR
    ensure_base_pack(log)

    log(
        f"Downloading Idol's full-precision denoiser ({IDOL_REPO}, 1.5 GB) and the Turbo adapter ({TURBO_REPO})"
    )
    denoiser_path = hf_hub_download(IDOL_REPO, "denoiser.safetensors")
    lora_path = hf_hub_download(TURBO_REPO, TURBO_LORA_FILE)

    log("Merging the Turbo adapter into the denoiser")
    building = TURBO_DIR.with_name(TURBO_DIR.name + ".building")
    shutil.rmtree(building, ignore_errors=True)
    building.mkdir(parents=True)
    layers = merge_and_quantize(Path(denoiser_path), Path(lora_path), building / "denoiser.safetensors")

    for entry in BASE_DIR.iterdir():
        if entry.name not in {"denoiser.safetensors", MARKER} and not entry.name.startswith("."):
            (building / entry.name).symlink_to(entry.resolve())
    (building / MARKER).write_text(
        json.dumps(
            {
                "base": BASE_REPO,
                "denoiser": IDOL_REPO,
                "adapter": f"{TURBO_REPO}/{TURBO_LORA_FILE}",
                "expert": EXPERT,
                "merged_layers": layers,
                "bits": BITS,
            },
            indent=2,
        )
    )
    shutil.rmtree(TURBO_DIR, ignore_errors=True)
    building.rename(TURBO_DIR)
    log(f"Turbo pack ready at {TURBO_DIR}")
    return TURBO_DIR


def merge_and_quantize(denoiser_path: Path, lora_path: Path, out_path: Path) -> int:
    """Merge the slow LoRA expert into the denoiser and quantize it like mlx-serve's converter. Returns layers merged."""
    import mlx.core as mx  # Apple Silicon only; imported here so the rest of the package loads anywhere

    weights = mx.load(str(denoiser_path))
    lora = mx.load(str(lora_path))
    scale = LORA_ALPHA / LORA_RANK

    layers = sorted({key.rsplit(".", 1)[0] for key in lora if key.endswith(f".{EXPERT}_A")})
    if not layers:
        raise ValueError(f"No '{EXPERT}' LoRA expert found in {lora_path}")
    for layer in layers:
        name = f"{layer}.weight"
        if name not in weights:
            raise ValueError(f"The adapter targets {layer}, which the denoiser doesn't have")
        base = weights[name]
        delta = scale * (lora[f"{layer}.{EXPERT}_B"] @ lora[f"{layer}.{EXPERT}_A"])
        if delta.shape != base.shape:
            raise ValueError(f"Shape mismatch at {layer}: adapter {delta.shape}, weight {base.shape}")
        weights[name] = (base.astype(mx.float32) + delta).astype(base.dtype)

    mx.save_safetensors(str(out_path), quantize_blocks(weights), metadata={"format": "mlx"})
    return len(layers)


def quantize_blocks(tensors: dict) -> dict:
    """Same layout as mlx-serve's tests/convert_idol_weights.py: block Linears to affine 8-bit group 64 where the
    input width allows (else float16), their biases to float16, everything else unchanged."""
    import mlx.core as mx

    def is_block_linear(name: str, shape) -> bool:
        return name.endswith(".weight") and len(shape) == 2 and name.startswith(BLOCK_PREFIXES)

    linear_bases = {k[: -len(".weight")] for k, v in tensors.items() if is_block_linear(k, v.shape)}
    out = {}
    for name, tensor in tensors.items():
        base = name.rsplit(".", 1)[0]
        if name.endswith(".weight") and base in linear_bases:
            if tensor.shape[1] % GROUP_SIZE == 0:
                q, scales, biases = mx.quantize(tensor.astype(mx.float32), group_size=GROUP_SIZE, bits=BITS)
                out[name] = q
                out[f"{base}.scales"] = scales.astype(mx.float16)
                out[f"{base}.biases"] = biases.astype(mx.float16)
            else:
                out[name] = tensor.astype(mx.float16)
        elif name.endswith(".bias") and base in linear_bases:
            out[name] = tensor.astype(mx.float16)
        else:
            out[name] = tensor
    return out


def remove_turbo_pack() -> bool:
    if not TURBO_DIR.exists():
        return False
    shutil.rmtree(TURBO_DIR)
    return True


def remove_base_pack() -> bool:
    if not BASE_DIR.exists():
        return False
    shutil.rmtree(BASE_DIR)
    return True
