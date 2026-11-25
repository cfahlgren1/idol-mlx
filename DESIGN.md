# idol-mlx: package design

Status: v0.1 implemented (local-merge pack, default and preset generation, folders, `view`, `setup`, `uninstall`). The pack is built locally on first run; publishing a merged pack is still open.

## Goal

One command on an Apple Silicon Mac turns a photo into a 3D Gaussian splat in about 15 s, using Idol with the Turbo adapter, fully local.

**Non-goals for v1:** NVIDIA support, training, a GUI app, a Python reimplementation of Idol. mlx-serve already runs the model; this package doesn't duplicate it.

## Layout

```
idol-mlx/
├── pyproject.toml            # hatchling; script: idol-mlx = idol_mlx.cli:main
├── README.md
├── assets/hero.png
├── src/idol_mlx/
│   ├── __init__.py           # Idol, Splat (public API)
│   ├── cli.py                # argument parsing, presets, progress output
│   ├── client.py             # Idol class: generate(), streams stage events
│   ├── server.py             # find, start and health-check mlx-serve
│   ├── pack.py               # get the Turbo pack: download, or merge locally
│   ├── splat.py              # Splat: GLB (KHR_gaussian_splatting) → 3DGS .ply
│   └── viewer/
│       └── index.html        # Spark-based local viewer for --view
└── tests/
    ├── test_splat.py         # GLB→PLY round trip on a small real fixture
    └── test_e2e.py           # one real generation (skipped unless on Apple Silicon)
```

Roughly 300 lines of Python plus the viewer page.

## Modules

| Module | Responsibility | Notes |
|---|---|---|
| `server.py` | Reuse mlx-serve if it answers on `localhost:11234`; otherwise start `mlx-serve --serve --model-dir ~/.mlx-serve/models` in the background, wait for `/v1/models`, and stop it on exit if we started it. Offer `brew install mlx-serve` when missing, with a prompt. | `--keep-server` leaves it running for batch work |
| `pack.py` | Make sure `~/.mlx-serve/models/Idol-Turbo` exists. Strategy A: download the published merged pack. Strategy B (fallback): build it locally. | See "Pack strategy" |
| `client.py` | `Idol.generate(image, steps, seed, gaussians, mode, guidance)`. POSTs base64 image with `stream: true`, yields stage events (`idol-condition`, `idol-denoise i/n`, `idol-octree`, `idol-gaussians`), returns a `Splat`. | Uses only the public HTTP API |
| `splat.py` | Parse the GLB, write 3DGS `.ply`: positions, `f_dc` from SH0, `logit(opacity)`, `log(scale)`, quaternion reordered xyzw → wxyz. Also `save("x.glb")` passthrough. | numpy only. The prototype is `glb2ply.py` |
| `cli.py` | Presets, folder input, `--view`, `--format`, `uninstall`. Uses `rich` for the progress bar. | |
| `viewer/` | Serve the `.ply` and viewer on a random localhost port; open the browser. | v0.1 uses `@mkkellogg/gaussian-splats-3d`; Spark would match the Hub's viewer |

## CLI surface

```
idol-mlx IMAGE|DIR [-o OUT] [--view] [--format ply|glb]
         [--fast | --quality high | --original]
         [--steps N] [--seed N] [--gaussians N] [--mode synthetic|real]
         [--keep-server]
idol-mlx setup        # install mlx-serve + model without generating
idol-mlx uninstall    # remove the Turbo pack (asks before touching mlx-serve)
```

| Preset | Request |
|---|---|
| `--fast` | `steps 4, guidance 0, gaussians 65536` |
| default | `steps 5, guidance 0, gaussians 262144` |
| `--quality high` | `steps 8, guidance 0, gaussians 262144` |
| `--original` | base pack, `steps 50, guidance 3.0` (downloads the base pack's denoiser if missing) |

## Pack strategy

| | A. Published merged pack | B. Merge locally |
|---|---|---|
| First-run download | ~3 GB | ~4.6 GB (base pack + full-precision denoiser + adapter) |
| Needs `mlx` installed | No | Yes (for `mx.quantize`) |
| We host weights | Yes, a Hugging Face repo | No |
| Time | Download only | Download + ~1 min merge |

**Recommendation:** publish the pack (A) and keep B as the fallback and as the documented way to rebuild it. The pack is a full mlx-serve model folder: the base pack's files plus the merged `denoiser.safetensors`. It needs the DINOv3 license file and NOTICE, as the base pack has.

**8-bit or 16-bit:** 16-bit was ~1 s faster on the M3 Max (14.7–15.3 s vs 15.9–16.7 s), but the denoiser is 777 MB instead of 426 MB. Ship 8-bit by default to keep the download small. Revisit if M5 `--w8a8` changes the picture.

## Decisions

- **Name:** `idol-mlx` (free on PyPI). Not `idol-turbo`, which is the adapter's own name, and not `idol`, which is taken.
- **Unofficial**, stated in the README's first credits line. Credit Speri Labs, AK and ddalcu.
- **Depends on mlx-serve** instead of porting Idol to Python MLX. That's less code, but it relies on mlx-serve's HTTP API staying stable: pin a minimum version and check `/v1/models` capabilities.
- **PLY by default:** most splat tools read 3DGS `.ply`, and the Hugging Face file viewer renders it.

## Risks

| Risk | Mitigation |
|---|---|
| mlx-serve API or Idol plugin changes | Pin the version; e2e test in CI on a macOS runner |
| Homebrew tap install is a third-party trust step | Prompt before installing; document the manual path |
| Slow-only adapter is not the full Turbo recipe | Document it; contribute adapter or phase-switch support to mlx-serve |
| Disk: ~3 GB per pack | `uninstall` command; reuse the base pack's files via symlinks where both exist |

## Prototype (already working)

| Prototype file | Becomes |
|---|---|
| `merge_turbo.py` | `pack.py` local merge. Verified: re-quantizing the untouched release reproduces the shipped 8-bit pack exactly (784/784 tensors) |
| `glb2ply.py` | `splat.py` |
| `bench.sh` / `stages.py` | `client.py` (streaming + stage timing) |
| `viewer.html` | `viewer/index.html` (move from `@mkkellogg/gaussian-splats-3d` to Spark) |

Measured on an M3 Max (40-core GPU, 48 GB): Turbo 4 / 5 / 8 steps = 13.4 / ~16 / 22.7 s; base 50 steps + guidance = 191 s.

## Milestones

1. **v0.1:** `splat.py` + `client.py` + `server.py` + local-merge `pack.py`; default preset only; `--view`.
2. **v0.2:** publish the merged pack; presets; folders; Python API docs.
3. **Later:** upstream adapter support to mlx-serve (full slow→fast switch); `--push` to a Hub repo so the result renders in the Hugging Face viewer.
