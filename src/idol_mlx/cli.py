"""idol-mlx: photo to 3D Gaussian splat on Apple Silicon."""

from __future__ import annotations

import argparse
import platform
import shutil
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from . import __version__, pack, server
from .client import GenerationError, Idol, check_settings
from .client import Progress as StageProgress
from .server import ServerError
from .splat import Splat
from .turntable import TurntableError, check_turntable_tools, render_frames, write_animation

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".bmp", ".tif", ".tiff"}

PRESETS = {
    "fast": {"steps": 4, "guidance": 0.0, "gaussians": 65_536},
    "default": {"steps": 5, "guidance": 0.0, "gaussians": 262_144},
    "high": {"steps": 8, "guidance": 0.0, "gaussians": 262_144},
    "original": {"steps": 50, "guidance": 3.0, "gaussians": 262_144},
}

STAGE_LABELS = {
    "idol-preprocess": "preparing image",
    "idol-matte": "removing background",
    "idol-condition": "encoding image",
    "idol-denoise": "denoising",
    "idol-octree": "decoding octree",
    "idol-gaussians": "decoding Gaussians",
}

console = Console()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["setup"]:
        return _guarded(cmd_setup, _setup_parser().parse_args(argv[1:]))
    if argv[:1] == ["uninstall"]:
        return _guarded(cmd_uninstall, _uninstall_parser().parse_args(argv[1:]))
    if argv[:1] == ["view"]:
        return _guarded(cmd_view, _view_parser().parse_args(argv[1:]))
    if argv[:1] == ["turntable"]:
        return _guarded(cmd_turntable, _turntable_parser().parse_args(argv[1:]))
    return _guarded(cmd_generate, _generate_parser().parse_args(argv))


def _guarded(command, args) -> int:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        console.print("[red]idol-mlx needs an Apple Silicon Mac.[/red] On NVIDIA GPUs, use Idol's own SDK.")
        return 1
    try:
        return command(args)
    except KeyboardInterrupt:
        console.print("\nStopped.")
        return 130
    except (ServerError, GenerationError, TurntableError, ValueError, FileNotFoundError) as err:
        console.print(f"[red]Error:[/red] {err}")
        return 1


# ---------------------------------------------------------------- generate


def _generate_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="idol-mlx",
        description="Turn a photo of an object into a 3D Gaussian splat, locally on Apple Silicon.",
        epilog="Other commands: `idol-mlx view FILE.ply…` (open existing splats), "
        "`idol-mlx turntable FILE.ply` (spinning GIF/MP4 of an existing splat), "
        "`idol-mlx setup` (install everything without generating), `idol-mlx uninstall`.",
    )
    p.add_argument("inputs", nargs="+", type=Path, help="image files or folders of images")
    p.add_argument("-o", "--output", type=Path, help="output folder (default: next to each image)")
    p.add_argument("--format", choices=["ply", "glb"], default="ply", help="output format (default: ply)")
    p.add_argument("--view", action="store_true", help="open the results in a browser viewer")
    p.add_argument("--turntable", nargs="?", const="gif", choices=["gif", "mp4"],
                   help="also save a spinning animation of each result (default: gif)")  # fmt: skip
    preset = p.add_mutually_exclusive_group()
    preset.add_argument("--fast", action="store_true", help="4 steps, 65k Gaussians (~12 s)")
    preset.add_argument("--quality", choices=["high"], help="'high': 8 steps (~23 s)")
    preset.add_argument("--original", action="store_true", help="base Idol, 50 steps + guidance (~3 min)")
    p.add_argument("--steps", type=int, help="override the preset's step count")
    p.add_argument("--gaussians", type=int, help="override the preset's Gaussian count (32768 to 1048576)")
    p.add_argument("--seed", type=int, default=42, help="change for a different take (default: 42)")
    p.add_argument("--mode", choices=["real", "synthetic"], default="real",
                   help="'real' for photographs (default); 'synthetic' crops tighter, for renders and generated images")  # fmt: skip
    p.add_argument("--keep-server", action="store_true", help="leave mlx-serve running afterwards")
    p.add_argument("--skip-existing", action="store_true", help="skip images whose output already exists")
    p.add_argument("--version", action="version", version=f"idol-mlx {__version__}")
    return p


def cmd_generate(args: argparse.Namespace) -> int:
    images = _collect_images(args.inputs)
    preset_name = (
        "original" if args.original else "fast" if args.fast else "high" if args.quality else "default"
    )
    settings = dict(PRESETS[preset_name])
    if args.steps is not None:
        settings["steps"] = args.steps
    if args.gaussians is not None:
        settings["gaussians"] = args.gaussians
    check_settings(steps=settings["steps"], gaussians=settings["gaussians"], mode=args.mode)
    if args.turntable:
        check_turntable_tools(args.turntable)

    _ensure_mlx_serve_installed(assume_yes=False)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)

    if args.original:
        pack.ensure_base_pack(log=_log)
    else:
        pack.ensure_turbo_pack(log=_log)
    with console.status("Starting mlx-serve…"):
        idol = Idol(keep_server=args.keep_server, original=args.original)
    written: list[Path] = []
    spins: list[tuple[Path, Path]] = []  # (splat file, turntable file)
    failed: list[Path] = []
    try:
        for image in images:
            out_dir = args.output or image.parent
            out_path = out_dir / f"{image.stem}.{args.format}"
            spin_path = out_dir / f"{image.stem}.{args.turntable}" if args.turntable else None
            if args.skip_existing and out_path.exists():
                console.print(f"[dim]• {out_path} exists, skipping[/dim]")
                written.append(out_path)
                if spin_path and not spin_path.exists():
                    spins.append((out_path, spin_path))
                continue
            started = time.monotonic()
            try:
                splat = _generate_with_progress(idol, image, settings, args)
            except GenerationError as err:
                console.print(f"[red]✗[/red] {image}  [dim]{err}[/dim]")
                failed.append(image)
                continue
            splat.save(out_path)
            elapsed = time.monotonic() - started
            retried = f" · retried with seed {splat.seed}" if splat.seed != args.seed else ""
            console.print(
                f"[green]✓[/green] {out_path}  [dim]{len(splat):,} Gaussians · {elapsed:.1f} s{retried}[/dim]"
            )
            written.append(out_path)
            if spin_path:
                spins.append((out_path, spin_path))
    finally:
        idol.close()

    # Turntables render after the server stops, so the browser has the GPU to itself.
    for splat_path, spin_path in spins:
        try:
            _render_turntable(_ply_bytes(splat_path), spin_path)
        except TurntableError as err:
            console.print(f"[red]✗[/red] {spin_path}  [dim]{err}[/dim]")
            failed.append(splat_path)

    if failed:
        console.print(
            f"[red]{len(failed)} of {len(images)} failed:[/red] " + ", ".join(str(p) for p in failed)
        )

    if args.view:
        plys = [p for p in written if p.suffix == ".ply"]
        if not plys:
            console.print("The viewer shows .ply files; rerun without --format glb to view them.")
            return 1 if failed else 0
        from .viewer import serve

        serve(plys)
    return 1 if failed else 0


def _ply_bytes(splat_path: Path) -> bytes:
    if splat_path.suffix == ".glb":
        return Splat.from_glb(splat_path.read_bytes()).to_ply()
    return splat_path.read_bytes()


def _generate_with_progress(idol: Idol, image: Path, settings: dict, args: argparse.Namespace):
    columns = (
        TextColumn("[bold]{task.fields[name]}[/bold]"),
        TextColumn("{task.description}", style="dim"),
        BarColumn(bar_width=28),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    )
    with Progress(*columns, console=console, transient=True) as bar:
        task = bar.add_task("starting", total=None, name=image.name)

        def on_progress(p: StageProgress) -> None:
            label = STAGE_LABELS.get(p.stage, p.stage)
            if p.stage == "idol-denoise":
                label = f"denoising {p.step}/{p.total}"
            bar.update(task, description=label, completed=p.step, total=p.total)

        return idol.generate(image, seed=args.seed, mode=args.mode, on_progress=on_progress, **settings)


def _log(message: str) -> None:
    console.print(f"[dim]•[/dim] {message}")


def _collect_images(inputs: list[Path]) -> list[Path]:
    images: list[Path] = []
    for path in inputs:
        if path.is_dir():
            found = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            if not found:
                raise FileNotFoundError(f"No images in {path}")
            images.extend(found)
        elif path.is_file():
            images.append(path)
        else:
            raise FileNotFoundError(f"{path} doesn't exist")
    return images


# ---------------------------------------------------------------- view / setup / uninstall


def _turntable_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="idol-mlx turntable",
        description="Save a spinning animation of a .ply splat. Needs Google Chrome (or Chromium, Brave, Edge).",
    )
    p.add_argument("file", type=Path, help="a .ply splat")
    p.add_argument("-o", "--output", type=Path, help="output .gif or .mp4 (default: FILE.gif)")
    p.add_argument("--frames", type=int, default=72, help="frames per full turn (default: 72)")
    p.add_argument("--size", type=int, default=640, help="width and height in pixels (default: 640)")
    p.add_argument("--fps", type=int, default=18, help="frames per second (default: 18)")
    return p


def cmd_turntable(args: argparse.Namespace) -> int:
    if not args.file.is_file() or args.file.suffix.lower() != ".ply":
        raise FileNotFoundError(f"{args.file} isn't a .ply file")
    out_path = args.output or args.file.with_suffix(".gif")
    check_turntable_tools(out_path.suffix.lstrip(".").lower())
    _render_turntable(args.file.read_bytes(), out_path, frames=args.frames, size=args.size, fps=args.fps)
    return 0


def _render_turntable(ply: bytes, out_path: Path, frames: int = 72, size: int = 640, fps: int = 18) -> None:
    started = time.monotonic()
    with Progress(
        TextColumn("[bold]{task.fields[name]}[/bold]"),
        TextColumn("rendering turntable", style="dim"),
        BarColumn(bar_width=28),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as bar:
        task = bar.add_task("", total=frames, name=out_path.name)
        images = render_frames(
            ply, frames=frames, size=size, on_frame=lambda n, _: bar.update(task, completed=n)
        )
    write_animation(images, out_path, fps=fps)
    elapsed = time.monotonic() - started
    size_mb = out_path.stat().st_size / 1e6
    console.print(
        f"[green]✓[/green] {out_path}  [dim]{frames} frames · {size_mb:.1f} MB · {elapsed:.1f} s[/dim]"
    )


def _view_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="idol-mlx view", description="Open .ply splats in a browser viewer.")
    p.add_argument("files", nargs="+", type=Path, help=".ply files, or folders of them")
    p.add_argument("--no-browser", action="store_true", help="print the URL instead of opening it")
    return p


def cmd_view(args: argparse.Namespace) -> int:
    from .viewer import serve

    plys: list[Path] = []
    for path in args.files:
        if path.is_dir():
            plys.extend(sorted(path.glob("*.ply")))
        elif path.is_file() and path.suffix.lower() == ".ply":
            plys.append(path)
        else:
            raise FileNotFoundError(f"{path} isn't a .ply file or a folder")
    if not plys:
        raise FileNotFoundError("No .ply files to show")
    serve(plys, open_browser=not args.no_browser)
    return 0


def _setup_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="idol-mlx setup", description="Install mlx-serve and build the Turbo pack."
    )
    p.add_argument("-y", "--yes", action="store_true", help="install mlx-serve without asking")
    return p


def cmd_setup(args: argparse.Namespace) -> int:
    _ensure_mlx_serve_installed(assume_yes=args.yes)
    pack.ensure_turbo_pack(log=_log)
    command = "idol-mlx" if shutil.which("idol-mlx") else "uvx idol-mlx"
    console.print(f"[green]Ready.[/green] Try: {command} photo.jpg --view")
    return 0


def _uninstall_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="idol-mlx uninstall", description="Remove the model packs idol-mlx added."
    )
    p.add_argument("--all", action="store_true", help="also remove the base Idol pack (~3 GB)")
    return p


def cmd_uninstall(args: argparse.Namespace) -> int:
    if pack.remove_turbo_pack():
        console.print(f"Removed {pack.TURBO_DIR}")
    if args.all and pack.remove_base_pack():
        console.print(f"Removed {pack.BASE_DIR}")
    console.print("mlx-serve itself stays installed; remove it with `brew uninstall mlx-serve`.")
    return 0


def _ensure_mlx_serve_installed(assume_yes: bool) -> None:
    if server.find_binary():
        return
    console.print("idol-mlx runs the model with [bold]mlx-serve[/bold], which isn't installed.")
    console.print("It installs from its Homebrew tap: github.com/ddalcu/mlx-serve")
    if not assume_yes:
        answer = console.input("Install it now with Homebrew? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            raise ServerError(
                "mlx-serve is required. Install it with `idol-mlx setup` or from its GitHub page."
            )
    server.install_with_homebrew()


if __name__ == "__main__":
    raise SystemExit(main())
