"""Find, start and stop the local mlx-serve that does the model work."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Self

DEFAULT_URL = "http://127.0.0.1:11234"
MODELS_DIR = Path.home() / ".mlx-serve" / "models"
LOG_PATH = Path.home() / ".mlx-serve" / "idol-mlx-server.log"
STARTUP_TIMEOUT_S = 120


class ServerError(RuntimeError):
    pass


def find_binary() -> str | None:
    return shutil.which("mlx-serve")


def install_with_homebrew() -> None:
    """Install the mlx-serve command-line server from its Homebrew tap."""
    brew = shutil.which("brew")
    if not brew:
        raise ServerError(
            "mlx-serve isn't installed and Homebrew wasn't found. "
            "Install mlx-serve from https://github.com/ddalcu/mlx-serve, then run idol-mlx again."
        )
    commands = [
        [brew, "tap", "ddalcu/mlx-serve", "https://github.com/ddalcu/mlx-serve"],
        [brew, "trust", "ddalcu/mlx-serve"],
        [brew, "install", "mlx-serve"],
    ]
    for command in commands:
        result = subprocess.run(command, check=False)
        # `brew trust` only exists on newer Homebrew; older versions don't need it.
        if result.returncode != 0 and command[1] != "trust":
            raise ServerError(f"`{' '.join(command[1:])}` failed (exit {result.returncode})")


def list_models(url: str = DEFAULT_URL, timeout: float = 2.0) -> list[dict] | None:
    """The server's models, or None when nothing answers at `url`."""
    try:
        with urllib.request.urlopen(f"{url}/v1/models", timeout=timeout) as response:
            return json.load(response)["data"]
    except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError, KeyError):
        return None


def rescan(url: str = DEFAULT_URL) -> None:
    """Ask a running server to pick up models added after it started."""
    request = urllib.request.Request(f"{url}/v1/models/rescan", data=b"{}", method="POST")
    request.add_header("content-type", "application/json")
    with urllib.request.urlopen(request, timeout=30):
        pass


class Server:
    """A running mlx-serve: either one that was already up, or one this process started and will stop."""

    def __init__(self, url: str = DEFAULT_URL, process: subprocess.Popen | None = None):
        self.url = url
        self.process = process

    @property
    def started_here(self) -> bool:
        return self.process is not None

    def has_model(self, model_id: str) -> bool:
        models = list_models(self.url) or []
        return any(m.get("id") == model_id for m in models)

    def stop(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self.process = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def ensure_server(url: str = DEFAULT_URL, keep_running: bool = False) -> Server:
    """Reuse a server that already answers at `url`, or start one bound to localhost only."""
    if list_models(url) is not None:
        return Server(url)

    binary = find_binary()
    if binary is None:
        raise ServerError("mlx-serve isn't installed. Run `idol-mlx setup` to install it.")

    port = url.rsplit(":", 1)[-1]
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    command = [binary, "--serve", "--host", "127.0.0.1", "--port", port, "--model-dir", str(MODELS_DIR)]
    with open(LOG_PATH, "ab") as log:  # the child keeps its own handle to the log
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=keep_running,  # survive this process when the user wants it kept
        )
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise ServerError(f"mlx-serve exited during startup (exit {process.returncode}). See {LOG_PATH}")
        if list_models(url, timeout=1.0) is not None:
            return Server(url, None if keep_running else process)
        time.sleep(0.3)
    process.terminate()
    raise ServerError(f"mlx-serve didn't answer within {STARTUP_TIMEOUT_S} s. See {LOG_PATH}")
