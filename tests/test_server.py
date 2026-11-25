"""Reading the camera mlx-serve logs for each generation, on lines copied from a real mlx-serve 26.11.3 log."""

from pathlib import Path

from idol_mlx.client import ORTHOGRAPHIC_RADIUS
from idol_mlx.server import log_size, logged_camera_radius

FIRST_GENERATION = b"""[registry] model id=Idol-Turbo ready (4.12 GB resident)
[mesh] generating Idol steps=5 gaussians=262144 cfg=0.0 rescale_t=3.0 seed=42
[idol] camera yaw=140.4 pitch=14.7 radius=1.82 fov=49.6
[gen] idol -> 15730112 bytes
"""
SECOND_GENERATION = b"""[mesh] generating Idol steps=5 gaussians=262144 cfg=0.0 rescale_t=3.0 seed=7
[idol] camera yaw=117.5 pitch=7.3 radius=1000.00 fov=0.1
[gen] idol -> 15730112 bytes
"""


def test_reads_the_camera_of_the_generation_after_the_offset(tmp_path: Path):
    log = tmp_path / "server.log"
    log.write_bytes(FIRST_GENERATION)
    assert logged_camera_radius(0, log) == 1.82

    offset = log_size(log)
    with open(log, "ab") as f:
        f.write(SECOND_GENERATION)
    assert logged_camera_radius(offset, log) == ORTHOGRAPHIC_RADIUS


def test_no_camera_when_nothing_new_was_logged(tmp_path: Path):
    log = tmp_path / "server.log"
    log.write_bytes(FIRST_GENERATION)
    assert logged_camera_radius(log_size(log), log) is None


def test_no_camera_without_a_log(tmp_path: Path):
    missing = tmp_path / "missing.log"
    assert log_size(missing) == 0
    assert logged_camera_radius(0, missing) is None
