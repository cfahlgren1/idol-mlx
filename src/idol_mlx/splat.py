"""Gaussian splats returned by mlx-serve: a GLB using KHR_gaussian_splatting, saved as GLB or standard 3DGS .ply."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GLB_MAGIC = b"glTF"
JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942

ATTR_ROTATION = "KHR_gaussian_splatting:ROTATION"
ATTR_SCALE = "KHR_gaussian_splatting:SCALE"
ATTR_OPACITY = "KHR_gaussian_splatting:OPACITY"
ATTR_SH0 = "KHR_gaussian_splatting:SH_DEGREE_0_COEF_0"

PLY_PROPERTIES = [
    "x", "y", "z", "nx", "ny", "nz",
    "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
]  # fmt: skip

COMPONENTS = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}
FLOAT32 = 5126
EPS = 1e-7
MIN_OPAQUE_FRACTION = 0.01


class InvalidSplatError(ValueError):
    """The splat contains NaN or infinite values: the model diverged for this image and seed."""


@dataclass
class Splat:
    """One generated object. Linear scales and opacities, quaternions in glTF order (x, y, z, w)."""

    glb: bytes
    positions: np.ndarray
    rotations: np.ndarray
    scales: np.ndarray
    opacities: np.ndarray
    sh0: np.ndarray
    seed: int | None = None  # set by Idol.generate: the seed that produced this splat

    @classmethod
    def from_glb(cls, glb: bytes) -> Splat:
        doc, binary = _read_glb(glb)
        attributes = doc["meshes"][0]["primitives"][0]["attributes"]
        missing = {"POSITION", ATTR_ROTATION, ATTR_SCALE, ATTR_OPACITY, ATTR_SH0} - attributes.keys()
        if missing:
            raise ValueError(
                f"GLB is not a KHR_gaussian_splatting splat, missing attributes: {sorted(missing)}"
            )

        def read(name: str) -> np.ndarray:
            return _accessor(doc, binary, attributes[name])

        splat = cls(
            glb=glb,
            positions=read("POSITION"),
            rotations=read(ATTR_ROTATION),
            scales=read(ATTR_SCALE),
            opacities=read(ATTR_OPACITY).reshape(-1),
            sh0=read(ATTR_SH0),
        )
        if not all(np.isfinite(a).all() for a in (splat.positions, splat.scales, splat.rotations, splat.sh0)):
            raise InvalidSplatError("The splat contains NaN or infinite values")
        return splat

    def __len__(self) -> int:
        return len(self.positions)

    @property
    def opaque_fraction(self) -> float:
        """Share of Gaussians at least half opaque. Healthy objects are well above MIN_OPAQUE_FRACTION; a
        generation that collapsed (often after a degenerate camera guess) leaves almost none."""
        return float((self.opacities >= 0.5).mean()) if len(self) else 0.0

    def check_not_empty(self) -> None:
        if self.opaque_fraction < MIN_OPAQUE_FRACTION:
            raise InvalidSplatError(
                f"Only {self.opaque_fraction:.2%} of Gaussians are opaque; the generation collapsed"
            )

    def save(self, path: str | Path) -> Path:
        """Write `.ply` (standard 3DGS) or `.glb` (as mlx-serve returned it), chosen by the file extension."""
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".glb":
            path.write_bytes(self.glb)
        elif suffix == ".ply":
            path.write_bytes(self.to_ply())
        else:
            raise ValueError(
                f"Can't save a splat as {suffix or 'a file without an extension'}: use .ply or .glb"
            )
        return path

    def to_ply(self) -> bytes:
        """3DGS layout: log scales, logit opacities, quaternions w first, normals zeroed."""
        opacity = np.clip(self.opacities, EPS, 1 - EPS)
        columns = np.hstack([
            self.positions,
            np.zeros_like(self.positions),
            self.sh0,
            np.log(opacity / (1 - opacity))[:, None],
            np.log(np.maximum(self.scales, EPS)),
            self.rotations[:, [3, 0, 1, 2]],
        ]).astype("<f4")  # fmt: skip
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(self)}\n"
            + "".join(f"property float {name}\n" for name in PLY_PROPERTIES)
            + "end_header\n"
        )
        return header.encode() + columns.tobytes()


def _read_glb(glb: bytes) -> tuple[dict, bytes]:
    if len(glb) < 20 or glb[:4] != GLB_MAGIC:
        raise ValueError("Not a GLB file (missing the glTF header)")
    json_length, json_type = struct.unpack_from("<II", glb, 12)
    if json_type != JSON_CHUNK:
        raise ValueError("GLB's first chunk is not JSON")
    try:
        doc = json.loads(glb[20 : 20 + json_length])
    except json.JSONDecodeError as err:
        # mlx-serve writes accessor bounds as bare `nan` when the model's output diverged
        raise InvalidSplatError(f"The GLB header isn't valid JSON ({err.msg} at {err.pos})") from err
    bin_offset = 20 + json_length
    bin_length, bin_type = struct.unpack_from("<II", glb, bin_offset)
    if bin_type != BIN_CHUNK:
        raise ValueError("GLB has no binary chunk")
    start = bin_offset + 8
    return doc, glb[start : start + bin_length]


def _accessor(doc: dict, binary: bytes, index: int) -> np.ndarray:
    accessor = doc["accessors"][index]
    if accessor["componentType"] != FLOAT32:
        raise ValueError(
            f"Expected float32 for accessor {index}, got componentType {accessor['componentType']}"
        )
    view = doc["bufferViews"][accessor["bufferView"]]
    width = COMPONENTS[accessor["type"]]
    offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    values = np.frombuffer(binary, dtype="<f4", count=accessor["count"] * width, offset=offset)
    return values.reshape(accessor["count"], width)
