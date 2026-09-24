"""Observation-only interfaces. Simulator objects/branch labels are not accepted."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Observation:
    episode: str
    frame: int
    timestamp: float
    rgb: np.ndarray
    depth: np.ndarray
    c2w: np.ndarray
    intrinsics: np.ndarray
    pose_track: str = "privileged_simulator_pose"
    depth_track: str = "privileged_reconstructed_depth"

    def validate(self):
        if self.rgb.ndim != 3 or self.rgb.shape[-1] != 3 or self.rgb.dtype != np.uint8:
            raise ValueError("Expected RGB uint8 image")
        if self.depth.shape != self.rgb.shape[:2] or self.depth.dtype != np.float32:
            raise ValueError("Metric depth must be float32, never display uint8")
        if self.c2w.shape != (4, 4) or self.intrinsics.shape != (3, 3):
            raise ValueError("Invalid camera metadata")
        if not np.isfinite(self.depth).all() or (self.depth < 0).any():
            raise ValueError("Invalid depths must be encoded as zero")
        if self.frame < 0 or self.timestamp < 0:
            raise ValueError("Invalid observation time")


@dataclass(frozen=True)
class GroundedMission:
    instruction: str
    kind: str
    evidence_frames: tuple[int, ...]
    bbox_xyxy: tuple[float, float, float, float] | None
    unresolved: bool

    @classmethod
    def parse(cls, value, available_frames):
        required = {"instruction", "kind", "evidence_frames", "bbox_xyxy", "unresolved"}
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError("Mission output does not match the strict schema")
        if value["kind"] not in ("navigate", "inspect", "reobserve") or type(value["unresolved"]) is not bool:
            raise ValueError("Invalid mission type/status")
        frames = tuple(value["evidence_frames"])
        if not frames or any(type(f) is not int or f not in available_frames for f in frames):
            raise ValueError("Mission cites unavailable/future observations")
        bbox = value["bbox_xyxy"]
        if bbox is not None:
            if len(bbox) != 4 or not all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in bbox) or not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
                raise ValueError("Bounding box must be normalized and nonempty")
            bbox = tuple(bbox)
        elif not value["unresolved"]:
            raise ValueError("Resolved visual reference needs observed evidence")
        return cls(value["instruction"], value["kind"], frames, bbox, value["unresolved"])
