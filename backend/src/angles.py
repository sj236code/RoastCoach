# backend/src/angles.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from pose import FramePose

@dataclass
class FrameAngles:
    t: float
    angles: Dict[str, float]  # degrees
    conf: float

def _pt(kp: Dict[str, Tuple[float, float, float]], name: str) -> np.ndarray | None:
    if name not in kp:
        return None
    x, y, _ = kp[name]
    return np.array([x, y], dtype=float)

def _vis(kp: Dict[str, Tuple[float, float, float]], name: str) -> float:
    return float(kp.get(name, (0.0, 0.0, 0.0))[2])

def angle_deg(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
    v1 = A - B
    v2 = C - B
    n1 = np.linalg.norm(v1) + 1e-8
    n2 = np.linalg.norm(v2) + 1e-8
    cos = float(np.dot(v1, v2) / (n1 * n2))
    cos = max(-1.0, min(1.0, cos))
    return float(np.degrees(np.arccos(cos)))

def trunk_incline_deg(mid_hip: np.ndarray, mid_shoulder: np.ndarray) -> float:
    # angle between torso vector and vertical axis (0 = perfectly vertical)
    v = mid_shoulder - mid_hip
    v = v / (np.linalg.norm(v) + 1e-8)
    vertical = np.array([0.0, -1.0])  # up in image coords
    cos = float(np.dot(v, vertical))
    cos = max(-1.0, min(1.0, cos))
    return float(np.degrees(np.arccos(cos)))

def compute_angles(frames: List[FramePose], min_kp_vis: float = 0.3) -> List[FrameAngles]:
    out: List[FrameAngles] = []

    for f in frames:
        kp = f.kp
        angles: Dict[str, float] = {}

        # Left / Right knee flexion
        lh, lk, la = _pt(kp, "left_hip"), _pt(kp, "left_knee"), _pt(kp, "left_ankle")
        rh, rk, ra = _pt(kp, "right_hip"), _pt(kp, "right_knee"), _pt(kp, "right_ankle")

        if lh is not None and lk is not None and la is not None and min(_vis(kp,"left_hip"), _vis(kp,"left_knee"), _vis(kp,"left_ankle")) >= min_kp_vis:
            angles["left_knee_flex"] = angle_deg(lh, lk, la)

        if rh is not None and rk is not None and ra is not None and min(_vis(kp,"right_hip"), _vis(kp,"right_knee"), _vis(kp,"right_ankle")) >= min_kp_vis:
            angles["right_knee_flex"] = angle_deg(rh, rk, ra)

        # Left / Right elbow flexion (optional, but nice)
        ls, le, lw = _pt(kp, "left_shoulder"), _pt(kp, "left_elbow"), _pt(kp, "left_wrist")
        rs, re, rw = _pt(kp, "right_shoulder"), _pt(kp, "right_elbow"), _pt(kp, "right_wrist")

        if ls is not None and le is not None and lw is not None and min(_vis(kp,"left_shoulder"), _vis(kp,"left_elbow"), _vis(kp,"left_wrist")) >= min_kp_vis:
            angles["left_elbow_flex"] = angle_deg(ls, le, lw)

        if rs is not None and re is not None and rw is not None and min(_vis(kp,"right_shoulder"), _vis(kp,"right_elbow"), _vis(kp,"right_wrist")) >= min_kp_vis:
            angles["right_elbow_flex"] = angle_deg(rs, re, rw)

        # Trunk incline
        if lh is not None and rh is not None and ls is not None and rs is not None:
            mid_hip = (lh + rh) / 2.0
            mid_shoulder = (ls + rs) / 2.0
            angles["trunk_incline"] = trunk_incline_deg(mid_hip, mid_shoulder)

        out.append(FrameAngles(t=f.t, angles=angles, conf=f.conf))

    return out
