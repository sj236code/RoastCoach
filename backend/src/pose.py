# backend/src/pose.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import cv2
from mediapipe.python.solutions import pose as mp_pose

KP = Tuple[float, float, float]  # (x_norm, y_norm, visibility)

@dataclass
class FramePose:
    t: float
    kp: Dict[str, KP]
    conf: float  # mean visibility

_LM = mp_pose.PoseLandmark

LANDMARK_NAMES = {
    _LM.LEFT_SHOULDER: "left_shoulder",
    _LM.RIGHT_SHOULDER: "right_shoulder",
    _LM.LEFT_ELBOW: "left_elbow",
    _LM.RIGHT_ELBOW: "right_elbow",
    _LM.LEFT_WRIST: "left_wrist",
    _LM.RIGHT_WRIST: "right_wrist",
    _LM.LEFT_HIP: "left_hip",
    _LM.RIGHT_HIP: "right_hip",
    _LM.LEFT_KNEE: "left_knee",
    _LM.RIGHT_KNEE: "right_knee",
    _LM.LEFT_ANKLE: "left_ankle",
    _LM.RIGHT_ANKLE: "right_ankle",
}

def extract_pose_frames(video_path: str, max_frames: int | None = None) -> List[FramePose]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frames: List[FramePose] = []
    i = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            if max_frames and i > max_frames:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            t = (i - 1) / fps
            kp: Dict[str, KP] = {}
            confs = []

            if res.pose_landmarks:
                for lm_enum, name in LANDMARK_NAMES.items():
                    lm = res.pose_landmarks.landmark[int(lm_enum)]
                    kp[name] = (lm.x, lm.y, float(lm.visibility))
                    confs.append(float(lm.visibility))

            conf = sum(confs) / len(confs) if confs else 0.0
            frames.append(FramePose(t=t, kp=kp, conf=conf))

    cap.release()
    return frames
