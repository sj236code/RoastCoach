# app/streamlit_app.py
# RoastCoach — Streamlit UI with cached pipeline + stable widget state + Gemini personality toggle
# PLUS: S3 persistence for progress tracking (videos + analysis.json + artifacts)
# PLUS: History tab (loads manifests + analysis from S3)

from __future__ import annotations

import base64
import io
import json
import os
import sys
import uuid
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, continue without it - will use system environment variables
    pass
except Exception as e:
    # If .env loading fails, continue - might use system env vars
    import warnings
    warnings.warn(f"Could not load .env file: {e}")

# Optional S3
try:
    import boto3
    from botocore.exceptions import NoCredentialsError, ClientError
    BOTO3_AVAILABLE = True
except Exception:
    BOTO3_AVAILABLE = False
    boto3 = None
    NoCredentialsError = Exception  # type: ignore
    ClientError = Exception  # type: ignore


# ----------------------------- Page config -----------------------------
st.set_page_config(page_title="RoastCoach", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for dark, modern, sleek UI inspired by Cluely and Notion
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /* Dark theme base */
    .stApp {
        background: #191919;
        color: #e5e5e5;
    }

    .main {
        background: #191919;
    }

    .main > div {
        padding-top: 1.5rem;
        max-width: 1400px;
        margin: 0 auto;
        background: #191919;
    }

    /* Typography - light text on dark */
    h1 {
        color: #ffffff;
        font-weight: 700;
        font-size: 2.5rem;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
        line-height: 1.2;
    }

    h2 {
        color: #e5e5e5;
        font-weight: 600;
        font-size: 1.5rem;
        letter-spacing: -0.01em;
        margin-top: 2.5rem;
        margin-bottom: 1rem;
        line-height: 1.3;
    }

    h3 {
        color: #d4d4d4;
        font-weight: 600;
        font-size: 1.25rem;
        letter-spacing: -0.01em;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        line-height: 1.4;
    }

    p, .stMarkdown p {
        color: #b3b3b3;
        font-size: 0.95rem;
        line-height: 1.6;
        font-weight: 400;
    }

    /* Modern, clickable buttons - dark theme */
    .stButton > button {
        width: 100%;
        border-radius: 6px;
        font-weight: 500;
        font-size: 0.95rem;
        padding: 0.75rem 1.5rem;
        transition: all 0.2s ease;
        border: 1px solid #404040;
        background: #2a2a2a;
        color: #ffffff;
        cursor: pointer;
    }

    .stButton > button:hover {
        background: #353535;
        border-color: #4a4a4a;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }

    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
    }

    .stButton > button:focus {
        box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.1);
        outline: none;
    }

    .stButton > button:disabled {
        background: #2a2a2a;
        border-color: #2a2a2a;
        color: #666666;
        cursor: not-allowed;
        opacity: 0.5;
    }

    /* Dark theme metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.75rem;
        font-weight: 600;
        color: #ffffff;
        letter-spacing: -0.01em;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.875rem;
        font-weight: 500;
        color: #9a9a9a;
        text-transform: uppercase;
        letter-spacing: 0.02em;
    }

    [data-testid="stMetricDelta"] {
        font-size: 0.875rem;
        font-weight: 500;
    }

    /* Dark metric containers */
    div[data-testid="stMetricContainer"] {
        background: #252525;
        border: 1px solid #333333;
        border-radius: 8px;
        padding: 1.25rem;
        transition: all 0.2s ease;
    }

    div[data-testid="stMetricContainer"]:hover {
        border-color: #404040;
        background: #2a2a2a;
    }

    /* Dark theme alerts */
    .stSuccess {
        border-radius: 6px;
        padding: 0.875rem 1rem;
        border-left: 3px solid #10b981;
        background: rgba(16, 185, 129, 0.1);
        color: #6ee7b7;
    }

    .stInfo {
        border-radius: 6px;
        padding: 0.875rem 1rem;
        border-left: 3px solid #3b82f6;
        background: rgba(59, 130, 246, 0.1);
        color: #93c5fd;
    }

    .stWarning {
        border-radius: 6px;
        padding: 0.875rem 1rem;
        border-left: 3px solid #f59e0b;
        background: rgba(245, 158, 11, 0.1);
        color: #fcd34d;
    }

    .stError {
        border-radius: 6px;
        padding: 0.875rem 1rem;
        border-left: 3px solid #ef4444;
        background: rgba(239, 68, 68, 0.1);
        color: #fca5a5;
    }

    div[data-testid="stExpander"] {
        border-radius: 6px;
        border: 1px solid #333333;
        background: #252525;
    }

    .stCaption {
        color: #9a9a9a;
        font-size: 0.875rem;
        font-weight: 400;
    }

    /* Dark dividers */
    hr {
        border: none;
        border-top: 1px solid #333333;
        margin: 2rem 0;
    }

    /* Dark selectboxes and inputs */
    .stSelectbox > div > div {
        border-radius: 6px;
        border: 1px solid #333333;
        background: #252525;
        color: #e5e5e5;
    }

    .stSelectbox > div > div:hover {
        border-color: #404040;
        background: #2a2a2a;
    }

    .stTextInput > div > div > input {
        border-radius: 6px;
        border: 1px solid #333333;
        font-size: 0.95rem;
        background: #252525;
        color: #e5e5e5;
    }

    .stTextInput > div > div > input:focus {
        border-color: #4a4a4a;
        box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.05);
        background: #2a2a2a;
    }

    .stTextInput > div > div > input::placeholder {
        color: #666666;
    }

    /* Dark dataframes */
    .dataframe {
        border-radius: 6px;
        border: 1px solid #333333;
        background: #252525;
        color: #e5e5e5;
    }

    /* Dark links */
    a {
        color: #60a5fa;
        text-decoration: none;
        font-weight: 500;
    }

    a:hover {
        color: #93c5fd;
        text-decoration: underline;
    }

    /* Dark file uploader */
    .stFileUploader > div {
        border-radius: 6px;
        border: 2px dashed #404040;
        background: #252525;
        transition: all 0.2s ease;
    }

    .stFileUploader > div:hover {
        border-color: #4a4a4a;
        background: #2a2a2a;
    }

    /* Dark checkbox and slider */
    .stCheckbox label {
        color: #e5e5e5;
    }

    .stSlider label {
        color: #e5e5e5;
    }

    /* Dark tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #191919;
        border-bottom: 1px solid #333333;
    }

    .stTabs [data-baseweb="tab"] {
        color: #9a9a9a;
    }

    .stTabs [aria-selected="true"] {
        color: #ffffff;
    }

    /* Subtle animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(4px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .main > div {
        animation: fadeIn 0.3s ease-out;
    }

    /* Mobile detection class */
    .mobile-device {
        /* Additional mobile-specific styles can be added here */
    }

    /* Mobile Responsive Styles */
    @media screen and (max-width: 768px) {
        /* Stack columns vertically on mobile */
        [data-testid="column"] {
            width: 100% !important;
            margin-bottom: 1rem;
        }

        /* Adjust typography for mobile */
        h1 {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }

        h2 {
            font-size: 1.25rem;
            margin-top: 1.5rem;
            margin-bottom: 0.75rem;
        }

        h3 {
            font-size: 1.1rem;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }

        p, .stMarkdown p {
            font-size: 0.9rem;
            line-height: 1.5;
        }

        /* Touch-friendly buttons */
        .stButton > button {
            min-height: 44px;
            padding: 0.875rem 1.25rem;
            font-size: 1rem;
        }

        /* Full-width containers on mobile */
        .main > div {
            max-width: 100%;
            padding: 1rem;
        }

        /* Optimize metric cards for mobile */
        div[data-testid="stMetricContainer"] {
            padding: 1rem;
            margin-bottom: 0.75rem;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.5rem;
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.8rem;
        }

        /* Stack metric columns */
        [data-testid="column"] > div {
            width: 100% !important;
        }

        /* Optimize file uploader for mobile */
        .stFileUploader > div {
            padding: 1rem;
            min-height: 120px;
        }

        /* Make radio buttons more touch-friendly */
        .stRadio > label {
            padding: 0.5rem;
            font-size: 0.95rem;
        }

        /* Optimize video previews */
        video {
            max-width: 100% !important;
            height: auto !important;
        }

        /* Adjust spacing */
        .stDivider {
            margin: 1.5rem 0;
        }

        /* Make expanders more touch-friendly */
        div[data-testid="stExpander"] {
            margin-bottom: 0.75rem;
        }

        /* Optimize tables for mobile */
        .dataframe {
            font-size: 0.85rem;
            overflow-x: auto;
            display: block;
        }

        /* Adjust alert padding */
        .stSuccess, .stInfo, .stWarning, .stError {
            padding: 0.75rem;
            font-size: 0.9rem;
        }

        /* Make tabs more touch-friendly */
        .stTabs [data-baseweb="tab"] {
            padding: 0.75rem 1rem;
            font-size: 0.95rem;
        }
    }

    /* Extra small mobile devices */
    @media screen and (max-width: 480px) {
        h1 {
            font-size: 1.75rem;
        }

        h2 {
            font-size: 1.1rem;
        }

        h3 {
            font-size: 1rem;
        }

        .stButton > button {
            font-size: 0.95rem;
            padding: 0.75rem 1rem;
        }

        div[data-testid="stMetricContainer"] {
            padding: 0.875rem;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.25rem;
        }
    }

    /* History tab - Video card styling (real estate app aesthetic) */
    .history-video-card {
        background: #252525;
        border-radius: 12px;
        padding: 0;
        margin: 1rem 0;
        border: 1px solid #333333;
        overflow: hidden;
        position: relative;
    }

    .history-video-card img {
        width: 100%;
        height: auto;
        display: block;
        border-radius: 12px 12px 0 0;
    }

    .history-stat-card {
        background: #252525;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #333333;
    }

    .history-stat-card h3 {
        color: #ffffff;
        margin-top: 0;
        margin-bottom: 0.5rem;
    }

    .history-stat-card h4 {
        color: #e5e5e5;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-size: 1.1rem;
    }

    .history-progress {
        text-align: center;
        color: #9a9a9a;
        font-size: 0.95rem;
        font-weight: 500;
        margin: 1rem 0;
        padding: 0.75rem;
        background: #252525;
        border-radius: 8px;
        border: 1px solid #333333;
    }

    /* Navigation buttons styling */
    .stButton > button[kind="secondary"] {
        background: #2a2a2a;
        border: 1px solid #404040;
        color: #ffffff;
        font-size: 1.2rem;
        padding: 0.5rem 1rem;
        min-width: 60px;
    }

    .stButton > button[kind="secondary"]:hover:not(:disabled) {
        background: #353535;
        border-color: #4a4a4a;
        transform: translateY(-1px);
    }

    .stButton > button[kind="secondary"]:disabled {
        opacity: 0.3;
        cursor: not-allowed;
    }

    /* Video selector styling */
    .stSelectbox > div > div {
        background: #252525;
        border: 1px solid #333333;
        color: #e5e5e5;
    }

    .stSelectbox > div > div:hover {
        border-color: #404040;
        background: #2a2a2a;
    }
    </style>
""", unsafe_allow_html=True)

st.title("RoastCoach")
st.markdown("### Exercise-Agnostic Motion Fingerprinting")
st.caption("Upload a coach reference video, then a user attempt. We'll compare joint-angle trajectories and provide personalized feedback.")

# Mobile detection and enhancement script
st.markdown("""
<script>
(function() {
    // Detect mobile device
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;

    if (isMobile || isTouchDevice) {
        // Add mobile class to body
        document.body.classList.add('mobile-device');

        // Enhance touch targets
        const touchTargets = document.querySelectorAll('button, a, input[type="radio"], input[type="checkbox"]');
        touchTargets.forEach(target => {
            if (target.offsetHeight < 44) {
                target.style.minHeight = '44px';
                target.style.minWidth = '44px';
            }
        });

        // Optimize video elements for mobile
        const videos = document.querySelectorAll('video');
        videos.forEach(video => {
            video.setAttribute('playsinline', '');
            video.setAttribute('webkit-playsinline', '');
        });

        // Prevent zoom on input focus (iOS)
        const inputs = document.querySelectorAll('input, textarea, select');
        inputs.forEach(input => {
            if (input.style.fontSize === '') {
                input.style.fontSize = '16px'; // Prevents iOS zoom
            }
        });
    }

    // Store mobile status in sessionStorage for Python access
    sessionStorage.setItem('isMobile', isMobile || isTouchDevice ? 'true' : 'false');
})();
</script>
""", unsafe_allow_html=True)


# ----------------------------- Paths -----------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


# ----------------------------- Backend imports -----------------------------
BACKEND_SRC = ROOT / "backend" / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.append(str(BACKEND_SRC))

try:
    from pose import extract_pose_frames
    from angles import compute_angles
    from segment import (
        choose_driver_column,
        find_reps_from_driver,
        filter_reps,
        mean_std_normalized_rep,
        resample_rep_angles,
    )
except Exception as e:
    st.error("Could not import backend modules. Check your folder structure and filenames.")
    st.exception(e)
    st.stop()

# Try importing Gemini helper (optional)
LLM_AVAILABLE = True
_llm_err = None
try:
    # backend/src/llm_coach.py must expose generate_personality_tip(...)
    from llm_coach import generate_personality_tip
except Exception as e:
    LLM_AVAILABLE = False
    _llm_err = e


# ----------------------------- Session state init -----------------------------
def ss_init():
    defaults = {
        # stage flags
        "stage1_done": False,
        "stage2_done": False,
        # file signatures (detect change)
        "coach_file_sig": None,
        "user_file_sig": None,
        # stage1 outputs
        "coach_path": None,
        "user_path": None,
        "df_coach": None,
        "df_user": None,
        "shared_cols": None,
        # stage2 outputs
        "coach_driver": None,
        "coach_reps": None,
        "user_reps": None,
        "coach_driver_smooth": None,
        "user_driver_smooth": None,
        "angle_cols": None,
        "N": 100,
        "tt": None,
        "coach_mean": None,
        "coach_std": None,
        "coach_stack": None,
        "user_mean": None,
        "user_std": None,
        "user_stack": None,
        # run tracking (local + s3)
        "run_id": None,
        "run_created_at": None,
        "latest_analysis": None,
        "latest_tip": None,
        "saved_to_s3": False,
        "s3_last_error": None,
        "artifacts_local": {},  # file paths for this run (filled after analysis)
        "exercise_label": "",
        # video naming
        "user_video_name": None,  # Custom name for user video (optional)
        # processing flags
        "processing_complete": False,
        "summary_generated": False,
        "evolution_data": None,
        "coach_video_s3_key": None,
        "user_video_s3_key": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


ss_init()

# ----------------------------- Helpers -----------------------------
def utc_run_id() -> Tuple[str, str]:
    """
    Returns (run_id, iso_timestamp).
    run_id is filesystem-friendly but preserves time ordering.
    """
    now = datetime.now(timezone.utc)
    iso = now.isoformat(timespec="seconds")
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    return run_id, iso


def save_upload(uploaded_file, out_dir: Path, prefix: str, run_id: str | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix
    rid = run_id or uuid.uuid4().hex[:8]
    fname = f"{prefix}_{rid}{suffix}"
    out_path = out_dir / fname
    out_path.write_bytes(uploaded_file.getbuffer())
    return out_path


def file_signature(uploaded_file) -> Tuple[str, int]:
    data = uploaded_file.getbuffer()
    return (uploaded_file.name, int(len(data)))


def video_small(uploaded_file, width_px: int = 360):
    data = uploaded_file.getvalue()
    b64 = base64.b64encode(data).decode("utf-8")

    suffix = Path(uploaded_file.name).suffix.lower()
    mime = "video/mp4"
    if suffix == ".mov":
        mime = "video/quicktime"
    elif suffix == ".m4v":
        mime = "video/mp4"

    html = f"""
    <video width="{width_px}" controls style="border-radius: 12px; border: 1px solid #e5e7eb;">
      <source src="data:{mime};base64,{b64}" type="{mime}">
      Your browser does not support the video tag.
    </video>
    """
    st.markdown(html, unsafe_allow_html=True)


def angles_to_df(angle_frames) -> pd.DataFrame:
    rows = []
    for f in angle_frames:
        row = {"t": f.t, "conf": f.conf}
        row.update(f.angles)
        rows.append(row)
    return pd.DataFrame(rows)


def conf_summary(df: pd.DataFrame) -> str:
    if "conf" not in df.columns:
        return "conf: n/a"
    c = df["conf"].to_numpy(dtype=float)
    if np.isfinite(c).sum() == 0:
        return "conf: n/a"
    return (
        f"conf mean={np.nanmean(c):.2f}, "
        f"p10={np.nanpercentile(c,10):.2f}, "
        f"p50={np.nanpercentile(c,50):.2f}, "
        f"p90={np.nanpercentile(c,90):.2f}"
    )


def phase_from_t(idx_peak: int, idx_bottom: int, N: int, bottom_window_frac: float = 0.08) -> str:
    w = int(bottom_window_frac * N)
    lo = max(0, idx_bottom - w)
    hi = min(N - 1, idx_bottom + w)
    if idx_peak < lo:
        return "descent"
    if idx_peak <= hi:
        return "bottom"
    return "ascent"


def severity_from_deg(deg_outside: float, mild=5.0, moderate=12.0, severe=20.0) -> str:
    if deg_outside >= severe:
        return "severe"
    if deg_outside >= moderate:
        return "moderate"
    if deg_outside >= mild:
        return "mild"
    return "none"


def pretty_joint(j: str) -> str:
    return j.replace("_", " ")


def min_halfwidth_for_joint(j: str, knee_elbow: float, trunk: float) -> float:
    jl = j.lower()
    if "knee_flex" in jl or "elbow_flex" in jl:
        return float(knee_elbow)
    if "trunk_incline" in jl:
        return float(trunk)
    return 4.0


def apply_envelope_floor(angle_cols, coach_mean, ref_lo, ref_hi, knee_elbow_floor, trunk_floor):
    min_half = np.array(
        [min_halfwidth_for_joint(c, knee_elbow_floor, trunk_floor) for c in angle_cols],
        dtype=float,
    )[None, :]  # (1, J)
    half_width = 0.5 * (ref_hi - ref_lo)
    half_width = np.maximum(half_width, min_half)
    return coach_mean - half_width, coach_mean + half_width


def robust_joint_max(over: np.ndarray, q: float = 95.0) -> np.ndarray:
    return np.nanpercentile(over, q, axis=0)


def focus_joints_from_driver(driver: str, angle_cols: List[str]) -> List[str]:
    d = (driver or "").lower()
    if "knee" in d:
        focus = ["left_knee_flex", "right_knee_flex", "trunk_incline"]
    elif "elbow" in d:
        focus = ["left_elbow_flex", "right_elbow_flex", "trunk_incline"]
    elif "trunk" in d:
        focus = ["trunk_incline", "left_knee_flex", "right_knee_flex"]
    else:
        focus = [driver, "trunk_incline"]
    focus = [j for j in focus if j in angle_cols]
    if driver in angle_cols and driver not in focus:
        focus.insert(0, driver)
    return focus


def _event_priority(e: dict) -> tuple:
    sev_rank = {"severe": 3, "moderate": 2, "mild": 1, "none": 0}
    pers_rank = 1 if e.get("persistence_label") == "persistent" else 0
    return (
        sev_rank.get(e.get("severity", "none"), 0),
        pers_rank,
        float(e.get("persistence", 0.0)),
        float(e.get("deg_outside", 0.0)),
    )


def _event_to_tip(e: dict) -> dict:
    joint = e.get("joint", "n/a")
    phase = e.get("phase", "n/a")
    direction = e.get("direction", "")
    jl = joint.lower()

    if "knee_flex" in jl or "elbow_flex" in jl:
        if direction == "above_reference":
            msg = f"Bend your {pretty_joint(joint)} more during {phase}."
        elif direction == "below_reference":
            msg = f"Bend your {pretty_joint(joint)} less during {phase}."
        else:
            msg = f"Keep your {pretty_joint(joint)} steadier during {phase}."
    elif "trunk_incline" in jl:
        if direction == "above_reference":
            msg = f"Stay more upright during {phase}."
        elif direction == "below_reference":
            msg = f"Lean slightly more forward during {phase}."
        else:
            msg = f"Keep your torso steadier during {phase}."
    else:
        if direction == "above_reference":
            msg = f"Reduce {pretty_joint(joint)} motion during {phase}."
        elif direction == "below_reference":
            msg = f"Increase {pretty_joint(joint)} motion during {phase}."
        else:
            msg = f"Keep {pretty_joint(joint)} steadier during {phase}."

    words = msg.split()
    if len(words) > 20:
        msg = " ".join(words[:20])

    return {"one_sentence_tip": msg, "target_joint": joint, "phase": phase}


def make_tip_from_events(
    events: List[dict],
    stability: float,
    confidence: float,
    min_stability: float,
    min_conf: float,
    allow_quick_mode: bool,
) -> dict:
    if confidence < min_conf:
        return {
            "one_sentence_tip": "Re-record: keep your whole body in frame with good lighting and steady camera.",
            "target_joint": "n/a",
            "phase": "n/a",
        }
    if not events:
        return {
            "one_sentence_tip": "Nice work—your motion stayed within the coach envelope for the main joints.",
            "target_joint": "n/a",
            "phase": "n/a",
        }
    if stability < min_stability and not allow_quick_mode:
        return {
            "one_sentence_tip": "Re-record: do 3–5 consistent coach reps so the reference envelope is stable.",
            "target_joint": "n/a",
            "phase": "n/a",
        }
    best = sorted(events, key=_event_priority, reverse=True)[0]
    return _event_to_tip(best)


# ----------------------------- Summary Dashboard Helpers -----------------------------
def extract_analysis_summary(analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract summary statistics from analyses list."""
    if not analyses:
        return {
            "total_reps": 0,
            "avg_confidence": 0.0,
            "severity_counts": {"severe": 0, "moderate": 0, "mild": 0, "none": 0},
            "top_issues": [],
            "max_deviation": 0.0,
            "avg_stability": 0.0,
        }

    total_reps = len(analyses)
    confidences = [a.get("confidence", 0.0) for a in analyses]
    avg_confidence = float(np.nanmean(confidences)) if confidences else 0.0

    severity_counts = {"severe": 0, "moderate": 0, "mild": 0, "none": 0}
    max_deviation = 0.0
    top_issues = []
    stabilities = []

    for analysis in analyses:
        events = analysis.get("events", [])
        stabilities.append(analysis.get("reference_stability", 0.0))

        for event in events:
            sev = event.get("severity", "none")
            if sev in severity_counts:
                severity_counts[sev] += 1

            deg = float(event.get("deg_outside", 0.0))
            if deg > max_deviation:
                max_deviation = deg

            if sev in ("severe", "moderate"):
                top_issues.append({
                    "joint": event.get("joint", "n/a"),
                    "phase": event.get("phase", "n/a"),
                    "severity": sev,
                    "deg_outside": deg,
                })

    # Sort top issues by severity and degree
    top_issues.sort(key=lambda x: (3 if x["severity"] == "severe" else 2 if x["severity"] == "moderate" else 1, x["deg_outside"]), reverse=True)
    top_issues = top_issues[:5]  # Top 5 issues

    avg_stability = float(np.nanmean(stabilities)) if stabilities else 0.0

    return {
        "total_reps": total_reps,
        "avg_confidence": avg_confidence,
        "severity_counts": severity_counts,
        "top_issues": top_issues,
        "max_deviation": max_deviation,
        "avg_stability": avg_stability,
    }


def render_executive_summary(analyses: List[Dict[str, Any]], run_id: str, created_at: str, exercise_label: str) -> None:
    """Render executive summary card with key stats."""
    summary = extract_analysis_summary(analyses)

    st.markdown("### Executive Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Run ID", run_id[:16] + "..." if len(run_id) > 16 else run_id)
        if created_at:
            st.caption(f"Created: {created_at}")

    with col2:
        st.metric("Avg Confidence", f"{summary['avg_confidence']:.2f}")
        st.metric("Reference Stability", f"{summary['avg_stability']:.2f}")

    with col3:
        st.metric("Total Reps", summary["total_reps"])
        st.metric("Max Deviation", f"{summary['max_deviation']:.1f}°")

    with col4:
        sev = summary["severity_counts"]
        st.markdown("**Deviations**")
        st.markdown(f"<span style='color: #dc2626; font-weight: 500;'>Severe: {sev['severe']}</span>", unsafe_allow_html=True)
        st.markdown(f"<span style='color: #f59e0b; font-weight: 500;'>Moderate: {sev['moderate']}</span>", unsafe_allow_html=True)
        st.markdown(f"<span style='color: #10b981; font-weight: 500;'>Mild: {sev['mild']}</span>", unsafe_allow_html=True)

    if exercise_label:
        st.caption(f"Exercise: {exercise_label}")


def render_metrics_grid(analyses: List[Dict[str, Any]], coach_reps: List, user_reps: List) -> None:
    """Render 4-column metrics grid."""
    summary = extract_analysis_summary(analyses)

    st.markdown("### Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Coach Reps", len(coach_reps) if coach_reps else 0)
        st.metric("User Reps", len(user_reps) if user_reps else 0)

    with col2:
        st.metric("Avg Confidence", f"{summary['avg_confidence']:.2%}")
        confidences = [a.get("confidence", 0.0) for a in analyses]
        if confidences:
            st.metric("Min Confidence", f"{min(confidences):.2%}")

    with col3:
        st.metric("Max Deviation", f"{summary['max_deviation']:.1f}°")
        total_events = sum(len(a.get("events", [])) for a in analyses)
        st.metric("Total Events", total_events)

    with col4:
        sev = summary["severity_counts"]
        total_sev = sev["severe"] + sev["moderate"] + sev["mild"]
        if total_sev > 0:
            persistence_pct = (total_sev / (len(analyses) * 10)) * 100  # Rough estimate
            st.metric("Persistence", f"{persistence_pct:.1f}%")
        else:
            st.metric("Persistence", "0%")
        st.metric("Issues Found", total_sev)


def render_analysis_table(analyses: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create formatted analysis table from analyses list."""
    rows = []
    for analysis in analyses:
        events = analysis.get("events", [])
        sev_counts = {"severe": 0, "moderate": 0, "mild": 0}
        for event in events:
            sev = event.get("severity", "none")
            if sev in sev_counts:
                sev_counts[sev] += 1

        # Find top issue
        top_issue = "None"
        if events:
            top_event = max(events, key=lambda e: float(e.get("deg_outside", 0.0)))
            top_issue = f"{pretty_joint(top_event.get('joint', 'n/a'))} ({top_event.get('phase', 'n/a')})"

        rows.append({
            "Rep ID": analysis.get("rep_id", 0),
            "Confidence": f"{analysis.get('confidence', 0.0):.2%}",
            "Events": len(events),
            "Severe": sev_counts["severe"],
            "Moderate": sev_counts["moderate"],
            "Mild": sev_counts["mild"],
            "Top Issue": top_issue,
        })

    return pd.DataFrame(rows)


def load_evolution_metrics(run_id: str, user_id: str, bucket: str, region: str, prefix: str, max_runs: int = 10) -> List[Dict[str, Any]]:
    """Load recent runs from S3 and extract evolution metrics."""
    if not s3_enabled():
        return []

    try:
        manifest_objs = s3_list_manifests_cached(bucket, region, prefix, user_id, max_items=max_runs)

        evolution_data = []
        for obj in manifest_objs:
            key = obj.get("key", "")
            try:
                manifest = s3_get_json_cached(bucket, region, key)
                m_run_id = manifest.get("run_id", "")

                # Skip current run
                if m_run_id == run_id:
                    continue

                created_at = manifest.get("created_at_utc", "")
                exercise_label = manifest.get("exercise_label", "")

                # Try to load analysis
                analysis_key = manifest.get("outputs", {}).get("analysis_s3_key")
                if analysis_key:
                    try:
                        analysis = s3_get_json_cached(bucket, region, analysis_key)
                        summary = extract_analysis_summary(analysis)

                        evolution_data.append({
                            "run_id": m_run_id,
                            "created_at": created_at,
                            "exercise_label": exercise_label,
                            "avg_confidence": summary["avg_confidence"],
                            "severity_counts": summary["severity_counts"],
                            "max_deviation": summary["max_deviation"],
                            "total_reps": summary["total_reps"],
                        })
                    except Exception:
                        pass  # Skip if analysis can't be loaded
            except Exception:
                pass  # Skip if manifest can't be loaded

        # Sort by created_at descending
        evolution_data.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return evolution_data[:max_runs]
    except Exception:
        return []


def render_evolution_section(run_id: str, user_id: str, bucket: str, region: str, prefix: str, current_summary: Dict[str, Any]) -> None:
    """Render evolution tracking charts and comparison."""
    evolution_data = load_evolution_metrics(run_id, user_id, bucket, region, prefix, max_runs=10)

    if not evolution_data:
        st.info("No previous runs found. Save more runs to see evolution tracking.")
        return

    st.markdown("### Evolution Tracking")
    st.caption(f"Comparing with {len(evolution_data)} previous run(s)")

    # Prepare data for charts
    run_ids = [d["run_id"][:12] + "..." for d in evolution_data]
    confidences = [d["avg_confidence"] for d in evolution_data]
    max_deviations = [d["max_deviation"] for d in evolution_data]
    severe_counts = [d["severity_counts"]["severe"] for d in evolution_data]
    moderate_counts = [d["severity_counts"]["moderate"] for d in evolution_data]
    mild_counts = [d["severity_counts"]["mild"] for d in evolution_data]

    # Add current run
    run_ids.insert(0, "Current")
    confidences.insert(0, current_summary["avg_confidence"])
    max_deviations.insert(0, current_summary["max_deviation"])
    severe_counts.insert(0, current_summary["severity_counts"]["severe"])
    moderate_counts.insert(0, current_summary["severity_counts"]["moderate"])
    mild_counts.insert(0, current_summary["severity_counts"]["mild"])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Confidence Trend")
        fig_conf = plt.figure(figsize=(8, 4))
        plt.plot(range(len(run_ids)), confidences, marker="o", linewidth=2, markersize=8)
        plt.xticks(range(len(run_ids)), run_ids, rotation=45, ha="right")
        plt.ylabel("Confidence")
        plt.title("Average Confidence Over Time")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_conf)

    with col2:
        st.markdown("#### Max Deviation Trend")
        fig_dev = plt.figure(figsize=(8, 4))
        plt.plot(range(len(run_ids)), max_deviations, marker="s", color="orange", linewidth=2, markersize=8)
        plt.xticks(range(len(run_ids)), run_ids, rotation=45, ha="right")
        plt.ylabel("Max Deviation (°)")
        plt.title("Max Deviation Over Time")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_dev)

    st.markdown("#### Severity Breakdown Over Time")
    fig_sev = plt.figure(figsize=(10, 5))
    x = range(len(run_ids))
    width = 0.25
    plt.bar([i - width for i in x], severe_counts, width, label="Severe", color="red", alpha=0.7)
    plt.bar(x, moderate_counts, width, label="Moderate", color="orange", alpha=0.7)
    plt.bar([i + width for i in x], mild_counts, width, label="Mild", color="yellow", alpha=0.7)
    plt.xticks(x, run_ids, rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title("Severity Distribution Across Runs")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    st.pyplot(fig_sev)

    # Comparison with previous run
    if len(evolution_data) > 0:
        prev = evolution_data[0]
        st.markdown("#### Comparison with Previous Run")
        comp_col1, comp_col2, comp_col3 = st.columns(3)

        with comp_col1:
            conf_diff = current_summary["avg_confidence"] - prev["avg_confidence"]
            st.metric("Confidence Change", f"{conf_diff:+.2%}", delta=f"{conf_diff:+.2%}")

        with comp_col2:
            dev_diff = current_summary["max_deviation"] - prev["max_deviation"]
            st.metric("Max Deviation Change", f"{dev_diff:+.1f}°", delta=f"{dev_diff:+.1f}°")

        with comp_col3:
            sev_diff = current_summary["severity_counts"]["severe"] - prev["severity_counts"]["severe"]
            st.metric("Severe Issues Change", f"{sev_diff:+d}", delta=f"{sev_diff:+d}")


# ----------------------------- Cache expensive steps -----------------------------
@st.cache_data(show_spinner=False)
def cached_pose(video_path: str):
    return extract_pose_frames(video_path)


@st.cache_data(show_spinner=False)
def cached_angles(frames):
    return compute_angles(frames)


# ----------------------------- S3 helpers -----------------------------
def s3_config() -> Dict[str, str]:
    bucket_name = os.getenv("ROASTCOACH_S3_BUCKET", "").strip()
    if not bucket_name:
        bucket_name = os.getenv("S3_BUCKET_NAME", "").strip()
    return {
        "bucket": bucket_name,
        "region": os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1")).strip(),
        "prefix": os.getenv("ROASTCOACH_S3_PREFIX", "roastcoach").strip().strip("/"),
        "user_id": os.getenv("ROASTCOACH_USER_ID", "anonymous").strip() or "anonymous",
    }


def s3_enabled() -> bool:
    cfg = s3_config()
    return BOTO3_AVAILABLE and bool(cfg["bucket"])


def s3_client():
    cfg = s3_config()
    if not BOTO3_AVAILABLE:
        raise RuntimeError("boto3 not available")
    return boto3.client("s3", region_name=cfg["region"])


def sanitize_video_name(name: str, original_filename: str) -> str:
    """
    Sanitize video name for filesystem and S3 compatibility.

    Args:
        name: Custom name provided by user (can be empty)
        original_filename: Original filename from upload

    Returns:
        Sanitized filename with proper extension
    """
    import re

    # Get extension from original filename
    original_path = Path(original_filename)
    ext = original_path.suffix.lower()

    # If no custom name provided, use original filename (without path)
    if not name or not name.strip():
        return original_path.name

    # Sanitize the custom name
    sanitized = name.strip()

    # Remove/replace invalid characters for filesystem and S3
    # S3 key restrictions: alphanumeric, forward slash, hyphen, underscore, period
    # Filesystem: avoid special chars that cause issues
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', sanitized)

    # Replace multiple underscores/spaces with single underscore
    sanitized = re.sub(r'[_\s]+', '_', sanitized)

    # Remove leading/trailing dots and underscores
    sanitized = sanitized.strip('._')

    # Limit length (S3 key length limit is 1024, but keep filename reasonable)
    # Reserve space for extension and path
    max_name_length = 200
    if len(sanitized) > max_name_length:
        sanitized = sanitized[:max_name_length].rstrip('._')

    # Ensure we have a name after sanitization
    if not sanitized:
        sanitized = original_path.stem

    # Add extension if not present
    if not sanitized.endswith(ext):
        sanitized += ext

    return sanitized


def s3_key(run_id: str, filename: str, kind: str) -> str:
    """
    kind examples:
      - raw/coach_video
      - raw/user_video
      - processed/analysis
      - processed/angles
      - processed/envelope
    """
    cfg = s3_config()
    return f"{cfg['prefix']}/{cfg['user_id']}/{run_id}/{kind}/{filename}"


def s3_put_file(local_path: Path, key: str, content_type: str | None = None, metadata: Dict[str, str] | None = None):
    cli = s3_client()
    extra: Dict[str, Any] = {}
    if content_type:
        extra["ContentType"] = content_type
    if metadata:
        extra["Metadata"] = {str(k).lower(): str(v) for k, v in metadata.items()}
    try:
        with local_path.open("rb") as f:
            cli.put_object(Bucket=s3_config()["bucket"], Key=key, Body=f, **extra)
    except ClientError as e:
        # Re-raise ClientError directly so it can be caught properly by upload handlers
        raise
    except Exception as e:
        raise Exception(f"Failed to upload to S3: {str(e)}")


def s3_put_bytes(data: bytes, key: str, content_type: str | None = None, metadata: Dict[str, str] | None = None):
    cli = s3_client()
    extra: Dict[str, Any] = {}
    if content_type:
        extra["ContentType"] = content_type
    if metadata:
        extra["Metadata"] = {str(k).lower(): str(v) for k, v in metadata.items()}
    try:
        cli.put_object(Bucket=s3_config()["bucket"], Key=key, Body=data, **extra)
    except ClientError as e:
        # Re-raise ClientError directly so it can be caught properly by upload handlers
        raise
    except Exception as e:
        raise Exception(f"Failed to upload to S3: {str(e)}")


@st.cache_data(ttl=30, show_spinner=False)
def s3_list_manifests_cached(bucket: str, region: str, prefix: str, user_id: str, max_items: int = 50) -> List[Dict[str, Any]]:
    """
    Lists manifest.json under: {prefix}/{user_id}/.../manifest.json
    Returns newest-first list of objects: {key, last_modified, size}
    """
    if not BOTO3_AVAILABLE:
        return []
    s3 = boto3.client("s3", region_name=region)
    base = f"{prefix}/{user_id}/"
    out: List[Dict[str, Any]] = []

    token = None
    while True:
        kwargs: Dict[str, Any] = {"Bucket": bucket, "Prefix": base, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)

        for obj in resp.get("Contents", []):
            key = obj.get("Key", "")
            if key.endswith("manifest.json"):
                out.append(
                    {
                        "key": key,
                        "last_modified": obj.get("LastModified"),
                        "size": obj.get("Size", 0),
                    }
                )

        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break

    out.sort(key=lambda x: x["last_modified"] or datetime(1970, 1, 1, tzinfo=timezone.utc), reverse=True)
    return out[:max_items]


@st.cache_data(ttl=30, show_spinner=False)
def s3_get_json_cached(bucket: str, region: str, key: str) -> Dict[str, Any]:
    if not BOTO3_AVAILABLE:
        raise RuntimeError("boto3 not available")
    s3 = boto3.client("s3", region_name=region)
    resp = s3.get_object(Bucket=bucket, Key=key)
    data = resp["Body"].read().decode("utf-8")
    return json.loads(data)


def guess_video_content_type(p: Path) -> str:
    suf = p.suffix.lower()
    if suf == ".mov":
        return "video/quicktime"
    if suf in (".m4v", ".mp4"):
        return "video/mp4"
    return "application/octet-stream"


def reset_run_state_on_new_videos():
    st.session_state["stage1_done"] = False
    st.session_state["stage2_done"] = False
    st.session_state["df_coach"] = None
    st.session_state["df_user"] = None
    st.session_state["shared_cols"] = None
    st.session_state["coach_driver"] = None
    st.session_state["coach_reps"] = None
    st.session_state["user_reps"] = None
    st.session_state["coach_mean"] = None
    st.session_state["user_mean"] = None
    st.session_state["latest_analysis"] = None
    st.session_state["latest_tip"] = None
    st.session_state["artifacts_local"] = {}
    st.session_state["saved_to_s3"] = False
    st.session_state["s3_last_error"] = None
    st.session_state["run_id"] = None
    st.session_state["run_created_at"] = None


def parse_manifest_row(manifest: Dict[str, Any], manifest_key: str, last_modified=None) -> Dict[str, Any]:
    created = manifest.get("created_at_utc") or ""
    label = manifest.get("exercise_label") or ""
    run_id = manifest.get("run_id") or ""
    style = (manifest.get("settings", {}).get("gemini", {}) or {}).get("style")
    tip = (manifest.get("outputs", {}).get("tip", {}) or {}).get("one_sentence_tip", "")
    coach_driver = manifest.get("outputs", {}).get("coach_driver", "")
    return {
        "created_at_utc": created,
        "exercise_label": label,
        "coach_driver": coach_driver,
        "llm_style": style if style else "—",
        "tip_preview": (tip[:80] + "…") if isinstance(tip, str) and len(tip) > 80 else (tip or ""),
        "run_id": run_id,
        "manifest_s3_key": manifest_key,
        "last_modified": str(last_modified) if last_modified else "",
    }


# ----------------------------- Settings (Always Enabled) -----------------------------
# Initialize default settings in session state if not present
if "personality_style" not in st.session_state:
    st.session_state["personality_style"] = "Supportive coach"
if "personality_intensity" not in st.session_state:
    st.session_state["personality_intensity"] = 1

# Gemini is always enabled (if available)
use_gemini = LLM_AVAILABLE  # Always use Gemini if available

# S3 is always enabled (if configured)
enable_s3 = s3_enabled()  # Always use S3 if configured


# ----------------------------- Tabs -----------------------------
tab_run, tab_history = st.tabs(["Run", "History"])


# ==============================
#             RUN
# ==============================
with tab_run:
    # ----------------------------- Upload UI -----------------------------
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1) Coach Reference Video")
        coach_file = st.file_uploader("Upload coach demo (.mp4/.mov/.m4v)", type=["mp4", "mov", "m4v"], key="coach")
    with col2:
        st.subheader("2) User Attempt Video")
        # Toggle between upload and record for user video
        user_mode = st.radio(
            "Choose input method:",
            ["Upload Video", "Record Video"],
            horizontal=True,
            key="user_video_mode"
        )

        if user_mode == "Upload Video":
            user_file = st.file_uploader("Upload user attempt (.mp4/.mov/.m4v)", type=["mp4", "mov", "m4v"], key="user")
        else:
            # Mobile video recording - enhanced file uploader with capture attribute
            st.markdown("**Record Video (Mobile)**")
            st.caption("On mobile devices, this will open your camera to record directly")
            user_file = st.file_uploader(
                "Record or upload video (.mp4/.mov/.m4v)",
                type=["mp4", "mov", "m4v"],
                key="user_record",
                help="Tap to record a video using your device camera (mobile) or upload a file"
            )

            # Add JavaScript to enhance the file uploader with capture attribute on mobile
            st.markdown("""
            <script>
            (function() {
                function enhanceFileUploader() {
                    // Find the file uploader input for user_record
                    const fileInputs = document.querySelectorAll('input[type="file"]');
                    fileInputs.forEach(input => {
                        // Check if this is a video input
                        if (input.accept && (input.accept.includes('video') || input.accept.includes('mp4') || input.accept.includes('mov'))) {
                            // Check if mobile device
                            const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
                            if (isMobile && !input.hasAttribute('capture')) {
                                input.setAttribute('capture', 'environment');
                                input.setAttribute('accept', 'video/*');
                            }
                        }
                    });
                }

                // Run on page load
                if (document.readyState === 'loading') {
                    document.addEventListener('DOMContentLoaded', enhanceFileUploader);
                } else {
                    enhanceFileUploader();
                }

                // Also run after Streamlit reruns (observe mutations)
                const observer = new MutationObserver(enhanceFileUploader);
                observer.observe(document.body, { childList: true, subtree: true });
            })();
            </script>
            """, unsafe_allow_html=True)

        # Video naming (after upload)
        if user_file:
            st.markdown("**Name Your Video (Optional)**")
            st.caption("Give your video a custom name for easier identification in AWS storage")

            # Get original filename
            original_filename = user_file.name

            # Initialize video name if not set or if file changed
            if st.session_state.get("user_video_name") is None or st.session_state.get("user_file_sig") != file_signature(user_file)[1]:
                # Default to original filename without extension
                default_name = Path(original_filename).stem
                st.session_state["user_video_name"] = default_name

            # Video name input
            video_name_input = st.text_input(
                "Video Name:",
                value=st.session_state.get("user_video_name", Path(original_filename).stem),
                key="user_video_name_input",
                help="Custom name for your video. Will be sanitized for filesystem compatibility.",
                placeholder="e.g., My_Squat_Attempt_1"
            )

            # Update session state with sanitized name
            if video_name_input:
                sanitized_name = sanitize_video_name(video_name_input, original_filename)
                st.session_state["user_video_name"] = sanitized_name

                # Show preview of sanitized name if it changed
                if sanitized_name != video_name_input:
                    st.caption(f"Sanitized name: `{sanitized_name}`")
            else:
                # Use original filename if empty
                st.session_state["user_video_name"] = original_filename

    # Reset stages only when videos change
    if coach_file and user_file:
        sig_c = file_signature(coach_file)
        sig_u = file_signature(user_file)
        if (st.session_state["coach_file_sig"] != sig_c) or (st.session_state["user_file_sig"] != sig_u):
            st.session_state["coach_file_sig"] = sig_c
            st.session_state["user_file_sig"] = sig_u
            reset_run_state_on_new_videos()

    # ----------------------------- Previews -----------------------------
    st.divider()
    st.subheader("Previews")
    left, right = st.columns(2)

    with left:
        st.markdown("**Coach Preview**")
        if coach_file:
            video_small(coach_file, width_px=360)
            with st.expander("View coach video large"):
                st.video(coach_file)
        else:
            st.info("Upload a coach reference video to preview it here.")

    with right:
        st.markdown("**User Preview**")
        if user_file:
            video_small(user_file, width_px=360)
            with st.expander("View user video large"):
                st.video(user_file)
        else:
            st.info("Upload a user attempt video to preview it here.")

    # ----------------------------- Confirm & Process -----------------------------
    st.divider()

    confirm_btn = st.button(
        "Confirm & Process",
        type="primary",
        disabled=not (coach_file and user_file),
        width='stretch',
    )

    # Initialize run when confirm is clicked
    if confirm_btn:
        run_id, iso_ts = utc_run_id()
        st.session_state["run_id"] = run_id
        st.session_state["run_created_at"] = iso_ts
        st.session_state["saved_to_s3"] = False
        st.session_state["s3_last_error"] = None
        st.session_state["artifacts_local"] = {}
        st.session_state["processing_complete"] = False
        st.session_state["summary_generated"] = False
        st.session_state["coach_video_s3_key"] = None
        st.session_state["user_video_s3_key"] = None

        # Save videos locally
        coach_path = save_upload(coach_file, DATA_RAW, prefix="coach", run_id=run_id)
        user_path = save_upload(user_file, DATA_RAW, prefix="user", run_id=run_id)

        st.session_state["coach_path"] = coach_path
        st.session_state["user_path"] = user_path

        # Upload to S3 immediately if enabled
        if enable_s3:
            try:
                cfg = s3_config()
                coach_key = s3_key(run_id, coach_path.name, "raw/coach_video")

                # Use custom video name if provided, otherwise use original filename
                user_video_filename = user_path.name
                if st.session_state.get("user_video_name"):
                    # Ensure the custom name has the correct extension
                    custom_name = st.session_state["user_video_name"]
                    if not custom_name.endswith(Path(user_path.name).suffix):
                        custom_name = sanitize_video_name(custom_name, user_path.name)
                    user_video_filename = custom_name

                user_key = s3_key(run_id, user_video_filename, "raw/user_video")

                s3_put_file(
                    coach_path,
                    coach_key,
                    content_type=guess_video_content_type(coach_path),
                    metadata={"run_id": run_id, "role": "coach", "created_at": iso_ts},
                )
                # Prepare metadata with custom video name
                user_metadata = {
                    "run_id": run_id,
                    "role": "user",
                    "created_at": iso_ts,
                    "original_filename": user_path.name,
                }
                if st.session_state.get("user_video_name") and st.session_state["user_video_name"] != user_path.name:
                    user_metadata["video_name"] = st.session_state["user_video_name"]
                    user_metadata["custom_name"] = "true"
                else:
                    user_metadata["custom_name"] = "false"

                s3_put_file(
                    user_path,
                    user_key,
                    content_type=guess_video_content_type(user_path),
                    metadata=user_metadata,
                )

                st.session_state["coach_video_s3_key"] = coach_key
                st.session_state["user_video_s3_key"] = user_key
                st.success("Videos uploaded to S3")
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "Unknown")
                error_message = e.response.get("Error", {}).get("Message", str(e))

                st.error(f"❌ **S3 Upload Error**")
                if error_code == "AccessDenied":
                    st.error(f"Access Denied: Your AWS credentials don't have permission to upload objects to bucket '{cfg['bucket']}'. Please ensure your IAM user has the 's3:PutObject' permission for this bucket.")

                    st.warning("⚠️ **Video processing will continue locally, but S3 upload failed.**")
                    st.info("💡 **How to fix this in AWS Console:**")

                    with st.expander("📋 **Step-by-Step Instructions**", expanded=True):
                        st.markdown("""
                        **Step 1:** Go to [AWS IAM Console](https://console.aws.amazon.com/iam/) and sign in.

                        **Step 2:** Click on "Users" in the left sidebar, then find and click on your IAM user (`daniel-aws`).

                        **Step 3:** Click on the "Permissions" tab, then click "Add permissions" → "Create inline policy".

                        **Step 4:** Click on the "JSON" tab and paste this policy:
                        """)

                        bucket_name = cfg["bucket"]
                        policy_json = f'''{{
    "Version": "2012-10-17",
    "Statement": [
        {{
            "Effect": "Allow",
            "Action": [
                "s3:ListBucket",
                "s3:GetObject",
                "s3:PutObject"
            ],
            "Resource": [
                "arn:aws:s3:::{bucket_name}",
                "arn:aws:s3:::{bucket_name}/*"
            ]
        }}
    ]
}}'''

                        st.code(policy_json, language="json")

                        st.markdown("""
                        **Step 5:** Click "Next", give the policy a name (e.g., `RoastCoachS3Access`), then click "Create policy".

                        **Step 6:** Wait a few seconds for the policy to propagate, then try uploading again.
                        """)
                else:
                    st.error(f"AWS Error ({error_code}): {error_message}")
                    st.info("💡 Check your AWS credentials and bucket configuration in the `.env` file.")
            except Exception as e:
                error_str = str(e)
                if "AWS S3 Error" in error_str and "AccessDenied" in error_str:
                    # Show full instructions for AccessDenied errors
                    st.error(f"❌ **S3 Upload Error**")
                    st.error(f"Access Denied: Your AWS credentials don't have permission to upload objects to bucket '{cfg['bucket']}'. Please ensure your IAM user has the 's3:PutObject' permission for this bucket.")

                    st.warning("⚠️ **Video processing will continue locally, but S3 upload failed.**")
                    st.info("💡 **How to fix this in AWS Console:**")

                    with st.expander("📋 **Step-by-Step Instructions**", expanded=True):
                        st.markdown("""
                        **Step 1:** Go to [AWS IAM Console](https://console.aws.amazon.com/iam/) and sign in.

                        **Step 2:** Click on "Users" in the left sidebar, then find and click on your IAM user (`daniel-aws`).

                        **Step 3:** Click on the "Permissions" tab, then click "Add permissions" → "Create inline policy".

                        **Step 4:** Click on the "JSON" tab and paste this policy:
                        """)

                        bucket_name = cfg["bucket"]
                        policy_json = f'''{{
    "Version": "2012-10-17",
    "Statement": [
        {{
            "Effect": "Allow",
            "Action": [
                "s3:ListBucket",
                "s3:GetObject",
                "s3:PutObject"
            ],
            "Resource": [
                "arn:aws:s3:::{bucket_name}",
                "arn:aws:s3:::{bucket_name}/*"
            ]
        }}
    ]
}}'''

                        st.code(policy_json, language="json")

                        st.markdown("""
                        **Step 5:** Click "Next", give the policy a name (e.g., `RoastCoachS3Access`), then click "Create policy".

                        **Step 6:** Wait a few seconds for the policy to propagate, then try uploading again.
                        """)
                elif "AWS S3 Error" in error_str or "AccessDenied" in error_str:
                    st.error(f"❌ **S3 Upload Error**")
                    st.error(error_str)
                    st.info("💡 Check your AWS IAM permissions. You need `s3:PutObject` permission for the bucket.")
                else:
                    st.warning(f"Video upload to S3 failed: {error_str}")
                    st.info("💡 Videos are still saved locally and processing will continue.")

        # Reset stage flags to trigger processing
        st.session_state["stage1_done"] = False
        st.session_state["stage2_done"] = False

    # Auto-process if videos are uploaded and stages not done
    should_process = (
        coach_file and user_file and
        st.session_state.get("coach_path") and
        st.session_state.get("user_path") and
        not st.session_state.get("processing_complete", False)
    )

    if should_process and not st.session_state.get("stage1_done", False):
        # ----------------------------- Stage 1: Auto-trigger -----------------------------
        st.info("Processing Stage 1: Pose extraction + angle computation...")

        coach_path = st.session_state["coach_path"]
        user_path = st.session_state["user_path"]

        with st.spinner("Extracting pose (coach)..."):
            coach_frames = cached_pose(str(coach_path))
        with st.spinner("Extracting pose (user)..."):
            user_frames = cached_pose(str(user_path))
        with st.spinner("Computing angles..."):
            coach_angles = cached_angles(coach_frames)
            user_angles = cached_angles(user_frames)

        df_coach = angles_to_df(coach_angles)
        df_user = angles_to_df(user_angles)

        run_id = st.session_state["run_id"] or "run_unknown"
        coach_csv_run = DATA_PROCESSED / f"{run_id}_angles_coach.csv"
        user_csv_run = DATA_PROCESSED / f"{run_id}_angles_user.csv"
        coach_csv_latest = DATA_PROCESSED / "angles_coach.csv"
        user_csv_latest = DATA_PROCESSED / "angles_user.csv"

        df_coach.to_csv(coach_csv_run, index=False)
        df_user.to_csv(user_csv_run, index=False)
        df_coach.to_csv(coach_csv_latest, index=False)
        df_user.to_csv(user_csv_latest, index=False)

        st.session_state["df_coach"] = df_coach
        st.session_state["df_user"] = df_user
        st.session_state["stage1_done"] = True

        st.session_state["artifacts_local"].update(
            {
                "coach_video": str(coach_path),
                "user_video": str(user_path),
                "angles_coach_csv": str(coach_csv_run),
                "angles_user_csv": str(user_csv_run),
            }
        )

        st.success("Stage 1 complete: Pose + angles computed")

    # Continue to Stage 2 if Stage 1 is done
    if st.session_state.get("stage1_done", False) and not st.session_state.get("stage2_done", False):
        df_coach = st.session_state["df_coach"]
        df_user = st.session_state["df_user"]

        angle_cols_coach = [c for c in df_coach.columns if c not in ("t", "conf")]
        angle_cols_user = [c for c in df_user.columns if c not in ("t", "conf")]
        shared_cols = [c for c in angle_cols_coach if c in angle_cols_user]
        st.session_state["shared_cols"] = shared_cols

        if not shared_cols:
            st.error("No shared angle columns found. Pose landmarks may be failing.")
            st.stop()

        # ----------------------------- Stage 2: Auto-trigger -----------------------------
        if should_process and not st.session_state.get("stage2_done", False):
            st.info("Processing Stage 2: Rep segmentation + normalization...")

        # Use default settings for auto-processing
        min_q = st.session_state.get("min_q", 0.35)
        drop_first = st.session_state.get("drop_first", True)
        N = int(st.session_state.get("N", 100))

        coach_driver = choose_driver_column(df_coach)
        user_driver = coach_driver

        coach_reps_raw, coach_driver_smooth = find_reps_from_driver(df_coach, coach_driver)
        user_reps_raw, user_driver_smooth = find_reps_from_driver(df_user, user_driver)

        coach_reps = filter_reps(coach_reps_raw, min_quality=min_q, drop_first=drop_first)
        user_reps = filter_reps(user_reps_raw, min_quality=min_q, drop_first=drop_first)

        if not (coach_reps and user_reps):
            st.error("Not enough reps kept after filtering. Try adjusting video quality or settings.")
            st.stop()

        angle_cols = shared_cols
        tt = np.linspace(0, 1, int(N))

        coach_mean, coach_std, coach_stack = mean_std_normalized_rep(df_coach, coach_reps, angle_cols, N=int(N))
        user_mean, user_std, user_stack = mean_std_normalized_rep(df_user, user_reps, angle_cols, N=int(N))

        st.session_state.update(
            {
                "coach_driver": coach_driver,
                "coach_reps": coach_reps,
                "user_reps": user_reps,
                "coach_driver_smooth": coach_driver_smooth,
                "user_driver_smooth": user_driver_smooth,
                "angle_cols": angle_cols,
                "N": int(N),
                "tt": tt,
                "coach_mean": coach_mean,
                "coach_std": coach_std,
                "coach_stack": coach_stack,
                "user_mean": user_mean,
                "user_std": user_std,
                "user_stack": user_stack,
                "stage2_done": True,
            }
        )

        st.success("Stage 2 complete: Reference envelope built")

    # Continue to Stage 3 if Stage 2 is done
    if st.session_state.get("stage2_done", False) and not st.session_state.get("processing_complete", False):
        # ----------------------------- Stage 3: Auto-trigger -----------------------------
        st.info("Processing Stage 3: Envelope + deviation detection + coaching tip...")

        coach_driver = st.session_state["coach_driver"]
        coach_reps = st.session_state["coach_reps"]
        user_reps = st.session_state["user_reps"]
        angle_cols = st.session_state["angle_cols"]
        tt = st.session_state["tt"]
        coach_mean = st.session_state["coach_mean"]
        coach_std = st.session_state["coach_std"]
        coach_stack = st.session_state["coach_stack"]
        user_mean = st.session_state["user_mean"]
        N = int(st.session_state["N"])
        df_user = st.session_state["df_user"]

        # Use default settings
        env_method = st.session_state.get("env_method", "std*k")
        k = float(st.session_state.get("k_std", 1.5))
        use_floor = st.session_state.get("use_floor", True)
        knee_floor = float(st.session_state.get("knee_floor", 5.0))
        trunk_floor = float(st.session_state.get("trunk_floor", 3.0))
        mild = float(st.session_state.get("mild", 5.0))
        moderate = float(st.session_state.get("moderate", 12.0))
        severe = float(st.session_state.get("severe", 20.0))
        persistent_thresh = float(st.session_state.get("persist_pct", 20.0)) / 100.0
        bottom_window = float(st.session_state.get("bottom_window", 8.0)) / 100.0
        use_robust = st.session_state.get("use_robust", True)
        robust_q = float(st.session_state.get("robust_q", 95.0))
        min_stability = float(st.session_state.get("min_stability", 0.6))
        min_conf_required = float(st.session_state.get("min_conf_required", 0.6))
        allow_quick = st.session_state.get("allow_quick", True)

        # Build envelope
        if env_method == "std*k":
            tol = coach_std * k
            ref_lo = coach_mean - tol
            ref_hi = coach_mean + tol
            stability = float(1.0 / (1.0 + np.nanmean(coach_std)))
        else:
            ref_lo = np.nanpercentile(coach_stack, 10, axis=0)
            ref_hi = np.nanpercentile(coach_stack, 90, axis=0)
            band_width = float(np.nanmean(ref_hi - ref_lo))
            stability = float(1.0 / (1.0 + band_width))

        if use_floor:
            ref_lo, ref_hi = apply_envelope_floor(angle_cols, coach_mean, ref_lo, ref_hi, knee_floor, trunk_floor)

        # Deviation detection
        driver_idx = angle_cols.index(coach_driver) if coach_driver in angle_cols else 0

        analyses: List[Dict[str, Any]] = []
        for ridx, r in enumerate(user_reps):
            user_mat = resample_rep_angles(df_user, r, angle_cols, N=int(N))
            idx_bottom = int(np.nanargmin(user_mat[:, driver_idx]))

            below = ref_lo - user_mat
            above = user_mat - ref_hi
            over = np.maximum(0.0, np.maximum(below, above))

            if use_robust:
                max_over = robust_joint_max(over, float(robust_q))
            else:
                max_over = np.nanmax(over, axis=0)

            persist = np.nanmean(over > 0, axis=0)

            focus_cols = focus_joints_from_driver(coach_driver, angle_cols)
            focus_idx = [angle_cols.index(c) for c in focus_cols]
            ranked_focus = sorted(focus_idx, key=lambda jj: (float(max_over[jj]), float(persist[jj])), reverse=True)
            top_idx = ranked_focus[:3]

            events = []
            for jj in top_idx:
                deg = float(max_over[jj]) if np.isfinite(max_over[jj]) else 0.0
                sev_label = severity_from_deg(deg, mild=mild, moderate=moderate, severe=severe)
                if sev_label == "none":
                    continue

                idx_peak = int(np.nanargmax(over[:, jj]))
                phase = phase_from_t(idx_peak, idx_bottom, int(N), bottom_window_frac=float(bottom_window))

                diff_val = float(user_mat[idx_peak, jj] - coach_mean[idx_peak, jj])
                direction = "above_reference" if diff_val > 0 else "below_reference"

                events.append(
                    {
                        "joint": angle_cols[jj],
                        "phase": phase,
                        "direction": direction,
                        "deg_outside": deg,
                        "persistence": float(persist[jj]),
                        "severity": sev_label,
                        "persistence_label": "persistent" if float(persist[jj]) >= persistent_thresh else "brief",
                    }
                )

            pose_conf = float(np.nanmean(df_user["conf"])) if "conf" in df_user.columns else 1.0
            rep_conf = float(np.clip(r.quality, 0.0, 1.0))
            confidence = float(0.5 * pose_conf + 0.5 * rep_conf)

            analyses.append(
                {
                    "rep_id": ridx,
                    "confidence": confidence,
                    "reference_stability": stability,
                    "events": events,
                }
            )

        # Generate coaching tip
        chosen = analyses[0] if analyses else {}
        base_tip = make_tip_from_events(
            chosen.get("events", []),
            stability=float(chosen.get("reference_stability", stability)),
            confidence=float(chosen.get("confidence", 0.5)),
            min_stability=float(min_stability),
            min_conf=float(min_conf_required),
            allow_quick_mode=bool(allow_quick),
        )

        final_tip = base_tip
        if LLM_AVAILABLE:
            try:
                final_tip = generate_personality_tip(
                    analysis=chosen,
                    base_tip=base_tip,
                    style="MEGA ROAST",
                    intensity=2,
                )
                if not isinstance(final_tip, dict) or "one_sentence_tip" not in final_tip:
                    final_tip = base_tip
            except Exception:
                final_tip = base_tip

        # Save analysis and envelope
        run_id = st.session_state["run_id"] or "run_unknown"
        analysis_path_run = DATA_PROCESSED / f"{run_id}_analysis.json"
        analysis_path_latest = DATA_PROCESSED / "analysis.json"
        analysis_path_run.write_text(json.dumps(analyses, indent=2))
        analysis_path_latest.write_text(json.dumps(analyses, indent=2))

        env_path_run = DATA_PROCESSED / f"{run_id}_reference_envelope_coach.npz"
        env_path_latest = DATA_PROCESSED / "reference_envelope_coach.npz"
        np.savez(
            env_path_run,
            angle_cols=np.array(angle_cols, dtype=object),
            coach_mean=coach_mean,
            ref_lo=ref_lo,
            ref_hi=ref_hi,
            stability=np.array([stability]),
            N=np.array([N]),
        )
        np.savez(
            env_path_latest,
            angle_cols=np.array(angle_cols, dtype=object),
            coach_mean=coach_mean,
            ref_lo=ref_lo,
            ref_hi=ref_hi,
            stability=np.array([stability]),
            N=np.array([N]),
        )

        # Update session state
        st.session_state["latest_analysis"] = analyses
        st.session_state["latest_tip"] = final_tip
        st.session_state["artifacts_local"]["analysis_json"] = str(analysis_path_run)
        st.session_state["artifacts_local"]["envelope_npz"] = str(env_path_run)
        st.session_state["ref_lo"] = ref_lo
        st.session_state["ref_hi"] = ref_hi
        st.session_state["processing_complete"] = True

        st.success("Stage 3 complete: Analysis + coaching tip generated")

    # Show summary dashboard if processing is complete
    if st.session_state.get("processing_complete", False) and not st.session_state.get("summary_generated", False):
        st.session_state["summary_generated"] = True
        st.rerun()  # Trigger rerun to show dashboard

    if st.session_state.get("processing_complete", False):
        # Show summary dashboard
        st.divider()
        st.markdown("# Summary Dashboard")

        analyses = st.session_state.get("latest_analysis", [])
        run_id = st.session_state.get("run_id", "unknown")
        run_created_at = st.session_state.get("run_created_at", "")
        exercise_label = st.session_state.get("exercise_label", "")
        final_tip = st.session_state.get("latest_tip", {})

        # Executive Summary
        render_executive_summary(analyses, run_id, run_created_at, exercise_label)

        st.divider()

        # Key Metrics Grid
        coach_reps = st.session_state.get("coach_reps", [])
        user_reps = st.session_state.get("user_reps", [])
        render_metrics_grid(analyses, coach_reps, user_reps)

        st.divider()

        # Gemini Feedback (prominent)
        st.markdown("### Coaching Feedback")
        if final_tip.get("one_sentence_tip"):
            tip_style = st.session_state.get("personality_style", "Supportive coach") if LLM_AVAILABLE else "Supportive coach"

            # Color code based on style
            if tip_style == "MEGA ROAST":
                st.error(final_tip['one_sentence_tip'])
            elif tip_style == "Slight roast":
                st.warning(final_tip['one_sentence_tip'])
            else:
                st.success(final_tip['one_sentence_tip'])

            with st.expander("View tip details"):
                st.json(final_tip)
        else:
            st.info("No coaching tip available.")

        st.divider()

        # Graphs and Analysis Table
        col_graphs, col_table = st.columns([2, 1])

        with col_graphs:
            st.markdown("### Visualizations")

            df_coach = st.session_state["df_coach"]
            df_user = st.session_state["df_user"]
            angle_cols = st.session_state.get("shared_cols", [])
            default_sanity = "left_knee_flex" if "left_knee_flex" in angle_cols else (angle_cols[0] if angle_cols else None)

            if default_sanity:
                # Sanity Plot
                st.markdown("#### Angle Over Time")
                fig = plt.figure(figsize=(10, 4))
                plt.plot(df_coach["t"], df_coach[default_sanity], label="coach", linewidth=2)
                plt.plot(df_user["t"], df_user[default_sanity], label="user", linewidth=2)
                plt.xlabel("time (s)")
                plt.ylabel("degrees")
                plt.title(f"{default_sanity} over time")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig, width='stretch')

            # Normalized Rep Plot
            if st.session_state.get("stage2_done", False):
                coach_mean = st.session_state["coach_mean"]
                user_mean = st.session_state["user_mean"]
                user_std = st.session_state["user_std"]
                tt = st.session_state["tt"]
                angle_cols = st.session_state["angle_cols"]
                coach_driver = st.session_state.get("coach_driver", "")

                if coach_driver in angle_cols:
                    j = angle_cols.index(coach_driver)
                    st.markdown("#### Normalized Rep Comparison")
                    fig3 = plt.figure(figsize=(10, 4))
                    plt.plot(tt, coach_mean[:, j], label="coach mean", linewidth=2)
                    plt.plot(tt, user_mean[:, j], label="user mean", linewidth=2)
                    plt.fill_between(tt, user_mean[:, j] - user_std[:, j], user_mean[:, j] + user_std[:, j], alpha=0.2)
                    plt.xlabel("normalized time (0→1)")
                    plt.ylabel("degrees")
                    plt.title(f"Mean normalized rep (±1 std) — {coach_driver}")
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig3, width='stretch')

                # Envelope Plot
                ref_lo = st.session_state.get("ref_lo")
                ref_hi = st.session_state.get("ref_hi")
                if ref_lo is not None and ref_hi is not None and coach_driver in angle_cols:
                    j = angle_cols.index(coach_driver)
                    st.markdown("#### Coach Envelope vs User")
                    fig_env = plt.figure(figsize=(10, 4))
                    plt.plot(tt, coach_mean[:, j], label="coach mean", linewidth=2)
                    plt.fill_between(tt, ref_lo[:, j], ref_hi[:, j], alpha=0.18, label="coach envelope")
                    plt.plot(tt, user_mean[:, j], label="user mean", linewidth=2)
                    plt.xlabel("normalized time (0→1)")
                    plt.ylabel("degrees")
                    plt.title(f"Coach envelope vs user mean — {coach_driver}")
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig_env, width='stretch')

        with col_table:
            st.markdown("### Analysis Summary")
            df_analysis = render_analysis_table(analyses)
            st.dataframe(df_analysis, width='stretch', hide_index=True)

            # Rep selector for details
            if len(analyses) > 0:
                rep_pick = st.selectbox("View rep details", options=list(range(len(analyses))), index=0, key="summary_rep_pick")
                chosen_rep = analyses[rep_pick]
                with st.expander("Rep events"):
                    st.json(chosen_rep.get("events", []))

        st.divider()

        # Evolution Tracking
        if enable_s3:
            cfg = s3_config()
            current_summary = extract_analysis_summary(analyses)
            render_evolution_section(run_id, cfg["user_id"], cfg["bucket"], cfg["region"], cfg["prefix"], current_summary)

        st.divider()

        # JSON Summaries (expandable)
        st.markdown("### Detailed Data")

        with st.expander("Analysis JSON (first 3 reps)"):
            st.json(analyses[:3] if len(analyses) >= 3 else analyses)

        with st.expander("Manifest Summary"):
            manifest_summary = {
                "run_id": run_id,
                "created_at": run_created_at,
                "exercise_label": exercise_label,
                "coach_driver": st.session_state.get("coach_driver"),
                "coach_reps": len(coach_reps),
                "user_reps": len(user_reps),
                "settings": {
                    "min_q": st.session_state.get("min_q", 0.35),
                    "N": st.session_state.get("N", 100),
                    "env_method": st.session_state.get("env_method", "std*k"),
                }
            }
            st.json(manifest_summary)

        # Auto-save to S3 if enabled
        if enable_s3 and not st.session_state.get("saved_to_s3", False):
            st.divider()
            st.markdown("### Save to S3")

            exercise_label_input = st.text_input(
                "Exercise label (optional)",
                value=st.session_state.get("exercise_label", ""),
                key="summary_exercise_label",
                help="Example: squat, pushup, lunge, deadlift. Stored in manifest.json.",
            )
            st.session_state["exercise_label"] = exercise_label_input

            if st.button("Save this run to S3", type="primary", key="auto_save_s3"):
                st.session_state["s3_last_error"] = None

                try:
                    cfg = s3_config()
                    coach_path = Path(st.session_state["coach_path"])
                    user_path = Path(st.session_state["user_path"])

                    # Videos may already be uploaded, use stored keys or upload
                    coach_key = st.session_state.get("coach_video_s3_key") or s3_key(run_id, coach_path.name, "raw/coach_video")

                    # Use custom video name if provided for user video
                    user_video_filename = user_path.name
                    if st.session_state.get("user_video_name"):
                        custom_name = st.session_state["user_video_name"]
                        if not custom_name.endswith(Path(user_path.name).suffix):
                            custom_name = sanitize_video_name(custom_name, user_path.name)
                        user_video_filename = custom_name

                    user_key = st.session_state.get("user_video_s3_key") or s3_key(run_id, user_video_filename, "raw/user_video")

                    # Upload videos if not already uploaded
                    if not st.session_state.get("coach_video_s3_key"):
                        s3_put_file(
                            coach_path,
                            coach_key,
                            content_type=guess_video_content_type(coach_path),
                            metadata={"run_id": run_id, "role": "coach", "created_at": run_created_at},
                        )
                        st.session_state["coach_video_s3_key"] = coach_key

                    if not st.session_state.get("user_video_s3_key"):
                        # Prepare metadata with custom video name
                        user_metadata = {
                            "run_id": run_id,
                            "role": "user",
                            "created_at": run_created_at,
                            "original_filename": user_path.name,
                        }
                        if st.session_state.get("user_video_name") and st.session_state["user_video_name"] != user_path.name:
                            user_metadata["video_name"] = st.session_state["user_video_name"]
                            user_metadata["custom_name"] = "true"
                        else:
                            user_metadata["custom_name"] = "false"

                        s3_put_file(
                            user_path,
                            user_key,
                            content_type=guess_video_content_type(user_path),
                            metadata=user_metadata,
                        )
                        st.session_state["user_video_s3_key"] = user_key

                    # Upload artifacts
                    artifacts = st.session_state.get("artifacts_local", {})

                    if "angles_coach_csv" in artifacts:
                        p = Path(artifacts["angles_coach_csv"])
                        s3_put_file(p, s3_key(run_id, p.name, "processed/angles"), content_type="text/csv")
                    if "angles_user_csv" in artifacts:
                        p = Path(artifacts["angles_user_csv"])
                        s3_put_file(p, s3_key(run_id, p.name, "processed/angles"), content_type="text/csv")

                    if "envelope_npz" in artifacts:
                        p = Path(artifacts["envelope_npz"])
                        s3_put_file(p, s3_key(run_id, p.name, "processed/envelope"), content_type="application/octet-stream")

                    if "analysis_json" in artifacts:
                        p = Path(artifacts["analysis_json"])
                        s3_put_file(p, s3_key(run_id, p.name, "processed/analysis"), content_type="application/json")

                    # Create manifest
                    manifest = {
                        "run_id": run_id,
                        "created_at_utc": run_created_at,
                        "user_id": cfg["user_id"],
                        "exercise_label": exercise_label_input.strip() or None,
                        "inputs": {
                            "coach_video_s3_key": coach_key,
                            "user_video_s3_key": user_key,
                            "coach_video_local": str(coach_path),
                            "user_video_local": str(user_path),
                            "user_video_name": st.session_state.get("user_video_name") or user_path.name,
                            "user_video_original_filename": user_path.name,
                        },
                        "settings": {
                            "min_q": float(st.session_state.get("min_q", 0.35)),
                            "drop_first": bool(st.session_state.get("drop_first", True)),
                            "N": int(st.session_state.get("N", 100)),
                            "env_method": str(st.session_state.get("env_method", "std*k")),
                            "k_std": float(st.session_state.get("k_std", 1.5)),
                            "use_floor": bool(st.session_state.get("use_floor", True)),
                            "knee_floor": float(st.session_state.get("knee_floor", 5.0)),
                            "trunk_floor": float(st.session_state.get("trunk_floor", 3.0)),
                            "mild": float(st.session_state.get("mild", 5.0)),
                            "moderate": float(st.session_state.get("moderate", 12.0)),
                            "severe": float(st.session_state.get("severe", 20.0)),
                            "persist_pct": float(st.session_state.get("persist_pct", 20.0)),
                            "bottom_window": float(st.session_state.get("bottom_window", 8.0)),
                            "use_robust": bool(st.session_state.get("use_robust", True)),
                            "robust_q": int(st.session_state.get("robust_q", 95)),
                            "min_stability": float(st.session_state.get("min_stability", 0.6)),
                            "min_conf_required": float(st.session_state.get("min_conf_required", 0.6)),
                            "allow_quick": bool(st.session_state.get("allow_quick", True)),
                            "gemini": {
                                "enabled": bool(st.session_state.get("use_gemini", False)),
                                "style": str(st.session_state.get("personality_style", "Supportive coach")),
                                "intensity": int(st.session_state.get("personality_intensity", 1)),
                                "llm_available": bool(LLM_AVAILABLE),
                            },
                        },
                        "outputs": {
                            "coach_driver": st.session_state.get("coach_driver"),
                            "tip": st.session_state.get("latest_tip"),
                            "analysis_s3_key": s3_key(run_id, Path(artifacts["analysis_json"]).name, "processed/analysis")
                            if "analysis_json" in artifacts else None,
                            "envelope_s3_key": s3_key(run_id, Path(artifacts["envelope_npz"]).name, "processed/envelope")
                            if "envelope_npz" in artifacts else None,
                        },
                    }

                    manifest_key = f"{cfg['prefix']}/{cfg['user_id']}/{run_id}/manifest.json"
                    s3_put_bytes(
                        json.dumps(manifest, indent=2).encode("utf-8"),
                        manifest_key,
                        content_type="application/json",
                        metadata={"run_id": run_id, "created_at": run_created_at},
                    )

                    st.session_state["saved_to_s3"] = True
                    st.success("Uploaded: Videos + analysis + manifest saved to S3.")
                    st.caption(f"Manifest: `s3://{cfg['bucket']}/{manifest_key}`")
                    st.rerun()

                except NoCredentialsError:
                    st.session_state["s3_last_error"] = "No AWS credentials found."
                    st.error("AWS credentials not found. Make sure you configured credentials.")
                except ClientError as e:
                    error_code = e.response.get("Error", {}).get("Code", "Unknown")
                    error_message = e.response.get("Error", {}).get("Message", str(e))
                    st.session_state["s3_last_error"] = str(e)

                    st.error(f"❌ **S3 Upload Error**")
                    if error_code == "AccessDenied":
                        st.error(f"Access Denied: Your AWS credentials don't have permission to upload objects to bucket '{cfg['bucket']}'. Please ensure your IAM user has the 's3:PutObject' permission for this bucket.")

                        st.warning("⚠️ **Your run data is saved locally, but S3 upload failed.**")
                        st.info("💡 **How to fix this in AWS Console:**")

                        with st.expander("📋 **Step-by-Step Instructions**", expanded=True):
                            st.markdown("""
                            **Step 1:** Go to [AWS IAM Console](https://console.aws.amazon.com/iam/) and sign in.

                            **Step 2:** Click on "Users" in the left sidebar, then find and click on your IAM user (`daniel-aws`).

                            **Step 3:** Click on the "Permissions" tab, then click "Add permissions" → "Create inline policy".

                            **Step 4:** Click on the "JSON" tab and paste this policy:
                            """)

                            bucket_name = cfg["bucket"]
                            policy_json = f'''{{
    "Version": "2012-10-17",
    "Statement": [
        {{
            "Effect": "Allow",
            "Action": [
                "s3:ListBucket",
                "s3:GetObject",
                "s3:PutObject"
            ],
            "Resource": [
                "arn:aws:s3:::{bucket_name}",
                "arn:aws:s3:::{bucket_name}/*"
            ]
        }}
    ]
}}'''

                            st.code(policy_json, language="json")

                            st.markdown("""
                            **Step 5:** Click "Next", give the policy a name (e.g., `RoastCoachS3Access`), then click "Create policy".

                            **Step 6:** Wait a few seconds for the policy to propagate, then try saving again.
                            """)
                    else:
                        st.error(f"AWS Error ({error_code}): {error_message}")
                        st.info("💡 Check your AWS credentials and bucket configuration in the `.env` file.")
                except Exception as e:
                    error_str = str(e)
                    st.session_state["s3_last_error"] = error_str
                    if "AWS S3 Error" in error_str and "AccessDenied" in error_str:
                        # Show full instructions for AccessDenied errors
                        st.error(f"❌ **S3 Upload Error**")
                        st.error(f"Access Denied: Your AWS credentials don't have permission to upload objects to bucket '{cfg['bucket']}'. Please ensure your IAM user has the 's3:PutObject' permission for this bucket.")

                        st.warning("⚠️ **Your run data is saved locally, but S3 upload failed.**")
                        st.info("💡 **How to fix this in AWS Console:**")

                        with st.expander("📋 **Step-by-Step Instructions**", expanded=True):
                            st.markdown("""
                            **Step 1:** Go to [AWS IAM Console](https://console.aws.amazon.com/iam/) and sign in.

                            **Step 2:** Click on "Users" in the left sidebar, then find and click on your IAM user (`daniel-aws`).

                            **Step 3:** Click on the "Permissions" tab, then click "Add permissions" → "Create inline policy".

                            **Step 4:** Click on the "JSON" tab and paste this policy:
                            """)

                            bucket_name = cfg["bucket"]
                            policy_json = f'''{{
    "Version": "2012-10-17",
    "Statement": [
        {{
            "Effect": "Allow",
            "Action": [
                "s3:ListBucket",
                "s3:GetObject",
                "s3:PutObject"
            ],
            "Resource": [
                "arn:aws:s3:::{bucket_name}",
                "arn:aws:s3:::{bucket_name}/*"
            ]
        }}
    ]
}}'''

                            st.code(policy_json, language="json")

                            st.markdown("""
                            **Step 5:** Click "Next", give the policy a name (e.g., `RoastCoachS3Access`), then click "Create policy".

                            **Step 6:** Wait a few seconds for the policy to propagate, then try saving again.
                            """)
                    elif "AWS S3 Error" in error_str or "AccessDenied" in error_str:
                        st.error(f"❌ **S3 Upload Error**")
                        st.error(error_str)
                        st.info("💡 Check your AWS IAM permissions. You need `s3:PutObject` permission for the bucket.")
                    else:
                        st.error(f"Unexpected error: {type(e).__name__}: {error_str}")
                        st.info("💡 Your run data is saved locally. Check your AWS configuration and try again.")
        elif st.session_state.get("saved_to_s3", False):
            st.divider()
            st.success("This run is already saved to S3.")


# ==============================
#           HISTORY
# ==============================
with tab_history:
    st.subheader("History — Calendar View")

    cfg = s3_config()
    if not s3_enabled():
        if not cfg["bucket"]:
            st.warning("Set ROASTCOACH_S3_BUCKET to enable History.")
            st.code('export ROASTCOACH_S3_BUCKET="your-bucket-name"')
        elif not BOTO3_AVAILABLE:
            st.warning("Install boto3 to enable History.")
            st.code("pip install boto3")
        st.stop()

    # Load manifests from S3
    manifest_objs = []
    error_msg = None
    with st.spinner("Loading runs from S3..."):
        try:
            manifest_objs = s3_list_manifests_cached(
                bucket=cfg["bucket"],
                region=cfg["region"],
                prefix=cfg["prefix"],
                user_id=cfg["user_id"],
                max_items=200,
            )
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))
            if error_code == "AccessDenied":
                error_msg = f"Access Denied: Your AWS credentials don't have permission to list objects in bucket '{cfg['bucket']}'. Please ensure your IAM user has the 's3:ListBucket' permission for this bucket."
            else:
                error_msg = f"AWS Error ({error_code}): {error_message}"
        except NoCredentialsError:
            error_msg = "AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in your .env file."
        except Exception as e:
            error_msg = f"Unexpected error accessing S3: {str(e)}"

    if error_msg:
        st.error(f"❌ **S3 Access Error**")
        st.error(error_msg)
        st.warning("⚠️ **The History tab requires AWS S3 permissions to list your saved runs.**")
        st.info("💡 **How to fix this in AWS Console:**")
        with st.expander("📋 **Step-by-Step Instructions**", expanded=True):
            st.markdown("""
            **Step 1:** Go to [AWS IAM Console](https://console.aws.amazon.com/iam/) and sign in.

            **Step 2:** Click on "Users" in the left sidebar, then find and click on your IAM user.

            **Step 3:** Click on the "Permissions" tab, then click "Add permissions" → "Create inline policy".

            **Step 4:** Click on the "JSON" tab and paste this policy:
            """)
            bucket_name = cfg["bucket"]
            policy_json = f'''{{
    "Version": "2012-10-17",
    "Statement": [
        {{
            "Effect": "Allow",
            "Action": [
                "s3:ListBucket",
                "s3:GetObject"
            ],
            "Resource": [
                "arn:aws:s3:::{bucket_name}",
                "arn:aws:s3:::{bucket_name}/*"
            ]
        }}
    ]
}}'''
            st.code(policy_json, language="json")
        st.stop()

    if not manifest_objs:
        st.info("No runs found yet. Save a run to S3 from the Run tab, then come back here.")
        st.stop()

    # Load and parse all manifests, group by date
    runs_by_date: Dict[str, List[Dict[str, Any]]] = {}

    with st.spinner("Loading run details..."):
        for obj in manifest_objs:
            key = obj.get("key", "")
            try:
                manifest = s3_get_json_cached(cfg["bucket"], cfg["region"], key)
                created_at_str = manifest.get("created_at_utc", "")

                # Parse date from created_at_utc (format: "2026-02-01T10:30:00Z")
                try:
                    if created_at_str:
                        created_dt = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                        date_key = created_dt.date().isoformat()
                    else:
                        # Fallback to last_modified if created_at_utc is missing
                        last_modified = obj.get("last_modified")
                        if last_modified:
                            date_key = last_modified.date().isoformat()
                        else:
                            continue
                except Exception:
                    continue

                # Load analysis if available
                analysis_data = None
                analysis_summary = None
                analysis_key = manifest.get("outputs", {}).get("analysis_s3_key")
                if analysis_key:
                    try:
                        analysis_data = s3_get_json_cached(cfg["bucket"], cfg["region"], analysis_key)
                        analysis_summary = extract_analysis_summary(analysis_data)
                    except Exception:
                        pass  # Skip if analysis can't be loaded

                # Get tip
                tip = (manifest.get("outputs", {}).get("tip", {}) or {}).get("one_sentence_tip", "")

                # Build run data
                run_data = {
                    "run_id": manifest.get("run_id", ""),
                    "exercise": manifest.get("exercise_label", "Unknown"),
                    "created_at": created_at_str,
                    "manifest_key": key,
                    "tip": tip,
                    "analysis_data": analysis_data,
                    "analysis_summary": analysis_summary,
                }

                if date_key not in runs_by_date:
                    runs_by_date[date_key] = []
                runs_by_date[date_key].append(run_data)

            except Exception:
                continue  # Skip if manifest can't be loaded

    # Sort runs within each date by created_at (newest first)
    for date_key in runs_by_date:
        runs_by_date[date_key].sort(key=lambda x: x.get("created_at", ""), reverse=True)

    # Get available dates
    available_dates = sorted([date.fromisoformat(d) for d in runs_by_date.keys()], reverse=True)

    if not available_dates:
        st.info("No runs found with valid dates. Save a run to S3 from the Run tab, then come back here.")
        st.stop()

    # Initialize session state for calendar and carousel
    if "history_selected_date" not in st.session_state:
        st.session_state["history_selected_date"] = available_dates[0]

    if "history_video_index" not in st.session_state:
        st.session_state["history_video_index"] = {}

    # Calendar date selection
    col_cal1, col_cal2 = st.columns([2, 1])
    with col_cal1:
        selected_date = st.date_input(
            "Select Date",
            value=st.session_state.get("history_selected_date") or available_dates[0],
            min_value=min(available_dates) if available_dates else date.today(),
            max_value=max(available_dates) if available_dates else date.today(),
            key="history_date_picker"
        )
        st.session_state["history_selected_date"] = selected_date

    with col_cal2:
        st.caption(f"**{len(available_dates)}** days with videos")
        total_runs = sum(len(runs) for runs in runs_by_date.values())
        st.caption(f"**{total_runs}** total runs")

    # Get runs for selected date
    date_key = selected_date.isoformat()
    runs_for_date = runs_by_date.get(date_key, [])

    if not runs_for_date:
        st.info(f"No runs found for {selected_date.strftime('%B %d, %Y')}. Select a different date.")
        st.stop()

    # Initialize video index for this date if not set
    if date_key not in st.session_state["history_video_index"]:
        st.session_state["history_video_index"][date_key] = 0

    current_index = st.session_state["history_video_index"][date_key]
    if current_index >= len(runs_for_date):
        current_index = 0
        st.session_state["history_video_index"][date_key] = 0

    # Carousel navigation
    col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
    with col_nav1:
        if st.button("◀", disabled=current_index == 0, key="prev_video", use_container_width=True):
            st.session_state["history_video_index"][date_key] = max(0, current_index - 1)
            st.rerun()

    with col_nav2:
        run_options = [f"Run {i+1}: {r.get('exercise', 'Unknown')}" for i, r in enumerate(runs_for_date)]
        selected_run_idx = st.selectbox(
            f"Select run ({len(runs_for_date)} total)",
            range(len(runs_for_date)),
            format_func=lambda x: run_options[x],
            index=current_index,
            key="run_selector"
        )
        if selected_run_idx != current_index:
            st.session_state["history_video_index"][date_key] = selected_run_idx
            st.rerun()

    with col_nav3:
        if st.button("▶", disabled=current_index >= len(runs_for_date) - 1, key="next_video", use_container_width=True):
            st.session_state["history_video_index"][date_key] = min(len(runs_for_date) - 1, current_index + 1)
            st.rerun()

    # Get current run
    current_run = runs_for_date[current_index]
    summary = current_run.get("analysis_summary", {})
    tip = current_run.get("tip", "")

    # Summary statistics card
    st.markdown("---")
    st.markdown("""
    <div class="history-stat-card">
    """, unsafe_allow_html=True)

    # Exercise name and date
    col_title1, col_title2 = st.columns([2, 1])
    with col_title1:
        st.markdown(f"### {current_run['exercise']}")
        created_at_str = current_run.get("created_at", "")
        if created_at_str:
            try:
                created_dt = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                st.caption(f"Uploaded: {created_dt.strftime('%B %d, %Y at %I:%M %p')}")
            except Exception:
                st.caption(f"Uploaded: {created_at_str}")

    with col_title2:
        st.markdown(f"**Run ID**")
        st.caption(f"`{current_run['run_id']}`")

    # Performance metrics
    if summary:
        st.markdown("#### Performance Metrics")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        with metric_col1:
            st.metric("Confidence", f"{summary.get('avg_confidence', 0.0):.0%}")

        with metric_col2:
            st.metric("Total Reps", summary.get("total_reps", 0))

        with metric_col3:
            st.metric("Max Deviation", f"{summary.get('max_deviation', 0.0):.1f}°")

        with metric_col4:
            st.metric("Stability", f"{summary.get('avg_stability', 0.0):.0%}")

        # Severity breakdown
        severity_counts = summary.get("severity_counts", {})
        st.markdown("#### Form Analysis")
        severity_col1, severity_col2, severity_col3 = st.columns(3)

        with severity_col1:
            st.metric("Mild Issues", severity_counts.get("mild", 0), delta=None)

        with severity_col2:
            st.metric("Moderate Issues", severity_counts.get("moderate", 0), delta=None)

        with severity_col3:
            st.metric("Severe Issues", severity_counts.get("severe", 0), delta=None)
    else:
        st.info("Analysis data not available for this run.")

    # Coaching tip
    if tip:
        st.markdown("#### Coaching Feedback")
        st.info(f"💡 {tip}")

    # Additional stats
    if summary:
        with st.expander("View Detailed Statistics"):
            st.json({
                "Average Confidence": f"{summary.get('avg_confidence', 0.0):.2%}",
                "Total Reps": summary.get("total_reps", 0),
                "Max Deviation": f"{summary.get('max_deviation', 0.0):.2f}°",
                "Average Stability": f"{summary.get('avg_stability', 0.0):.2%}",
                "Severity Breakdown": summary.get("severity_counts", {}),
                "Upload Time": current_run.get("created_at", ""),
            })

    st.markdown("</div>", unsafe_allow_html=True)

    # Progress indicator
    st.markdown("---")
    st.markdown(f'<p class="history-progress">{current_index + 1} of {len(runs_for_date)} runs for {selected_date.strftime("%B %d, %Y")}</p>', unsafe_allow_html=True)
