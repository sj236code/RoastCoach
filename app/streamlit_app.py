# app/streamlit_app.py
# RoastCoach — Streamlit UI with cached pipeline + stable widget state + Gemini personality toggle
# PLUS: S3 persistence for progress tracking (videos + analysis.json + artifacts)

from __future__ import annotations

import base64
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

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
st.set_page_config(page_title="RoastCoach", layout="wide")

st.title("RoastCoach — Exercise-Agnostic Motion Fingerprinting")
st.caption("Upload a coach reference video, then a user attempt. We'll compare joint-angle trajectories.")

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
    # stable signature across reruns
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
        # interior angle: above_reference => too straight
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


# ----------------------------- Cache expensive steps -----------------------------
@st.cache_data(show_spinner=False)
def cached_pose(video_path: str):
    return extract_pose_frames(video_path)


@st.cache_data(show_spinner=False)
def cached_angles(frames):
    return compute_angles(frames)


# ----------------------------- S3 helpers -----------------------------
def s3_config() -> Dict[str, str]:
    return {
        "bucket": os.getenv("ROASTCOACH_S3_BUCKET", "").strip(),
        "region": os.getenv("AWS_REGION", "us-east-1").strip(),
        "prefix": os.getenv("ROASTCOACH_S3_PREFIX", "roastcoach").strip().strip("/"),
        "user_id": os.getenv("ROASTCOACH_USER_ID", "anonymous").strip(),
    }


def s3_enabled() -> bool:
    cfg = s3_config()
    return BOTO3_AVAILABLE and bool(cfg["bucket"])


def s3_client():
    cfg = s3_config()
    return boto3.client("s3", region_name=cfg["region"])


def s3_key(run_id: str, filename: str, kind: str) -> str:
    """
    kind examples:
      - raw/coach_video.mp4
      - raw/user_video.mp4
      - processed/analysis.json
      - processed/angles_coach.csv
      - processed/envelope.npz
      - manifest.json
    """
    cfg = s3_config()
    # layout: prefix/user_id/run_id/kind
    return f"{cfg['prefix']}/{cfg['user_id']}/{run_id}/{kind}/{filename}"


def s3_put_file(local_path: Path, key: str, content_type: str | None = None, metadata: Dict[str, str] | None = None):
    cli = s3_client()
    extra = {}
    if content_type:
        extra["ContentType"] = content_type
    if metadata:
        # S3 metadata keys must be strings, lowercase recommended
        extra["Metadata"] = {str(k).lower(): str(v) for k, v in metadata.items()}
    with local_path.open("rb") as f:
        cli.put_object(Bucket=s3_config()["bucket"], Key=key, Body=f, **extra)


def s3_put_bytes(data: bytes, key: str, content_type: str | None = None, metadata: Dict[str, str] | None = None):
    cli = s3_client()
    extra = {}
    if content_type:
        extra["ContentType"] = content_type
    if metadata:
        extra["Metadata"] = {str(k).lower(): str(v) for k, v in metadata.items()}
    cli.put_object(Bucket=s3_config()["bucket"], Key=key, Body=data, **extra)


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


# ----------------------------- Sidebar controls -----------------------------
st.sidebar.header("Controls")
st.sidebar.caption("These persist across reruns.")

use_gemini = st.sidebar.checkbox("Use Gemini personality", key="use_gemini", value=False)
personality = st.sidebar.selectbox(
    "Personality style",
    ["Supportive coach", "Slight roast", "MEGA ROAST"],
    key="personality_style",
    index=0,
)
intensity = st.sidebar.slider(
    "Personality intensity",
    min_value=0,
    max_value=2,
    value=1,
    step=1,
    help="0 = soft, 1 = normal, 2 = max intensity",
    key="personality_intensity",
)

if use_gemini and not LLM_AVAILABLE:
    st.sidebar.warning("Gemini helper not imported. Falling back to constrained cue.")
    if _llm_err is not None:
        st.sidebar.caption(f"Import error: {type(_llm_err).__name__}: {_llm_err}")

# ---- S3 sidebar section ----
st.sidebar.divider()
st.sidebar.subheader("Progress tracking (S3)")

cfg = s3_config()
st.sidebar.caption(
    f"Bucket: `{cfg['bucket'] or 'NOT SET'}`\n\n"
    f"Region: `{cfg['region']}`\n\n"
    f"Prefix: `{cfg['prefix']}`\n\n"
    f"User ID: `{cfg['user_id']}`"
)

enable_s3 = st.sidebar.checkbox(
    "Enable S3 saving for this session",
    value=False,
    key="enable_s3_saving",
    help="Uploads videos + analysis.json + artifacts to S3 when you click 'Save this run to S3'.",
)

if enable_s3 and not s3_enabled():
    if not BOTO3_AVAILABLE:
        st.sidebar.error("boto3 not available. Install with: `pip install boto3`")
    if not cfg["bucket"]:
        st.sidebar.error("Set env var: ROASTCOACH_S3_BUCKET")


# ----------------------------- Upload UI -----------------------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("1) Coach Reference Video")
    coach_file = st.file_uploader("Upload coach demo (.mp4/.mov/.m4v)", type=["mp4", "mov", "m4v"], key="coach")
with col2:
    st.subheader("2) User Attempt Video")
    user_file = st.file_uploader("Upload user attempt (.mp4/.mov/.m4v)", type=["mp4", "mov", "m4v"], key="user")

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

# Save uploads locally only AFTER we have a run_id (created on Stage 1 click).
# But we still want previews now — that's fine; we can show them directly.
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


# ----------------------------- Stage 1 -----------------------------
st.divider()
st.subheader("Stage 1 — Compute pose + angles (expensive)")

compute_btn = st.button(
    "Compute pose + angles",
    type="primary",
    disabled=not (coach_file and user_file),
)

if compute_btn:
    # Create a new run id for tracking this attempt
    run_id, iso_ts = utc_run_id()
    st.session_state["run_id"] = run_id
    st.session_state["run_created_at"] = iso_ts
    st.session_state["saved_to_s3"] = False
    st.session_state["s3_last_error"] = None
    st.session_state["artifacts_local"] = {}

    # Save uploads locally with run_id in filename (so you can also track locally)
    coach_path = save_upload(coach_file, DATA_RAW, prefix="coach", run_id=run_id)
    user_path = save_upload(user_file, DATA_RAW, prefix="user", run_id=run_id)

    st.session_state["coach_path"] = coach_path
    st.session_state["user_path"] = user_path

    st.session_state["stage1_done"] = False
    st.session_state["stage2_done"] = False

    st.info("Running pose extraction + angle computation (cached by video path)...")

    with st.spinner("Extracting pose (coach)..."):
        coach_frames = cached_pose(str(coach_path))
    with st.spinner("Extracting pose (user)..."):
        user_frames = cached_pose(str(user_path))
    with st.spinner("Computing angles..."):
        coach_angles = cached_angles(coach_frames)
        user_angles = cached_angles(user_frames)

    df_coach = angles_to_df(coach_angles)
    df_user = angles_to_df(user_angles)

    # Save CSVs (run-specific + latest)
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

    st.success("Stage 1 complete ✅ Pose + angles cached")
    st.caption(f"Run ID: `{run_id}`  |  Created (UTC): `{iso_ts}`")
    st.caption(f"Saved locally: `{coach_path.name}`, `{user_path.name}`")

if not st.session_state["stage1_done"]:
    st.info("Click **Compute pose + angles** once. After that, widgets won’t re-run expensive steps.")
    st.stop()

df_coach: pd.DataFrame = st.session_state["df_coach"]
df_user: pd.DataFrame = st.session_state["df_user"]
run_id = st.session_state["run_id"] or "run_unknown"
run_created_at = st.session_state["run_created_at"] or ""


# ----------------------------- Sanity plot -----------------------------
st.subheader("Angle Sanity Plot")
angle_cols_coach = [c for c in df_coach.columns if c not in ("t", "conf")]
angle_cols_user = [c for c in df_user.columns if c not in ("t", "conf")]
shared_cols = [c for c in angle_cols_coach if c in angle_cols_user]
st.session_state["shared_cols"] = shared_cols

if not shared_cols:
    st.warning("No shared angle columns found. Pose landmarks may be failing.")
    st.write("Try a clearer video (full body visible, good lighting, minimal occlusion).")
    st.stop()

default_sanity = "left_knee_flex" if "left_knee_flex" in shared_cols else shared_cols[0]
sanity_joint = st.selectbox("Sanity plot joint", shared_cols, index=shared_cols.index(default_sanity), key="sanity_joint")

fig = plt.figure()
plt.plot(df_coach["t"], df_coach[sanity_joint], label="coach")
plt.plot(df_user["t"], df_user[sanity_joint], label="user")
plt.xlabel("time (s)")
plt.ylabel("degrees")
plt.title(f"{sanity_joint} over time")
plt.legend()
st.pyplot(fig)

# ----------------------------- Stage 2 -----------------------------
st.divider()
st.subheader("Stage 2 — Rep segmentation + reference (medium)")

cA, cB, cC, cD = st.columns([1, 1, 1, 2])
with cA:
    min_q = st.slider("Min rep quality", 0.0, 1.0, 0.35, 0.05, key="min_q")
with cB:
    drop_first = st.checkbox("Drop first rep (setup)", value=True, key="drop_first")
with cC:
    N = st.slider("Normalization length (N)", 50, 200, int(st.session_state.get("N", 100)), 10, key="N_norm")
with cD:
    st.caption("After changing these, click **Build reference** again.")

build_btn = st.button("Build reference (segmentation + normalization)")

if build_btn:
    st.session_state["stage2_done"] = False

    coach_driver = choose_driver_column(df_coach)
    user_driver = coach_driver  # force same driver

    coach_reps_raw, coach_driver_smooth = find_reps_from_driver(df_coach, coach_driver)
    user_reps_raw, user_driver_smooth = find_reps_from_driver(df_user, user_driver)

    coach_reps = filter_reps(coach_reps_raw, min_quality=min_q, drop_first=drop_first)
    user_reps = filter_reps(user_reps_raw, min_quality=min_q, drop_first=drop_first)

    if not (coach_reps and user_reps):
        st.warning("Not enough reps kept after filtering. Lower min quality or uncheck 'drop first rep'.")
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

    st.success("Stage 2 complete ✅ Reference cached")

if not st.session_state["stage2_done"]:
    st.info("Click **Build reference** to enable envelope + deviation + coaching.")
    st.stop()

coach_driver: str = st.session_state["coach_driver"]
coach_reps = st.session_state["coach_reps"]
user_reps = st.session_state["user_reps"]
angle_cols = st.session_state["angle_cols"]
tt = st.session_state["tt"]
coach_mean = st.session_state["coach_mean"]
coach_std = st.session_state["coach_std"]
coach_stack = st.session_state["coach_stack"]
user_mean = st.session_state["user_mean"]
user_std = st.session_state["user_std"]
N = int(st.session_state["N"])

st.write(f"Coach driver: `{coach_driver}` (forced for user)")
st.caption(f"Coach {conf_summary(df_coach)} | User {conf_summary(df_user)}")
st.write(f"Kept reps — coach: **{len(coach_reps)}** | user: **{len(user_reps)}**")

# Normalized joint plot
default_idx = angle_cols.index(coach_driver) if coach_driver in angle_cols else 0
plot_joint = st.selectbox("Plot joint (normalized)", options=angle_cols, index=default_idx, key="plot_joint_norm")
j = angle_cols.index(plot_joint)

fig3 = plt.figure()
plt.plot(tt, coach_mean[:, j], label="coach mean")
plt.plot(tt, user_mean[:, j], label="user mean")
plt.fill_between(tt, user_mean[:, j] - user_std[:, j], user_mean[:, j] + user_std[:, j], alpha=0.2)
plt.xlabel("normalized time (0→1)")
plt.ylabel("degrees")
plt.title(f"Mean normalized rep (±1 std) — {plot_joint}")
plt.legend()
st.pyplot(fig3)

# ----------------------------- Stage 3 (live) -----------------------------
st.divider()
st.subheader("Stage 3 — Envelope + deviation + coaching (fast, live)")

env_method = st.selectbox("Envelope method", ["std*k", "percentile (10–90)"], index=0, key="env_method")
k = st.slider("k (std multiplier)", 0.5, 3.0, 1.5, 0.1, key="k_std")

use_floor = st.checkbox("Use minimum envelope width floor (recommended)", value=True, key="use_floor")
knee_floor = st.slider("Min half-width for knee/elbow (deg)", 1.0, 15.0, 5.0, 0.5, key="knee_floor")
trunk_floor = st.slider("Min half-width for trunk_incline (deg)", 1.0, 10.0, 3.0, 0.5, key="trunk_floor")

# Build envelope (fast)
if env_method == "std*k":
    tol = coach_std * float(k)
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

st.write(f"Reference stability (proxy): **{stability:.2f}**")
min_stability = st.slider("Warn if stability < ", 0.1, 1.0, 0.6, 0.05, key="min_stability")
if stability < min_stability:
    st.warning("Coach reference is unstable (few reps or variable form). You can still coach in Quick Mode.")

# Envelope viz
fig_env = plt.figure()
plt.plot(tt, coach_mean[:, j], label="coach mean")
plt.fill_between(tt, ref_lo[:, j], ref_hi[:, j], alpha=0.18, label="coach envelope")
plt.plot(tt, user_mean[:, j], label="user mean")
plt.xlabel("normalized time (0→1)")
plt.ylabel("degrees")
plt.title(f"Coach envelope vs user mean — {plot_joint}")
plt.legend()
st.pyplot(fig_env)

# Save envelope artifact (run-specific + latest)
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
st.session_state["artifacts_local"]["envelope_npz"] = str(env_path_run)
st.caption(f"Saved envelope: `{env_path_run.name}`")

# ----------------------------- Deviation detection (live) -----------------------------
st.subheader("Deviation Detection (user reps vs coach envelope)")

mild = st.slider("Mild threshold (deg outside)", 1.0, 20.0, 5.0, 0.5, key="mild")
moderate = st.slider("Moderate threshold (deg outside)", 5.0, 40.0, 12.0, 0.5, key="moderate")
severe = st.slider("Severe threshold (deg outside)", 10.0, 60.0, 20.0, 0.5, key="severe")
persistent_thresh = st.slider("Persistent if outside > (%)", 0, 100, 20, 5, key="persist_pct") / 100.0
bottom_window = st.slider("Bottom window size (normalized %)", 2, 20, 8, 1, key="bottom_window") / 100.0

use_robust = st.checkbox("Use robust max (percentile) for deg_outside", value=True, key="use_robust")
robust_q = st.slider("Robust percentile (q)", 80, 100, 95, 1, key="robust_q")

driver_idx = angle_cols.index(coach_driver) if coach_driver in angle_cols else 0

analyses: List[Dict[str, Any]] = []
for ridx, r in enumerate(user_reps):
    user_mat = resample_rep_angles(df_user, r, angle_cols, N=int(N))  # (N,J)

    # bottom index: min interior angle = deepest bend
    idx_bottom = int(np.nanargmin(user_mat[:, driver_idx]))

    below = ref_lo - user_mat
    above = user_mat - ref_hi
    over = np.maximum(0.0, np.maximum(below, above))  # (N,J)

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

st.write(f"Analyzed **{len(analyses)}** user reps.")
with st.expander("Preview analysis JSON (first 2 reps)"):
    st.json(analyses[:2])

# Save analysis (run-specific + latest)
analysis_path_run = DATA_PROCESSED / f"{run_id}_analysis.json"
analysis_path_latest = DATA_PROCESSED / "analysis.json"
analysis_path_run.write_text(json.dumps(analyses, indent=2))
analysis_path_latest.write_text(json.dumps(analyses, indent=2))
st.session_state["latest_analysis"] = analyses
st.session_state["artifacts_local"]["analysis_json"] = str(analysis_path_run)
st.caption(f"Saved analysis: `{analysis_path_run.name}`")

# ----------------------------- Coaching cue -----------------------------
st.divider()
st.subheader("Coaching cue")

min_conf_required = st.slider("Min confidence required", 0.0, 1.0, 0.6, 0.05, key="min_conf_required")
rep_pick = st.selectbox("Choose rep to coach", options=list(range(len(analyses))), index=0, key="rep_pick")
allow_quick = st.checkbox("Allow Quick Mode when stability is low", value=True, key="allow_quick")

chosen = analyses[rep_pick]

base_tip = make_tip_from_events(
    chosen["events"],
    stability=float(chosen["reference_stability"]),
    confidence=float(chosen["confidence"]),
    min_stability=float(min_stability),
    min_conf=float(min_conf_required),
    allow_quick_mode=bool(allow_quick),
)

final_tip = base_tip

# Optional Gemini personality transform
if use_gemini and LLM_AVAILABLE:
    try:
        final_tip = generate_personality_tip(
            analysis=chosen,
            base_tip=base_tip,
            style=personality,
            intensity=int(intensity),
        )
        if not isinstance(final_tip, dict) or "one_sentence_tip" not in final_tip:
            final_tip = base_tip
    except Exception as e:
        st.warning("Gemini personality failed; showing constrained cue.")
        st.caption(f"{type(e).__name__}: {e}")
        final_tip = base_tip

st.session_state["latest_tip"] = final_tip

st.success(final_tip["one_sentence_tip"])
with st.expander("Tip JSON"):
    st.json(final_tip)


# ----------------------------- S3 Save (button) -----------------------------
st.divider()
st.subheader("Save this run (videos + analysis) to S3")

cfg = s3_config()

if not enable_s3:
    st.info("Enable S3 saving from the sidebar to upload this run for progress tracking.")
else:
    if not s3_enabled():
        st.error("S3 is not ready. Make sure boto3 is installed and ROASTCOACH_S3_BUCKET is set.")
    else:
        # Let the user optionally label the exercise (for future progress UI)
        exercise_label = st.text_input(
            "Exercise label (optional)",
            value=st.session_state.get("exercise_label", ""),
            key="exercise_label",
            help="Example: squat, pushup, lunge, deadlift. Stored in manifest.json.",
        )

        # Show target S3 folder
        base_prefix = f"{cfg['prefix']}/{cfg['user_id']}/{run_id}/"
        st.caption(f"S3 destination prefix: `{base_prefix}`")

        save_btn = st.button("Save this run to S3", type="primary", disabled=st.session_state.get("saved_to_s3", False))

        if st.session_state.get("saved_to_s3", False):
            st.success("This run is already saved to S3 ✅ (you can start a new run by uploading new videos).")

        if save_btn:
            st.session_state["s3_last_error"] = None

            try:
                coach_path = Path(st.session_state["coach_path"])
                user_path = Path(st.session_state["user_path"])

                # Upload videos
                coach_key = s3_key(run_id, coach_path.name, "raw/coach_video")
                user_key = s3_key(run_id, user_path.name, "raw/user_video")

                s3_put_file(
                    coach_path,
                    coach_key,
                    content_type=guess_video_content_type(coach_path),
                    metadata={"run_id": run_id, "role": "coach", "created_at": run_created_at},
                )
                s3_put_file(
                    user_path,
                    user_key,
                    content_type=guess_video_content_type(user_path),
                    metadata={"run_id": run_id, "role": "user", "created_at": run_created_at},
                )

                # Upload processed artifacts (analysis + angles + envelope)
                artifacts = st.session_state.get("artifacts_local", {})

                # angles
                if "angles_coach_csv" in artifacts:
                    p = Path(artifacts["angles_coach_csv"])
                    s3_put_file(p, s3_key(run_id, p.name, "processed/angles"), content_type="text/csv")
                if "angles_user_csv" in artifacts:
                    p = Path(artifacts["angles_user_csv"])
                    s3_put_file(p, s3_key(run_id, p.name, "processed/angles"), content_type="text/csv")

                # envelope
                if "envelope_npz" in artifacts:
                    p = Path(artifacts["envelope_npz"])
                    s3_put_file(p, s3_key(run_id, p.name, "processed/envelope"), content_type="application/octet-stream")

                # analysis
                if "analysis_json" in artifacts:
                    p = Path(artifacts["analysis_json"])
                    s3_put_file(p, s3_key(run_id, p.name, "processed/analysis"), content_type="application/json")

                # manifest.json (super important for progress tracking)
                manifest = {
                    "run_id": run_id,
                    "created_at_utc": run_created_at,
                    "user_id": cfg["user_id"],
                    "exercise_label": exercise_label.strip() or None,
                    "inputs": {
                        "coach_video_s3_key": coach_key,
                        "user_video_s3_key": user_key,
                        "coach_video_local": str(coach_path),
                        "user_video_local": str(user_path),
                    },
                    "settings": {
                        "min_q": float(st.session_state.get("min_q", 0.35)),
                        "drop_first": bool(st.session_state.get("drop_first", True)),
                        "N": int(st.session_state.get("N", N)),
                        "env_method": str(st.session_state.get("env_method", env_method)),
                        "k_std": float(st.session_state.get("k_std", k)),
                        "use_floor": bool(st.session_state.get("use_floor", use_floor)),
                        "knee_floor": float(st.session_state.get("knee_floor", knee_floor)),
                        "trunk_floor": float(st.session_state.get("trunk_floor", trunk_floor)),
                        "mild": float(st.session_state.get("mild", mild)),
                        "moderate": float(st.session_state.get("moderate", moderate)),
                        "severe": float(st.session_state.get("severe", severe)),
                        "persist_pct": float(st.session_state.get("persist_pct", persistent_thresh * 100.0)),
                        "bottom_window": float(st.session_state.get("bottom_window", bottom_window * 100.0)),
                        "use_robust": bool(st.session_state.get("use_robust", use_robust)),
                        "robust_q": int(st.session_state.get("robust_q", robust_q)),
                        "min_stability": float(st.session_state.get("min_stability", min_stability)),
                        "min_conf_required": float(st.session_state.get("min_conf_required", min_conf_required)),
                        "allow_quick": bool(st.session_state.get("allow_quick", allow_quick)),
                        "gemini": {
                            "enabled": bool(st.session_state.get("use_gemini", use_gemini)),
                            "style": str(st.session_state.get("personality_style", personality)),
                            "intensity": int(st.session_state.get("personality_intensity", intensity)),
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
                st.success("Uploaded ✅ Videos + analysis + manifest saved to S3.")

                st.caption("Saved keys:")
                st.code(
                    "\n".join(
                        [
                            f"s3://{cfg['bucket']}/{coach_key}",
                            f"s3://{cfg['bucket']}/{user_key}",
                            f"s3://{cfg['bucket']}/{manifest_key}",
                        ]
                    )
                )

            except NoCredentialsError:
                st.session_state["s3_last_error"] = "No AWS credentials found."
                st.error(
                    "AWS credentials not found. Make sure you ran `aws configure` (or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)."
                )
            except ClientError as e:
                st.session_state["s3_last_error"] = str(e)
                st.error(f"S3 ClientError: {e}")
            except Exception as e:
                st.session_state["s3_last_error"] = str(e)
                st.error(f"Unexpected error uploading to S3: {type(e).__name__}: {e}")


# ----------------------------- Debug -----------------------------
with st.expander("Debug: App state"):
    st.write(
        {
            "run_id": st.session_state.get("run_id"),
            "created_at_utc": st.session_state.get("run_created_at"),
            "saved_to_s3": st.session_state.get("saved_to_s3"),
            "s3_last_error": st.session_state.get("s3_last_error"),
            "LLM_AVAILABLE": LLM_AVAILABLE,
            "use_gemini": use_gemini,
            "personality": personality,
            "intensity": intensity,
            "stage1_done": st.session_state["stage1_done"],
            "stage2_done": st.session_state["stage2_done"],
            "coach_driver": st.session_state.get("coach_driver"),
            "N": st.session_state.get("N"),
            "artifacts_local": st.session_state.get("artifacts_local", {}),
        }
    )
    if _llm_err is not None:
        st.write({"llm_import_error": f"{type(_llm_err).__name__}: {_llm_err}"})
