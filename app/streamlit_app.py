import streamlit as st
from pathlib import Path
import base64
import sys
import uuid
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="RoastCoach", layout="wide")

st.title("RoastCoach — Exercise-Agnostic Motion Fingerprinting")
st.caption("Upload a coach reference video, then a user attempt. We'll compare joint-angle trajectories.")


# ---------- Project paths (absolute, robust) ----------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


# ---------- Helpers ----------
def save_upload(uploaded_file, out_dir: Path, prefix: str) -> Path:
    """Save uploaded file with a unique name to avoid overwriting."""
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix
    fname = f"{prefix}_{uuid.uuid4().hex[:8]}{suffix}"
    out_path = out_dir / fname
    out_path.write_bytes(uploaded_file.getbuffer())
    return out_path


def video_small(uploaded_file, width_px: int = 360):
    """Smaller preview using HTML <video> so it doesn't take over the page."""
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
    good = np.isfinite(c)
    if good.sum() == 0:
        return "conf: n/a"
    return (
        f"conf mean={np.nanmean(c):.2f}, "
        f"p10={np.nanpercentile(c,10):.2f}, "
        f"p50={np.nanpercentile(c,50):.2f}, "
        f"p90={np.nanpercentile(c,90):.2f}"
    )


def phase_from_t(idx_peak: int, idx_bottom: int, N: int, bottom_window_frac: float = 0.08) -> str:
    """Classify deviation timepoint as descent/bottom/ascent using bottom index."""
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


# -------------------- NEW: safer coaching cue logic --------------------
def _event_priority(e: dict) -> tuple:
    """
    Higher is better. Sort by:
    1) severity rank
    2) persistence label (persistent > brief)
    3) persistence fraction
    4) degrees outside
    """
    sev_rank = {"severe": 3, "moderate": 2, "mild": 1, "none": 0}
    pers_rank = 1 if e.get("persistence_label") == "persistent" else 0
    return (
        sev_rank.get(e.get("severity", "none"), 0),
        pers_rank,
        float(e.get("persistence", 0.0)),
        float(e.get("deg_outside", 0.0)),
    )


def _event_to_tip(e: dict) -> dict:
    """
    Convert event to a human coaching cue that matches your angle conventions:
      - knee/elbow angles are interior angles: smaller = more bend, larger = straighter
      - trunk_incline: larger = more forward lean
    """
    joint = e.get("joint", "n/a")
    phase = e.get("phase", "n/a")
    direction = e.get("direction", "")

    j = joint.lower()

    if "knee_flex" in j or "elbow_flex" in j:
        # interior angle: above_reference => too straight (not enough bend)
        if direction == "above_reference":
            msg = f"Bend your {joint.replace('_', ' ')} more during {phase}."
        elif direction == "below_reference":
            msg = f"Bend your {joint.replace('_', ' ')} less during {phase}."
        else:
            msg = f"Keep your {joint.replace('_', ' ')} steadier during {phase}."

    elif "trunk_incline" in j:
        # trunk_incline is angle from vertical: above_reference => leaning too far forward
        if direction == "above_reference":
            msg = f"Stay more upright during {phase}."
        elif direction == "below_reference":
            msg = f"Lean slightly more forward during {phase}."
        else:
            msg = f"Keep your torso steadier during {phase}."
    else:
        # fallback
        if direction == "below_reference":
            msg = f"Increase {joint} motion during {phase}."
        elif direction == "above_reference":
            msg = f"Reduce {joint} motion during {phase}."
        else:
            msg = f"Keep {joint} steadier during {phase}."

    # <= 20 words hard cap
    words = msg.split()
    if len(words) > 20:
        msg = " ".join(words[:20])

    return {"one_sentence_tip": msg, "target_joint": joint, "phase": phase}


def focus_joints_from_driver(driver: str, angle_cols: list[str]) -> list[str]:
    """
    Focus only on joints that actually exist in angle_cols.
    With current angles.py, the only possible joints are:
      - left/right_knee_flex
      - left/right_elbow_flex
      - trunk_incline
    """
    d = driver.lower()

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


def make_tip_from_events(
    events,
    stability,
    confidence,
    min_stability=0.6,
    min_conf=0.6,
    allow_quick_mode=True,
) -> dict:
    """
    Constrained, safe coaching tip (no LLM).

    NEW POLICY:
    - Hard gate only on confidence (tracking quality).
    - If stability is low, still coach in "quick mode" using top event
      (but UI can warn that reference is unstable).
    """
    # HARD GATE: low confidence => re-record
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

    # If stability is low, we still coach (quick mode) if allowed
    if stability < min_stability and not allow_quick_mode:
        return {
            "one_sentence_tip": "Re-record: do 3–5 consistent coach reps so the reference envelope is stable.",
            "target_joint": "n/a",
            "phase": "n/a",
        }

    # Choose top event
    events_sorted = sorted(events, key=_event_priority, reverse=True)
    best = events_sorted[0]
    return _event_to_tip(best)
# ----------------------------------------------------------------------


# ---------- Import backend pipeline (pose -> angles -> segmentation) ----------
BACKEND_SRC = ROOT / "backend" / "src"
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


# ---------- Upload UI ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("1) Coach Reference Video")
    coach_file = st.file_uploader(
        "Upload coach demo (.mp4/.mov/.m4v)",
        type=["mp4", "mov", "m4v"],
        key="coach",
    )

with col2:
    st.subheader("2) User Attempt Video")
    user_file = st.file_uploader(
        "Upload user attempt (.mp4/.mov/.m4v)",
        type=["mp4", "mov", "m4v"],
        key="user",
    )

coach_path = save_upload(coach_file, DATA_RAW, prefix="coach") if coach_file else None
user_path = save_upload(user_file, DATA_RAW, prefix="user") if user_file else None


# ---------- Previews ----------
st.divider()
st.subheader("Previews")

left, right = st.columns(2)

with left:
    st.markdown("**Coach Preview**")
    if coach_file:
        video_small(coach_file, width_px=360)
        with st.expander("View coach video large"):
            st.video(coach_file)
        st.caption(f"Saved as: `{coach_path.name}`")
    else:
        st.info("Upload a coach reference video to preview it here.")

with right:
    st.markdown("**User Preview**")
    if user_file:
        video_small(user_file, width_px=360)
        with st.expander("View user video large"):
            st.video(user_file)
        st.caption(f"Saved as: `{user_path.name}`")
    else:
        st.info("Upload a user attempt video to preview it here.")


# ---------- Action Button ----------
st.divider()
run = st.button("Run analysis", type="primary", disabled=not (coach_path and user_path))

if run:
    st.info("Running pose extraction + angle computation...")

    with st.spinner("Extracting pose (coach)..."):
        coach_frames = extract_pose_frames(str(coach_path))

    with st.spinner("Extracting pose (user)..."):
        user_frames = extract_pose_frames(str(user_path))

    with st.spinner("Computing joint angles..."):
        coach_angles = compute_angles(coach_frames)
        user_angles = compute_angles(user_frames)

    df_coach = angles_to_df(coach_angles)
    df_user = angles_to_df(user_angles)

    # Save CSVs
    coach_csv = DATA_PROCESSED / "angles_coach.csv"
    user_csv = DATA_PROCESSED / "angles_user.csv"
    df_coach.to_csv(coach_csv, index=False)
    df_user.to_csv(user_csv, index=False)

    st.success("Saved angle CSVs.")
    st.write(f"Coach: `{coach_csv}`")
    st.write(f"User: `{user_csv}`")

    # ---------- Angle sanity plot ----------
    target = "left_knee_flex"
    angle_cols_coach = [c for c in df_coach.columns if c not in ("t", "conf")]
    angle_cols_user = [c for c in df_user.columns if c not in ("t", "conf")]
    shared_cols = [c for c in angle_cols_coach if c in angle_cols_user]

    if not shared_cols:
        st.warning("No shared angle columns found. Pose landmarks may be failing.")
        st.write("Try a clearer video (full body visible, good lighting, minimal occlusion).")
        st.stop()

    if target not in shared_cols:
        target = shared_cols[0]

    st.subheader("Angle Sanity Plot")
    st.caption(f"Showing: `{target}` (coach vs user)")

    fig = plt.figure()
    plt.plot(df_coach["t"], df_coach[target], label="coach")
    plt.plot(df_user["t"], df_user[target], label="user")
    plt.xlabel("time (s)")
    plt.ylabel("degrees")
    plt.title(f"{target} over time")
    plt.legend()
    st.pyplot(fig)

    # ---------- Rep segmentation ----------
    st.divider()
    st.subheader("Rep Segmentation")

    coach_driver = choose_driver_column(df_coach)
    user_driver = coach_driver

    st.write(f"Coach driver: `{coach_driver}`")
    st.write(f"User driver (forced): `{user_driver}`")
    st.caption(f"Coach {conf_summary(df_coach)} | User {conf_summary(df_user)}")

    coach_reps_raw, coach_driver_smooth = find_reps_from_driver(df_coach, coach_driver)
    user_reps_raw, user_driver_smooth = find_reps_from_driver(df_user, user_driver)

    st.markdown("### Rep filtering")
    cA, cB, cC = st.columns([1, 1, 2])
    with cA:
        min_q = st.slider("Min rep quality", 0.0, 1.0, 0.35, 0.05)
    with cB:
        drop_first = st.checkbox("Drop first rep (setup)", value=True)
    with cC:
        st.caption("Tip: Raise min quality to tighten the reference band (cleaner demo).")

    coach_reps = filter_reps(coach_reps_raw, min_quality=min_q, drop_first=drop_first)
    user_reps = filter_reps(user_reps_raw, min_quality=min_q, drop_first=drop_first)

    st.write(
        f"Detected coach reps: **{len(coach_reps_raw)}** → kept **{len(coach_reps)}** | "
        f"user reps: **{len(user_reps_raw)}** → kept **{len(user_reps)}**"
    )

    def plot_driver_with_reps(df: pd.DataFrame, driver_col: str, x_s: np.ndarray, reps, title: str):
        fig = plt.figure()
        plt.plot(df["t"], x_s, label=f"{driver_col} (smoothed)")
        for r in reps:
            plt.axvline(df["t"].iloc[r.start_idx], linestyle="--")
            plt.axvline(df["t"].iloc[r.end_idx], linestyle="--")
        plt.xlabel("time (s)")
        plt.ylabel("degrees")
        plt.title(title)
        plt.legend()
        return fig

    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_driver_with_reps(df_coach, coach_driver, coach_driver_smooth, coach_reps, "Coach driver + rep boundaries"))
    with c2:
        st.pyplot(plot_driver_with_reps(df_user, user_driver, user_driver_smooth, user_reps, "User driver + rep boundaries"))

    # ---------- Normalization (mean ± std) ----------
    st.divider()
    st.subheader("Normalized reps (mean ± 1 std)")

    if not (coach_reps and user_reps):
        st.warning("Not enough reps kept after filtering. Lower min quality or uncheck 'drop first rep'.")
        st.stop()

    angle_cols = shared_cols
    N = st.slider("Normalization length (N)", 50, 200, 100, 10)
    tt = np.linspace(0, 1, N)

    coach_mean, coach_std, coach_stack = mean_std_normalized_rep(df_coach, coach_reps, angle_cols, N=N)
    user_mean, user_std, user_stack = mean_std_normalized_rep(df_user, user_reps, angle_cols, N=N)

    default_idx = angle_cols.index(coach_driver) if coach_driver in angle_cols else 0
    plot_joint = st.selectbox("Plot joint (normalized)", options=angle_cols, index=default_idx)
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

    # ---------- Reference Envelope Builder (Coach) ----------
    st.divider()
    st.subheader("Reference Envelope (Coach) + Stability")

    env_method = st.selectbox("Envelope method", ["std*k", "percentile (10–90)"], index=0)
    k = st.slider("k (std multiplier)", 0.5, 3.0, 1.5, 0.1)

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

    st.write(f"Reference stability (proxy): **{stability:.2f}**")
    min_stability = st.slider("Warn if stability < ", 0.1, 1.0, 0.6, 0.05)
    if stability < min_stability:
        st.warning("Coach reps vary a lot (or you have few reps). We'll still coach in Quick Mode.")

    # Visualize envelope for selected joint
    fig_env = plt.figure()
    plt.plot(tt, coach_mean[:, j], label="coach mean")
    plt.fill_between(tt, ref_lo[:, j], ref_hi[:, j], alpha=0.18, label="coach envelope")
    plt.plot(tt, user_mean[:, j], label="user mean")
    plt.xlabel("normalized time (0→1)")
    plt.ylabel("degrees")
    plt.title(f"Coach envelope vs user mean — {plot_joint}")
    plt.legend()
    st.pyplot(fig_env)

    # Save envelope artifact
    env_path = DATA_PROCESSED / "reference_envelope_coach.npz"
    np.savez(
        env_path,
        angle_cols=np.array(angle_cols, dtype=object),
        coach_mean=coach_mean,
        ref_lo=ref_lo,
        ref_hi=ref_hi,
        stability=np.array([stability]),
        N=np.array([N]),
    )
    st.caption(f"Saved envelope: `{env_path}`")

    # ---------- Deviation Detection ----------
    st.divider()
    st.subheader("Deviation Detection (User reps vs Coach envelope)")

    mild = st.slider("Mild threshold (deg outside)", 1.0, 20.0, 5.0, 0.5)
    moderate = st.slider("Moderate threshold (deg outside)", 5.0, 40.0, 12.0, 0.5)
    severe = st.slider("Severe threshold (deg outside)", 10.0, 60.0, 20.0, 0.5)
    persistent_thresh = st.slider("Persistent if outside > (%)", 0, 100, 20, 5) / 100.0
    bottom_window = st.slider("Bottom window size (normalized %)", 2, 20, 8, 1) / 100.0

    driver_idx = angle_cols.index(coach_driver) if coach_driver in angle_cols else 0

    analyses = []
    for ridx, r in enumerate(user_reps):
        user_mat = resample_rep_angles(df_user, r, angle_cols, N=N)  # (N, J)

        idx_bottom = int(np.nanargmin(user_mat[:, driver_idx]))

        below = ref_lo - user_mat
        above = user_mat - ref_hi
        over = np.maximum(0.0, np.maximum(below, above))  # (N, J)

        max_over = np.nanmax(over, axis=0)
        persist = np.nanmean(over > 0, axis=0)

        focus_cols = focus_joints_from_driver(coach_driver, angle_cols)
        focus_idx = [angle_cols.index(c) for c in focus_cols]

        # rank only within focus joints
        ranked_focus = sorted(focus_idx, key=lambda jj: (max_over[jj], persist[jj]), reverse=True)
        top_idx = ranked_focus[:3]

        events = []
        for jj in top_idx:
            deg = float(max_over[jj]) if np.isfinite(max_over[jj]) else 0.0
            sev_label = severity_from_deg(deg, mild=mild, moderate=moderate, severe=severe)
            if sev_label == "none":
                continue

            idx_peak = int(np.nanargmax(over[:, jj]))
            phase = phase_from_t(idx_peak, idx_bottom, N, bottom_window_frac=bottom_window)

            diff_val = float(user_mat[idx_peak, jj] - coach_mean[idx_peak, jj])
            direction = "above_reference" if diff_val > 0 else "below_reference"

            events.append({
                "joint": angle_cols[jj],
                "phase": phase,
                "direction": direction,
                "deg_outside": deg,
                "persistence": float(persist[jj]),
                "severity": sev_label,
                "persistence_label": "persistent" if float(persist[jj]) >= persistent_thresh else "brief",
            })

        pose_conf = float(np.nanmean(df_user["conf"])) if "conf" in df_user.columns else 1.0
        rep_conf = float(np.clip(r.quality, 0.0, 1.0))
        confidence = float(0.5 * pose_conf + 0.5 * rep_conf)

        analyses.append({
            "rep_id": ridx,
            "confidence": confidence,
            "reference_stability": stability,
            "events": events
        })

    st.write(f"Analyzed **{len(analyses)}** user reps.")
    with st.expander("Preview analysis JSON (first 2 reps)"):
        st.json(analyses[:2])

    analysis_path = DATA_PROCESSED / "analysis.json"
    analysis_path.write_text(json.dumps(analyses, indent=2))
    st.caption(f"Saved analysis: `{analysis_path}`")

    # ---------- One-sentence coaching cue ----------
    st.divider()
    st.subheader("Coaching cue (constrained, no LLM yet)")

    min_conf_required = st.slider("Min confidence required", 0.0, 1.0, 0.6, 0.05)
    rep_pick = st.selectbox("Choose rep to coach", options=list(range(len(analyses))), index=0)

    chosen = analyses[rep_pick]

    # NEW toggle
    allow_quick = st.checkbox("Allow Quick Mode when stability is low", value=True)

    tip = make_tip_from_events(
        chosen["events"],
        stability=chosen["reference_stability"],
        confidence=chosen["confidence"],
        min_stability=min_stability,
        min_conf=min_conf_required,
        allow_quick_mode=allow_quick,
    )

    st.success(tip["one_sentence_tip"])
    with st.expander("Tip JSON"):
        st.json(tip)
