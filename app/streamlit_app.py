import streamlit as st
import tempfile
from pathlib import Path
import base64
import sys
import uuid

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
    """
    Render a smaller, size-controlled video preview using an HTML <video> tag.
    This avoids Streamlit's default behavior where st.video can be very large.
    """
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


# ---------- Import backend pipeline (pose -> angles -> segmentation) ----------

# Assumes repo layout:
# RoastCoach/
#   app/streamlit_app.py
#   backend/src/pose.py
#   backend/src/angles.py
#   backend/src/segment.py
BACKEND_SRC = ROOT / "backend" / "src"
sys.path.append(str(BACKEND_SRC))

try:
    from pose import extract_pose_frames
    from angles import compute_angles
    from segment import choose_driver_column, find_reps_from_driver, resample_rep_angles
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

coach_path = None
user_path = None

if coach_file:
    coach_path = save_upload(coach_file, DATA_RAW, prefix="coach")

if user_file:
    user_path = save_upload(user_file, DATA_RAW, prefix="user")


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
run = st.button(
    "Run analysis",
    type="primary",
    disabled=not (coach_path and user_path),
)

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

    # Save CSVs (robust absolute paths)
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
    shared = [c for c in angle_cols_coach if c in angle_cols_user]

    if target not in shared and shared:
        target = shared[0]

    if not shared:
        st.warning("No angle columns found. This usually means pose landmarks were not detected reliably.")
        st.write("Try a clearer video (full body visible, good lighting, minimal occlusion).")
        st.stop()

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

    with st.expander("Show data preview"):
        st.write("Coach angles (head):")
        st.dataframe(df_coach.head(10), use_container_width=True)
        st.write("User angles (head):")
        st.dataframe(df_user.head(10), use_container_width=True)

    # ---------- Rep segmentation ----------
    st.divider()
    st.subheader("Rep Segmentation")

    # Choose driver angle automatically per clip
    coach_driver = choose_driver_column(df_coach)
    user_driver = choose_driver_column(df_user)

    st.write(f"Coach driver: `{coach_driver}`")
    st.write(f"User driver: `{user_driver}`")

    # Segment
    coach_reps, coach_driver_smooth = find_reps_from_driver(df_coach, coach_driver)
    user_reps, user_driver_smooth = find_reps_from_driver(df_user, user_driver)

    st.write(f"Detected coach reps: **{len(coach_reps)}**")
    st.write(f"Detected user reps: **{len(user_reps)}**")

    # Plot driver with rep boundaries for debugging
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
        st.pyplot(plot_driver_with_reps(
            df_coach, coach_driver, coach_driver_smooth, coach_reps,
            "Coach driver + rep boundaries"
        ))
    with c2:
        st.pyplot(plot_driver_with_reps(
            df_user, user_driver, user_driver_smooth, user_reps,
            "User driver + rep boundaries"
        ))

    # Resample first rep as proof-of-normalization
    if coach_reps and user_reps:
        st.subheader("Rep Normalization (Resampling)")

        angle_cols = [c for c in df_coach.columns if c not in ("t", "conf")]
        N = 100

        coach_rep0 = resample_rep_angles(df_coach, coach_reps[0], angle_cols, N=N)
        user_rep0 = resample_rep_angles(df_user, user_reps[0], angle_cols, N=N)

        st.success(f"Resampled first rep to {N} timesteps: coach={coach_rep0.shape}, user={user_rep0.shape}")

        # Plot same target if possible
        plot_joint = target if target in angle_cols else angle_cols[0]
        j = angle_cols.index(plot_joint)

        fig2 = plt.figure()
        plt.plot(np.linspace(0, 1, N), coach_rep0[:, j], label="coach rep0")
        plt.plot(np.linspace(0, 1, N), user_rep0[:, j], label="user rep0")
        plt.xlabel("normalized time (0→1)")
        plt.ylabel("degrees")
        plt.title(f"Resampled rep (N={N}) — {plot_joint}")
        plt.legend()
        st.pyplot(fig2)

        with st.expander("Show resampled arrays (first 10 rows)"):
            st.write("Angle columns:", angle_cols)
            st.write("Coach rep0 (first 10):")
            st.dataframe(pd.DataFrame(coach_rep0[:10], columns=angle_cols), use_container_width=True)
            st.write("User rep0 (first 10):")
            st.dataframe(pd.DataFrame(user_rep0[:10], columns=angle_cols), use_container_width=True)
    else:
        st.warning("No reps detected in one or both videos. Try adjusting segmentation params or using a clearer, more repetitive motion.")
        st.caption("Next tuning knobs: smooth_window, min_rep_seconds, max_rep_seconds in find_reps_from_driver().")
