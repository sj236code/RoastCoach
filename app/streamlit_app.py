import streamlit as st
import tempfile
from pathlib import Path
import base64
import sys

import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="RoastCoach", layout="wide")

st.title("RoastCoach — Exercise-Agnostic Motion Fingerprinting")
st.caption("Upload a coach reference video, then a user attempt. We'll compare joint-angle trajectories.")


# ---------- Helpers ----------

def save_upload(uploaded_file, out_dir: Path) -> Path:
    """Save an uploaded file to disk and return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = Path(tmp.name)

    out_path = out_dir / uploaded_file.name
    out_path.write_bytes(tmp_path.read_bytes())
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


# ---------- Import backend pipeline (pose -> angles) ----------

# Assumes repo layout:
# roastcoach/
#   app/streamlit_app.py
#   backend/src/pose.py
#   backend/src/angles.py
BACKEND_SRC = Path(__file__).resolve().parents[1] / "backend" / "src"
sys.path.append(str(BACKEND_SRC))

try:
    from pose import extract_pose_frames
    from angles import compute_angles
except Exception as e:
    st.error("Could not import backend modules. Check your folder structure and filenames.")
    st.exception(e)
    st.stop()


def angles_to_df(angle_frames) -> pd.DataFrame:
    rows = []
    for f in angle_frames:
        row = {"t": f.t, "conf": f.conf}
        row.update(f.angles)
        rows.append(row)
    return pd.DataFrame(rows)


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
    coach_path = save_upload(coach_file, Path("../data/raw"))

if user_file:
    user_path = save_upload(user_file, Path("../data/raw"))

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

    processed_dir = Path("../data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    with st.spinner("Extracting pose (coach)..."):
        coach_frames = extract_pose_frames(str(coach_path))

    with st.spinner("Extracting pose (user)..."):
        user_frames = extract_pose_frames(str(user_path))

    with st.spinner("Computing joint angles..."):
        coach_angles = compute_angles(coach_frames)
        user_angles = compute_angles(user_frames)

    df_coach = angles_to_df(coach_angles)
    df_user = angles_to_df(user_angles)

    coach_csv = processed_dir / "angles_coach.csv"
    user_csv = processed_dir / "angles_user.csv"
    df_coach.to_csv(coach_csv, index=False)
    df_user.to_csv(user_csv, index=False)

    st.success("Saved angle CSVs.")
    st.write(f"Coach: `{coach_csv}`")
    st.write(f"User: `{user_csv}`")

    # Plot a default angle for sanity
    target = "left_knee_flex"

    # If it doesn't exist, choose the first angle column that exists in both
    angle_cols_coach = [c for c in df_coach.columns if c not in ("t", "conf")]
    angle_cols_user = [c for c in df_user.columns if c not in ("t", "conf")]
    shared = [c for c in angle_cols_coach if c in angle_cols_user]

    if target not in shared and shared:
        target = shared[0]

    if not shared:
        st.warning("No angle columns found. This usually means pose landmarks were not detected reliably.")
        st.write("Try a clearer video (full body visible, good lighting, minimal occlusion).")
    else:
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
