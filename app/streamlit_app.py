import streamlit as st
import tempfile
from pathlib import Path
import base64

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

    # Try to infer type, default to mp4
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


# ---------- Upload UI ----------

col1, col2 = st.columns(2)

with col1:
    st.subheader("1) Coach Reference Video")
    coach_file = st.file_uploader(
        "Upload coach demo (.mp4/.mov)",
        type=["mp4", "mov", "m4v"],
        key="coach",
    )

with col2:
    st.subheader("2) User Attempt Video")
    user_file = st.file_uploader(
        "Upload user attempt (.mp4/.mov)",
        type=["mp4", "mov", "m4v"],
        key="user",
    )

coach_path = None
user_path = None

if coach_file:
    coach_path = save_upload(coach_file, Path("../data/raw"))

if user_file:
    user_path = save_upload(user_file, Path("../data/raw"))

# ---------- Previews (Smaller) ----------

st.divider()
st.subheader("Previews")

left, right = st.columns(2)

with left:
    st.markdown("**Coach Preview**")
    if coach_file:
        # Small preview
        video_small(coach_file, width_px=360)

        # Optional: expandable large view
        with st.expander("View coach video large"):
            st.video(coach_file)

        st.caption(f"Saved as: `{coach_path.name}`")
    else:
        st.info("Upload a coach reference video to preview it here.")

with right:
    st.markdown("**User Preview**")
    if user_file:
        # Small preview
        video_small(user_file, width_px=360)

        # Optional: expandable large view
        with st.expander("View user video large"):
            st.video(user_file)

        st.caption(f"Saved as: `{user_path.name}`")
    else:
        st.info("Upload a user attempt video to preview it here.")

# ---------- Action Button ----------

st.divider()
run = st.button(
    "Run analysis (coming next)",
    type="primary",
    disabled=not (coach_path and user_path),
)

if run:
    st.info("Next: we'll extract pose → compute angles → build reference envelope → score deviations.")
