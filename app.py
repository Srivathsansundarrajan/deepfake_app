import streamlit as st
import tempfile
import os

from detector import run_inference

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Deepfake FaceSwap Detection",
    page_icon="",
    layout="centered"
)

# ------------------ MAIN PAGE ------------------
st.title("AI-Based Deepfake FaceSwap Detection")
st.markdown("""
Upload a **video file** to analyze whether it contains manipulated or face-swapped content.
The system extracts face frames and runs them through our best-performing Deep Learning model.
""")

uploaded_video = st.file_uploader(
    "Upload a video file",
    type=["mp4", "avi", "mov"]
)

if uploaded_video:
    # Save uploaded video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        video_path = tmp.name

    st.video(video_path)

    if st.button("Run Deepfake Detection"):
        with st.spinner("Analyzing video..."):
            result = run_inference(video_path)

        st.success("Analysis complete!")

        # --- Display RGB Frames ---
        frames = result.get("frames", [])
        if frames:
            st.markdown("###  Extracted Face Frames")
            with st.expander("View Face Regions Sequence"):
                cols = st.columns(4)
                for i, frame in enumerate(frames):
                    cols[i % 4].image(frame, use_container_width=True, caption=f"Frame {i+1}")

        # --- Display DCT Frames ---
        dct_frames = result.get("dct_frames", [])
        if dct_frames:
            st.markdown("###DCT Frequency Heatmaps")
            with st.expander("View DCT Frequency Frames"):
                st.caption("These show the frequency-domain (DCT) representation of each face frame using an INFERNO colormap. Deepfake artifacts are often most visible in this view.")
                cols = st.columns(4)
                for i, dct_frame in enumerate(dct_frames):
                    cols[i % 4].image(dct_frame, use_container_width=True, caption=f"DCT {i+1}")
        
        # --- Display Predictions ---
        st.markdown("###  Model Predictions")

        model1 = result["Model_1"]
        model2 = result["Model_2"]

        col1, col2 = st.columns(2)

        # -------- Model 1 --------
        with col1:
            st.subheader("Model 1 (Exp 1)")
            if model1["label"] == "FAKE":
                st.error(f"FAKE (Confidence: {model1['confidence']:.2f}%)")
            else:
                st.success(f"REAL (Confidence: {model1['confidence']:.2f}%)")

        # -------- Model 2 --------
        with col2:
            st.subheader("Model 2 (Exp 3)")
            if model2["label"] == "FAKE":
                st.error(f"FAKE (Confidence: {model2['confidence']:.2f}%)")
            else:
                st.success(f"REAL (Confidence: {model2['confidence']:.2f}%)")

        # ✅ remove temp file
        os.remove(video_path)