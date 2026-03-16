import cv2
import torch
import numpy as np
from scipy.fftpack import dct
from config import Phase2Config as C
from insightface.app import FaceAnalysis

# ============================
# ImageNet normalization
# ============================
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3,1,1)

# ================= FACE DETECTOR =================
import os
import gdown
import zipfile

def download_insightface_model():
    model_dir = os.path.expanduser("~/.insightface/models/buffalo_l")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

        print("Downloading InsightFace buffalo_l models...")

        url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
        output = "buffalo_l.zip"

        gdown.download(url, output, quiet=False)

        with zipfile.ZipFile(output, "r") as zip_ref:
            zip_ref.extractall(model_dir)

        if os.path.exists(output):
            os.remove(output)

download_insightface_model()

class FastFaceDetector:
    def __init__(self):
        try:
            self.app = FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"]
            )
        except TypeError:
            # Fallback for older versions like 0.2.1 on local machine
            self.app = FaceAnalysis(
                name="buffalo_l"
            )
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

        # CPU warm-up (prevents first-video lag)
        _ = self.app.get(np.zeros((640, 640, 3), dtype=np.uint8))

    def detect(self, img):
        faces = self.app.get(img)
        dets = []
        for f in faces:
            x1, y1, x2, y2 = map(int, f.bbox)
            dets.append({
                "box": (x1, y1, x2 - x1, y2 - y1),
                "score": float(f.det_score)
            })
        return dets

import streamlit as st

@st.cache_resource
def get_face_detector():
    return FastFaceDetector()

# ============================
# Fast block DCT
# ============================
def dct2(x):
    return dct(dct(x.T, norm="ortho").T, norm="ortho")

def extract_dct_y(frame):
    y = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)[:, :, 0]
    y = y.astype(np.float32) / 255.0

    out = np.zeros_like(y)

    for i in range(0, C.IMG_SIZE, 8):
        for j in range(0, C.IMG_SIZE, 8):
            out[i:i+8, j:j+8] = dct2(y[i:i+8, j:j+8])

    out = (out - out.mean()) / (out.std() + 1e-6)

    return out.astype(np.float32)

# ================= EXTRACT FRAMES =================
def sample_indices(total, n):
    if total <= n:
        return list(range(total))
    return np.linspace(0, total - 1, n, dtype=int).tolist()

def extract_faces(video_path, T=16):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Failed to open video")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_indices(total, T)
    crops = [None] * T

    for i, frame_id in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            continue

        detector = get_face_detector()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = detector.detect(rgb)

        if dets:
            best = max(dets, key=lambda x: x["score"])
            x, y, w, h = best["box"]
            face = rgb[max(0, y):y + h, max(0, x):x + w]
            if face.size != 0:
                crops[i] = cv2.resize(face, (C.IMG_SIZE, C.IMG_SIZE))

    cap.release()

    valid_faces = [c for c in crops if c is not None]
    if len(valid_faces) == 0:
        raise Exception("No face detected in the video")

    # Fill missing frames gracefully
    last_valid = valid_faces[-1]
    for i in range(len(crops)):
        if crops[i] is None:
            crops[i] = last_valid

    return crops

# ================= MAIN PREPROCESS =================
def preprocess_video(video_path):
    frames = extract_faces(video_path, T=C.SEQ_LEN)
    frames = frames[:C.SEQ_LEN]

    rgb_tensor = torch.zeros(C.SEQ_LEN, 3, C.IMG_SIZE, C.IMG_SIZE)
    dct_tensor = torch.zeros(C.SEQ_LEN, 1, C.IMG_SIZE, C.IMG_SIZE)
    mask = torch.zeros(C.SEQ_LEN, dtype=torch.bool)
    dct_frames = []  # Displayable DCT images (uint8, RGB)

    for i, frame in enumerate(frames):
        # ===== RGB =====
        rgb = torch.as_tensor(frame).permute(2,0,1).float().div_(255.0)
        rgb.sub_(IMAGENET_MEAN).div_(IMAGENET_STD)
        rgb_tensor[i] = rgb

        # ===== DCT =====
        dct_map = extract_dct_y(frame)
        dct_tensor[i].copy_(torch.from_numpy(dct_map)).unsqueeze_(0)

        # ===== DCT Visualization =====
        # Normalize the DCT map to 0-255 for display
        dct_vis = dct_map.copy()
        dct_vis -= dct_vis.min()
        dct_vis /= (dct_vis.max() + 1e-6)
        dct_vis = (dct_vis * 255).astype(np.uint8)
        # Apply INFERNO colormap to make it visually meaningful (BGR -> RGB)
        dct_color = cv2.applyColorMap(dct_vis, cv2.COLORMAP_INFERNO)
        dct_frames.append(cv2.cvtColor(dct_color, cv2.COLOR_BGR2RGB))

        mask[i] = True

    # Expected shape by model: (Batch, Seq_len, ...)
    # Add batch dimension
    rgb_tensor = rgb_tensor.unsqueeze(0)
    dct_tensor = dct_tensor.unsqueeze(0)
    mask = mask.unsqueeze(0)

    return rgb_tensor, dct_tensor, mask, frames, dct_frames