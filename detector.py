import torch
import os

from preprocess import preprocess_video
from model_architecture import Phase2Model
from config import Phase2Config as C

DEVICE = torch.device(C.DEVICE)

import gdown

# =====================================================
# DOWNLOAD MODELS FROM GOOGLE DRIVE IF MISSING
# =====================================================
# REPLACE 'YOUR_FILE_ID_HERE' WITH YOUR ACTUAL SHARE LINKS FROM GOOGLE DRIVE
GDRIVE_IDS = {
    "exp1.pth": "1JHUXVDQl7I0h07kLlUpLJEU1afpnwrT6",
    "best.pth": "1Hj_6viDlZNGQCdUXnBB1Si4lONCzdCJH"
}

os.makedirs("models", exist_ok=True)

for model_name, file_id in GDRIVE_IDS.items():
    model_path = f"models/{model_name}"
    if not os.path.exists(model_path):
        print(f"Downloading {model_name} from Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)


import gc

def load_model(ckpt_name, mode):
    model = Phase2Model(mode=mode, phase1_ckpt=None, freeze_backbone=False)
    model_path = f"models/{ckpt_name}"
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    else:
        print(f"⚠️ {model_path} not found. Using untrained {mode} model.")
    model.to(DEVICE)
    model.eval()
    return model

# =====================================================
# RUN INFERENCE (CALLED BY STREAMLIT)
# =====================================================
def run_inference(video_path):

    # -------- preprocessing --------
    # Runs the InsightFace models (uses ~300MB RAM) via @st.cache_resource
    rgb, dct, mask, frames = preprocess_video(video_path)

    rgb = rgb.to(DEVICE)
    dct = dct.to(DEVICE)
    mask = mask.to(DEVICE)

    # We load PyTorch models sequentially to prevent Streamlit Cloud Out-Of-Memory (OOM) crashes!
    # A single model takes ~260MB. Loading both concurrently alongside InsightFace pushes past the 1GB limit.
    with torch.no_grad():
        # -------- Model 1 --------
        model1 = load_model("exp1.pth", "imagenet")
        out1 = model1(rgb, dct, mask)
        prob1 = torch.softmax(out1, dim=1)
        pred1 = prob1.argmax(1).item()
        conf1 = prob1.max().item() * 100
        
        # Free memory immediately before loading the second model!
        del model1
        gc.collect()
        
        # -------- Model 2 --------
        model2 = load_model("best.pth", "phase1")
        out2 = model2(rgb, dct, mask)
        prob2 = torch.softmax(out2, dim=1)
        pred2 = prob2.argmax(1).item()
        conf2 = prob2.max().item() * 100
        
        del model2
        gc.collect()

    labels = ["REAL", "FAKE"]

    return {
        "Model_1": {
            "label": labels[pred1],
            "confidence": conf1
        },
        "Model_2": {
            "label": labels[pred2],
            "confidence": conf2
        },
        "label": labels[pred2],        # main display (using Exp3 as primary)
        "confidence": conf2,
        "frames": frames               # pass raw frames back to app.py
    }