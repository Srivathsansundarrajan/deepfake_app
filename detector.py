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


# =====================================================
# LOAD MODEL 1 (Experiment 1: ImageNet)
# =====================================================
model1 = Phase2Model(mode="imagenet", phase1_ckpt=None, freeze_backbone=False)

if os.path.exists("models/exp1.pth"):
    ckpt1 = torch.load("models/exp1.pth", map_location=DEVICE, weights_only=False)
    model1.load_state_dict(ckpt1["model"] if "model" in ckpt1 else ckpt1, strict=False)
else:
    print("⚠️ models/exp1.pth not found. Using untrained Exp1 model.")

model1.to(DEVICE)
model1.eval()

# =====================================================
# LOAD MODEL 2 (Experiment 3: Phase1 Finetuned)
# =====================================================
model2 = Phase2Model(mode="phase1", phase1_ckpt=None, freeze_backbone=False)

if os.path.exists("models/best.pth"):
    ckpt2 = torch.load("models/best.pth", map_location=DEVICE, weights_only=False)
    model2.load_state_dict(ckpt2["model"] if "model" in ckpt2 else ckpt2, strict=False)
else:
    print("⚠️ models/best.pth not found. Using untrained Exp3 model.")

model2.to(DEVICE)
model2.eval()


# =====================================================
# RUN INFERENCE (CALLED BY STREAMLIT)
# =====================================================
def run_inference(video_path):

    # -------- preprocessing --------
    rgb, dct, mask, frames = preprocess_video(video_path)

    rgb = rgb.to(DEVICE)
    dct = dct.to(DEVICE)
    mask = mask.to(DEVICE)

    with torch.no_grad():
        out1 = model1(rgb, dct, mask)
        prob1 = torch.softmax(out1, dim=1)
        
        out2 = model2(rgb, dct, mask)
        prob2 = torch.softmax(out2, dim=1)

    # predictions
    pred1 = prob1.argmax(1).item()
    conf1 = prob1.max().item() * 100
    
    pred2 = prob2.argmax(1).item()
    conf2 = prob2.max().item() * 100

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