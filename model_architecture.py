import torch
import torch.nn as nn
import timm
from torch.nn.utils.rnn import pack_padded_sequence


# ===================================
# Temporal Attention for frequency
# ===================================
class FreqTemporalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1)
        )

    def forward(self, x, mask):
        # x: (B,T,dim)
        w = self.fc(x).squeeze(-1)
        w = w.masked_fill(mask == 0, -1e4)
        a = torch.softmax(w, dim=1).unsqueeze(-1)
        return (x * a).sum(1)


# ===================================
# Lightweight Frequency CNN
# ===================================
class LightweightFreqCNN(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.fc = nn.Linear(128, out_dim)
        self.dim = out_dim

    def forward(self, x):
        f = self.features(x).flatten(1)
        return self.fc(f)


# ===================================
# Full Frequency Stream
# ===================================
class FrequencyStream(nn.Module):
    def __init__(self, fusion_dim=1280, freq_dim=512):
        super().__init__()

        self.cnn = LightweightFreqCNN(freq_dim)
        self.temporal = FreqTemporalAttention(freq_dim)

        # projection to match spatial dim
        self.proj = nn.Linear(freq_dim, fusion_dim)
        self.dim = fusion_dim

    def forward(self, dct, mask):
        """
        dct: (B,T,1,H,W)
        mask: (B,T)
        """
        B, T, _, H, W = dct.shape

        qt = []
        for t in range(T):
            qt.append(self.cnn(dct[:, t]))

        qt = torch.stack(qt, dim=1)  # (B,T,512)

        ff = self.temporal(qt, mask)  # (B,512)
        ff = self.proj(ff)            # (B,1280)

        return ff

# ===================================
# RGB Backbone
# ===================================
class RGBBackbone(nn.Module):
    def __init__(self, mode="imagenet", phase1_ckpt=None, freeze=False):
        super().__init__()

        if mode == "imagenet":
            self.backbone = timm.create_model(
                "efficientnet_b0",
                pretrained=True,
                num_classes=0
            )

        elif mode == "phase1":
            self.backbone = timm.create_model(
                "efficientnet_b0",
                pretrained=False,
                num_classes=0
            )
            # during inference, the final model weight usually contains backbone weights already
            # so we might not need to load phase1_ckpt if the whole Phase2 model is loaded together
            if phase1_ckpt:
                self._load_phase1(phase1_ckpt)

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.dim = self.backbone.num_features

    def _load_phase1(self, ckpt_path):
        import os
        if not os.path.exists(ckpt_path):
            print(f"⚠️ Phase-1 checkpoint not found at {ckpt_path}, skipping.")
            return

        ckpt = torch.load(ckpt_path, map_location="cpu")

        # handle both formats
        if "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt

        bb = {}

        for k, v in state.items():
            if k.startswith("backbone."):
                bb[k.replace("backbone.", "")] = v
            else:
                bb[k] = v

        self.backbone.load_state_dict(bb, strict=False)
        print("✅ Phase-1 backbone loaded")

    def forward(self, x):
        return self.backbone(x)


# ===================================
# Temporal Encoder
# ===================================
class TemporalEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lstm = nn.LSTM(
            dim, dim//2,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x, mask):
        lengths = torch.clamp(mask.sum(1), min=1).long().detach().cpu()
        packed = pack_padded_sequence(
            x, lengths,
            batch_first=True,
            enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        return torch.cat([h[-2], h[-1]], dim=1)


# ===================================
# Gated Fusion (stabilized)
# ===================================
class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim * 3)
        self.gate = nn.Sequential(
            nn.Linear(dim*3, dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, 3)
        )

    def forward(self, s, t, f):
        cat = torch.cat([s, t, f], 1)
        cat = self.norm(cat)
        w = torch.softmax(self.gate(cat), dim=-1)
        return w[:,0:1]*s + w[:,1:2]*t + w[:,2:3]*f


# ===================================
# Phase-2 Model
# ===================================
class Phase2Model(nn.Module):
    def __init__(self, mode, phase1_ckpt=None, freeze_backbone=False):
        super().__init__()

        self.rgb_backbone = RGBBackbone(
            mode=mode,
            phase1_ckpt=phase1_ckpt,
            freeze=freeze_backbone
        )

        dim = self.rgb_backbone.dim

        self.temporal = TemporalEncoder(dim)
        self.frequency = FrequencyStream(fusion_dim=dim)

        self.fusion = GatedFusion(dim)

        self.stream_dropout = nn.Dropout(0.3)
        self.final_norm = nn.LayerNorm(dim)

        self.classifier = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(dim//2, 2)
        )

    def forward(self, rgb, dct, mask):

        B,T,_,H,W = rgb.shape

        # ===== RGB =====
        rgb_flat = rgb.reshape(B*T,3,H,W)

        feat = self.rgb_backbone(rgb_flat)
        feat = feat.view(B,T,-1)

        mask_f = mask.unsqueeze(-1).float()

        # masked spatial pooling
        spatial = (feat * mask_f).sum(1) / (mask_f.sum(1) + 1e-6)

        # temporal stream
        temporal = self.temporal(feat, mask)

        # frequency stream
        freq = self.frequency(dct, mask)

        # stream regularization
        spatial = self.stream_dropout(spatial)
        temporal = self.stream_dropout(temporal)
        freq = self.stream_dropout(freq)

        fused = self.fusion(spatial, temporal, freq)

        fused = self.final_norm(fused)

        out = self.classifier(fused)

        return out