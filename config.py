import torch

class Phase2Config:
    SEQ_LEN = 16

    IMG_SIZE = 224 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
