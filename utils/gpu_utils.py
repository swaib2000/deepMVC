import torch

def set_gpu():
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")