import torch
import numpy as np


def cos(y, i, j):
    if isinstance(y, torch.cuda.Tensor):
        y = y.cpu().numpy()
        y = np.mat(y)
    num = y[i] * y[j].T
    denom = np.linalg.norm(y[i])*np.linalg.norm(y[j])
    return num /denom
