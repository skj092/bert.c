import torch
import numpy as np
from pathlib import Path
path = Path('/home/sonu/code/bert.c')

eps = 1e-5


class LayerNorm:

    @staticmethod
    def forward(x, w, b):
        B, T, C = x.shape
        mean = x.sum(-1, keepdim=True) / C  # B,T,1
        xshift = x - mean  # B,T,C
        var = (xshift**2).sum(-1, keepdim=True) / C  # B,T,1
        rstd = (var + eps) ** -0.5  # B,T,1
        norm = xshift * rstd  # B,T,C
        out = norm * w + b  # B,T,C

        cache = (x, w, mean, rstd)
        return out, cache


# create a small dummy example and check w.r.t PyTorch forward
B = 2
T = 128
C = 768
x_path = path/'bins/combined_embeddings.bin'
b_path = path/'bins/ln_b.bin'
w_path = path/'bins/ln_w.bin'
assert x_path.exists()
assert b_path.exists()
assert w_path.exists()
b = torch.from_numpy(np.fromfile(b_path, dtype=np.float32))
w = torch.from_numpy(np.fromfile(w_path, dtype=np.float32))
x = torch.from_numpy(np.fromfile(x_path, dtype=np.float32)).view(B, T, C)

out, cache = LayerNorm.forward(x, w, b)

# PyTorch LayerNorm
ln = torch.nn.LayerNorm(C, elementwise_affine=True)
ln.weight.data = w.clone()
ln.bias.data = b.clone()
out_torch = ln(x)

assert torch.allclose(out, out_torch, atol=1e-5)
