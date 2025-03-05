import torch

eps = 1e-5


class LayerNorm:

    @staticmethod
    def forward(x, w, b):
        B, T, C = x.size()
        mean = x.sum(-1, keepdim=True) / C  # B,T,1
        xshift = x - mean  # B,T,C
        var = (xshift**2).sum(-1, keepdim=True) / C  # B,T,1
        rstd = (var + eps) ** -0.5  # B,T,1
        norm = xshift * rstd  # B,T,C
        out = norm * w + b  # B,T,C

        cache = (x, w, mean, rstd)
        return out, cache


# create a small dummy example and check w.r.t PyTorch backward
B = 2
T = 3
C = 4
x = torch.randn(B, T, C, requires_grad=True)
w = torch.randn(C, requires_grad=True)
b = torch.randn(C, requires_grad=True)
out, cache = LayerNorm.forward(x, w, b)

# PyTorch LayerNorm
ln = torch.nn.LayerNorm(C, elementwise_affine=True)
ln.weight.data = w.clone()
ln.bias.data = b.clone()
out_torch = ln(x)

# Check forward pass
print("Forward pass error custom vs pytorch:", (out - out_torch).abs().max().item())


dout = torch.randn(B, T, C)

# compare to PyTorch autograd
fakeloss = (out * dout).sum()
fakeloss.backward()

# for reference checking in C also
x, w, mean, rstd = cache


def write(tensor, handle):
    handle.write(tensor.detach().numpy().astype("float32").tobytes())


# Write to file
with open('ln.bin', 'wb') as file:
    write(x, file)  # (B, T, C)
    write(w, file)  # (C, )
    write(b, file)  # (C, )
    write(out, file)  # (B, T, C)
    write(mean, file)  # (B, T)
    write(rstd, file)  # (B, T)
