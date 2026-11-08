"""PyTorch stand-in for the Windows q8_kernels CUDA extension.

The 12GB archive ships only .pyd binaries. FP8 per-channel matmul is
dequantized and run with torch.nn.functional.linear.
"""
import torch
import torch.nn.functional as F


def _dequant(x, scale):
    values = x.to(torch.float32)
    if scale is None:
        return values
    scale = scale.to(dtype=torch.float32, device=values.device)
    while scale.ndim < values.ndim:
        scale = scale.unsqueeze(-1)
    return values * scale


def fp8_linear(a, b, bias=None, scale_a=None, scale_b=None, use_hadamard=False, out_dtype=None):
    if use_hadamard:
        raise RuntimeError("hadamard FP8 matmul is not implemented in the PyTorch fallback")
    left = _dequant(a, scale_a)
    right = _dequant(b, scale_b)
    if not getattr(fp8_linear, "_logged", False):
        print(
            "FP8_FALLBACK",
            tuple(a.shape),
            tuple(b.shape),
            None if scale_a is None else tuple(scale_a.shape),
            None if scale_b is None else tuple(scale_b.shape),
            None if bias is None else tuple(bias.shape),
            flush=True,
        )
        fp8_linear._logged = True
    out = F.linear(left, right, None if bias is None else bias.to(torch.float32))
    if out_dtype is not None:
        out = out.to(out_dtype)
    return out
