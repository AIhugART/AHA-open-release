"""Linear layers over the community ConvRot weights of MiniMax H3, computed with comfy-kitchen (Apache-2.0).

Storage, as read from the checkpoints themselves (docs/05_models.md section 2.0):
  * ``int8_tensorwise`` + ``convrot``: INT8 weight [N, K], one float32 scale per output row, the weight was rotated with a
    Hadamard transform over groups of 256 input features;
  * ``convrot_w4a4`` with ``linear_dtype: int8``: the same, but the weight holds two signed INT4 values per byte
    ([N, K/2], low nibble first) and runs through the INT8 matrix product.
``comfy_kitchen.int8_linear(..., convrot=True)`` rotates the activation the same way, quantizes it per row to INT8, runs the
integer product and applies both scales, so the math here is "unpack, then call it".
"""

from __future__ import annotations

import torch
import torch.nn as nn
import comfy_kitchen as ck

from lenient_safetensors import SafetensorsFile

CONVROT_GROUP = 256
# The integer product of an INT8 [M, K] by [K, N] makes an int32 [M, N]: 4 bytes per element. A 37k-token video row block
# against the 28672-wide feed-forward would be 4 GB, so long inputs go through in slices.
CHUNK_ROWS = 4096


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Two signed 4-bit values per byte (low nibble first) -> INT8 [.., 2 * K]. Checked against the INT8 file's weights."""
    p = packed.to(torch.int16) & 0xFF
    low, high = p & 0xF, (p >> 4) & 0xF
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    return torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], -1).to(torch.int8)


def unpack_int4_fast(packed: torch.Tensor) -> torch.Tensor:
    """Same result as ``unpack_int4`` with int8 shifts only (a left shift puts the low nibble in the sign position, an arithmetic
    right shift brings it back signed); no int16 or ``where`` intermediates, so it costs about one pass over the weight."""
    out = torch.empty(*packed.shape[:-1], packed.shape[-1] * 2, dtype=torch.int8, device=packed.device)
    out[..., 0::2] = (packed << 4) >> 4
    out[..., 1::2] = packed >> 4
    return out


def int8_tensor_cores(device: torch.device) -> bool:
    """Turing (7.5) and newer have INT8 tensor cores; ``torch._int_mm`` needs them to be worth using."""
    return torch.cuda.get_device_capability(device) >= (7, 5)


def convrot_matmul(
    x: torch.Tensor, weight_i8: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None, group: int = CONVROT_GROUP
) -> torch.Tensor:
    """The INT8 product (activation rotated and quantized per row). Kept as the reference the faster path is tested against."""
    lead = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1])
    parts = [
        ck.int8_linear(flat[i : i + CHUNK_ROWS], weight_i8, scale, bias, out_dtype=x.dtype, convrot=True, convrot_groupsize=group)
        for i in range(0, flat.shape[0], CHUNK_ROWS)
    ]
    return torch.cat(parts, dim=0).reshape(*lead, weight_i8.shape[0])


_HADAMARD: dict[tuple[int, torch.device], torch.Tensor] = {}
DEQUANT_ROWS = 8192  # rows dequantized at a time: bounds the float32 intermediate (8192 x 28672 x 4 B would be 0.9 GB)


def hadamard(size: int, device: torch.device) -> torch.Tensor:
    """The normalised regular Hadamard matrix of a power-of-4 size (Kronecker powers of the 4x4 one). Symmetric and its own inverse."""
    key = (size, device)
    if key not in _HADAMARD:
        h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32, device=device)
        h = h4
        while h.shape[0] < size:
            h = torch.kron(h, h4)
        if h.shape[0] != size:
            raise ValueError(f"{size} is not a power of 4")
        _HADAMARD[key] = h / size**0.5
    return _HADAMARD[key]


def dequantize(weight_i8: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype, group: int = CONVROT_GROUP) -> torch.Tensor:
    """INT8 [N, K] + per-row scale, rotated per group of K -> the weight in its original basis, in ``dtype``.

    The file stores ``W @ H`` (H symmetric, H @ H = I), so ``W = stored @ H``. Done in float32 by blocks of rows.
    """
    n, k = weight_i8.shape
    h = hadamard(group, weight_i8.device)
    out = torch.empty(n, k, dtype=dtype, device=weight_i8.device)
    for i in range(0, n, DEQUANT_ROWS):
        block = weight_i8[i : i + DEQUANT_ROWS].float() * scale[i : i + DEQUANT_ROWS].float().reshape(-1, 1)
        out[i : i + DEQUANT_ROWS] = (block.view(-1, k // group, group) @ h).view(-1, k).to(dtype)
    return out


def dequantized_linear(x: torch.Tensor, weight_i8: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """``x @ W.T`` with W decoded to x's dtype just before the product.

    Why not the INT8 product: on GPUs without comfy-kitchen's cuBLASLt kernels (the RTX 3090 Ti, and ComfyUI's own build)
    ``int8_linear`` rotates and quantizes the activation with separate kernels on every call, which measured slower than a
    bf16 product of the same shape (block 0, 37k tokens: 545 ms of INT8 linears + ~600 ms of elementwise work, against
    343 ms of bf16 products). Decoding the weight also skips quantizing the activation, so the result is closer to the
    unquantized model, not further.
    """
    return torch.nn.functional.linear(x, dequantize(weight_i8, scale, x.dtype), bias)


class ConvRotLinear(nn.Module):
    """Weights resident on the device (the transformer). ``rows`` makes a view of a row range without copying."""

    def __init__(self, weight: torch.Tensor, scale: torch.Tensor, packed4: bool, bias: torch.Tensor | None = None):
        super().__init__()
        self.packed4 = packed4
        self.out_features = weight.shape[0]
        self.in_features = weight.shape[1] * (2 if packed4 else 1)
        if self.in_features % CONVROT_GROUP:
            raise ValueError(f"in_features {self.in_features} is not a multiple of {CONVROT_GROUP}")
        scale = scale.reshape(-1)
        if scale.numel() != self.out_features:
            raise ValueError(f"expected one scale per output row ({self.out_features}), got {scale.numel()}")
        self.register_buffer("weight", weight, persistent=False)
        self.register_buffer("scale", scale.float(), persistent=False)
        self.register_buffer("bias", bias, persistent=False)

    def rows(self, start: int, stop: int) -> "ConvRotLinear":
        return ConvRotLinear(self.weight[start:stop], self.scale[start:stop], self.packed4)

    def dense(self, dtype: torch.dtype) -> torch.Tensor:
        """The weight decoded to ``dtype`` (original basis), for a caller that reuses it over several input slices."""
        return dequantize(unpack_int4(self.weight) if self.packed4 else self.weight, self.scale, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x @ W.T`` by rotating the ACTIVATION instead of un-rotating the weight (plan Q0).

        The file stores ``S = W @ H`` with H symmetric and ``H @ H = I``, so ``x @ W.T = (x @ H) @ S.T``. Rotating x costs
        ``2*M*K*256`` flops against the product's ``2*M*K*N``, while un-rotating the weight cost ``2*N*K*256`` on every call
        (measured: 66 s instead of ~40 s per 1024 px image). H has entries +-1/16, exact in bfloat16.

        On a GPU with INT8 tensor cores the product itself is INT8 (the checkpoints were made for it: the activation is
        quantized per row after the rotation); otherwise the weight is decoded to x's dtype and the product runs there.
        """
        k = x.shape[-1]
        h = hadamard(CONVROT_GROUP, x.device).to(x.dtype)
        rotated = (x.reshape(-1, k // CONVROT_GROUP, CONVROT_GROUP) @ h).reshape(-1, k)
        weight = unpack_int4_fast(self.weight) if self.packed4 else self.weight
        if x.is_cuda and rotated.shape[0] > 16 and int8_tensor_cores(x.device):
            out = torch.cat([self._int8_product(rotated[i : i + CHUNK_ROWS], weight, x.dtype) for i in range(0, rotated.shape[0], CHUNK_ROWS)])
            if self.bias is not None:
                out = out + self.bias
        else:
            w = (weight.float() * self.scale.reshape(-1, 1)).to(x.dtype)
            out = torch.nn.functional.linear(rotated, w, self.bias)
        return out.reshape(*x.shape[:-1], self.out_features)

    def _int8_product(self, rotated: torch.Tensor, weight: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Per-row INT8 activation x INT8 weight through cuBLASLt. ``weight.t()`` is passed as a view: cuBLASLt wants the
        right operand column-major, and a contiguous [K, N] copy measured 3.4x slower (5.9 ms against 1.7 ms, 4608x4096x12288)."""
        a_scale = rotated.abs().amax(dim=-1, keepdim=True).float().clamp_min(1e-8) / 127.0
        q = (rotated.float() / a_scale).round_().clamp_(-127, 127).to(torch.int8)
        y = torch._int_mm(q, weight.t())
        return (y.float() * a_scale * self.scale.reshape(1, -1)).to(dtype)


class StreamedConvRotLinear(nn.Module):
    """Same math, weights stay in the file and are read when called, so a 27 GB encoder never sits in RAM or on the GPU."""

    def __init__(self, file: SafetensorsFile, prefix: str, device: torch.device | str):
        super().__init__()
        meta = file.get_json(prefix + ".comfy_quant")
        if meta.get("format") != "int8_tensorwise" or not meta.get("convrot"):
            raise ValueError(f"{prefix}: unsupported quantization {meta}")
        self._file, self._prefix, self._device = file, prefix, device
        out_features, in_features = file.shape(prefix + ".weight")
        self.out_features, self.in_features = out_features, in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._file.get(self._prefix + ".weight").to(self._device)
        s = self._file.get(self._prefix + ".weight_scale").to(self._device).reshape(-1)
        return dequantized_linear(x, w, s, None)
