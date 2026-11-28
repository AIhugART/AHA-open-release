"""Linear layers over the community ConvRot weights of MiniMax H3, computed with comfy-kitchen (Apache-2.0).

Storage, as read from the checkpoints themselves (docs/product/05_models.md section 2.0):
  * ``int8_tensorwise`` + ``convrot``: INT8 weight [N, K], one float32 scale per output row, the weight was rotated with a
    Hadamard transform over groups of 256 input features;
  * ``convrot_w4a4`` with ``linear_dtype: int8``: the same, but the weight holds two signed INT4 values per byte
    ([N, K/2], low nibble first) and runs through the INT8 matrix product;
  * ``nvfp4`` (not ConvRot): two FP4 (E2M1) values per byte ([N, K/2], **high** nibble first), one FP8 (E4M3) scale per 16
    inputs of a row, stored in cuBLAS' tiled block layout, and one float32 scale for the whole tensor. Checked layer by layer
    against the INT8 file of the same model: this layout is 9.5 % off (4-bit noise), every other one 44-146 %.
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

    def swapped(self, first: int) -> "ConvRotLinear":
        """Rows [first:] then rows [:first] (the SwiGLU halves)."""
        return ConvRotLinear(torch.cat([self.weight[first:], self.weight[:first]]),
                             torch.cat([self.scale[first:], self.scale[:first]]), self.packed4)

    def dense(self, dtype: torch.dtype) -> torch.Tensor:
        """The weight decoded to ``dtype`` (original basis), for a caller that reuses it over several input slices."""
        return dequantize(unpack_int4(self.weight) if self.packed4 else self.weight, self.scale, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.dense(x.dtype), self.bias)


NVFP4_BLOCK = 16  # inputs per FP8 block scale
NVFP4_ROWS = 2048  # rows decoded at a time: the int64 index into the pair table is 8 bytes per byte of weight
_FP4 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
_PAIRS: dict[torch.device, torch.Tensor] = {}


def _fp4_pairs(device: torch.device) -> torch.Tensor:
    """[256, 2]: the two FP4 values of every byte, high nibble first."""
    if device not in _PAIRS:
        values = torch.tensor(_FP4, device=device)
        byte = torch.arange(256, device=device)
        _PAIRS[device] = torch.stack([values[byte >> 4], values[byte & 15]], dim=1)
    return _PAIRS[device]


class Nvfp4Linear(nn.Module):
    """NVFP4 weights, decoded to the input's dtype before the product (the same route as ``ConvRotLinear``).

    Why not NVIDIA's FP4 kernels: they need compute capability 10 (Blackwell). Decoding runs on every NVIDIA GPU, and the
    activation is not quantized, so the result is closer to the unquantized model, not further.
    """

    def __init__(self, weight: torch.Tensor, block_scale: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None = None):
        super().__init__()
        rows, half = weight.shape
        self.out_features, self.in_features = rows, half * 2
        if self.in_features % NVFP4_BLOCK or tuple(block_scale.shape) != (rows, self.in_features // NVFP4_BLOCK):
            raise ValueError(f"NVFP4 weight {tuple(weight.shape)} does not match block scales {tuple(block_scale.shape)}")
        self.register_buffer("weight", weight, persistent=False)
        # Row-major (so rows can be sliced) and kept in FP8, one byte each: held as float32 these scales were 4 GiB of the
        # transformer's 16.5 GiB peak, half of it again on top of the 4-bit weights.
        self.register_buffer("block_scale", block_scale, persistent=False)
        self.register_buffer("scale", scale.float().reshape(()), persistent=False)
        self.register_buffer("bias", bias, persistent=False)

    @classmethod
    def from_file(cls, weight: torch.Tensor, tiled_block_scale: torch.Tensor, scale: torch.Tensor) -> "Nvfp4Linear":
        """``tiled_block_scale`` as the checkpoint stores it (FP8, cuBLAS block layout)."""
        from comfy_kitchen.float_utils import from_blocked

        rows, blocks = weight.shape[0], weight.shape[1] * 2 // NVFP4_BLOCK
        return cls(weight, from_blocked(tiled_block_scale, rows, blocks).contiguous(), scale)

    def rows(self, start: int, stop: int) -> "Nvfp4Linear":
        return Nvfp4Linear(self.weight[start:stop], self.block_scale[start:stop], self.scale)

    def swapped(self, first: int) -> "Nvfp4Linear":
        return Nvfp4Linear(torch.cat([self.weight[first:], self.weight[:first]]),
                           torch.cat([self.block_scale[first:], self.block_scale[:first]]), self.scale)

    def dense(self, dtype: torch.dtype) -> torch.Tensor:
        n, k = self.out_features, self.in_features
        table = _fp4_pairs(self.weight.device)
        out = torch.empty(n, k, dtype=dtype, device=self.weight.device)
        for i in range(0, n, NVFP4_ROWS):
            values = table[self.weight[i : i + NVFP4_ROWS].long()].reshape(-1, k // NVFP4_BLOCK, NVFP4_BLOCK)
            values = values * (self.block_scale[i : i + NVFP4_ROWS].float() * self.scale).unsqueeze(-1)
            out[i : i + NVFP4_ROWS] = values.reshape(-1, k).to(dtype)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.dense(x.dtype), self.bias)


QUANTIZED_LINEARS = (ConvRotLinear, Nvfp4Linear)


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
