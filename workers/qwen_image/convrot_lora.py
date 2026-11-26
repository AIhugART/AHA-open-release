"""LoRA for the ConvRot variants (plan D6).

peft can only wrap the layer types it knows, and ``ConvRotLinear`` is not one of them, so the adapters are applied here: a forward
hook adds ``scale * (x @ A.T) @ B.T`` to the output of every layer a LoRA names. ``x`` is the layer's own input, which is what a LoRA
was trained against (the rotation inside ``ConvRotLinear`` is internal to the layer). The same hook serves the plain ``nn.Linear``
layers of the transformer, so a LoRA that names one of them is not half applied.

Nothing is added while no adapter is active, so a job without LoRAs is bit-identical to a server that never loaded one.
The adapters are registered submodules (buffers), so ``enable_model_cpu_offload`` moves them together with the layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVE: dict[str, float] = {}  # adapter name -> scale, for the job that is running; empty = no LoRA at all


class _Slot(nn.Module):
    def __init__(self, name: str, a: torch.Tensor, b: torch.Tensor):
        super().__init__()
        self.adapter = name
        self.register_buffer("a", a, persistent=False)  # [rank, in_features]
        self.register_buffer("b", b, persistent=False)  # [out_features, rank], alpha / rank already applied by lora_state_for


def _key(name: str) -> str:
    return name.encode().hex()  # a ModuleDict key may not contain "."; adapter ids are free text


def _hook(module: nn.Module, args: tuple, output: torch.Tensor) -> torch.Tensor:
    if not ACTIVE:
        return output
    x = args[0]
    for slot in module.aha_lora.values():
        scale = ACTIVE.get(slot.adapter)
        if scale:
            output = output + F.linear(F.linear(x, slot.a), slot.b) * scale
    return output


def attach(model: nn.Module, name: str, state: dict[str, torch.Tensor]) -> int:
    """Register the adapter ``name`` from the dict ``lora_state_for`` returns. All of it or none of it: returns the layer count."""
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in state.items():
        module, _, rest = key.removeprefix("transformer.").rpartition(".lora_")
        pairs.setdefault(module, {})[rest.split(".")[0]] = tensor  # "A" or "B"
    modules = dict(model.named_modules())
    staged = []
    for module_name, parts in pairs.items():
        layer = modules.get(module_name)
        if layer is None or "A" not in parts or "B" not in parts:
            raise ValueError(f"this LoRA names {module_name!r}, which is missing or incomplete in this model")
        a, b = parts["A"], parts["B"]
        if a.shape[1] != layer.in_features or b.shape[0] != layer.out_features or a.shape[0] != b.shape[1]:
            raise ValueError(
                f"this LoRA does not fit {module_name!r}: A {tuple(a.shape)} and B {tuple(b.shape)} against "
                f"{layer.in_features} -> {layer.out_features}"
            )
        staged.append((layer, _Slot(name, a.to(torch.bfloat16).contiguous(), b.to(torch.bfloat16).contiguous())))
    for layer, slot in staged:  # only now, so a file that fails halfway leaves nothing behind
        if not hasattr(layer, "aha_lora"):
            layer.aha_lora = nn.ModuleDict()
            layer.register_forward_hook(_hook)
        layer.aha_lora[_key(name)] = slot
    return len(staged)


def delete(model: nn.Module, name: str) -> None:
    for module in model.modules():
        slots = getattr(module, "aha_lora", None)
        if slots is not None and _key(name) in slots:
            del slots[_key(name)]
    ACTIVE.pop(name, None)


def activate(loras: list[dict]) -> None:
    """Exactly these adapters, at these scales, for the next job."""
    ACTIVE.clear()
    ACTIVE.update({item["id"]: float(item["scale"]) for item in loras})
