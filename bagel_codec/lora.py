from __future__ import annotations
import re
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear expects nn.Linear")
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(1, self.r)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        self.bias = None
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)
        self.A = nn.Parameter(torch.zeros(self.r, self.in_features))
        self.B = nn.Parameter(torch.zeros(self.out_features, self.r))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora = F.linear(self.dropout(x), self.A)
        lora = F.linear(lora, self.B)
        return base + self.scaling * lora

def apply_lora_by_regex(root: nn.Module, patterns: List[str], r: int, alpha: float, dropout: float) -> List[str]:
    replaced = []
    regexes = [re.compile(p) for p in patterns]
    stack = [("", root)]
    modules = []
    while stack:
        prefix, mod = stack.pop()
        for name, child in mod.named_children():
            full = f"{prefix}.{name}" if prefix else name
            modules.append((mod, name, full, child))
            stack.append((full, child))
    for parent, child_name, full_name, child in modules:
        if isinstance(child, nn.Linear) and any(rgx.search(full_name) for rgx in regexes):
            setattr(parent, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
            replaced.append(full_name)
    return replaced
