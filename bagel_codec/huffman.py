from __future__ import annotations
import heapq
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from .utils import pack_bits, unpack_bits

@dataclass
class _Node:
    w: float
    sym: Optional[int] = None
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None

def build_huffman_code(probs: List[float]) -> Dict[int, str]:
    heap: List[Tuple[float, int, _Node]] = []
    uid = 0
    for s, p in enumerate(probs):
        w = float(p) if p > 0 else 1e-12
        heapq.heappush(heap, (w, uid, _Node(w=w, sym=s)))
        uid += 1
    if len(heap) == 1:
        return {0: "0"}
    while len(heap) > 1:
        w1, _, n1 = heapq.heappop(heap)
        w2, _, n2 = heapq.heappop(heap)
        parent = _Node(w=w1+w2, left=n1, right=n2)
        heapq.heappush(heap, (parent.w, uid, parent))
        uid += 1
    root = heap[0][2]
    code: Dict[int, str] = {}
    def dfs(n: _Node, prefix: str):
        if n.sym is not None:
            code[n.sym] = prefix or "0"
            return
        assert n.left is not None and n.right is not None
        dfs(n.left, prefix + "0")
        dfs(n.right, prefix + "1")
    dfs(root, "")
    return code

def encode_symbols(symbols: List[int], code: Dict[int, str]) -> bytes:
    return pack_bits(code[s] for s in symbols)

def decode_symbols(data: bytes, code: Dict[int, str], n_symbols: int) -> List[int]:
    inv: Dict[str, int] = {v: k for k, v in code.items()}
    bits = unpack_bits(data)
    out: List[int] = []
    buf = ""
    for b in bits:
        buf += b
        if buf in inv:
            out.append(inv[buf])
            buf = ""
            if len(out) == n_symbols:
                break
    if len(out) != n_symbols:
        raise ValueError(f"Huffman decode failed: expected {n_symbols}, got {len(out)}")
    return out
