import math
from typing import Iterable

def pack_bits(bitstrings: Iterable[str]) -> bytes:
    # Pack '0'/'1' strings into bytes (MSB first).
    bits = "".join(bitstrings)
    pad = (-len(bits)) % 8
    if pad:
        bits += "0" * pad
    out = bytearray()
    for i in range(0, len(bits), 8):
        out.append(int(bits[i:i+8], 2))
    return bytes([pad]) + bytes(out)

def unpack_bits(data: bytes) -> str:
    if len(data) == 0:
        return ""
    pad = data[0]
    bits = "".join(f"{b:08b}" for b in data[1:])
    if pad:
        bits = bits[:-pad]
    return bits

def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b
