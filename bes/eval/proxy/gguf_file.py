"""Minimal GGUF (version 2/3) reader and writer, enough to inspect and rewrite opponent files.

GGUF layout (little endian): magic ``b"GGUF"``, ``uint32`` version, ``uint64`` tensor count,
``uint64`` key/value count, the key/value pairs (key string, ``uint32`` value type, value), the
tensor descriptors (name, ``uint32`` number of dims, ``uint64`` dims in ggml order ``ne0, ne1,
...``, ``uint32`` ggml type, ``uint64`` offset relative to the data section), padding to
``general.alignment`` (default 32) and the tensor data. Strings are ``uint64`` length + UTF-8
bytes; arrays are ``uint32`` element type + ``uint64`` count + elements.

Only the ggml types listed in ``GGML_BLOCK`` can be rewritten (their byte size must be known);
reading the header works for any file. Nothing here imports llama.cpp.
"""
from __future__ import annotations

import mmap
import os
import struct
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32

# value type codes
UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64 = range(13)
_SCALAR_FMT = {UINT8: "B", INT8: "b", UINT16: "H", INT16: "h", UINT32: "I", INT32: "i",
               FLOAT32: "f", BOOL: "?", UINT64: "Q", INT64: "q", FLOAT64: "d"}

# ggml tensor types: code -> (elements per block, bytes per block)
GGML_F32, GGML_F16, GGML_Q8_0, GGML_BF16 = 0, 1, 8, 30
GGML_BLOCK: dict[int, tuple[int, int]] = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34),
    9: (32, 36), 10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210),
    15: (256, 292), 30: (1, 2),
}


@dataclass
class Value:
    """One metadata value with its GGUF type (``subtype`` is the element type of an array)."""

    vtype: int
    value: Any
    subtype: int | None = None


@dataclass
class TensorInfo:
    name: str
    dims: tuple[int, ...]   # ggml order: dims[0] = ne0 (contiguous), dims[1] = ne1 (rows), ...
    ggml_type: int
    offset: int = 0         # relative to the start of the data section

    @property
    def n_elements(self) -> int:
        return int(np.prod(self.dims, dtype=np.int64)) if self.dims else 1

    @property
    def nbytes(self) -> int:
        if self.ggml_type not in GGML_BLOCK:
            raise ValueError(f"tensor {self.name}: unsupported ggml type {self.ggml_type}")
        per, size = GGML_BLOCK[self.ggml_type]
        if self.n_elements % per:
            raise ValueError(f"tensor {self.name}: {self.n_elements} elements not a multiple of {per}")
        return self.n_elements // per * size

    @property
    def row_bytes(self) -> int:
        """Bytes of one row (``ne0`` elements)."""
        per, size = GGML_BLOCK[self.ggml_type]
        return self.dims[0] // per * size


@dataclass
class GGUF:
    path: Path
    version: int
    kv: dict[str, Value] = field(default_factory=dict)
    tensors: list[TensorInfo] = field(default_factory=list)
    alignment: int = DEFAULT_ALIGNMENT
    data_start: int = 0

    def get(self, key: str, default: Any = None) -> Any:
        item = self.kv.get(key)
        return default if item is None else item.value

    def tensor(self, name: str) -> TensorInfo:
        for info in self.tensors:
            if info.name == name:
                return info
        raise KeyError(name)

    def tensor_bytes(self, info: TensorInfo) -> bytes:
        with open(self.path, "rb") as f:
            f.seek(self.data_start + info.offset)
            data = f.read(info.nbytes)
        if len(data) != info.nbytes:
            raise ValueError(f"tensor {info.name}: file truncated")
        return data

    def tensor_array(self, info: TensorInfo) -> np.ndarray:
        """Dequantised float32 copy of an F32 / F16 / Q8_0 tensor, shape ``reversed(dims)``."""
        raw = np.frombuffer(self.tensor_bytes(info), dtype=np.uint8)
        if info.ggml_type == GGML_F32:
            out = raw.view(np.float32).copy()
        elif info.ggml_type == GGML_F16:
            out = raw.view(np.float16).astype(np.float32)
        elif info.ggml_type == GGML_Q8_0:
            blocks = raw.reshape(-1, 34)
            scale = blocks[:, :2].copy().view(np.float16).astype(np.float32)
            quant = blocks[:, 2:].copy().view(np.int8).astype(np.float32)
            out = (scale * quant).reshape(-1)
        else:
            raise ValueError(f"tensor {info.name}: cannot dequantise ggml type {info.ggml_type}")
        return out.reshape(tuple(reversed(info.dims)))


# --------------------------------------------------------------------------------- reading
class _Reader:
    def __init__(self, buf: memoryview):
        self.buf, self.pos = buf, 0

    def take(self, n: int) -> memoryview:
        if self.pos + n > len(self.buf):
            raise ValueError("unexpected end of GGUF header")
        out = self.buf[self.pos:self.pos + n]
        self.pos += n
        return out

    def scalar(self, vtype: int) -> Any:
        fmt = "<" + _SCALAR_FMT[vtype]
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        return bytes(self.take(self.scalar(UINT64))).decode("utf-8", errors="replace")

    def value(self, vtype: int) -> Value:
        if vtype in _SCALAR_FMT:
            return Value(vtype, self.scalar(vtype))
        if vtype == STRING:
            return Value(vtype, self.string())
        if vtype == ARRAY:
            subtype, count = self.scalar(UINT32), self.scalar(UINT64)
            if subtype in _SCALAR_FMT:
                fmt = _SCALAR_FMT[subtype]
                size = struct.calcsize(fmt)
                items = list(struct.unpack(f"<{count}{fmt}", self.take(size * count))) if count else []
            elif subtype == STRING:
                items = [self.string() for _ in range(count)]
            else:
                items = [self.value(subtype).value for _ in range(count)]
            return Value(vtype, items, subtype)
        raise ValueError(f"unknown GGUF value type {vtype}")


def read(path: str | os.PathLike) -> GGUF:
    """Parse the header of a GGUF file (tensor data stays on disk)."""
    path = Path(path)
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        r = _Reader(memoryview(mm))
        try:
            if bytes(r.take(4)) != MAGIC:
                raise ValueError(f"{path}: not a GGUF file")
            version = r.scalar(UINT32)
            if version not in (2, 3):
                raise ValueError(f"{path}: unsupported GGUF version {version}")
            n_tensors, n_kv = r.scalar(UINT64), r.scalar(UINT64)
            kv: dict[str, Value] = {}
            for _ in range(n_kv):
                key = r.string()
                kv[key] = r.value(r.scalar(UINT32))
            tensors = []
            for _ in range(n_tensors):
                name = r.string()
                n_dims = r.scalar(UINT32)
                dims = tuple(r.scalar(UINT64) for _ in range(n_dims))
                tensors.append(TensorInfo(name, dims, r.scalar(UINT32), r.scalar(UINT64)))
            header_end = r.pos
        finally:
            del r
    alignment = int(kv["general.alignment"].value) if "general.alignment" in kv else DEFAULT_ALIGNMENT
    data_start = -(-header_end // alignment) * alignment
    return GGUF(path, version, kv, tensors, alignment, data_start)


# --------------------------------------------------------------------------------- writing
def _pack_scalar(vtype: int, value: Any) -> bytes:
    return struct.pack("<" + _SCALAR_FMT[vtype], value)


def _pack_string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _pack_value(item: Value) -> bytes:
    if item.vtype in _SCALAR_FMT:
        return _pack_scalar(item.vtype, item.value)
    if item.vtype == STRING:
        return _pack_string(item.value)
    if item.vtype == ARRAY:
        sub = item.subtype
        if sub is None:
            raise ValueError("array value without element type")
        head = struct.pack("<IQ", sub, len(item.value))
        if sub in _SCALAR_FMT:
            return head + struct.pack(f"<{len(item.value)}{_SCALAR_FMT[sub]}", *item.value)
        if sub == STRING:
            return head + b"".join(_pack_string(s) for s in item.value)
        return head + b"".join(_pack_value(Value(sub, v)) for v in item.value)
    raise ValueError(f"unknown GGUF value type {item.vtype}")


def write(path: str | os.PathLike, kv: dict[str, Value],
          tensors: Iterable[tuple[TensorInfo, Callable[[], bytes]]],
          version: int = 3, alignment: int = DEFAULT_ALIGNMENT) -> None:
    """Write a GGUF file atomically (temporary file, fsync, rename).

    ``tensors`` yields ``(info, data_fn)``; ``data_fn()`` returns exactly ``info.nbytes`` bytes.
    Offsets are recomputed; ``general.alignment`` in ``kv`` (if present) must equal ``alignment``.
    """
    path = Path(path)
    items = list(tensors)
    if "general.alignment" in kv and int(kv["general.alignment"].value) != alignment:
        raise ValueError("general.alignment disagrees with the alignment argument")
    offsets, cursor = [], 0
    for info, _ in items:
        cursor = -(-cursor // alignment) * alignment
        offsets.append(cursor)
        cursor += info.nbytes
    header = bytearray(MAGIC + struct.pack("<IQQ", version, len(items), len(kv)))
    for key, item in kv.items():
        header += _pack_string(key) + struct.pack("<I", item.vtype) + _pack_value(item)
    for (info, _), off in zip(items, offsets):
        header += _pack_string(info.name) + struct.pack("<I", len(info.dims))
        header += struct.pack(f"<{len(info.dims)}Q", *info.dims) + struct.pack("<IQ", info.ggml_type, off)
    header += b"\0" * (-len(header) % alignment)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(header)
        written = 0
        for (info, data_fn), off in zip(items, offsets):
            f.write(b"\0" * (off - written))
            data = data_fn()
            if len(data) != info.nbytes:
                raise ValueError(f"tensor {info.name}: got {len(data)} bytes, expected {info.nbytes}")
            f.write(data)
            written = off + info.nbytes
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def permute_rows(data: bytes, info: TensorInfo, order: np.ndarray) -> bytes:
    """Reorder the rows (``ne1`` index) of a 2-D tensor of any block type: row ``i`` of the
    result is row ``order[i]`` of the input. Rows are independent in every ggml block format,
    so this is exact for quantised tensors."""
    if len(info.dims) != 2:
        raise ValueError(f"tensor {info.name}: permute_rows needs a 2-D tensor")
    rows = np.frombuffer(data, dtype=np.uint8).reshape(info.dims[1], info.row_bytes)
    order = np.asarray(order, dtype=np.int64)
    if sorted(order.tolist()) != list(range(info.dims[1])):
        raise ValueError("order must be a permutation of the rows")
    return rows[order].tobytes()
