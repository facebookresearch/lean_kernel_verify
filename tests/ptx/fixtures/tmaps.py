# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Real CUtensorMap bytes for the decoder tests (words w0.., the rest 0).

CUTE442: descriptors CuTe DSL 4.4.2's JIT'd host code encoded in software, captured on a
CPU-only host at its `_cudaLaunchKernelEx` (design §5.1), from
`cpasync.make_tiled_tma_atom(op, T, smem_layout, tiler)` over a torch tensor T (CPU
addresses). `tensor` is T as a capture snapshot entry, `box` the expected KIR box (the
smem tile in T's elements: bytes for the packed 4-bit tile).

DRIVER: cuTensorMapEncodeTiled (GB300, driver 13000) with the given arguments (host
enums; parameter-array slots beyond the rank held garbage). expected(args) is what
launch.decode must return for such an encode, written from the driver's enums
independently of launch.py.
"""

import struct


def raw(*words: int) -> bytes:
    return struct.pack("<16Q", *words, *[0] * (16 - len(words)))


def _t(ptr: int, shape: list, strides: list, elem_bytes: int, storage=None) -> dict:
    span = 1 + sum((n - 1) * s for n, s in zip(shape, strides))
    return {
        "data_ptr": ptr,
        "storage": storage or [ptr, ptr + span * elem_bytes],
        "elem_bytes": elem_bytes,
        "shape": shape,
        "strides": strides,
    }


CUTE442 = {
    # bf16 view base[:, 16:80] of a (256, 96) tensor; tile (64, 32)
    "bf16_view": {
        "raw": raw(0xFFFC239C0020, 0xC00040512, 0, 0, 0xFF0000003F, 0, 0x1F00000000000000, 0x3F),
        "tensor": _t(0xFFFC239C0020, [256, 64], [96, 1], 2, [0xFFFC239C0000, 0xFFFC239CC000]),
        "box": [64, 32],
    },
    "f32_3d": {
        "raw": raw(0xFFFC23588000, 0x18000403A2, 0x600, 0, 0x3F0000005F, 0x3, 0x1F00000000000000, 0x1F),
        "tensor": _t(0xFFFC23588000, [4, 64, 96], [6144, 96, 1], 4),
        "box": [1, 32, 32],
    },
    # 128B-swizzled smem layout
    "bf16_swizzle128": {
        "raw": raw(0xFFFC239B0000, 0x1000046512, 0, 0, 0xFF0000007F, 0, 0x3F00000000000000, 0x7),
        "tensor": _t(0xFFFC239B0000, [256, 128], [128, 1], 2),
        "box": [8, 64],
    },
    # 256 KiB: CuTe sets the encoder flag w1[21]
    "u8_256k": {
        "raw": raw(0xFFFC23A20000, 0x1000240012, 0, 0, 0x3FF000000FF, 0, 0x7F00000000000000, 0x1F),
        "tensor": _t(0xFFFC23A20000, [1024, 256], [256, 1], 1),
        "box": [32, 128],
    },
    "bf16_store": {
        "raw": raw(0xFFFC279D4000, 0x800040512, 0, 0, 0x7F0000003F, 0, 0x1F00000000000000, 0x3F),
        "tensor": _t(0xFFFC279D4000, [128, 64], [64, 1], 2),
        "box": [64, 32],
    },
    # internal_type=Uint16 over u8 data: dim 0 in 16-bit units
    "u16_format_over_u8": {
        "raw": raw(0xFFFC2375E000, 0x800040092, 0, 0, 0x3F0000003F, 0, 0x1F00000000000000, 0xF),
        "tensor": _t(0xFFFC2375E000, [64, 128], [128, 1], 1),
        "box": [16, 64],
    },
    # recast_tensor(T, Float4E2M1FN) of a u8 tensor: 16U4_ALIGN8B, tile (16, 128) fp4
    "fp4": {
        "raw": raw(0xFFFC23766000, 0x800040592, 0, 0, 0x3F000000FF, 0, 0x7F00000000000000, 0xF),
        "tensor": _t(0xFFFC23766000, [64, 128], [128, 1], 1),
        "box": [16, 64],
    },
}  # fmt: skip

# dtype, interleave, swizzle, l2, oob: CUtensorMapDataType / ...Interleave / ...Swizzle /
# ...L2promotion / ...FloatOOBfill enums.
DRIVER = {
    "f16_rank5": {
        "args": {"dtype": 6, "address": 0x7F0000004000, "dims": [64, 3, 5, 7, 2], "strides": [128, 384, 1920, 13440], "box": [32, 3, 5, 1, 2], "estrides": [1, 2, 3, 1, 8], "interleave": 0, "swizzle": 6, "l2": 3, "oob": 1},
        "raw": raw(0x7F0000004000, 0x8001EE340, 0x7800000018, 0x348, 0x20000003F, 0x600000004, 0x1F00708800000001, 0x1000402, 0x0, 0x400),
    },
    "tf32_ftz_2e32": {
        "args": {"dtype": 12, "address": 0x7F0000010000, "dims": [1 << 32, 2], "strides": [1 << 34], "box": [32, 2], "estrides": [1, 1], "interleave": 0, "swizzle": 4, "l2": 2, "oob": 1},
        "raw": raw(0x7F0000010000, 0x40000000002DE410, 0, 0, 0x1FFFFFFFF, 0, 0x1F00000000000000, 0x1, 0x100, 0x400),
    },
    "u6_rank3": {
        "args": {"dtype": 15, "address": 0x7F0000020000, "dims": [128, 64, 3], "strides": [96, 6144], "box": [128, 8, 1], "estrides": [1, 1, 1], "interleave": 0, "swizzle": 3, "l2": 0, "oob": 0},
        "raw": raw(0x7F0000020000, 0x6000066A0, 0x180, 0, 0x3F0000007F, 0x2, 0x7F00000000000000, 0x7, 0x0, 0x400),
    },
    "u4_align16b": {
        "args": {"dtype": 14, "address": 0x7F0000030000, "dims": [256, 32], "strides": [128], "box": [128, 16], "estrides": [1, 1], "interleave": 0, "swizzle": 3, "l2": 1, "oob": 0},
        "raw": raw(0x7F0000030000, 0x800026610, 0, 0, 0x1F000000FF, 0, 0x7F00000000000000, 0xF, 0x0, 0x400),
    },
    "bf16_interleave16b": {
        "args": {"dtype": 9, "address": 0x7F0000040000, "dims": [4, 64, 256], "strides": [1024, 65536], "box": [4, 32, 64], "estrides": [1, 1, 1], "interleave": 1, "swizzle": 0, "l2": 0, "oob": 0},
        "raw": raw(0x7F0000040000, 0x4000200D20, 0x1000, 0, 0x3F00000003, 0xFF, 0x300000000000000, 0x3F1F, 0x20000, 0x10),
    },
    "f64_rank1": {
        "args": {"dtype": 8, "address": 0x7F0000050000, "dims": [1000], "strides": [], "box": [16], "estrides": [1], "interleave": 0, "swizzle": 5, "l2": 0, "oob": 0},
        "raw": raw(0x7F0000050000, 0x106480, 0, 0, 0x3E7, 0, 0xF00000000000000, 0x0, 0x80, 0x400),
    },
}  # fmt: skip

# CUtensorMapDataType -> (decoded dtype, element bits, TF32)
HOST_DTYPES = [
    ("UINT8", 8, False), ("UINT16", 16, False), ("UINT32", 32, False),
    ("INT32", 32, False), ("UINT64", 64, False), ("INT64", 64, False),
    ("FLOAT16", 16, False), ("FLOAT32", 32, False), ("FLOAT64", 64, False),
    ("BFLOAT16", 16, False), ("FLOAT32_FTZ", 32, False), ("FLOAT32", 32, True),
    ("FLOAT32_FTZ", 32, True), ("16U4_ALIGN8B", 4, False), ("16U4_ALIGN16B", 4, False),
    ("16U6_ALIGN16B", 6, False),
]  # fmt: skip
# CUtensorMapSwizzle -> (swizzle, swizzle_atom)
HOST_SWIZZLES = [
    ("NONE", "16B"), ("32B", "16B"), ("64B", "16B"), ("128B", "16B"),
    ("128B", "32B"), ("128B", "32B_FLIP_8B"), ("128B", "64B"),
]  # fmt: skip
INTERLEAVES = ("NONE", "16B", "32B")
L2S = ("NONE", "64B", "128B", "256B")


def expected(a: dict) -> dict:
    dtype, bits, tf32 = HOST_DTYPES[a["dtype"]]
    swizzle, atom = HOST_SWIZZLES[a["swizzle"]]
    r = len(a["dims"])
    return {
        "dtype": dtype,
        "elem_bits": bits,
        "tf32": tf32,
        "rank": r,
        "address": a["address"],
        "dims": list(a["dims"]),
        "strides_bytes": list(a["strides"][: r - 1]),
        "box": list(a["box"]),
        "estrides": list(a["estrides"]),
        "interleave": INTERLEAVES[a["interleave"]],
        "swizzle": swizzle,
        "swizzle_atom": atom,
        "oob_nan": bool(a["oob"]),
        "l2": L2S[a["l2"]],
        "encoder_variant": 0,
    }
