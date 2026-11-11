# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The CUtensorMap decoder (frontend/ptx/launch.py) against the real driver.

    python3 -I tests/ptx/test_gpu_tmap.py   (or the buck target :test_gpu_tmap)

Needs libcuda.so.1 of a real driver and a visible GPU; skips otherwise (the empty stub
libcuda of CPU-only runs has no cuTensorMapEncodeTiled). Encodes random tiled descriptors
with cuTensorMapEncodeTiled over every dtype (the packed 16U4 / 16U6 formats included),
ranks 1-5, element strides, interleave, every swizzle (the 128B atoms included), L2
promotion and OOB fill, with garbage in the parameter-array slots beyond the rank;
decodes each one the driver accepts and compares every field, and checks that every
dtype and every value of the other fields was accepted and compared. The descriptors
CuTe DSL 4.4.2 encoded in software (fixtures/tmaps.py, CUTE442) must equal the driver's
encoding of their decoded parameters outside the whitelisted w1[0:4] (2 vs 0), w8 and w9.
"""

import ctypes
import random
import sys
import unittest
from collections import Counter
from pathlib import Path
from types import ModuleType

if __package__ in (None, ""):
    # Run as a script: the project root becomes the package `lkv_root`, so the relative
    # imports below resolve as they do in the buck target.
    sys.dont_write_bytecode = True
    _root = ModuleType("lkv_root")
    _root.__path__ = [str(Path(__file__).resolve().parents[2])]
    sys.modules["lkv_root"] = _root
    __package__ = "lkv_root.tests.ptx"  # noqa: A001

from ...frontend.ptx import launch  # noqa: E402
from .fixtures.tmaps import (  # noqa: E402
    CUTE442,
    expected,
    HOST_DTYPES,
    HOST_SWIZZLES,
    INTERLEAVES,
    L2S,
)

PER_DTYPE = 300  # accepted encodes per dtype
MAX_TRIES = 200_000

U32, U64 = ctypes.c_uint32, ctypes.c_uint64


def _driver():
    """cuTensorMapEncodeTiled of the real driver with a current context, or None."""
    try:
        cu = ctypes.CDLL("libcuda.so.1")
        encode = cu.cuTensorMapEncodeTiled
    except (OSError, AttributeError):
        return None
    dev, ctx = ctypes.c_int(), ctypes.c_void_p()
    if (
        cu.cuInit(0)
        or cu.cuDeviceGet(ctypes.byref(dev), 0)
        or cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev)
        or cu.cuCtxSetCurrent(ctx)
    ):
        return None
    encode.restype = ctypes.c_int
    encode.argtypes = [
        ctypes.c_void_p, ctypes.c_int, U32, ctypes.c_void_p, ctypes.POINTER(U64),
        ctypes.POINTER(U64), ctypes.POINTER(U32), ctypes.POINTER(U32),
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]  # fmt: skip
    return encode


def _encode(encode, a: dict, pad: list) -> tuple:
    """(rc, 128 bytes) of cuTensorMapEncodeTiled(**a); the arrays are padded to 5 entries
    with `pad`, which the driver must ignore."""
    buf = (ctypes.c_uint8 * 192)()
    # a CUtensorMap is 64-byte aligned
    at = ctypes.addressof(buf) + (-ctypes.addressof(buf)) % 64

    def arr(t, xs):
        return (t * 5)(*(list(xs) + pad[len(xs) : 5]))

    rc = encode(
        at, a["dtype"], len(a["dims"]), a["address"], arr(U64, a["dims"]),
        arr(U64, a["strides"]), arr(U32, a["box"]), arr(U32, a["estrides"]),
        a["interleave"], a["swizzle"], a["l2"], a["oob"],
    )  # fmt: skip
    return rc, ctypes.string_at(at, 128)


def _draw(rng: random.Random, dtype: int) -> dict:
    """Random tiled-encode arguments, biased towards what the driver accepts (16-byte
    multiples, the packed formats' 128-element boxes)."""
    bits = HOST_DTYPES[dtype][1]
    rank = rng.randint(1, 5)
    dims = [
        rng.choice((1, 2, 3, 128, 256, rng.randint(1, 1 << rng.choice((4, 8, 16, 31)))))
        for _ in range(rank)
    ]
    if rng.random() < 0.05:
        dims[rng.randrange(rank)] = 1 << 32  # the largest extent the 32-bit field holds
    strides, acc = [], (dims[0] * bits + 7) // 8
    for d in dims[1:]:
        acc = (acc + 15) // 16 * 16 * rng.choice((1, 1, 2, 3))
        strides.append(acc)
        acc *= d
    box = [rng.randint(1, min(256, d)) for d in dims]
    if rng.random() < 0.5:
        box[0] = min(dims[0], rng.choice((16, 32, 64, 128, 256)) * 8 // bits)
    return {
        "dtype": dtype,
        "address": rng.randrange(1 << 20, 1 << 47) & ~rng.choice((0xF, 0x1F, 0xFF)),
        "dims": dims,
        "strides": strides,
        "box": box,
        "estrides": [rng.randint(1, 8) if rng.random() < 0.3 else 1 for _ in dims],
        "interleave": rng.choice((0, 0, 0, 0, 1, 2)),
        "swizzle": rng.randrange(len(HOST_SWIZZLES)),
        "l2": rng.randrange(len(L2S)),
        "oob": rng.randrange(2),
    }


class DriverRoundTripTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encode = _driver()
        if cls.encode is None:
            raise unittest.SkipTest("no CUDA driver with a visible GPU")

    def test_random_encodes_decode_exactly(self):
        rng = random.Random(20261103)
        ok, refused, seen = Counter(), Counter(), Counter()
        bad = []
        for _ in range(MAX_TRIES):
            todo = [t for t in range(len(HOST_DTYPES)) if ok[t] < PER_DTYPE]
            if not todo:
                break
            a = _draw(rng, rng.choice(todo))
            rc, raw = _encode(
                self.encode, a, [rng.randrange(1, 1 << 31) for _ in range(5)]
            )
            if rc:
                refused[a["dtype"]] += 1
                continue
            ok[a["dtype"]] += 1
            try:
                got = launch.decode(raw)
            except launch.LaunchError as e:
                got = {"error": str(e)}
            if got != expected(a):
                bad.append((a, raw.hex(), got))
                continue
            self._check_kir(got)
            for key in ("rank", "interleave", "swizzle", "swizzle_atom", "l2"):
                seen[key, got[key]] += 1
            seen["oob_nan", got["oob_nan"]] += 1
            seen["estride>1", any(e > 1 for e in got["estrides"])] += 1
            seen["dim 2^32", (1 << 32) in got["dims"]] += 1
        print(
            "\ncuTensorMapEncodeTiled round trip: "
            + ", ".join(
                f"{name}{'/tf32' if tf32 else ''} {ok[t]} ok {refused[t]} refused"
                for t, (name, _, tf32) in enumerate(HOST_DTYPES)
            )
            + f"; {sum(ok.values())} decoded, {len(bad)} mismatches\naccepted: "
            + ", ".join(f"{k}={v}: {n}" for (k, v), n in sorted(seen.items(), key=str)),
            file=sys.stderr,
        )
        self.assertEqual(bad[:3], [])
        self.assertEqual([t for t in range(len(HOST_DTYPES)) if ok[t] < PER_DTYPE], [])
        must = (
            [("rank", r) for r in range(1, 6)]
            + [("swizzle", s) for s, _ in HOST_SWIZZLES]
            + [("swizzle_atom", a) for _, a in HOST_SWIZZLES]
            + [("l2", x) for x in L2S]
            + [("interleave", x) for x in INTERLEAVES]
            + [("oob_nan", True), ("estride>1", True)]
            + [("dim 2^32", True)]
        )
        self.assertEqual([m for m in must if not seen[m]], [])

    def _check_kir(self, d: dict):
        """A dense descriptor maps to KIR over its own element size (bytes for the packed
        formats): dims reversed, dim 0 converted by dim0_ratio."""
        if d["interleave"] != "NONE" or set(d["estrides"]) != {1}:
            with self.assertRaises(launch.LaunchError):
                launch.kir_desc(d, 1)
            return
        eb = max(1, d["elem_bits"] // 8)
        try:
            k = launch.kir_desc(d, eb)
        except launch.LaunchError:
            # only a dim-0 extent or box of 4/6-bit elements that is not whole bytes
            self.assertNotEqual(
                (d["dims"][0] * d["elem_bits"] % 8, d["box"][0] * d["elem_bits"] % 8),
                (0, 0),
            )
            return
        num, den = k["dim0_ratio"]
        self.assertEqual(k["shape"], d["dims"][:0:-1] + [d["dims"][0] * num // den])
        self.assertEqual(k["box"], d["box"][:0:-1] + [d["box"][0] * num // den])
        self.assertEqual(
            k["strides"], [s // eb for s in d["strides_bytes"][::-1]] + [1]
        )

    def test_cute_encodings_match_driver(self):
        for name, v in CUTE442.items():
            with self.subTest(name):
                d = launch.decode(v["raw"])
                dtype = next(
                    t for t, (n, _, tf32) in enumerate(HOST_DTYPES)
                    if (n, tf32) == (d["dtype"], d["tf32"])
                )  # fmt: skip
                a = {
                    "dtype": dtype, "address": d["address"], "dims": d["dims"],
                    "strides": d["strides_bytes"], "box": d["box"],
                    "estrides": d["estrides"],
                    "interleave": INTERLEAVES.index(d["interleave"]),
                    "swizzle": HOST_SWIZZLES.index((d["swizzle"], d["swizzle_atom"])),
                    "l2": L2S.index(d["l2"]), "oob": int(d["oob_nan"]),
                }  # fmt: skip
                rc, raw = _encode(self.encode, a, [0] * 5)
                self.assertEqual(rc, 0)
                cute = memoryview(v["raw"]).cast("Q")
                drv = memoryview(raw).cast("Q")
                self.assertLessEqual(
                    {i for i in range(16) if cute[i] != drv[i]}, {1, 8, 9}
                )
                self.assertEqual((cute[1] & 0xF, drv[1] & 0xF), (2, 0))
                self.assertEqual(cute[1] & ~0xF, drv[1] & ~0xF)
                self.assertEqual(launch.decode(raw), {**d, "encoder_variant": 0})


if __name__ == "__main__":
    unittest.main()
