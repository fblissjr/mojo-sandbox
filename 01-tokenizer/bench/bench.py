"""Phase A + B bench harness for the mojo_tokenizer extension module.

Phase A: count_bytes (length round-trip vs len()).
Phase B: count_byte_* (scalar baseline + SIMD-16/32/64) vs Python's C-optimized
str.count(). Honest about overhead: each Mojo call pays a Python-str -> Mojo
String copy. Phase C will explore zero-copy via the buffer protocol.
"""

from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_BUILD = os.path.normpath(os.path.join(_HERE, "..", "build"))
sys.path.insert(0, _BUILD)

import mojo_tokenizer  # type: ignore[import-not-found]  # noqa: E402  # built artifact


def fmt(ms: float, nbytes: int) -> str:
    gbs = (nbytes / 1e9) / (ms / 1000) if ms > 0 else float("inf")
    return f"{ms:8.3f} ms/call   {gbs:6.2f} GB/s"


def bench(label: str, fn, iters: int, nbytes: int) -> int:
    # Warm-up.
    result = fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        result = fn()
    ms = (time.perf_counter() - t0) / iters * 1000
    print(f"  {label:28s}  {fmt(ms, nbytes)}   -> {result}")
    return result


def phase_a() -> int:
    print("=== Phase A: count_bytes (len round-trip) ===")
    sample = "hello, mojo"
    py = len(sample)
    mj = mojo_tokenizer.count_bytes(sample)
    print(f"  sample             : {sample!r}")
    print(f"  python len()       : {py}")
    print(f"  mojo  count_bytes(): {mj}")
    if py != mj:
        print("  FAIL: lengths disagree")
        return 1
    print("  PASS")
    return 0


def phase_b() -> int:
    print()
    print("=== Phase B: count_byte_* (SIMD byte scan) ===")

    # Mostly-ASCII payload with a known target frequency. ~12 MB.
    # 's' appears once per " mojo " repeat (in "mojo"), plus once per "hello",
    # so we just count it with Python and use that as the oracle.
    text = ("hello, mojo " * 1_000_000)
    nbytes = len(text)
    target = ord("o")
    py_oracle = text.count(chr(target))
    print(f"  text size          : {nbytes} bytes")
    print(f"  target             : {chr(target)!r} (0x{target:02x})")
    print(f"  python .count()    : {py_oracle}")

    # Correctness first -- bail loudly on disagreement.
    results = {
        "scalar":  mojo_tokenizer.count_byte_scalar(text, target),
        "simd_16": mojo_tokenizer.count_byte_simd_16(text, target),
        "simd_32": mojo_tokenizer.count_byte_simd_32(text, target),
        "simd_64": mojo_tokenizer.count_byte_simd_64(text, target),
    }
    for name, got in results.items():
        if got != py_oracle:
            print(f"  FAIL: mojo {name} returned {got}, expected {py_oracle}")
            return 1
    print("  correctness: scalar + simd_{16,32,64} all match Python oracle")

    # Microbench. Iterations chosen so total wallclock stays small.
    iters = 30
    print()
    print("  --- bench ---")
    bench("python str.count() (C)",  lambda: text.count(chr(target)),                 iters, nbytes)
    bench("mojo count_byte_scalar",  lambda: mojo_tokenizer.count_byte_scalar(text, target),  iters, nbytes)
    bench("mojo count_byte_simd_16", lambda: mojo_tokenizer.count_byte_simd_16(text, target), iters, nbytes)
    bench("mojo count_byte_simd_32", lambda: mojo_tokenizer.count_byte_simd_32(text, target), iters, nbytes)
    bench("mojo count_byte_simd_64", lambda: mojo_tokenizer.count_byte_simd_64(text, target), iters, nbytes)

    # Pure-Python loop on a smaller slice (full-size would take minutes).
    small = text[: 200_000]
    py_loop_iters = 5
    n = 0
    t0 = time.perf_counter()
    for _ in range(py_loop_iters):
        n = sum(1 for c in small if c == chr(target))
    ms = (time.perf_counter() - t0) / py_loop_iters * 1000
    gbs = (len(small) / 1e9) / (ms / 1000)
    print(f"  {'python loop (200 KB only)':28s}  {ms:8.3f} ms/call   {gbs:6.4f} GB/s   -> {n}")

    print()
    print("  PASS (Phase B)")
    return 0


def main() -> int:
    rc = phase_a()
    if rc:
        return rc
    rc = phase_b()
    return rc


if __name__ == "__main__":
    sys.exit(main())
