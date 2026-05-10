"""Phase A: smoke + micro-bench for the mojo_tokenizer extension module.

Imports the built .so from ../build and exercises count_bytes against pure
Python's len(). Pure-Python len() is O(1) on str (cached length), so it'll
crush us here -- the point of this phase is just proving the call pipeline
works. Real benchmarks land in Phase B (SIMD byte scanning) and Phase C (BPE).
"""

from __future__ import annotations

import os
import sys
import time

# 01-tokenizer/build sits beside this script's parent.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BUILD = os.path.normpath(os.path.join(_HERE, "..", "build"))
sys.path.insert(0, _BUILD)

import mojo_tokenizer  # type: ignore[import-not-found]  # noqa: E402  # built artifact


def main() -> int:
    sample = "hello, mojo"
    py_len = len(sample)
    mj_len = mojo_tokenizer.count_bytes(sample)
    print(f"sample             : {sample!r}")
    print(f"python len()       : {py_len}")
    print(f"mojo  count_bytes(): {mj_len}")
    if py_len != mj_len:
        print("FAIL: lengths disagree")
        return 1

    # Larger payload + microbench. len() will win; that's expected.
    big = ("hello, mojo " * 1_000_000)
    n = len(big)
    print(f"big text           : {n} bytes")

    iters = 50
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = mojo_tokenizer.count_bytes(big)
    mj_ms = (time.perf_counter() - t0) / iters * 1000

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = len(big)
    py_ms = (time.perf_counter() - t0) / iters * 1000

    print(f"mojo  count_bytes  : {mj_ms:.3f} ms/call")
    print(f"python len()       : {py_ms:.6f} ms/call (cached attr; expected to win)")
    print("PASS: pipeline works end-to-end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
