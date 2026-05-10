# 01-tokenizer

Goal: ship a Mojo function as a Python extension module, end-to-end on macOS arm64.

The plan is staged. Each phase is independently committable so failure surfaces are small.

| Phase | What it proves | Status |
| --- | --- | --- |
| **A — pipeline smoke test** | Mojo `.so` builds, Python imports it, calls a Mojo function. | done |
| **B — SIMD byte scan** | `count_byte_simd[width]` comptime-specialized; beats Python's C-optimized `str.count()`. | done |
| **C — BPE encoder/decoder** | Real-ish tokenizer: load merges file, encode → ids, decode → text, round-trip and bench vs `tiktoken`. | not started |

## Numbers on M2 Ultra (12 MB ASCII payload, target byte `'o'`)

```
  python str.count() (C)           5.125 ms/call     2.34 GB/s
  mojo count_byte_scalar           4.592 ms/call     2.61 GB/s   (1.1x over .count())
  mojo count_byte_simd_16          1.569 ms/call     7.65 GB/s   (3.3x)
  mojo count_byte_simd_32          1.250 ms/call     9.60 GB/s   (4.1x)
  mojo count_byte_simd_64          1.119 ms/call    10.72 GB/s   (4.6x)
  python loop (200 KB only)        5.894 ms/call   0.0339 GB/s   (~316x slower than SIMD-64)
```

Numbers include a Python-str → Mojo-String copy on every call (~12 MB at memcpy speed ≈ 0.4 ms baked in). Phase C will explore zero-copy via Python's buffer protocol for an apples-to-apples comparison.

### Caveats worth knowing

- **NEON width.** M-series silicon has 128-bit SIMD registers = 16 uint8 lanes. `width=16` is the actual hardware vector. `width=32` and `width=64` are loop-unroll variations — the speedup beyond 16 comes from instruction-level parallelism / better pipelining, not wider hardware.
- **Lane-accumulator trick didn't help here.** The standard popcount-style optimization (accumulate lane-locally, flush horizontal reduce every 255 chunks) was implemented and benchmarked — it slowed everything by ~10%. With ~0.4 ms of the budget eaten by `String(text)` copy, the per-chunk `reduce_add` is not the bottleneck; the compiler already vectorizes it well. The simpler code is also the faster code on this hardware/workload. Phase C's zero-copy fix is what should actually move the needle.

## Run

```bash
# from repo root
uvp run tokenizer-build      # mojo build --emit shared-lib -> build/mojo_tokenizer.so
uvp run tokenizer-bench      # imports the .so and runs Phase A + B harness
# or:
uvp run tokenizer            # build + bench in sequence
```

## Layout

```
01-tokenizer/
├── README.md
├── src/mojo_tokenizer.mojo     # count_bytes, count_byte_scalar, count_byte_simd[w]
├── bench/bench.py              # Phase A + B harness
└── build/                      # gitignored: mojo_tokenizer.so lands here
```

## Concepts exercised

**Phase A (Python extension pipeline):**
- `PythonModuleBuilder` from `std.python.bindings` — register a Python module from Mojo.
- `@export def PyInit_<name>() -> PythonObject` — the standard Python C-extension entry point.
- `t"..."` template string interpolation (new in 1.0) for error messages via `abort`.
- `mojo build --emit shared-lib` — produces a `.so` Python can import directly.

**Phase B (SIMD + comptime):**
- `def count_byte_simd[width: Int](...)` — comptime parameter for SIMD lane count.
- Each width is a separate compiled specialization, registered as its own Python symbol (`count_byte_simd_16/32/64`).
- `s.unsafe_ptr()` + `ptr.load[width=width](i)` — vectorized byte loads from a Mojo `String`.
- `SIMD.eq(other)` returns an elementwise bool mask — `==` reduces to a scalar `Bool` in 1.0.
- `mask.cast[DType.uint8]().reduce_add()` — count matches across lanes.

## Known gotchas

- The built artifact must be named `mojo_tokenizer.so` (matches the `PyInit_<name>` symbol).
- macOS arm64: if `import mojo_tokenizer` complains about libpython at runtime, set `MOJO_PYTHON_LIBRARY` to the active Python's `libpython3.X.dylib`.
- Each `def_function[fn]("name", ...)` call adds one symbol — no comptime parameter passthrough into the Python signature, so SIMD widths get separate entry points.
- `PythonObject` → `Int` is `Int(py=obj)`, not `Int(obj)` — the latter looks for `Intable`/`IntableRaising` trait conformance which `PythonObject` doesn't have.
- `def` is not implicitly raising in 1.0 — calls to `Int(py=…)` or other raising ops force an explicit `raises` annotation.

## Next phase

Phase C: load a small BPE merges table (GPT-2 public domain), implement encode/decode with the SIMD scan reused as the pre-tokenizer inner loop, round-trip vs `tiktoken`.
