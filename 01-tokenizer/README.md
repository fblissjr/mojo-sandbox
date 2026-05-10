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
  python str.count() (C)           5.131 ms/call     2.34 GB/s
  mojo count_byte_scalar           4.686 ms/call     2.56 GB/s   (1.1x over .count())
  mojo count_byte_simd_16          1.624 ms/call     7.39 GB/s   (3.2x)
  mojo count_byte_simd_32          1.287 ms/call     9.33 GB/s   (4.0x)
  mojo count_byte_simd_64          1.153 ms/call    10.41 GB/s   (4.5x)
  python loop (200 KB only)        6.062 ms/call   0.0330 GB/s   (~315x slower than SIMD-64)
```

Numbers include a Python-str → Mojo-String copy on every call (~12 MB at memcpy speed ≈ 0.4 ms baked in). Phase C will explore zero-copy via Python's buffer protocol for an apples-to-apples comparison.

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
