# 01-tokenizer

Goal: ship a Mojo function as a Python extension module, end-to-end on macOS arm64.

The plan is staged. Each phase is independently committable so failure surfaces are small.

| Phase | What it proves | Status |
| --- | --- | --- |
| **A — pipeline smoke test** | Mojo `.so` builds, Python imports it, calls a Mojo function. | in progress |
| **B — SIMD byte scan** | `count_byte_simd[width]` parameterized at compile time over SIMD width; benchmarked against a pure-Python loop. | not started |
| **C — BPE encoder/decoder** | Real-ish tokenizer: load merges file, encode → ids, decode → text, round-trip and bench vs `tiktoken`. | not started |

## Run (Phase A)

```bash
# from repo root
uvp run tokenizer-build      # mojo build --emit shared-lib -> build/mojo_tokenizer.so
uvp run tokenizer-bench      # imports the .so and calls count_bytes
# or:
uvp run tokenizer            # build + bench in sequence
```

## Layout

```
01-tokenizer/
├── README.md
├── src/mojo_tokenizer.mojo     # Phase A entry point + PyInit
├── bench/bench.py              # Python harness
└── build/                      # gitignored: mojo_tokenizer.so lands here
```

## Concepts under test (Phase A)

- `PythonModuleBuilder` from `std.python.bindings` — the canonical 1.0 way to register a Python module from Mojo.
- `@export def PyInit_<name>() -> PythonObject` — the standard Python C-extension entry point.
- `t"..."` template string interpolation (new in 1.0) for error messages via `abort`.
- `mojo build --emit shared-lib` — produces a `.so` Python can import directly.

## Known gotchas

- The built artifact must be named `mojo_tokenizer.so` (matches the `PyInit_<name>` symbol).
- macOS arm64: if `import mojo_tokenizer` complains about libpython at runtime, set `MOJO_PYTHON_LIBRARY` to the active Python's `libpython3.X.dylib`.
- Each `def_function[fn]("name", ...)` call adds one symbol to the Python module — no comptime parameter passthrough yet, so a SIMD-parameterized function gets registered per-width in Phase B.

## Next phase

When Phase A passes: extend `mojo_tokenizer.mojo` with `count_byte_simd[width: Int = 32](text, target)` and SIMD-load the bytes via `unsafe_ptr().load[width](i)`.
