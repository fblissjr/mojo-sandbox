# 00-setup — Toolchain validation

Goal: prove pixi + Mojo 1.0 beta + GPU access actually work on this hardware before writing anything that matters. Also: re-acclimate to the 1.0 syntax (origins, traits, closures) after time away.

## What's here

| File | Purpose |
| --- | --- |
| `hello_kernel.mojo` | Smallest possible end-to-end GPU program. Vector add on the local accelerator (Metal on M2 Ultra, CUDA on RTX 4090). Falls back with a clear message if no accelerator is detected. |
| `test_smoke.mojo`   | CPU-only `mojo test` cases. Validates the compile and test pipeline even without GPU access. |
| `NOTES.md`          | Running notes from the [Mojo GPU Puzzles](https://puzzles.modular.com/) walkthrough — what tripped me up coming back from 0.x. |

## Run

```bash
# from the repo root
pixi install              # first time only
pixi run versions         # confirms mojo + pixi installed
pixi run hello-kernel     # builds and runs the GPU smoke test
pixi run hello-test       # runs the CPU test file
```

## What success looks like

`pixi run hello-kernel` prints something like:

```
Detected accelerator: <Metal | CUDA>
Launching vec_add: N=1048576, block=256, grid=4096
out[0] = 0.0
out[1] = 3.0
out[2] = 6.0
PASS: vector add matches reference on 1048576 elements
```

`pixi run hello-test` prints `1 test passed`.

## Mojo GPU Puzzles (recommended next)

Don't vendor it — clone alongside this repo:

```bash
cd ~/code  # or wherever
git clone https://github.com/modular/mojo-gpu-puzzles
cd mojo-gpu-puzzles
pixi run -e apple p01     # M2 Ultra
# or
pixi run -e nvidia p01    # RTX 4090
```

Take notes in `NOTES.md` here. Aim for puzzles 1–8: that covers thread/block indexing, shared memory, reductions, and tiling — the conceptual basis for `02-gpu-softmax`.

## What this phase exercises

- Pixi workspace install end-to-end on both `osx-arm64` and `linux-64`.
- `DeviceContext` lifecycle, host↔device buffer transfer.
- Compile-time `@parameter` branching on `has_accelerator()`.
- The `mojo test` runner.

## Gaps it will surface

- macOS arm64: if `mojo build` complains about C++ compiler, install/refresh Xcode CLT.
- Apple Metal: `print()` inside kernels is a single-string-only call (vs. variadic on CUDA). Plan debugging accordingly.
- If Python interop is needed later and libpython can't be found on macOS, set `MOJO_PYTHON_LIBRARY` to the active Python's lib path.
