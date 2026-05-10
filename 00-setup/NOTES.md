# Setup notes

A running log of what's surprising or different coming back to Mojo 1.0 from ~0.x.

## What changed since I last looked

Tested against `Mojo 1.0.0b2.dev2026051006`.

### Verified by the compiler on this hardware

- **`fn` is removed.** `def` is the only function form. Old `fn` ↔ new `def` (with explicit `raises` when needed).
- **`alias` → `comptime`.** `comptime N = 1 << 20` declares a compile-time constant.
- **`@parameter if` → `comptime if`.** Same semantics, cleaner spelling. Likewise the `@parameter` decorator pattern is gone.
- **Implicit stdlib imports are deprecated.** Every stdlib import needs the `std.` prefix:
  - `from std.gpu.host import DeviceContext`
  - `from std.gpu import thread_idx, block_idx, block_dim, global_idx`
  - `from std.sys import has_accelerator`
  - `from std.memory import UnsafePointer`
  - `from std.testing import assert_equal, assert_almost_equal`
  - **Exception:** `from layout import Layout, LayoutTensor` — `layout` is a separate package, NOT under `std`.
- **`out` is reserved as an argument name.** Use `output`, `dst`, etc.
- **`def` is no longer implicitly raising.** If you call a raising fn (`DeviceContext()`, `assert_*`), the surrounding fn needs an explicit `raises` annotation.
- **`mojo test` is gone.** Test files are just scripts: `mojo run test_smoke.mojo`. A `def main() raises:` calls each test fn and prints a summary.
- **`let` is gone** (already removed pre-1.0). `var` is the only binding form.
- **`assert_almost_equal` wants typed args.** `assert_almost_equal(v.reduce_add(), Float32(10.0))` — bare `10.0` infers as Float64 and the comparison errors.

### Kernel-arg pattern in 1.0

Raw `UnsafePointer[Float32, _]` is NOT `DevicePassable`, so you can't pass `dev_buf.unsafe_ptr()` directly to `enqueue_function`. The working pattern uses `LayoutTensor` with `MutAnyOrigin`:

```mojo
from layout import Layout, LayoutTensor
from std.gpu import global_idx
from std.gpu.host import DeviceContext

comptime N = 1 << 20
comptime layout = Layout.row_major(N)
comptime Tensor = LayoutTensor[DType.float32, layout, MutAnyOrigin]

def vec_add_kernel(a: Tensor, b: Tensor, output: Tensor):
    var i = global_idx.x
    if i < N:
        output[i] = a[i] + b[i]

def main() raises:
    var ctx = DeviceContext()
    var buf = ctx.enqueue_create_buffer[DType.float32](N)
    var t = Tensor(buf)              # wrap, then pass:
    ctx.enqueue_function[vec_add_kernel](t, ..., grid_dim=GRID, block_dim=BLOCK)
```

Key bits:
- `MutAnyOrigin` is implicitly available — no import.
- `comptime Tensor = LayoutTensor[...]` is a compile-time type alias.
- `global_idx.x` replaces the manual `block_idx.x * block_dim.x + thread_idx.x` calculation (since 1.0b1 these are `Int`, not `UInt`).
- The buffer comes from `ctx.enqueue_create_buffer[dtype](N)`; wrap with `Tensor(buf)` for the kernel call.

### Still unconfirmed on this version

- Origin/lifetime model — `MutAnyOrigin` is the easy escape hatch; tighter scoping TBD.
- Closure capture syntax.
- Trait conditional-conformance spelling.
- `TileTensor` (referenced in roadmap, separate from `LayoutTensor`) — explore in 02-gpu-softmax.

## Mojo GPU Puzzles log

> Fill in as you go through `pixi run -e apple pNN` (or `nvidia`).

- [ ] **Puzzle 01 — Map**: thread_idx / block_idx layout.
- [ ] **Puzzle 02 — Zip**: two input arrays.
- [ ] **Puzzle 03 — Guard**: bounds check.
- [ ] **Puzzle 04 — 2D map**: 2D grid indexing.
- [ ] **Puzzle 05 — Broadcast**: rank mismatch.
- [ ] **Puzzle 06 — Blocks**: when N > block size.
- [ ] **Puzzle 07 — 2D blocks**: 2D grid + 2D blocks.
- [ ] **Puzzle 08 — Shared memory**: first taste of `__shared__`-equivalent.

## Gotchas

(populate during runs)

- Apple Metal `print()` inside a kernel: single-string only — concat eagerly.
- `MOJO_PYTHON_LIBRARY` env var: set if libpython auto-discovery fails on macOS arm64.
- Channel ordering matters in `pixi.toml` — `max-nightly` first, `conda-forge` last.
