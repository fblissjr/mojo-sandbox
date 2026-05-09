# Setup notes

A running log of what's surprising or different coming back to Mojo 1.0 from ~0.x.

## What changed since I last looked

- `let` is gone. `var` is the only binding form. Most "constants" become `alias` (compile-time) or `var` (runtime).
- `fn` vs `def`: `fn` is strictly typed and propagates errors only via `raises`; `def` is Python-shaped (dynamic, implicitly `raises`). Mostly use `fn` outside `main`.
- Origins (formerly "lifetimes") are now first-class in the type system. References carry an origin parameter; the borrow checker is real and rejects use-after-move at compile time.
- Closures capture explicitly — there's a syntax for declaring captures, which is new.
- Traits with conditional conformance: `impl Foo for Bar where T: Baz` style.
- `TileTensor` is the new GPU-side layout type — encodes shape + strides at the type level so kernels can be specialized at compile time.

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
