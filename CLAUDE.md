# CLAUDE.md — Project instructions

Repo-specific instructions for Claude Code working in this sandbox. The user's global instructions still apply; this file adds project context.

## What this repo is

A hands-on sandbox for **Mojo 1.0 beta** (Modular 26.3, released 2026-05-07). The goal is to feel out the language a year after the user last touched it, surface what 1.0 actually delivers vs. what's still missing, and ship a few small things worth keeping. Not a library, not a product — a deliberate learning vehicle.

Order of work, by anchor area, is:

1. `00-setup/` — toolchain validation, GPU smoke test, syntax-migration notes. **done.**
2. `01-tokenizer/` — Mojo → Python `.so` extension with comptime SIMD. Phases A + B **done**, Phase C (BPE merges + tiktoken) **not started**.
3. `02-gpu-softmax/` — Metal/CUDA softmax kernel. **not started.**
4. `03-baby-llama/` — small transformer inference, INT8 quant, ties 01 + 02 together. **not started.**

Working language gotchas live in `00-setup/NOTES.md` — read it before writing Mojo against this repo. Each subdir's `README.md` records what 1.0 features that phase exercised and what gaps it surfaced.

## Hardware targets

- **Mac Studio M2 Ultra, 192 GB** — primary dev box. `osx-arm64`. Metal target works on M1+; NEON SIMD register is 128-bit (16 uint8 lanes).
- **Ubuntu + RTX 4090** — secondary, portability target. `linux-64`. Mojo's most mature backend; better profiler (Nsight). Not used yet in this branch.

Code is written to compile and run on both platforms unmodified. That portability is half the point.

## Workflow

- **Package manager:** pixi (≥ 0.68). Modular channels are wired into `pixi.toml`.
- **uv-style wrapper:** the user has a `uvp` command on their `$PATH` that dispatches to pixi with uv-shaped subcommands (`uvp sync`, `uvp run <task>`, etc.). When suggesting commands in chat, prefer `uvp run <task>` over raw `pixi run <task>`. Both work; `uvp` matches the user's muscle memory.
- **Tasks** are declared in the root `pixi.toml`:
  - `versions`, `hello-test`, `hello-kernel` — 00-setup smoke tests.
  - `tokenizer-build`, `tokenizer-bench`, `tokenizer` — 01-tokenizer Phase A + B.
  - 02 and 03 tasks land when those phases start.
- **Branch convention:** all work for this sandbox happens on `claude/mojo-1.0-research-rslbA`. `main` is empty-ish (initial commit only).

## Mojo 1.0 syntax landmines (top hits)

These bit us on first contact; full list in `00-setup/NOTES.md`. Reference them when writing new Mojo:

- **`fn` is removed.** `def` only. `def` is **not** implicitly raising — annotate `def foo() raises:` explicitly when calling anything that raises.
- **`alias` → `comptime`.** Compile-time constants. `comptime FOO = 5`.
- **`@parameter if` → `comptime if`.**
- **Stdlib imports need `std.` prefix.** `from std.gpu.host import DeviceContext`, `from std.python.bindings import PythonModuleBuilder`, etc. The `layout` package is the exception — it's not under `std`.
- **`out` is a reserved argument name.** Use `output`, `dst`, etc.
- **`mojo test` is gone.** Test files are scripts run via `mojo run`; the `def main()` calls each test and prints a summary.
- **`SIMD == SIMD` is scalar Bool** ("are they fully equal"). For elementwise lane masks: `v.eq(other)`.
- **`PythonObject → Int` is `Int(py=obj)`**, not `Int(obj)`. The bare-positional form looks for `Intable`/`IntableRaising` conformance which `PythonObject` doesn't have.
- **Kernel-arg type is `LayoutTensor[dtype, layout, MutAnyOrigin]`**, not raw `UnsafePointer` — raw pointers aren't `DevicePassable`. Wrap a buffer via `Tensor(buf)` before passing to `enqueue_function`. `global_idx.x` replaces the manual `block_idx.x * block_dim.x + thread_idx.x`.
- **Template strings `t"..."`** are the 1.0 way to format messages in `abort()` and similar.

## Commit conventions

- The user's global rule: **never push unless explicitly told to**. Local commits are fine.
- **No `Co-Authored-By` lines** on commits (user preference).
- The pre-commit and commit-msg hooks reject any path in repo content that resolves outside the repo root. Don't reference shell rc files, plan files under the user's home, or absolute paths to other directories in commits, README, or other tracked files. Use repo-relative paths (`./01-tokenizer/src/...`), generic placeholders (`<HOME>/<config-file>`), or rephrase to avoid naming the path at all.
- Commit messages should explain the "why" not just the "what". Recent commits in this branch are good references for tone and length.

## Things to avoid

- **Don't push.** Wait for explicit instruction.
- **Don't add `Co-Authored-By`** to commits.
- **Don't reinvent things in `std.algorithm.functional` / `std.algorithm.reduction`** unless the explicit form is more didactic for the current phase. We did this deliberately for Phase B's SIMD scan (raw `while` + `.load[width]()` was clearer than `elementwise[]` for first-principles teaching).
- **Don't add async / networking / web framework code.** All three are weak or absent in 1.0 beta. The plan deliberately stays in numerics + GPU + interop.
- **Don't bypass pre-commit hooks** with `--no-verify`. If a hook blocks, fix the underlying issue (almost always a path leak).
- **Don't recommend the `context7` MCP server** (user's standing rule).

## Useful references

- `00-setup/NOTES.md` — full Mojo 1.0 migration log, kernel-arg pattern, GPU puzzles checklist.
- `01-tokenizer/README.md` — Python extension + SIMD numbers and caveats (NEON width, lane-accumulator results).
- `pixi.toml` — channel config, dep pins (`modular >=26.3,<26.4`), task definitions.
- Installed Mojo source under `.pixi/envs/default/lib/python3.12/site-packages/max/` is a useful reference for canonical 1.0 patterns (kernel registration, Python interop, comparison ops). Grep there before guessing.
