# mojo-sandbox

Hands-on with **Mojo 1.0 beta** (Modular 26.3, released 2026-05-07). Coming back to the language after roughly a year away — the goal is to feel out what 1.0 actually delivers, where it still falls short, and produce a few small things worth keeping.

## Current state

| Phase | What it proves | Status |
| --- | --- | --- |
| `00-setup/` | pixi + Mojo 1.0 toolchain, 1M-element vector add on Metal | done |
| `01-tokenizer/` A | Mojo function shipped as a Python `.so`, end-to-end | done |
| `01-tokenizer/` B | comptime-SIMD byte scanner, 4.6× over `str.count()` | done |
| `01-tokenizer/` C | BPE encoder + tiktoken round-trip | not started |
| `02-gpu-softmax/` | Metal/CUDA softmax kernel + bench | not started |
| `03-baby-llama/` | small transformer inference + INT8 quant | not started |

### Headline numbers (M2 Ultra, 12 MB ASCII payload, target byte `'o'`)

```
  python str.count() (C)           5.125 ms/call     2.34 GB/s
  mojo count_byte_scalar           4.592 ms/call     2.61 GB/s   (1.1x over .count())
  mojo count_byte_simd_16          1.569 ms/call     7.65 GB/s   (3.3x)
  mojo count_byte_simd_32          1.250 ms/call     9.60 GB/s   (4.1x)
  mojo count_byte_simd_64          1.119 ms/call    10.72 GB/s   (4.6x)
  python loop (200 KB only)        5.894 ms/call   0.0339 GB/s   (~316x slower)
```

NEON hardware vector on M-series is 128-bit = 16 uint8 lanes; width=32 and width=64 are loop-unroll variations, not wider lanes. See `01-tokenizer/README.md` for details and caveats.

## Hardware targets

| Machine | Role | Why |
| --- | --- | --- |
| **Mac Studio M2 Ultra, 192 GB** | Primary dev box | Unified memory means even mid-sized models live entirely in GPU-addressable RAM. Metal target works on M1+, no MMA dependency. |
| **Ubuntu + RTX 4090** | Portability target | NVIDIA is Mojo's most mature backend. Validates the same kernels run on CUDA via MAX. Better profiler (Nsight). |

Project code is written to compile and run on both `osx-arm64` and `linux-64`. The whole point of Mojo's GPU model is unmodified portability — these two boxes prove it.

## Layout

```
.
├── README.md
├── CLAUDE.md                  # project-level instructions for Claude
├── pixi.toml                  # workspace deps + task definitions
├── 00-setup/                  # toolchain validation + GPU vector-add smoke test
├── 01-tokenizer/              # SIMD byte scan, shipped as a Python module
├── 02-gpu-softmax/            # Metal + CUDA softmax kernel (not started)
└── 03-baby-llama/             # capstone: small transformer + INT8 quant (not started)
```

Each subdir is independently runnable, has its own README, and records which 1.0 features it exercised and what gaps it surfaced. The Mojo language migration log lives in `00-setup/NOTES.md`.

## Why this sequencing

1. **Tokenizer first** — exercises the language's identity (comptime + SIMD + Python interop) and produces a shippable `.so`. Most generalizable skill.
2. **GPU softmax second** — Mojo's actual differentiator. Same kernel runs on Metal and CUDA without changes.
3. **Baby-llama capstone** — pulls 01 and 02 together. Concrete answer to "can I ship LLM inference in Mojo today?"

## What this sandbox deliberately avoids

`async`/`await`, networking, web, autodiff/training, classes-with-inheritance — all weak or absent in 1.0 beta. Stay in the part of the language that's stable, fast, and uniquely Mojo's.

## Quick start

```bash
# install pixi if you don't have it
curl -fsSL https://pixi.sh/install.sh | bash

# from the repo root
pixi install
pixi run versions          # mojo + pixi versions

# 00-setup smoke tests
pixi run hello-test        # CPU-only SIMD/int sanity
pixi run hello-kernel      # 1M-element vector add on GPU

# 01-tokenizer build + bench
pixi run tokenizer         # mojo build --emit shared-lib + Python harness
```

If you use the `uvp` uv-shaped wrapper, every `pixi run X` becomes `uvp run X` and `pixi install` becomes `uvp sync`.

## Plan notes

The original plan + rationale was hashed out in conversation and lives in the commit history of this branch (start at the first commit on `claude/mojo-1.0-research-rslbA`). Per-phase rationale also lives in each subdir's README.
