# mojo-sandbox

Hands-on with **Mojo 1.0 beta** (Modular 26.3, released 2026-05-07). Coming back to the language after roughly a year away — the goal is to feel out what 1.0 actually delivers, where it still falls short, and produce a few small things worth keeping.

## Hardware targets

Two machines, two reasons:

| Machine | Role | Why |
| --- | --- | --- |
| **Mac Studio M2 Ultra, 192 GB** | Primary dev box | Unified memory means even mid-sized models live entirely in GPU-addressable RAM. Metal target works on M1+, no MMA dependency. |
| **Ubuntu + RTX 4090** | Portability target | NVIDIA is Mojo's most mature backend. Validates the same kernels run on CUDA via MAX. Better profiler (Nsight). |

Project code is written to compile and run on both `osx-arm64` and `linux-64`. The whole point of Mojo's GPU model is unmodified portability — these two boxes prove it.

## Layout

```
.
├── pixi.toml                 # workspace deps: mojo, max
├── 00-setup/                 # toolchain validation + Mojo GPU Puzzles
├── 01-tokenizer/             # SIMD BPE, shipped as a Python module
├── 02-gpu-softmax/           # Metal + CUDA softmax kernel
└── 03-baby-llama/            # capstone: small transformer + INT8 quant
```

Each subdir is independently runnable, has its own README, and lists which 1.0 features it exercises and which gaps it surfaced.

## Why this sequencing

1. **Tokenizer first** — exercises the language's identity (comptime + SIMD + ownership) and produces a shippable `.dylib`/`.so`. Most generalizable skill.
2. **GPU softmax second** — Mojo's actual differentiator. Same kernel runs on Metal and CUDA without changes.
3. **Baby-llama capstone** — pulls 01 and 02 together. Concrete answer to "can I ship LLM inference in Mojo today?"

## What this sandbox deliberately avoids

`async`/`await`, networking, web, autodiff/training, classes-with-inheritance — all weak or absent in 1.0 beta. Stay in the part of the language that's stable, fast, and uniquely Mojo's.

## Quick start

On either machine:

```bash
# install pixi if you don't have it
curl -fsSL https://pixi.sh/install.sh | bash

# from the repo root
pixi install
pixi run mojo --version

# enter a project
cd 00-setup
pixi run hello-kernel
```

## Plan

Full plan with rationale, concept coverage per project, and verification steps lives at `~/.claude/plans/modular-s-mojo-programming-language-valiant-pumpkin.md` (in the dev environment that produced this repo).
