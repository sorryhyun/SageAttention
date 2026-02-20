# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SageAttention is an accurate, efficient plug-and-play low-bit quantized attention mechanism for NVIDIA GPUs. It quantizes Q/K to INT8 and optionally V to FP8, achieving significant speedup without losing accuracy. The project spans four papers: SageAttention (ICLR 2025), SageAttention2 (ICML 2025), SageAttention2++, and SageAttention3 (NeurIPS 2025 Spotlight).

## Build & Install

Requires Python >=3.9, PyTorch >=2.3.0, Triton >=3.0.0, and CUDA >=12.0.

```bash
# From source (recommended for development)
python setup.py install

# Faster builds with parallelism
export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32
python setup.py install

# From PyPI
pip install sageattention==2.2.0 --no-build-isolation

# Skip CUDA compilation (e.g., for sdist or CI without GPU)
SAGEATTN_SKIP_CUDA_BUILD=1 python setup.py install
```

GPU architecture is auto-detected from available GPUs, or override with `TORCH_CUDA_ARCH_LIST` (e.g., `"8.0;8.9;9.0"`).

## Benchmarking

No formal test suite exists. Validation is done through benchmarking scripts:

```bash
# Benchmark specific backends
python bench/bench_qk_int8_pv_fp8_cuda.py --pv_accum_dtype fp32+fp16 --quant_gran per_warp
python bench/bench_qk_int8_pv_fp16_cuda.py
python bench/bench_qk_int8_pv_fp8_cuda_sm90.py
python bench/bench_qk_int8_pv_fp16_triton.py

# Compare against FlashAttention
python bench/bench_fa3.py
python bench/bench_baseline.py
```

## Architecture

### Public API (`sageattention/__init__.py`)

Six exported functions, all sharing the signature `(q, k, v, tensor_layout="HND", is_causal=False, ...)`:

- **`sageattn`** — Auto-dispatch based on GPU compute capability (the main entry point)
- **`sageattn_qk_int8_pv_fp16_triton`** — Triton backend, INT8 QK + FP16 PV
- **`sageattn_qk_int8_pv_fp16_cuda`** — CUDA backend, INT8 QK + FP16 PV (Ampere)
- **`sageattn_qk_int8_pv_fp8_cuda`** — CUDA backend, INT8 QK + FP8 PV (Ada/Blackwell)
- **`sageattn_qk_int8_pv_fp8_cuda_sm90`** — CUDA backend, optimized for Hopper (SM90)
- **`sageattn_varlen`** — Variable sequence length support (Triton)

### GPU Dispatch (`core.py:sageattn`)

`sageattn()` detects the GPU and routes to the best backend:
- SM80 (A100) → `pv_fp16_cuda` with fp32 accum
- SM86 (RTX4090/3090) → `pv_fp16_triton`
- SM89 (L20/L40S) → `pv_fp8_cuda` with fp32+fp16 accum (SageAttention2++)
- SM90 (H100/H20) → `pv_fp8_cuda_sm90` with fp32+fp32 accum
- SM120/121 (Blackwell) → `pv_fp8_cuda` with per_warp quantization

### Module Layout

- **`sageattention/core.py`** — All attention implementations and the auto-dispatch logic (~1000 lines). This is the central file.
- **`sageattention/quant.py`** — Python wrappers around CUDA fused quantization ops (`per_block_int8`, `per_warp_int8`, `sub_mean`, `per_channel_fp8`).
- **`sageattention/sm{80,89,90}_compile.py`** — `torch.library.custom_op` registrations binding C++/CUDA kernels to Python, enabling `torch.compile` compatibility.
- **`sageattention/triton/`** — Triton kernel implementations for quantization and attention (per-block, per-thread, causal/non-causal, variable-length variants).

### CUDA Kernels (`csrc/`)

- **`csrc/fused/`** — Fused quantization kernels: INT8 quantization with scale, mean subtraction (outlier smoothing), FP8 quantization, tensor transpose/pad/permute. Bound via `pybind.cpp` as `sageattention._fused`.
- **`csrc/qattn/`** — Attention computation kernels, organized by GPU architecture:
  - `qk_int_sv_f16_cuda_sm80.cu` — SM80 attention (INT8 QK, FP16 PV)
  - `sm89_qk_int8_sv_f8_*.cu` — SM89 attention variants (multiple accumulation and fusion strategies)
  - `qk_int_sv_f8_cuda_sm90.cu` — SM90 attention using WGMMA instructions
  - `pybind_sm{80,89,90}.cpp` — Per-architecture Python bindings as `sageattention._qattn_sm{80,89,90}`
- **`csrc/*.cuh`** — Shared utilities: async copy (`cp_async.cuh`), MMA wrappers (`mma.cuh`, `wgmma.cuh`), numeric conversion (`numeric_conversion.cuh`), shared memory permutation (`permuted_smem.cuh`).

### Build Extensions (from `setup.py`)

Four CUDA extension modules, conditionally compiled based on detected GPU architecture:
1. `sageattention._qattn_sm80` — Built for SM80+ (always built if any supported GPU detected)
2. `sageattention._qattn_sm89` — Built for SM89+
3. `sageattention._qattn_sm90` — Built for SM90 only
4. `sageattention._fused` — Always built (quantization kernels)

### Key Technical Concepts

- **Outlier smoothing**: Mean subtraction from K (controlled by `smooth_k`) reduces quantization error. Optional `smooth_v` for V.
- **Quantization granularity**: `per_block` (coarser, faster), `per_warp`, or `per_thread` (finer, more accurate).
- **Two-level accumulation** (`pv_accum_dtype`): PV accumulated in lower precision then periodically flushed to FP32. Options like `"fp32+fp16"` mean FP16 intermediate with FP32 flush.
- **Tensor layouts**: `"HND"` = `[batch, heads, seq, dim]`, `"NHD"` = `[batch, seq, heads, dim]`.
- Input tensors must be FP16/BF16, CUDA, with contiguous last dimension. Supports GQA (num_qo_heads divisible by num_kv_heads).

### SageAttention3 (`sageattention3_blackwell/`)

Separate experimental module for Blackwell GPUs using FP4 microscaling. Not part of the main `sageattention` package.
