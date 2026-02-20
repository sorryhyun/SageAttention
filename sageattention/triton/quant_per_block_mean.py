"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _per_block_q_mean_kernel(
    Q, Q_out, Qm,
    L,
    stride_qb, stride_qh, stride_ql,
    stride_qmb, stride_qmh, stride_qml,
    GROUP_SIZE: tl.constexpr,
    D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_group = tl.program_id(2)

    group_start = pid_group * GROUP_SIZE
    offsets = group_start + tl.arange(0, GROUP_SIZE)
    mask = offsets < L

    # Effective count for this group (handles partial last group)
    remaining = L - group_start
    count = tl.where(remaining < GROUP_SIZE, remaining, GROUP_SIZE)

    # Load Q block: [GROUP_SIZE, D]
    q_ptrs = pid_b * stride_qb + pid_h * stride_qh + offsets[:, None] * stride_ql + tl.arange(0, D)[None, :]
    q_block = tl.load(Q + q_ptrs, mask=mask[:, None], other=0.0)

    # Compute mean in fp32: [D]
    qm = tl.sum(q_block.to(tl.float32), axis=0) / count.to(tl.float32)

    # Subtract mean and store in original dtype
    q_centered = q_block.to(tl.float32) - qm[None, :]
    tl.store(Q_out + q_ptrs, q_centered.to(Q_out.dtype.element_ty), mask=mask[:, None])

    # Store mean as fp32
    qm_ptrs = pid_b * stride_qmb + pid_h * stride_qmh + pid_group * stride_qml + tl.arange(0, D)
    tl.store(Qm + qm_ptrs, qm)


def per_block_q_mean(q, tensor_layout="HND", GROUP_SIZE=128):
    """Compute per-block mean of Q and subtract it.

    Args:
        q: Query tensor in HND [B, H, L, D] or NHD [B, L, H, D] layout.
        tensor_layout: "HND" or "NHD".
        GROUP_SIZE: Block size for mean computation (should match BLOCK_M=128).

    Returns:
        q_centered: Q with per-block mean subtracted, same shape/dtype as input.
        qm: Per-block means, shape [B, H, num_groups, D] in float32.
    """
    if tensor_layout == "HND":
        B, H, L, D = q.shape
        stride_b, stride_h, stride_l = q.stride(0), q.stride(1), q.stride(2)
    elif tensor_layout == "NHD":
        B, L, H, D = q.shape
        stride_b, stride_h, stride_l = q.stride(0), q.stride(2), q.stride(1)
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    num_groups = (L + GROUP_SIZE - 1) // GROUP_SIZE

    q_out = torch.empty_like(q)
    qm = torch.empty(B, H, num_groups, D, device=q.device, dtype=torch.float32)

    grid = (B, H, num_groups)
    _per_block_q_mean_kernel[grid](
        q, q_out, qm,
        L,
        stride_b, stride_h, stride_l,
        qm.stride(0), qm.stride(1), qm.stride(2),
        GROUP_SIZE=GROUP_SIZE,
        D=D,
    )

    return q_out, qm
