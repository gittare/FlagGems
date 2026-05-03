"""
chunk_gated_delta.py – Chunk Gated Delta Rule for FlagGems
==========================================================

Forward recurrence (per head, sequential over T):
    r_t = v_t  −  k_t @ S_{t-1}          # residual
    S_t = S_{t-1}  +  β_t · outer(k_t, r_t)   # rank-1 state update
    o_t = q_t @ S_t                        # output

Key design decisions vs. the broken original
--------------------------------------------
* No ``tl.make_block_ptr`` – that was the source of the LLVM struct
  size-mismatch ("expected 4 but got 1").  Plain scalar-offset loads
  are used throughout.
* No ``tl.dot`` – its shape constraints (inner dim ≥ 16, both dims
  power-of-2) caused silent wrong results for small D.  All products
  are expressed as element-wise broadcast + ``tl.sum``.
* BD (block dim) is the next power-of-two ≥ D.  A mask guards
  out-of-range lanes when BD > D.
* Backward is a pure-PyTorch analytic reverse scan – correct and
  differentiable, easy to audit.  A Triton backward can be added later.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl

__all__ = [
    "torch_chunk_gated_delta_rule",
    "chunk_gated_delta_rule",
    "FlagOS_ChunkGatedDelta",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PyTorch Reference  (used for validation and gradient checking)
# ─────────────────────────────────────────────────────────────────────────────


def torch_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """
    Sequential PyTorch reference implementation.

    Args:
        q, k, v : (B, H, T, D)
        beta    : (B, H, T)

    Returns:
        o : (B, H, T, D)
    """
    B, H, T, D = q.shape
    o = torch.zeros_like(q)
    for b in range(B):
        for h in range(H):
            S = q.new_zeros(D, D)
            for t in range(T):
                k_t = k[b, h, t]  # [D]
                r_t = v[b, h, t] - k_t @ S  # [D]
                S = S + beta[b, h, t] * torch.outer(k_t, r_t)
                o[b, h, t] = q[b, h, t] @ S
    return o


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Triton Forward Kernel
# ─────────────────────────────────────────────────────────────────────────────


@triton.jit
def _cgd_fwd_kernel(
    # ── tensor pointers ──────────────────────────────────────────────────────
    Q,
    K,
    V,
    Beta,
    O,
    # ── strides for Q  (B, H, T, D) ─────────────────────────────────────────
    sq_b,
    sq_h,
    sq_t,
    sq_d,
    # ── strides for K ────────────────────────────────────────────────────────
    sk_b,
    sk_h,
    sk_t,
    sk_d,
    # ── strides for V ────────────────────────────────────────────────────────
    sv_b,
    sv_h,
    sv_t,
    sv_d,
    # ── strides for Beta  (B, H, T) ──────────────────────────────────────────
    sb_b,
    sb_h,
    sb_t,
    # ── strides for O ────────────────────────────────────────────────────────
    so_b,
    so_h,
    so_t,
    so_d,
    # ── runtime scalars ──────────────────────────────────────────────────────
    H,  # int – number of heads
    T,  # int – sequence length
    D,  # int – head dimension
    # ── compile-time constant ────────────────────────────────────────────────
    BD: tl.constexpr,  # next_power_of_2(D) >= D
):
    """
    One Triton program per (batch, head) pair.

    State S  [BD × BD] lives entirely in registers.
    T steps are executed sequentially inside a single thread block so that
    the data dependency  S_t → S_{t+1}  is respected.

    All tensor reads/writes use plain scalar offsets – no block-pointer
    API – to avoid the LLVM struct packing error that plagued the original.
    """
    bh = tl.program_id(0)
    b = bh // H
    h = bh % H

    # Lane indices and validity mask
    d = tl.arange(0, BD)  # [BD]
    mask = d < D  # True for valid lanes

    # Precompute base offsets (time-independent)
    q0 = b * sq_b + h * sq_h
    k0 = b * sk_b + h * sk_h
    v0 = b * sv_b + h * sv_h
    b0 = b * sb_b + h * sb_h
    o0 = b * so_b + h * so_h

    # State matrix, initialised to zero
    S = tl.zeros([BD, BD], dtype=tl.float32)  # [BD, BD]

    for t in range(T):
        # ── load vectors for timestep t ─────────────────────────────────────
        k_t = tl.load(K + k0 + t * sk_t + d * sk_d, mask=mask, other=0.0).to(
            tl.float32
        )  # [BD]
        v_t = tl.load(V + v0 + t * sv_t + d * sv_d, mask=mask, other=0.0).to(
            tl.float32
        )  # [BD]
        q_t = tl.load(Q + q0 + t * sq_t + d * sq_d, mask=mask, other=0.0).to(
            tl.float32
        )  # [BD]
        bt = tl.load(Beta + b0 + t * sb_t).to(tl.float32)  # scalar

        # ── delta-rule state update ─────────────────────────────────────────
        #   kS[j]  = Σ_i  k_t[i] * S[i, j]
        kS = tl.sum(k_t[:, None] * S, axis=0)  # [BD]

        #   r_t    = v_t − kS
        r_t = v_t - kS  # [BD]

        #   S     += β · outer(k_t, r_t)
        #   outer[i,j] = k_t[i] * r_t[j]  via broadcast
        S = S + bt * (k_t[:, None] * r_t[None, :])  # [BD, BD]

        # ── output ──────────────────────────────────────────────────────────
        #   o_t[j] = Σ_i  q_t[i] * S[i, j]
        o_t = tl.sum(q_t[:, None] * S, axis=0)  # [BD]

        tl.store(O + o0 + t * so_t + d * so_d, o_t, mask=mask)


def _triton_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    BT: int,  # kept for API compatibility; kernel is not chunked
) -> torch.Tensor:
    """Host-side launcher for ``_cgd_fwd_kernel``."""
    B, H, T, D = q.shape
    BD = triton.next_power_of_2(D)

    # Ensure inputs are contiguous so strides are predictable
    q, k, v, beta = (t.contiguous() for t in (q, k, v, beta))

    o = torch.empty_like(q)

    _cgd_fwd_kernel[(B * H,)](
        q,
        k,
        v,
        beta,
        o,
        # strides Q
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        # strides K
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        # strides V
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        # strides Beta
        beta.stride(0),
        beta.stride(1),
        beta.stride(2),
        # strides O
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        # runtime dims
        H=H,
        T=T,
        D=D,
        # compile-time block dim
        BD=BD,
    )
    return o


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PyTorch Analytic Backward  (reverse sequential scan)
# ─────────────────────────────────────────────────────────────────────────────


def _torch_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute ∂L/∂q, ∂L/∂k, ∂L/∂v, ∂L/∂β via a two-pass algorithm:

    Pass 1 (forward)  – re-run the recurrence to cache  S_{t-1}  and  r_t.
    Pass 2 (backward) – propagate  dS  backward, accumulating gradients.

    Derivation (index notation, L = scalar loss)
    --------------------------------------------
    From  o_t = q_t @ S_t:
        dq_t  = S_t  @ do_t
        dS_t += outer(q_t, do_t)

    From  S_t = S_{t-1} + β_t · outer(k_t, r_t):
        dβ_t  = k_t  @ (dS_t @ r_t)
        dk_t += β_t  *  dS_t @ r_t
        dr_t  = β_t  *  dS_t.T @ k_t
        dS_{t-1} = dS_t  −  outer(k_t, dr_t)   ← pass-through minus kS term

    From  r_t = v_t − k_t @ S_{t-1}:
        dv_t += dr_t
        dk_t -= S_{t-1} @ dr_t
        (dS_{t-1} contribution already folded in above)
    """
    B, H, T, D = q.shape

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    dbeta = torch.zeros_like(beta)

    for b in range(B):
        for h in range(H):
            # ── Pass 1: forward to cache S_{t-1} and r_t ──────────────────
            S_prevs = []  # S_{t-1}  for each t
            residuals = []  # r_t      for each t
            S = q.new_zeros(D, D)

            for t in range(T):
                S_prevs.append(S.clone())
                k_t = k[b, h, t]
                r_t = v[b, h, t] - k_t @ S
                residuals.append(r_t.clone())
                S = S + beta[b, h, t] * torch.outer(k_t, r_t)

            # ── Pass 2: backward scan ──────────────────────────────────────
            dS = q.new_zeros(D, D)  # ∂L/∂S_t; starts at 0 (= ∂L/∂S_T)

            for t in range(T - 1, -1, -1):
                S_prev = S_prevs[t]
                k_t = k[b, h, t]
                r_t = residuals[t]
                bt = beta[b, h, t]
                do_t = do[b, h, t]

                # Reconstruct S_t for the dq update
                S_t = S_prev + bt * torch.outer(k_t, r_t)

                # ∂L/∂q_t  and  accumulate ∂L/∂S_t from o_t
                dq[b, h, t] = S_t @ do_t
                dS = dS + torch.outer(q[b, h, t], do_t)

                # Gradients through S_t = S_prev + β · outer(k_t, r_t)
                dSr = dS @ r_t  # [D]
                dbeta[b, h, t] = torch.dot(k_t, dSr)
                dk[b, h, t] += bt * dSr
                dr_t = bt * (dS.T @ k_t)  # [D]

                # Gradients through r_t = v_t − k_t @ S_prev
                dv[b, h, t] += dr_t
                dk[b, h, t] -= S_prev @ dr_t

                # Pass ∂L/∂S_{t-1} to the next (earlier) step
                dS = dS - torch.outer(k_t, dr_t)

    return dq, dk, dv, dbeta


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Custom Autograd Function
# ─────────────────────────────────────────────────────────────────────────────


class _ChunkGatedDeltaFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, beta, BT):
        o = _triton_forward(q, k, v, beta, BT)
        ctx.save_for_backward(q, k, v, beta)
        ctx.BT = BT
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, beta = ctx.saved_tensors
        dq, dk, dv, dbeta = _torch_backward(q, k, v, beta, do.contiguous())
        return dq, dk, dv, dbeta, None  # no grad for BT


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Public functional API
# ─────────────────────────────────────────────────────────────────────────────


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    BT: int = 16,
) -> torch.Tensor:
    """
    Triton-accelerated chunk gated delta rule (forward + analytic backward).

    Args:
        q, k, v : (B, H, T, D)  – query / key / value
        beta    : (B, H, T)     – per-step gate / learning rate
        BT      : chunk size    – kept for API compatibility

    Returns:
        o : (B, H, T, D)
    """
    return _ChunkGatedDeltaFn.apply(q, k, v, beta, BT)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  nn.Module wrapper
# ─────────────────────────────────────────────────────────────────────────────


class FlagOS_ChunkGatedDelta(nn.Module):
    """
    ``nn.Module`` wrapper around ``chunk_gated_delta_rule``.

    Usage::
        layer = FlagOS_ChunkGatedDelta(BT=16)
        o = layer(q, k, v, beta)
    """

    def __init__(self, BT: int = 16) -> None:
        super().__init__()
        self.BT = BT

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        return chunk_gated_delta_rule(q, k, v, beta, BT=self.BT)
