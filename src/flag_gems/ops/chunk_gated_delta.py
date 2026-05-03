import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


def _next_power_of_2(n: int) -> int:
    """Return smallest power of 2 >= n (minimum 16)."""
    n = max(n, 1)
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    return max(n, 16)


@triton.jit
def _chunk_gated_delta_fwd_kernel(
    q_ptr, k_ptr, v_ptr, beta_ptr, out_ptr, state_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_betab, stride_betah, stride_betas,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_stateb, stride_stateh, stride_statec, stride_stated1, stride_stated2,
    BATCH, HEADS, SEQ_LEN, DIM, NUM_CHUNKS,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    q_base = q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    k_base = k_ptr + batch_idx * stride_kb + head_idx * stride_kh
    v_base = v_ptr + batch_idx * stride_vb + head_idx * stride_vh
    beta_base = beta_ptr + batch_idx * stride_betab + head_idx * stride_betah
    o_base = out_ptr + batch_idx * stride_ob + head_idx * stride_oh
    state_base = state_ptr + batch_idx * stride_stateb + head_idx * stride_stateh

    offs_d = tl.arange(0, BLOCK_SIZE_D)
    offs_c = tl.arange(0, BLOCK_SIZE_C)
    dim_mask = offs_d < DIM
    causal_mask = offs_c[:, None] >= offs_c[None, :]

    # Running state S_c, accumulated across chunks (fp32 for precision)
    state = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_D), dtype=tl.float32)

    for chunk_idx in range(NUM_CHUNKS):
        seq_start = chunk_idx * BLOCK_SIZE_C
        seq_offs = seq_start + offs_c
        seq_mask = seq_offs < SEQ_LEN

        # Save state BEFORE this chunk; the backward pass needs it for
        # the inter-chunk gradient (dq_inter) and dbeta.
        state_ptrs = (
            state_base
            + chunk_idx * stride_statec
            + offs_d[:, None] * stride_stated1
            + offs_d[None, :] * stride_stated2
        )
        tl.store(
            state_ptrs,
            state.to(state_ptr.dtype.element_ty),
            mask=dim_mask[:, None] & dim_mask[None, :],
        )

        # ---- Load inputs ----
        q_ptrs = q_base + seq_offs[:, None] * stride_qs + offs_d[None, :] * stride_qd
        k_ptrs = k_base + seq_offs[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = v_base + seq_offs[:, None] * stride_vs + offs_d[None, :] * stride_vd
        beta_ptrs = beta_base + seq_offs * stride_betas
        o_ptrs = o_base + seq_offs[:, None] * stride_os + offs_d[None, :] * stride_od

        q = tl.load(q_ptrs, mask=seq_mask[:, None] & dim_mask[None, :], other=0.0)
        k = tl.load(k_ptrs, mask=seq_mask[:, None] & dim_mask[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=seq_mask[:, None] & dim_mask[None, :], other=0.0)
        beta = tl.load(beta_ptrs, mask=seq_mask, other=0.0).to(tl.float32)

        # ---- Decay factors ----
        # cum_log_beta[j] = log(beta[0]) + ... + log(beta[j])  (inclusive)
        log_beta = tl.log(beta + 1e-6)
        cum_log_beta = tl.cumsum(log_beta, axis=0)
        # Total decay over the full chunk (= cum_log_beta at the last position)
        total_log_beta = tl.sum(log_beta, axis=0)
        # Factor applied to state contribution at position j: exp(cum[j])
        decay_from_start = tl.math.exp(cum_log_beta)
        # Factor for key decaying to end of chunk: exp(total - cum[j])
        decay_to_end = tl.math.exp(total_log_beta - cum_log_beta)

        # ---- Intra-chunk causal attention with decay ----
        # decay_matrix[j, l] = exp(cum[j] - cum[l]) for j >= l, else 0
        decay_matrix = tl.math.exp(cum_log_beta[:, None] - cum_log_beta[None, :])
        decay_masked = tl.where(causal_mask, decay_matrix, 0.0)
        qk = tl.dot(q.to(tl.float32), tl.trans(k).to(tl.float32))
        attn_weights = (qk * decay_masked).to(q.dtype)
        out_intra = tl.dot(attn_weights, v)

        # ---- Inter-chunk output ----
        # out_inter[j] = exp(cum[j]) * (q[j] @ S_c)
        q_scaled = (q.to(tl.float32) * decay_from_start[:, None]).to(q.dtype)
        out_inter = tl.dot(q_scaled, state.to(q_scaled.dtype))

        out = out_intra.to(tl.float32) + out_inter.to(tl.float32)
        tl.store(
            o_ptrs,
            out.to(q_ptr.dtype.element_ty),
            mask=seq_mask[:, None] & dim_mask[None, :],
        )

        # ---- Update running state ----
        # S_{c+1} = exp(total) * S_c + sum_j (k[j] * exp(total - cum[j])) ⊗ v[j]
        k_decayed = k.to(tl.float32) * decay_to_end[:, None]
        state = (
            state * tl.math.exp(total_log_beta)
            + tl.dot(tl.trans(k_decayed), v.to(tl.float32))
        )


@triton.jit
def _chunk_gated_delta_bwd_kernel(
    q_ptr, k_ptr, v_ptr, beta_ptr, do_ptr, fwd_state_ptr,
    dq_ptr, dk_ptr, dv_ptr, dbeta_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_betab, stride_betah, stride_betas,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_fsb, stride_fsh, stride_fsc, stride_fsd1, stride_fsd2,
    BATCH, HEADS, SEQ_LEN, DIM, NUM_CHUNKS,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    base_qkv = batch_idx * stride_qb + head_idx * stride_qh
    base_beta = batch_idx * stride_betab + head_idx * stride_betah
    base_do = batch_idx * stride_ob + head_idx * stride_oh
    base_fs = batch_idx * stride_fsb + head_idx * stride_fsh

    q_base = q_ptr + base_qkv
    k_base = k_ptr + base_qkv
    v_base = v_ptr + base_qkv
    beta_base = beta_ptr + base_beta
    do_base = do_ptr + base_do
    dq_base = dq_ptr + base_qkv
    dk_base = dk_ptr + base_qkv
    dv_base = dv_ptr + base_qkv
    dbeta_base = dbeta_ptr + base_beta
    fs_base = fwd_state_ptr + base_fs

    offs_d = tl.arange(0, BLOCK_SIZE_D)
    offs_c = tl.arange(0, BLOCK_SIZE_C)
    dim_mask = offs_d < DIM
    causal_mask = offs_c[:, None] >= offs_c[None, :]

    # d_state = G_{c+1}: gradient w.r.t. state entering chunk c+1 (fp32)
    d_state = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_D), dtype=tl.float32)

    for chunk_idx in range(NUM_CHUNKS - 1, -1, -1):
        seq_start = chunk_idx * BLOCK_SIZE_C
        seq_offs = seq_start + offs_c
        seq_mask = seq_offs < SEQ_LEN

        # ---- Load inputs ----
        q_ptrs = q_base + seq_offs[:, None] * stride_qs + offs_d[None, :] * stride_qd
        k_ptrs = k_base + seq_offs[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = v_base + seq_offs[:, None] * stride_vs + offs_d[None, :] * stride_vd
        do_ptrs = do_base + seq_offs[:, None] * stride_os + offs_d[None, :] * stride_od
        beta_ptrs = beta_base + seq_offs * stride_betas
        fs_ptrs = (
            fs_base
            + chunk_idx * stride_fsc
            + offs_d[:, None] * stride_fsd1
            + offs_d[None, :] * stride_fsd2
        )

        q = tl.load(q_ptrs, mask=seq_mask[:, None] & dim_mask[None, :], other=0.0)
        k = tl.load(k_ptrs, mask=seq_mask[:, None] & dim_mask[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=seq_mask[:, None] & dim_mask[None, :], other=0.0)
        do = tl.load(do_ptrs, mask=seq_mask[:, None] & dim_mask[None, :], other=0.0)
        beta = tl.load(beta_ptrs, mask=seq_mask, other=0.0).to(tl.float32)
        fwd_state = tl.load(
            fs_ptrs,
            mask=dim_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # ---- Recompute decay factors (same as forward) ----
        log_beta = tl.log(beta + 1e-6)
        cum_log_beta = tl.cumsum(log_beta, axis=0)
        total_log_beta = tl.sum(log_beta, axis=0)
        decay_from_start = tl.math.exp(cum_log_beta)
        decay_to_end = tl.math.exp(total_log_beta - cum_log_beta)
        decay_matrix = tl.math.exp(cum_log_beta[:, None] - cum_log_beta[None, :])
        decay_masked = tl.where(causal_mask, decay_matrix, 0.0)

        # Recompute forward attention weights (needed for dv, dk, dbeta)
        qk = tl.dot(q.to(tl.float32), tl.trans(k).to(tl.float32))
        attn_weights = (qk * decay_masked).to(q.dtype)
        k_decayed = k.to(tl.float32) * decay_to_end[:, None]

        # ---- dv ----
        # dv_intra: gradient through intra-chunk attention output
        dv_intra = tl.dot(tl.trans(attn_weights), do)
        # dv_inter: gradient through state update (k_decayed ⊗ v contribution)
        dv_inter = tl.dot(k_decayed.to(do.dtype), d_state.to(do.dtype))
        dv = dv_intra.to(tl.float32) + dv_inter.to(tl.float32)
        dv_ptrs = dv_base + seq_offs[:, None] * stride_vs + offs_d[None, :] * stride_vd
        tl.store(
            dv_ptrs,
            dv.to(v_ptr.dtype.element_ty),
            mask=seq_mask[:, None] & dim_mask[None, :],
        )

        # ---- dq ----
        do_v_t = tl.dot(do.to(tl.float32), tl.trans(v).to(tl.float32))  # (BT, BT)
        # dq_intra: gradient through intra-chunk causal attention
        dq_attn = tl.where(causal_mask, do_v_t * decay_matrix, 0.0)
        dq_intra = tl.dot(dq_attn.to(q.dtype), k)
        # dq_inter: gradient through exp(cum[j]) * q[j] @ S_c term
        dq_inter = (
            tl.dot(do.to(tl.float32), tl.trans(fwd_state))
            * decay_from_start[:, None]
        )
        dq = dq_intra.to(tl.float32) + dq_inter
        dq_ptrs = dq_base + seq_offs[:, None] * stride_qs + offs_d[None, :] * stride_qd
        tl.store(
            dq_ptrs,
            dq.to(q_ptr.dtype.element_ty),
            mask=seq_mask[:, None] & dim_mask[None, :],
        )

        # ---- dk ----
        # dk_intra: gradient through intra-chunk causal attention
        dk_intra = tl.dot(tl.trans(dq_attn).to(q.dtype), q)
        # dk_inter: gradient through state update (k * decay_to_end[j] contribution)
        v_decayed = v.to(tl.float32) * decay_to_end[:, None]
        dk_inter = tl.dot(v_decayed.to(do.dtype), tl.trans(d_state).to(do.dtype))
        dk = dk_intra.to(tl.float32) + dk_inter.to(tl.float32)
        dk_ptrs = dk_base + seq_offs[:, None] * stride_ks + offs_d[None, :] * stride_kd
        tl.store(
            dk_ptrs,
            dk.to(k_ptr.dtype.element_ty),
            mask=seq_mask[:, None] & dim_mask[None, :],
        )

        # ---- dbeta (mathematically exact gradient) ----
        #
        # beta[t] appears via log_beta[t] = log(beta[t] + eps) and
        # cum_log_beta[j] = sum_{i<=j} log_beta[i].  Three sources:
        #
        # 1. Intra-chunk decay: W[j,l] = qk[j,l]*exp(cum[j]-cum[l])*(do[j]·v[l])
        #    dL/d(log_beta[t]) = sum_{j>=t} row_sum_W[j] - sum_{l>=t} col_sum_W[l]
        #
        # 2. Inter-chunk output: val_inter[j] = exp(cum[j])*(q[j]@S_c)·do[j]
        #    dL/d(log_beta[t]) = sum_{j>=t} val_inter[j]  (reverse cumsum)
        #
        # 3. State update: S_{c+1} = exp(total)*S_c + sum_j k_decayed[j]⊗v[j]
        #    dL/d(log_beta[t]) = exp(total)*<G_{c+1},S_c>
        #                       + sum_{l<t} decay_to_end[l]*k[l]·(G_{c+1}@v[l])

        # Source 1
        W_elem = qk * decay_masked * do_v_t   # (BT, BT) fp32; causal from decay_masked
        row_sum_W = tl.sum(W_elem, axis=1)    # (BT,)
        col_sum_W = tl.sum(W_elem, axis=0)    # (BT,)
        # Reverse cumsum: sum_{j>=t} x[j] = total - inclusive_cumsum[t] + x[t]
        A = tl.sum(row_sum_W, axis=0) - tl.cumsum(row_sum_W, axis=0) + row_sum_W
        B = tl.sum(col_sum_W, axis=0) - tl.cumsum(col_sum_W, axis=0) + col_sum_W
        dlog_beta_intra = A - B

        # Source 2
        q_state = tl.dot(q.to(tl.float32), fwd_state)   # (BT, D)
        val_inter = decay_from_start * tl.sum(q_state * do.to(tl.float32), axis=1)
        dlog_beta_inter = (
            tl.sum(val_inter, axis=0) - tl.cumsum(val_inter, axis=0) + val_inter
        )

        # Source 3
        fs_dot_ds_rows = tl.sum(fwd_state * d_state, axis=1)  # (BD,)
        fs_dot_ds = tl.sum(fs_dot_ds_rows, axis=0)            # scalar
        const_state = tl.math.exp(total_log_beta) * fs_dot_ds
        k_d_v = tl.sum(
            tl.dot(k.to(tl.float32), d_state) * v.to(tl.float32), axis=1
        )  # (BT,)
        contrib = decay_to_end * k_d_v
        excl_contrib = tl.cumsum(contrib, axis=0) - contrib  # exclusive prefix sum
        dlog_beta_state = const_state + excl_contrib

        dlog_beta = dlog_beta_intra + dlog_beta_inter + dlog_beta_state
        dbeta = tl.where(seq_mask, dlog_beta / (beta + 1e-6), 0.0)
        dbeta_ptrs = dbeta_base + seq_offs * stride_betas
        tl.store(
            dbeta_ptrs,
            dbeta.to(beta_ptr.dtype.element_ty),
            mask=seq_mask,
        )

        # ---- Update d_state: G_c = exp(total)*G_{c+1} + (q*exp(cum))^T @ do ----
        q_scaled = q.to(tl.float32) * decay_from_start[:, None]
        d_state = (
            d_state * tl.math.exp(total_log_beta)
            + tl.dot(tl.trans(q_scaled), do.to(tl.float32))
        )


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, beta, BT):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        beta = beta.contiguous()

        BATCH, HEADS, SEQ_LEN, DIM = q.shape
        BLOCK_SIZE_C = BT
        BLOCK_SIZE_D = _next_power_of_2(DIM)
        NUM_CHUNKS = triton.cdiv(SEQ_LEN, BLOCK_SIZE_C)

        out = torch.empty_like(q)
        # fwd_states[b, h, chunk, d1, d2] = state BEFORE chunk (needed in backward)
        fwd_states = torch.zeros(
            (BATCH, HEADS, NUM_CHUNKS, DIM, DIM),
            dtype=torch.float32,
            device=q.device,
        )

        grid = (BATCH, HEADS)
        _chunk_gated_delta_fwd_kernel[grid](
            q, k, v, beta, out, fwd_states,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            beta.stride(0), beta.stride(1), beta.stride(2),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            fwd_states.stride(0), fwd_states.stride(1),
            fwd_states.stride(2),  # stride_statec  (chunk stride)
            fwd_states.stride(3), fwd_states.stride(4),
            BATCH, HEADS, SEQ_LEN, DIM, NUM_CHUNKS,
            BLOCK_SIZE_C=BLOCK_SIZE_C,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
        )

        ctx.save_for_backward(q, k, v, beta, fwd_states)
        ctx.dims = (BATCH, HEADS, SEQ_LEN, DIM, NUM_CHUNKS)
        ctx.blocks = (BLOCK_SIZE_C, BLOCK_SIZE_D)
        return out

    @staticmethod
    def backward(ctx, do):
        do = do.contiguous()
        q, k, v, beta, fwd_states = ctx.saved_tensors
        BATCH, HEADS, SEQ_LEN, DIM, NUM_CHUNKS = ctx.dims
        BLOCK_SIZE_C, BLOCK_SIZE_D = ctx.blocks

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dbeta = torch.empty_like(beta)

        grid = (BATCH, HEADS)
        _chunk_gated_delta_bwd_kernel[grid](
            q, k, v, beta, do, fwd_states,
            dq, dk, dv, dbeta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            beta.stride(0), beta.stride(1), beta.stride(2),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            fwd_states.stride(0), fwd_states.stride(1),
            fwd_states.stride(2),  # stride_fsc  (chunk stride)
            fwd_states.stride(3), fwd_states.stride(4),
            BATCH, HEADS, SEQ_LEN, DIM, NUM_CHUNKS,
            BLOCK_SIZE_C=BLOCK_SIZE_C,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
        )

        return dq, dk, dv, dbeta, None  # None for BT (not a tensor)


def chunk_gated_delta_rule(q, k, v, beta, BT: int = 64):
    """Chunked gated delta-rule linear attention.

    Recurrence:  S_t = beta_t * S_{t-1} + k_t ⊗ v_t
    Output:      o_t = q_t @ S_t

    The sequence is processed in chunks of size *BT* for efficiency.
    Sequences whose length is not a multiple of BT are zero-padded
    internally; the padding is not visible in the returned tensor.

    Args:
        q:    (Batch, Heads, SeqLen, Dim)
        k:    (Batch, Heads, SeqLen, Dim)
        v:    (Batch, Heads, SeqLen, Dim)
        beta: (Batch, Heads, SeqLen) – per-step decay gate, values in (0, 1]
        BT:   chunk size; must be a power of 2 and >= 16 (default 64)

    Returns:
        out:  (Batch, Heads, SeqLen, Dim)
    """
    B, H, T, D = q.shape
    T_pad = triton.cdiv(T, BT) * BT
    if T_pad > T:
        pad_len = T_pad - T
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        # beta = 1.0 at padded positions so the state passes through unchanged
        beta = F.pad(beta, (0, pad_len), value=1.0)
    out = ChunkGatedDeltaRuleFunction.apply(q, k, v, beta, BT)
    return out[:, :, :T, :]


class FlagOS_ChunkGatedDelta(nn.Module):
    """nn.Module wrapper for :func:`chunk_gated_delta_rule`."""

    def __init__(self, BT: int = 64):
        super().__init__()
        self.BT = BT

    def forward(self, q, k, v, beta):
        """
        Args:
            q:    (Batch, Heads, SeqLen, Dim)
            k:    (Batch, Heads, SeqLen, Dim)
            v:    (Batch, Heads, SeqLen, Dim)
            beta: (Batch, Heads, SeqLen) – decay gate
        Returns:
            out:  (Batch, Heads, SeqLen, Dim)
        """
        return chunk_gated_delta_rule(q, k, v, beta, BT=self.BT)
