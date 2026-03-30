# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton
import triton.language as tl

from fla.utils import autotune_cache_kwargs, input_guard


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BD': BD}, num_warps=num_warps)
        for BD in [32, 64, 128, 256]
        for num_warps in [1, 2, 4, 8]
    ],
    key=['D'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_gm_swa_fwd_kernel(
    x,
    g,
    o,
    h0,
    ht,
    cu_seqlens,
    T,
    HM: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_hm, i_n = tl.program_id(0), tl.program_id(1)

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = i_n * T

    o_d = tl.arange(0, BD)
    mask = o_d < D

    p_x = x + bos * HM * D + i_hm * D + o_d
    p_g = g + bos * HM + i_hm
    p_o = o + bos * HM * D + i_hm * D + o_d

    b_h = tl.zeros([BD], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + (i_n * HM + i_hm) * D + o_d
        b_h = tl.load(p_h0, mask=mask, other=0).to(tl.float32)

    for _ in range(0, T):
        b_x = tl.load(p_x, mask=mask, other=0).to(tl.float32)
        b_g = tl.load(p_g).to(tl.float32)
        b_h = b_g * b_h + (1.0 - b_g) * b_x
        tl.store(p_o, b_h.to(p_o.dtype.element_ty), mask=mask)

        p_x += HM * D
        p_g += HM
        p_o += HM * D

    if STORE_FINAL_STATE:
        p_ht = ht + (i_n * HM + i_hm) * D + o_d
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask)


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BD': BD}, num_warps=num_warps)
        for BD in [32, 64, 128, 256]
        for num_warps in [1, 2, 4, 8]
    ],
    key=['D'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_gm_swa_bwd_kernel(
    x,
    g,
    o,
    h0,
    dx,
    dg,
    do,
    dht,
    dh0,
    cu_seqlens,
    T,
    HM: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_hm, i_n = tl.program_id(0), tl.program_id(1)

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = i_n * T

    o_d = tl.arange(0, BD)
    mask = o_d < D

    b_dh = tl.zeros([BD], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = dht + (i_n * HM + i_hm) * D + o_d
        b_dh = tl.load(p_dht, mask=mask, other=0).to(tl.float32)

    for i in range(T - 1, -1, -1):
        idx = bos + i
        p_x = x + idx * HM * D + i_hm * D + o_d
        p_do = do + idx * HM * D + i_hm * D + o_d
        p_dx = dx + idx * HM * D + i_hm * D + o_d
        p_g = g + idx * HM + i_hm
        p_dg = dg + idx * HM + i_hm

        b_x = tl.load(p_x, mask=mask, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask, other=0).to(tl.float32)
        b_g = tl.load(p_g).to(tl.float32)

        if i > 0:
            p_prev = o + (idx - 1) * HM * D + i_hm * D + o_d
            b_prev = tl.load(p_prev, mask=mask, other=0).to(tl.float32)
        elif USE_INITIAL_STATE:
            p_h0 = h0 + (i_n * HM + i_hm) * D + o_d
            b_prev = tl.load(p_h0, mask=mask, other=0).to(tl.float32)
        else:
            b_prev = tl.zeros([BD], dtype=tl.float32)

        b_dh += b_do
        b_dx = b_dh * (1.0 - b_g)
        b_dg = tl.sum(b_dh * (b_prev - b_x), axis=0)
        b_dh = b_dh * b_g

        tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), mask=mask)
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty))

    if USE_INITIAL_STATE:
        p_dh0 = dh0 + (i_n * HM + i_hm) * D + o_d
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), mask=mask)


def fused_recurrent_gm_swa_fwd(
    x: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    B, T, HM, D = x.shape
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    o = torch.empty_like(x)
    final_state = torch.empty((N, HM, D), device=x.device, dtype=torch.float32) if output_final_state else None

    def grid(meta):
        return (HM, N)

    fused_recurrent_gm_swa_fwd_kernel[grid](
        x=x,
        g=g,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        T=T,
        HM=HM,
        D=D,
    )
    return o, final_state


def fused_recurrent_gm_swa_bwd(
    x: torch.Tensor,
    g: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    B, T, HM, D = x.shape
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    dx = torch.empty_like(o, dtype=torch.float)
    dg = torch.empty_like(g, dtype=torch.float)
    dh0 = torch.empty((N, HM, D), device=o.device, dtype=torch.float32) if initial_state is not None else None

    def grid(meta):
        return (HM, N)

    fused_recurrent_gm_swa_bwd_kernel[grid](
        x=x,
        g=g,
        o=o,
        h0=initial_state,
        dx=dx,
        dg=dg,
        do=do,
        dht=dht,
        dh0=dh0,
        cu_seqlens=cu_seqlens,
        T=T,
        HM=HM,
        D=D,
    )
    return dx, dg, dh0


class FusedRecurrentGMSWAFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        x: torch.Tensor,
        g: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
    ):
        o, ht = fused_recurrent_gm_swa_fwd(
            x=x,
            g=g,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
        ctx.save_for_backward(x, g, o, initial_state)
        ctx.cu_seqlens = cu_seqlens
        return o, ht

    @staticmethod
    @input_guard
    def backward(ctx, do, dht=None):
        x, g, o, initial_state = ctx.saved_tensors
        dx, dg, dh0 = fused_recurrent_gm_swa_bwd(
            x=x,
            g=g,
            o=o,
            do=do,
            dht=dht,
            initial_state=initial_state,
            cu_seqlens=ctx.cu_seqlens,
        )
        return dx, dg, dh0, None, None


@torch.compiler.disable
def fused_recurrent_gm_swa(
    x: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    return FusedRecurrentGMSWAFunction.apply(
        x,
        g,
        initial_state,
        output_final_state,
        cu_seqlens,
    )
