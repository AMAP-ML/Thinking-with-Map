# PKPO.py
from __future__ import annotations
import torch


def _m_normed(N: int, K: int, i: int, j: int) -> float:
    # 注意：这里严格用 “> K-1” 与你的 numpy reference 一致
    if i == j and i > K - 1:
        num = 1.0
        den = 1.0
        for a, b in zip(range(i - K + 2, i + 1), range(N - K + 2, N + 1)):
            num *= float(a)
            den *= float(b)
        return (K / (N - K + 1)) * (num / den)

    elif j > i and j > K - 1 and K >= 2:
        num = 1.0
        den = 1.0
        for a, b in zip(range(j - K + 2, j), range(N - K + 2, N)):
            num *= float(a)
            den *= float(b)
        return (K / (N - K + 1)) * ((K - 1) / N) * (num / den)

    return 0.0


def _m_diagonal(N: int, K: int, device=None, dtype=torch.float64) -> torch.Tensor:
    return torch.tensor([_m_normed(N, K, i, i) for i in range(N)],
                        device=device, dtype=dtype)


def _delta(N: int, K: int, i: int) -> float:
    return _m_normed(N, K, i, i + 1) - _m_normed(N, K, i + 1, i + 1)


def _deltas(N: int, K: int, device=None, dtype=torch.float64) -> torch.Tensor:
    return torch.tensor([_delta(N - 1, K, i) for i in range(N - 2)],
                        device=device, dtype=dtype)


def _sorted_apply_1d(x: torch.Tensor, func) -> torch.Tensor:
    sort_idx = torch.argsort(x)
    inv = torch.empty_like(sort_idx)
    inv[sort_idx] = torch.arange(sort_idx.numel(), device=x.device)
    y_sorted = func(x[sort_idx])
    return y_sorted[inv]


def _s_sorted(g_sorted: torch.Tensor, K: int) -> torch.Tensor:
    # 对齐 numpy：c[:N-1] += g[1:]*_deltas(N+1,K)（即使 N=1 也不会进来）
    N = g_sorted.numel()
    md = _m_diagonal(N, K, device=g_sorted.device, dtype=g_sorted.dtype)
    c = g_sorted * md
    if N >= 2:
        c[: N - 1] = c[: N - 1] + g_sorted[1:] * _deltas(N + 1, K, device=g_sorted.device, dtype=g_sorted.dtype)
    return torch.cumsum(torch.flip(c, dims=[0]), dim=0).flip(dims=[0])


def _b_sorted(g_sorted: torch.Tensor, K: int) -> torch.Tensor:
    N = g_sorted.numel()
    # numpy 对 N=1 实际不会调用到 b（因为 pass@k 至少2），这里仍做防御
    if N == 1:
        return torch.zeros_like(g_sorted)

    md = _m_diagonal(N - 1, K, device=g_sorted.device, dtype=g_sorted.dtype)  # (N-1,)
    w = (-md * torch.arange(1, N, device=g_sorted.device, dtype=g_sorted.dtype))

    if N > 2:
        w[1:] = w[1:] + _deltas(N, K, device=g_sorted.device, dtype=g_sorted.dtype) * torch.arange(
            1, N - 1, device=g_sorted.device, dtype=g_sorted.dtype
        )

    c1 = torch.tensor([(w * g_sorted[1:]).sum()], device=g_sorted.device, dtype=g_sorted.dtype)
    c2 = (g_sorted[:-1] - g_sorted[1:]) * w
    return torch.cumsum(torch.cat([c1, c2], dim=0), dim=0)


def _sloo_minus_one_sorted(g_sorted: torch.Tensor, K: int) -> torch.Tensor:
    if K <= 1:
        raise ValueError(f"K must be >=2 for sloo_minus_one, got K={K}")
    N = g_sorted.numel()
    return _s_sorted(g_sorted, K) - _b_sorted(g_sorted, K - 1) * (K / (K - 1)) / float(N)


def sloo_minus_one(g: torch.Tensor, K: int) -> torch.Tensor:
    # 输出dtype跟随输入dtype（建议测试用 float64）
    return _sorted_apply_1d(g, lambda g_sorted: _sloo_minus_one_sorted(g_sorted, K))
