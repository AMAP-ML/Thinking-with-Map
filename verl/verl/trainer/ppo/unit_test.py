# test_sloo_minus_one.py
import numpy as np
import torch


# -------------------------
# Numpy reference (from your image)
# -------------------------

def np__m_normed(N: int, K: int, i: int, j: int) -> float:
    if i == j and i > K - 1:
        return (
            K / (N - K + 1)
            * np.prod(np.arange(i - K + 2, i + 1) / np.arange(N - K + 2, N + 1))
        )
    elif j > i and j > K - 1 and K >= 2:
        return (
            K / (N - K + 1)
            * (K - 1) / N
            * np.prod(np.arange(j - K + 2, j) / np.arange(N - K + 2, N))
        )
    return 0.0


def np__m_diagonal(N: int, K: int) -> np.ndarray:
    return np.array([np__m_normed(N, K, i, i) for i in range(N)], dtype=np.float64)


def np_rho(g: np.ndarray, K: int) -> float:
    return (np.sort(g) * -np__m_diagonal(len(g), K)).sum()


def np_delta(N: int, K: int, i: int) -> float:
    return np__m_normed(N, K, i, i + 1) - np__m_normed(N, K, i + 1, i + 1)


def np__deltas(N: int, K: int) -> np.ndarray:
    return np.array([np_delta(N - 1, K, i) for i in range(N - 2)], dtype=np.float64)


def np__sorted_apply(func):
    def inner(x: np.ndarray, *args, **kwargs) -> np.ndarray:
        i_sort = np.argsort(x)
        func_x = np.zeros_like(x, dtype=np.float64)
        func_x[i_sort] = func(x[i_sort], *args, **kwargs)
        return func_x
    return inner


@np__sorted_apply
def np_s(g: np.ndarray, K: int) -> np.ndarray:
    N = len(g)
    c = g * np__m_diagonal(N, K)
    c[: (N - 1)] += g[1:] * np__deltas(N + 1, K)
    return np.cumsum(c[::-1])[::-1]


@np__sorted_apply
def np__b(g: np.ndarray, K: int) -> np.ndarray:
    N = len(g)
    w = (-np__m_diagonal(N - 1, K) * np.arange(1, N)).astype(np.float64)
    w[1:] += np__deltas(N, K) * np.arange(1, N - 1)
    c1 = np.array([(w * g[1:]).sum()], dtype=np.float64)
    c2 = (g[:-1] - g[1:]) * w
    return np.cumsum(np.concatenate((c1, c2)))


def np_sloo_minus_one(g: np.ndarray, K: int) -> np.ndarray:
    return np_s(g, K) - np__b(g, K - 1) * K / (K - 1) / len(g)


# -------------------------
# Torch implementation under test
# (import from your module in real usage)
# -------------------------

def torch_sloo_minus_one(g: torch.Tensor, K: int) -> torch.Tensor:
    # You should import your real implementation; inline minimal call here:
    from PKPO import sloo_minus_one  # <-- change this
    return sloo_minus_one(g, K)


def test_sloo_minus_one_matches_numpy():
    rng = np.random.default_rng(0)

    for N in [2, 3, 5, 8]:
        for K in [2, min(3, N)]:  # K must be >=2, and reasonable vs N
            g = rng.normal(size=(N,)).astype(np.float64)

            np_out = np_sloo_minus_one(g, K)  # (N,)

            tg = torch.tensor(g, dtype=torch.float64)
            torch_out = torch_sloo_minus_one(tg, K).detach().cpu().numpy()

            assert np.allclose(torch_out, np_out, rtol=1e-10, atol=1e-10), \
                f"Mismatch for N={N}, K={K}\nnp={np_out}\ntorch={torch_out}"


test_sloo_minus_one_matches_numpy()
