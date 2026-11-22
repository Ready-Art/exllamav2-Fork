from __future__ import annotations
import torch
import os, glob
from functools import lru_cache
from exllamav2.ext import exllamav2_ext as ext_c

had_dict: dict[int: torch.Tensor] | None = {}
primes: set[int]

def load_constants():
    global had_dict, primes

    module_dir = os.path.dirname(os.path.abspath(__file__))
    file_pattern = os.path.join(module_dir, 'hadamard_*.txt')
    files = glob.glob(file_pattern)
    had_dict = {}

    for file_path in files:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            lines = [line.strip() for line in lines if line.strip()]
            dim = len(lines)
            assert all(len(line) == dim for line in lines), "Non-square matrix in " + file_path
            matrix = [[1 if char == '+' else -1 for char in line] for line in lines]
            tensor = torch.tensor(matrix, dtype = torch.float16)
            had_dict[dim] = tensor

    prime_path = os.path.join(module_dir, 'primes.txt')
    with open(prime_path, "r") as f:
        lines = f.readlines()
        primes = set([int(line) for line in lines if line.strip()])

load_constants()

def sylvester(h: torch.Tensor):
    d = h.shape[0]
    assert d == h.shape[1], "h not square"
    s = torch.empty((d * 2, d * 2), dtype = h.dtype, device = h.device)
    s[:d, :d] = h
    s[:d, d:] = h
    s[d:, :d] = h
    s[d:, d:] = -h
    return s

def is_quadratic_residue(a: int, p: int):
    return pow(a, (p - 1) // 2, p) == 1

def paley_torch(n: int):
    h = torch.empty((n, n), dtype = torch.half)
    p = n - 1
    for i in range(p):
        for j in range(p):
            if i == j:
                h[i + 1][j + 1] = 1
            else:
                residue = (i - j) % p
                if is_quadratic_residue(residue, p):
                    h[i + 1][j + 1] = 1
                else:
                    h[i + 1][j + 1] = -1
    h[0, :] = 1
    h[:, 0] = -1
    h[0, 0] = 1
    return h

def paley(n: int):
    h = torch.empty((n, n), dtype = torch.half)
    ext_c.had_paley(h)
    # ref = paley_torch(n)
    # assert torch.all(h == ref)
    return h

def paley2_torch(n: int):
    h = torch.empty((n, n), dtype = torch.half)
    p = n // 2 - 1
    for i in range(n // 2):
        i0 = 2 * i + 0
        i1 = 2 * i + 1
        for j in range(n // 2):
            j0 = 2 * j + 0
            j1 = 2 * j + 1
            if j == i:
                h[i0, j0] = 1
                h[i0, j1] = -1
                h[i1, j0] = -1
                h[i1, j1] = -1
            else:
                residue = (i - j) % p
                if i == 0 or j == 0 or is_quadratic_residue(residue, p):
                    h[i0, j0] = 1
                    h[i0, j1] = 1
                    h[i1, j0] = 1
                    h[i1, j1] = -1
                else:
                    h[i0, j0] = -1
                    h[i0, j1] = -1
                    h[i1, j0] = -1
                    h[i1, j1] = 1
    return h

def paley2(n: int):
    h = torch.empty((n, n), dtype = torch.half)
    ext_c.had_paley2(h)
    # ref = paley2_torch(n)
    # assert torch.all(h == ref)
    return h

@lru_cache(maxsize = 100)
def get_hadamard(n: int):
    global had_dict, primes

    if n in had_dict: return had_dict[n]

    # Sylvester's construction
    if n % 2 == 0:
        s = get_hadamard(n // 2)
        if s is not None:
            s = sylvester(s)
            return s

    # Paley construction
    if n % 4 == 0 and (n - 1) % 4 == 3 and (n - 1) in primes:
        return paley(n)

    # Other Paley construction
    if n % 4 == 0 and (n // 2) - 1 in primes:
        return paley2(n)

    return None

def assert_hadamard(h):
    h = h.cuda()
    n = h.shape[0]
    assert n == h.shape[1], "h is not square"
    assert torch.all((h == 1) | (h == -1)), "h entries are not all +1 or -1"
    si = n * torch.eye(n, dtype = h.dtype, device = h.device)
    gram = h @ h.T
    # print (si)
    # print (gram)
    assert torch.allclose(si, gram), "h rows are not mutually orthogonal"

def test_hadamard_dims(min_dim: int, max_dim: int, step: int):

    for dim in range(min_dim, max_dim, step):
        h = get_hadamard(dim)
        if h is None:
            print(f"No hadamard matrix for n = {dim}")
        else:
            # print(f"Testing n = {dim}")
            assert_hadamard(h)

# test_hadamard_dims(128, 16384, 128)


# ---- EXL2.5 K-side incoherence rotation ----------------------------------------------
#
# Convention (matches the exl25 GEMM kernel exactly):
#   kernel activations : x' = x @ H_int          (UNNORMALIZED +/-1 butterfly, exact fp16)
#   stored weights     : W' = H_int @ W / block  (the 1/block folds into the quant scales)
#   identity           : (x H_int)(H_int W / block) = x W   since H_int^2 = block * I
#   inverse (unpack)   : W = H_int @ W' / block
# H_int is symmetric, so row-vector and column-vector forms use the same matrix.

def _had_int(block: int, device, dtype):
    h = get_hadamard(block)
    assert h is not None, f"No Hadamard matrix for block {block}"
    return h.to(device = device, dtype = dtype)


def had_rotate_activations(x: torch.Tensor, block: int = 128):
    """x: (..., k) -> x @ H_int block-diagonal along the last dim (k % block == 0)."""
    k = x.shape[-1]
    assert k % block == 0
    h = _had_int(block, x.device, x.dtype)
    shape = x.shape
    return (x.reshape(-1, k // block, block) @ h).reshape(shape)


def had_rotate_weights(w: torch.Tensor, block: int = 128, inverse: bool = False):
    """w: (k, n) -> H_int @ w / block (forward, stored space) or H_int @ w / block
    (inverse -- the transform is an involution up to the same 1/block)."""
    k = w.shape[0]
    assert k % block == 0
    h = _had_int(block, w.device, w.dtype)
    y = torch.matmul(h, w.reshape(k // block, block, w.shape[1]))
    return (y / block).reshape(w.shape)


def had_rotate_hessian(hess: torch.Tensor, block: int = 128):
    """Hessian of x @ H_int activations: H' = H_int @ H @ H_int (block-diagonal congruence).
    Chunked over row-blocks: a full (nb*nb, 128, 128) batched form would peak several GB
    of temporaries on top of the Hessian itself (down_proj k=21504 -> OOM next to a model
    replica). Per-chunk temporaries are ~nb*128*128 instead."""
    k = hess.shape[0]
    assert k % block == 0 and hess.shape[1] == k
    h = _had_int(block, hess.device, hess.dtype)
    nb = k // block
    out = torch.empty_like(hess)
    hv = hess.view(nb, block, nb, block)
    ov = out.view(nb, block, nb, block)
    for ib in range(nb):
        x = hv[ib].permute(1, 0, 2).contiguous()
        y = h @ x @ h
        ov[ib].copy_(y.permute(1, 0, 2))
    return out