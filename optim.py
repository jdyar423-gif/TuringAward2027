import torch

# Polar Express (Amsel et al. 2025, arXiv 2505.16932) per-iteration coefficients, 5-step schedule.
POLAR_EXPRESS = [
    (8.1566, -22.4833, 15.8788),
    (4.0429, -2.8089, 0.5000),
    (3.8917, -2.7725, 0.5061),
    (3.2858, -2.3681, 0.4645),
    (2.3465, -1.7098, 0.4232),
]
# Quintic Newton-Schulz coefficients (original Muon).
NS5 = (3.4445, -4.7750, 2.0315)


def _coeffs(steps, polar):
    if not polar:
        return [NS5] * steps
    if steps <= 5:
        return POLAR_EXPRESS[len(POLAR_EXPRESS) - steps:]
    return POLAR_EXPRESS + [POLAR_EXPRESS[-1]] * (steps - 5)


def use_gram(m, n, steps):
    """Gram-space iteration is cheaper when the matrix is wide enough (m <= n)."""
    m, n = sorted((m, n))
    std = steps * (4 * m * m * n + 2 * m ** 3)
    gram = 4 * m * m * n + (8 * steps - 6) * m ** 3
    return gram < std


@torch.no_grad()
def orthogonalize(G, steps, polar=True):
    if use_gram(G.size(0), G.size(1), steps):
        return orthogonalize_gram(G, steps, polar)
    return orthogonalize_std(G, steps, polar)


@torch.no_grad()
def orthogonalize_gram(G, steps, polar=True):
    """Same iterates as orthogonalize_std, computed in m x m Gram space:
    X_{k+1} = q_k(A_k) X_k  with  A_k = X_k X_k^T  =>  A_{k+1} = q_k(A_k)^2 A_k,  X_K = (prod q_k) X_0.
    Costs 4 m^2 n + (8K-6) m^3 instead of K (4 m^2 n + 2 m^3)."""
    X = G.float()
    tr = X.size(0) > X.size(1)
    if tr:
        X = X.T
    X = X / (X.norm() * 1.02 + 1e-6) if polar else X / (X.norm() + 1e-7)
    coeffs = _coeffs(steps, polar)
    A = X @ X.T
    I = torch.eye(A.size(0))
    Q = None
    for i, (a, b, c) in enumerate(coeffs):
        q = a * I + b * A + c * (A @ A)
        Q = q if Q is None else q @ Q
        if i < len(coeffs) - 1:
            A = q @ (q @ A)
            A = 0.5 * (A + A.T)
    X = Q @ X
    return X.T if tr else X


@torch.no_grad()
def orthogonalize_std(G, steps, polar=True):
    X = G.float()
    tr = X.size(0) > X.size(1)
    if tr:
        X = X.T
    X = X / (X.norm() * 1.02 + 1e-6) if polar else X / (X.norm() + 1e-7)
    coeffs = _coeffs(steps, polar)
    for a, b, c in coeffs:
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if tr:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Muon with Polar Express, optional NorMuon row normalisation and cautious weight decay."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, wd=0.0,
                 polar=True, normuon=False, beta2=0.95, cautious=True):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, wd=wd,
                                      polar=polar, normuon=normuon, beta2=beta2, cautious=cautious))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p)
                buf = st["buf"]
                buf.lerp_(p.grad, 1 - g["momentum"])
                u = p.grad.lerp(buf, g["momentum"]) if g["nesterov"] else buf
                o = orthogonalize(u, g["ns_steps"], g["polar"])
                if g["normuon"]:
                    if "v" not in st:
                        st["v"] = torch.zeros(p.size(0), 1)
                    nrm = o.norm()
                    st["v"].lerp_(o.square().mean(1, keepdim=True), 1 - g["beta2"])
                    o = o / (st["v"].sqrt() + 1e-10)
                    o = o * (nrm / (o.norm() + 1e-10))
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                if g["wd"]:
                    if g["cautious"]:
                        mask = (o * p) > 0
                        p.sub_(p * mask, alpha=g["lr"] * g["wd"])
                    else:
                        p.mul_(1 - g["lr"] * g["wd"])
                p.add_(o, alpha=-g["lr"] * scale)


class SparseRowAdam(torch.optim.Optimizer):
    """Adam with beta1=0 and a single second-moment scalar per row; only touched rows are updated.
    For the hashed n-gram tables (sparse gradients)."""

    def __init__(self, params, lr=0.1, beta2=0.95, eps=1e-8):
        super().__init__(params, dict(lr=lr, beta2=beta2, eps=eps))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "v" not in st:
                    st["v"] = torch.zeros(p.size(0))
                    st["k"] = torch.zeros(p.size(0))
                gr = p.grad.coalesce()
                rows, vals = gr.indices()[0], gr.values()
                k = st["k"][rows] + 1
                v = st["v"][rows] * g["beta2"] + (1 - g["beta2"]) * vals.square().mean(1)
                st["k"][rows] = k
                st["v"][rows] = v
                vhat = v / (1 - g["beta2"] ** k)
                p.index_add_(0, rows, vals / (vhat.sqrt() + g["eps"])[:, None], alpha=-g["lr"])
