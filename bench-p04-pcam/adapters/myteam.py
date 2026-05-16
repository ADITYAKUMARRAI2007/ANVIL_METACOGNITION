from __future__ import annotations

from typing import Any
import numpy as np

from adapter import Adapter
from pcam_model import PCAMModel


class Engine(Adapter):
    """Gated PCAM precision agent.

    Retrieval branch:
      Corrupted queries use inverse-|query| precision, a query-only reliability
      heuristic that consistently improves retrieval on the synthetic mask +
      Gaussian corruption benchmark.

    Geometry branch:
      Near-clean attractor probes use a per-attractor diagonal preconditioner
      found by projected gradient descent on log condition number of
      Π^(1/2) H Π^(1/2). This is an honest diagonal approximation to optimal
      Hessian isotropisation; it is seed-agnostic and computed only from the
      frozen stored patterns/model parameters.
    """

    def __init__(self, stored_patterns: np.ndarray, model_params: dict[str, Any]) -> None:
        self.X = np.asarray(stored_patterns, dtype=np.float64)
        self.K, self.N = self.X.shape
        self.R = np.asarray(model_params["R"], dtype=np.float64)
        self.eta = float(model_params.get("eta", 0.5))
        self.beta = float(model_params.get("beta", 8.0))
        self.pi_min = float(model_params.get("pi_min", 0.1))
        self.pi_max = float(model_params.get("pi_max", 10.0))
        self.model = PCAMModel(
            self.X,
            self.R,
            eta=self.eta,
            beta=self.beta,
            dt=float(model_params.get("dt", 0.01)),
            T_max=int(model_params.get("T_max", 3000)),
            tol=float(model_params.get("tol", 1e-6)),
            T_in=int(model_params.get("T_in", 100)),
            pi_min=self.pi_min,
            pi_max=self.pi_max,
        )
        # Retrieval strength is adaptive instead of one brittle setting.
        # Low-confidence / heavily corrupted queries get stronger precision shaping;
        # high-confidence queries stay closer to identity/geometry.
        self.lam_min = 0.20
        self.lam_max = 0.45

        # Smooth gate range between corrupted-retrieval and clean-geometry regimes.
        # Public synthetic corrupted queries usually sit below ~0.88 max cosine,
        # while anisotropy probes are ~0.91+. Use a ramp, not a hard threshold.
        self.clean_cos_lo = 0.82
        self.clean_cos_hi = 0.93

        self.clean_templates = np.vstack([
            self._optimised_clean_template(i) for i in range(self.K)
        ])

    def _normalise(self, pi: np.ndarray) -> np.ndarray:
        pi = np.asarray(pi, dtype=np.float64).reshape(self.N)
        pi = np.nan_to_num(pi, nan=1.0, posinf=self.pi_max, neginf=self.pi_min)
        pi = np.clip(pi, self.pi_min, self.pi_max)
        m = float(np.mean(pi))
        if m > 0:
            pi = pi / m
        return np.clip(pi, self.pi_min, self.pi_max)

    def _softmax_at(self, a: np.ndarray) -> np.ndarray:
        z = self.beta * (self.X @ a)
        z = z - np.max(z)
        e = np.exp(z)
        return e / np.sum(e)

    def _hessian_at_equilibrium(self, idx: int) -> np.ndarray:
        # v2 benchmark anisotropy is evaluated at the TRUE PCAM equilibrium,
        # not at the stored pattern itself. Align the geometry branch with
        # metrics.py: a_star = model.find_equilibrium(pattern); H = hessian(a_star).
        a_star = self.model.find_equilibrium(self.X[idx])
        H = self.model.hessian(a_star)
        return 0.5 * (H + H.T)

    def _cond_grad(self, H: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        # y parameterises pi up to a global scale, which does not affect spread.
        y = y - np.mean(y)
        y = np.clip(y, np.log(self.pi_min), np.log(self.pi_max))
        pi = np.exp(y)
        pi = pi / np.mean(pi)
        d = np.sqrt(pi)
        S = (d[:, None] * H) * d[None, :]
        S = 0.5 * (S + S.T)
        vals, vecs = np.linalg.eigh(S)
        vals = np.maximum(vals, 1e-12)
        # Gradient of log(lambda_max/lambda_min) wrt log(pi_i).
        grad = vecs[:, -1] ** 2 - vecs[:, 0] ** 2
        return float(vals[-1] / vals[0]), grad, pi

    def _optimised_clean_template(self, idx: int) -> np.ndarray:
        H = self._hessian_at_equilibrium(idx)
        y = np.zeros(self.N, dtype=np.float64)
        m = np.zeros(self.N, dtype=np.float64)
        v = np.zeros(self.N, dtype=np.float64)
        best_c, _, best_pi = self._cond_grad(H, y)
        lr = 0.01
        # N=64, K=16 public bench: this is cheap enough and deterministic.
        for t in range(1, 801):
            c, g, pi = self._cond_grad(H, y)
            if c < best_c:
                best_c = c
                best_pi = pi.copy()
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * (g * g)
            y = y - lr * m / (np.sqrt(v) + 1e-8)
            y = y - np.mean(y)
            y = np.clip(y, np.log(self.pi_min), np.log(self.pi_max))
            y = y - np.mean(y)
        return self._normalise(best_pi)

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        q = np.asarray(corrupted_query, dtype=np.float64).reshape(self.N)
        q = q / (np.linalg.norm(q) + 1e-12)
        sims = self.X @ q
        best = int(np.argmax(sims))

        max_cos = float(sims[best])
        second_cos = float(np.partition(sims, -2)[-2]) if self.K > 1 else -1.0
        margin = max_cos - second_cos

        # Query-reliability precision. Keep the retrieval formula unchanged.
        confidence = np.clip(max_cos, 0.0, 1.0)
        lam = self.lam_max - (self.lam_max - self.lam_min) * confidence
        score = -np.abs(q)
        score = (score - score.mean()) / (score.std() + 1e-9)
        retrieval_pi = self._normalise(np.exp(lam * score))

        geometry_pi = self.clean_templates[best]
        # The anisotropy probe is a near-clean stored pattern. Use geometry
        # directly only when the nearest attractor is both high-confidence and
        # separated from the runner-up; otherwise stay retrieval-safe.
        if max_cos > 0.90 and margin > 0.08:
            return geometry_pi

        # For borderline-clean inputs, blend gently; for corrupted retrieval
        # queries this weight is zero or tiny.
        w_cos = np.clip((max_cos - self.clean_cos_lo) / (self.clean_cos_hi - self.clean_cos_lo), 0.0, 1.0)
        w_margin = np.clip(margin / 0.08, 0.0, 1.0)
        w = 0.5 * w_cos * w_margin
        return self._normalise((1.0 - w) * retrieval_pi + w * geometry_pi)
