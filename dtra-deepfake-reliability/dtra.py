#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dual-Threshold, Risk-Aware Reliability Estimation (DTRA)
========================================================

A self-contained, paper-grade Python implementation of the DTRA method described in:
"Dual-Threshold Risk-Aware Reliability Estimation for Deepfake Detection".

Key features
------------
- Equal-frequency binning on a calibration set.
- Per-bin confusion estimation in class-space.
- Class-specific dual-threshold search (positive and negative classes).
- Risk-adjusted utility in the exact "paper" form:
      E = [exp(mu) * exp(lambda * width)] / [1 - exp(mu)]
  where exp(mu) is the interval precision (p̂), and width = q_up - q_low.
- Optional engineering baselines (width/cover/countinv/none) for ablation.
- Hoeffding deviation bound utility helpers (not used by the "paper" scorer unless you adapt it).
- Robust CLI and modular design for reproducibility and extension.


"""

from __future__ import annotations

import os
import sys
import math
import json
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable

import numpy as np

try:
    import pandas as pd  # optional but recommended
except Exception:
    pd = None



def as_np(a: Any) -> np.ndarray:
    return np.asarray(a)


def clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def hoeffding_lower_bound(phat: float, n: int, delta: float) -> float:
    """
    Hoeffding-style deviation lower bound for a Bernoulli mean estimate phat with n samples.

    LB = phat - sqrt(log(2/delta) / (2n))

    Returns 0.0 if n<=0.
    """
    if n <= 0:
        return 0.0
    eps = math.sqrt(max(0.0, math.log(2.0 / max(delta, 1e-12))) / (2.0 * max(1, n)))
    return max(0.0, phat - eps)


def beta_smooth(success: int, total: int, a: float, b: float) -> float:
    """
    Beta(a,b) posterior mean smoothing for a binomial rate.
    """
    return (success + a) / (total + a + b) if total >= 0 else 0.0


# ----------------------------------------------------------------------
# IO and column picking
# ----------------------------------------------------------------------

def load_any_table(path: str):
    if pd is None:
        raise RuntimeError("pandas is required to read tables; please install pandas.")
    ext = os.path.splitext(path)[-1].lower()
    if ext in (".csv", ".txt"):
        return pd.read_csv(path)
    if ext in (".tsv", ".tab"):
        return pd.read_csv(path, sep="\t")
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if ext == ".npy":
        arr = np.load(path, allow_pickle=True)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return pd.DataFrame(arr, columns=[f"col_{i}" for i in range(arr.shape[1])])
        raise ValueError("NPY must be 2D with >=2 columns: probability and label.")
    if ext == ".npz":
        z = np.load(path, allow_pickle=True)
        if {"probs", "labels"}.issubset(z.keys()):
            return pd.DataFrame({"prob": z["probs"], "label": z["labels"]})
        raise ValueError("NPZ must contain arrays 'probs' and 'labels'.")
    # fallback: try csv
    return pd.read_csv(path)


def pick_columns(df, label_col: Optional[str], prob_col: Optional[str]) -> Tuple[str, str]:
    label_candidates = [label_col] if label_col else [
        "label", "y", "gt", "target", "class", "类别", "真实标签", "真实标签(1=real,0=fake)"
    ]
    prob_candidates = [prob_col] if prob_col else [
        "prob", "p", "score", "prob_real", "P(real)", "P_real",
        "pred_prob", "probability", "预测为真实样本的概率"
    ]

    lab = next((c for c in label_candidates if c and c in df.columns), None)
    if lab is None:
        for c in df.columns[::-1]:
            s = df[c].dropna()
            if len(s) > 0:
                vals = as_np(s)
                try:
                    u = np.unique(vals.astype(int))
                    if set(u.tolist()).issubset({0, 1}):
                        lab = c
                        break
                except Exception:
                    pass
    if lab is None:
        raise KeyError("Label column not found. Use --label-col to specify.")

    pr = next((c for c in prob_candidates if c and c in df.columns), None)
    if pr is None:
        for c in df.columns:
            v = df[c].dropna()
            if len(v) > 0 and safe_float(v.iloc[0]) is not None:
                pr = c
                break
    if pr is None:
        raise KeyError("Probability column not found. Use --prob-col to specify.")
    return lab, pr


def coerce_arrays(df, lab: str, pr: str, label_one_means: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        P1: np.ndarray of shape [n], interpreted as P(Y=1|x)
        y : binary labels in {0,1}
        p_raw: the raw probability column after clipping to [0,1]
    """
    y = as_np(df[lab].values).astype(int)
    p_raw = as_np(df[pr].values).astype(float)
    p_raw = np.clip(p_raw, 0.0, 1.0)

    lom = label_one_means.strip().lower()
    if lom not in {"real", "fake"}:
        raise ValueError("--label-one-means must be 'real' or 'fake'")

    # If the raw prob means P(real), then label==1 should correspond to "real".
    # If the user says label-one-means='fake', then P1 = 1 - P(real).
    if lom == "real":
        P1 = p_raw
    else:
        P1 = 1.0 - p_raw
    return P1, y, p_raw


# ----------------------------------------------------------------------
# Equal-frequency bins in class-space
# ----------------------------------------------------------------------

@dataclass
class Bin:
    lo: float
    hi: float
    n: int
    succ: int  # number of correct predictions for the target class inside this bin


def build_class_space_bins(p_class: np.ndarray, y_class: np.ndarray, nmin: int) -> List[Bin]:
    """
    Builds equal-frequency bins in class-space over [0.5, 1].

    Args:
        p_class : class-space probability P(Y=c|x) for c in {0,1}
        y_class : indicator vector of length n: 1 if y==c else 0
        nmin    : minimal samples per bin

    Returns:
        list of Bin(lo, hi, n, succ)
    """
    mask = (p_class >= 0.5)
    if not np.any(mask):
        return []

    ps = p_class[mask]
    ys = y_class[mask]
    order = np.argsort(ps)
    ps = ps[order]
    ys = ys[order]
    n = len(ps)

    out: List[Bin] = []
    i = 0
    while i < n:
        j = min(n, i + max(1, nmin))
        seg_y = ys[i:j]
        out.append(Bin(float(ps[i]), float(ps[j - 1]), j - i, int(seg_y.sum())))
        i = j

    # Merge tail if last bin is too small (ensures minimal n per bin, except possibly when n < nmin).
    if len(out) >= 2 and out[-1].n < nmin:
        a, b = out[-2], out[-1]
        out[-2] = Bin(a.lo, b.hi, a.n + b.n, a.succ + b.succ)
        out.pop()

    return out


class Scorer:
    """
    Interval scorer.

    Supported strategies:
        - "paper": exact Eq.(5) from the manuscript:
            E = [exp(mu) * exp(lambda * width)] / [1 - exp(mu)],
          where exp(mu) = p_hat is the precision within the interval.
        - "width"    : score = LB + lambda * width
        - "cover"    : score = LB + lambda * coverage
        - "countinv" : score = LB - lambda / n
        - "none"     : score = LB
      LB is Hoeffding lower bound computed from p_hat, n, delta.

    Note: "paper" does not invoke LB unless you choose to adapt it.
    """

    def __init__(self, name: str, lam: float, delta: float, laplace: Optional[Tuple[float, float]]):
        self.name = name
        self.lam = float(lam)
        self.delta = float(delta)
        self.laplace = laplace
        self._eps = 1e-12
        self._cap = 1.0 - 1e-12

    def _phat(self, succ: int, tot: int) -> float:
        if tot <= 0:
            return 0.0
        if self.laplace is None:
            ph = succ / tot
        else:
            a, b = self.laplace
            ph = beta_smooth(succ, tot, a, b)
        return float(min(max(ph, self._eps), self._cap))

    def eval(self, succ: int, tot: int, lo: float, up: float, m_all: int) -> Tuple[float, Dict[str, Any]]:
        ph = self._phat(succ, tot)
        width = max(0.0, up - lo)
        cov = (tot / max(1, m_all))

        if self.name == "paper":
            mu = math.log(ph)              # exp(mu) == ph
            exp_mu = ph
            risk = max(self._eps, 1.0 - exp_mu)
            bonus = math.exp(self.lam * width)
            score = (exp_mu * bonus) / risk
            info = {
                "p_hat": ph,
                "mu": mu,
                "exp_mu": exp_mu,
                "risk": 1.0 - exp_mu,
                "bonus_width_term": bonus,
                "width": width,
                "cov": cov,
                "n": int(tot),
            }
            return float(score), info

        # Engineering baselines below (kept for ablations)
        lb = hoeffding_lower_bound(ph, tot, self.delta)
        if self.name == "width":
            sc = lb + self.lam * width
        elif self.name == "cover":
            sc = lb + self.lam * cov
        elif self.name == "countinv":
            sc = lb - (self.lam / max(1, tot))
        elif self.name == "none":
            sc = lb
        else:
            # Fallback to "paper" if strategy name is unknown
            mu = math.log(ph)
            exp_mu = ph
            risk = max(self._eps, 1.0 - exp_mu)
            bonus = math.exp(self.lam * width)
            sc = (exp_mu * bonus) / risk

        info = {
            "p_hat": ph,
            "lb": lb,
            "width": width,
            "cov": cov,
            "n": int(tot),
        }
        return float(sc), info


# ----------------------------------------------------------------------
# Interval search on consecutive bins
# ----------------------------------------------------------------------

def search_interval_on_bins(
    bins: List[Bin],
    scorer: Scorer,
    m_all: int,
    min_width: float,
    max_width: float
) -> Optional[Dict[str, Any]]:
    """
    Exhaustive search of consecutive-bin intervals [i..j] in class-space.

    Returns:
        best dict with fields:
            q_low, q_up, score, details{ p_hat|mu|lb..., width, cov, n }
        or None if no feasible interval.
    """
    if not bins:
        return None

    nB = len(bins)
    pref_n = np.zeros(nB + 1, dtype=int)
    pref_s = np.zeros(nB + 1, dtype=int)
    lows = [b.lo for b in bins]
    highs = [b.hi for b in bins]

    for i, b in enumerate(bins, start=1):
        pref_n[i] = pref_n[i - 1] + b.n
        pref_s[i] = pref_s[i - 1] + b.succ

    best = None
    best_score = -1e300
    best_n = -1

    for i in range(nB):
        for j in range(i, nB):
            tot = int(pref_n[j + 1] - pref_n[i])
            succ = int(pref_s[j + 1] - pref_s[i])
            lo = float(lows[i])
            up = float(highs[j])

            width = up - lo
            if width + 1e-12 < min_width:
                continue
            if max_width > 0.0 and width > max_width + 1e-12:
                continue

            sc, info = scorer.eval(succ, tot, lo, up, m_all)
            if (sc > best_score) or (abs(sc - best_score) < 1e-12 and info.get("n", 0) > best_n):
                best_score = sc
                best_n = info.get("n", 0)
                best = {"q_low": lo, "q_up": up, "score": float(sc), "details": info}

    return best


# ----------------------------------------------------------------------
# Per-class pipeline and top-level estimator
# ----------------------------------------------------------------------

def per_class_search(
    P1: np.ndarray,
    y: np.ndarray,
    target_class: int,
    nmin: int,
    scorer: Scorer,
    min_width: float,
    max_width: float
) -> Dict[str, Any]:
    """
    Runs the DTRA interval search for a specific class in class-space.

    Returns a structured dict with:
      - interval_class_space: best [q_low,q_up], width, size, coverage, precision (p_hat), score
      - mapped_to_input_prob: mapping back to input probability semantics
    """
    assert target_class in (0, 1)
    y_c = (y == target_class).astype(int)
    p_c = P1 if target_class == 1 else (1.0 - P1)

    m_all = int(np.sum(p_c >= 0.5))
    bins = build_class_space_bins(p_c, y_c, nmin=nmin)
    best = search_interval_on_bins(
        bins=bins, scorer=scorer, m_all=m_all if m_all > 0 else len(P1),
        min_width=min_width, max_width=max_width
    )
    if best is None:
        return {
            "class": int(target_class),
            "interval_class_space": None,
            "mapped_to_input_prob": None
        }

    d = best["details"]

    # Mapping to the raw input probability space depends on how the user defined the input column:
    # - If input column represents P(Y=1|x), then for class=1 the mapping is identity, and for class=0 it flips.
    # - If input column represents P(Y=0|x), the mapping is reversed.
    if target_class == 1:
        mapped_if_input_is_p1 = (best["q_low"], best["q_up"])
        mapped_if_input_is_p0 = (1.0 - best["q_up"], 1.0 - best["q_low"])
    else:
        mapped_if_input_is_p1 = (1.0 - best["q_up"], 1.0 - best["q_low"])
        mapped_if_input_is_p0 = (best["q_low"], best["q_up"])

    return {
        "class": int(target_class),
        "interval_class_space": {
            "q_low": round(best["q_low"], 10),
            "q_up": round(best["q_up"], 10),
            "width": round(best["q_up"] - best["q_low"], 10),
            "n_in_interval": int(d["n"]),
            "coverage_ratio": round(d.get("cov", 0.0), 10),
            "p_hat": round(d.get("p_hat", 0.0), 10),
            "hoeffding_lb": round(hoeffding_lower_bound(d.get("p_hat", 0.0), int(d["n"]), 1e-6), 10),  # optional diagnostic
            "score": round(best["score"], 10),
        },
        "mapped_to_input_prob": {
            "if_input_is_P(label=1)": {
                "low": round(mapped_if_input_is_p1[0], 10),
                "up": round(mapped_if_input_is_p1[1], 10),
            },
            "if_input_is_P(label=0)": {
                "low": round(mapped_if_input_is_p0[0], 10),
                "up": round(mapped_if_input_is_p0[1], 10),
            }
        }
    }


def run_dtra(
    P1: np.ndarray,
    y: np.ndarray,
    nmin: int = 50,
    strategy: str = "paper",
    lam: float = 0.05,
    delta: float = 0.05,
    laplace: Optional[Tuple[float, float]] = None,
    min_width: float = 0.0,
    max_width: float = 0.0
) -> Dict[str, Any]:
    """
    Executes DTRA search for both classes (1 and 0) in class-space [0.5,1].

    Returns a JSON-serializable dictionary with metadata and per-class intervals.
    """
    scorer = Scorer(strategy=strategy, lam=lam, delta=delta, laplace=laplace)
    out1 = per_class_search(
        P1=P1, y=y, target_class=1, nmin=nmin, scorer=scorer,
        min_width=min_width, max_width=max_width
    )
    out0 = per_class_search(
        P1=P1, y=y, target_class=0, nmin=nmin, scorer=scorer,
        min_width=min_width, max_width=max_width
    )
    return {
        "meta": {
            "n": int(len(P1)),
            "nmin": int(nmin),
            "strategy": strategy,
            "lambda": float(lam),
            "delta": float(delta),
            "laplace": None if laplace is None else {"a": float(laplace[0]), "b": float(laplace[1])},
            "search_domain": "[0.5, 1] for both classes in their class-spaces",
        },
        "class_1": out1,
        "class_0": out0
    }


# ----------------------------------------------------------------------
# Optional: synthetic calibration data
# ----------------------------------------------------------------------

def make_synthetic_calibration(n: int = 4000, seed: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a synthetic calibration set for demonstration purposes.

    Returns:
        P1: probabilities P(Y=1|x)
        y : labels in {0,1}
    """
    rng = np.random.default_rng(seed)
    # two mixture components with different class balance and calibration slope
    z = rng.uniform(0, 1, size=n)
    # latent score
    s = 2.0 * rng.beta(3.0, 6.0, size=n) + 0.5 * rng.beta(6.0, 3.0, size=n)
    s = 0.5 * s + 0.5 * rng.uniform(0, 1, size=n)
    # probability via a squashed function
    p1 = 1.0 / (1.0 + np.exp(-4.0 * (s - 0.5)))
    # heteroskedastic noise to mimic imperfect calibration
    noise = rng.normal(0.0, 0.08, size=n)
    p1 = np.clip(p1 + noise, 0.0, 1.0)

    # labels from p1
    y = (rng.uniform(0, 1, size=n) < p1).astype(int)
    return p1, y


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="DTRA (dual-threshold risk-aware reliability). Exact 'paper' utility by default.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--input", default=None, help="Calibration table (CSV/TSV/XLSX/Parquet/NPY/NPZ). If omitted, synthetic data are generated.")
    ap.add_argument("--label-col", default=None, help="Binary labels column name in the table.")
    ap.add_argument("--prob-col", default=None, help="Probability column name in the table.")
    ap.add_argument("--label-one-means", default="real", choices=["real", "fake"],
                    help="Meaning of label==1 relative to the probability column.")
    ap.add_argument("--nmin", type=int, default=150, help="Minimal samples per equal-frequency bin (class-space).")
    ap.add_argument("--strategy", default="paper",
                    choices=["paper", "width", "cover", "countinv", "none"],
                    help="Scoring strategy. 'paper' implements Eq.(5); others are engineering baselines.")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.10, help="Trade-off weight for width in the utility.")
    ap.add_argument("--delta", type=float, default=0.05, help="Hoeffding delta for deviation bounds (only used by baselines unless you adapt 'paper').")
    ap.add_argument("--laplace", type=str, default="", help="Optional 'a,b' for Beta smoothing of p̂ (e.g., '1,1').")
    ap.add_argument("--min-width", type=float, default=0.0, help="Minimum allowed interval width in class-space.")
    ap.add_argument("--max-width", type=float, default=0.0, help="Maximum allowed interval width in class-space (0 = no cap).")
    ap.add_argument("--out-json", default=None, help="Write JSON summary to this path.")
    ap.add_argument("--dump-bins", default=None, help="Write per-class bins to CSV (requires pandas).")
    return ap.parse_args(argv)


def dump_bins_csv(path: str, P1: np.ndarray, y: np.ndarray, nmin: int) -> None:
    if pd is None:
        return
    rows = []
    for cls in (0, 1):
        y_c = (y == cls).astype(int)
        p_c = P1 if cls == 1 else (1.0 - P1)
        bins = build_class_space_bins(p_c, y_c, nmin=nmin)
        for k, b in enumerate(bins):
            rows.append({"class": cls, "idx": k, "lo": b.lo, "hi": b.hi, "n": b.n, "succ": b.succ})
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)

    if args.input:
        if pd is None:
            raise RuntimeError("pandas is required to read input tables. Please install pandas.")
        df = load_any_table(args.input)
        lab, pr = pick_columns(df, args.label_col, args.prob_col)
        P1, y, p_raw = coerce_arrays(df, lab, pr, args.label_one_means)
    else:
        # Synthetic calibration set if no input is provided
        P1, y = make_synthetic_calibration(n=5000, seed=13)

    lap = None
    if args.laplace.strip():
        try:
            a, b = args.laplace.split(",")
            lap = (float(a), float(b))
        except Exception:
            raise ValueError("--laplace expects 'a,b', e.g., '1,1'.")

    summary = run_dtra(
        P1=P1,
        y=y,
        nmin=args.nmin,
        strategy=args.strategy,
        lam=args.lam,
        delta=args.delta,
        laplace=lap,
        min_width=args.min_width,
        max_width=args.max_width
    )

    js = json.dumps(summary, ensure_ascii=False, indent=2)
    print(js)

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            f.write(js)

    if args.dump_bins:
        dump_bins_csv(args.dump_bins, P1, y, args.nmin)


if __name__ == "__main__":
    main()


