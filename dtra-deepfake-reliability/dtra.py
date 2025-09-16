#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dual-Threshold, Risk-Aware Reliability Estimation (DTRA)
"""

from __future__ import annotations
import os, math, json, argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable
import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None


def as_np(a: Any) -> np.ndarray:
    return np.asarray(a)


def safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def hoeffding_epsilon(n: int, delta: float) -> float:
    """Hoeffding epsilon, Eq.(6) in paper"""
    if n <= 0:
        return 0.0
    return math.sqrt(max(0.0, math.log(1.0 / max(delta, 1e-12))) / (2.0 * max(1, n)))


def beta_smooth(success: int, total: int, a: float, b: float) -> float:
    """Beta(a,b) posterior mean smoothing for p̂"""
    return (success + a) / (total + a + b) if total >= 0 else 0.0


# ---------------- IO helpers ----------------

def load_any_table(path: str):
    if pd is None:
        raise RuntimeError("pandas is required to read tables; please install pandas.")
    ext = os.path.splitext(path)[-1].lower()
    if ext in (".csv", ".txt"): return pd.read_csv(path)
    if ext in (".tsv", ".tab"): return pd.read_csv(path, sep="\t")
    if ext in (".xlsx", ".xls"): return pd.read_excel(path)
    if ext in (".parquet", ".pq"): return pd.read_parquet(path)
    if ext == ".npy":
        arr = np.load(path, allow_pickle=True)
        return pd.DataFrame(arr, columns=["prob", "label"])
    if ext == ".npz":
        z = np.load(path, allow_pickle=True)
        return pd.DataFrame({"prob": z["probs"], "label": z["labels"]})
    return pd.read_csv(path)


def pick_columns(df, label_col: Optional[str], prob_col: Optional[str]) -> Tuple[str, str]:
    label_candidates = [label_col] if label_col else ["label","y","gt","target","class","真实标签","真实标签(1=real,0=fake)"]
    prob_candidates  = [prob_col]  if prob_col  else ["prob","p","score","prob_real","P(real)","pred_prob","probability","预测为真实样本的概率"]

    lab = next((c for c in label_candidates if c and c in df.columns), None)
    pr  = next((c for c in prob_candidates  if c and c in df.columns), None)
    if lab is None: raise KeyError("Label column not found.")
    if pr  is None: raise KeyError("Probability column not found.")
    return lab, pr


def coerce_arrays(df, lab: str, pr: str, label_one_means: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = as_np(df[lab].values).astype(int)
    p_raw = np.clip(as_np(df[pr].values).astype(float), 0.0, 1.0)
    if label_one_means.lower() == "real": P1 = p_raw
    else: P1 = 1.0 - p_raw
    return P1, y, p_raw


# ---------------- Binning ----------------

@dataclass
class Bin:
    lo: float
    hi: float
    n: int
    succ: int


def build_class_space_bins(p_class: np.ndarray, y_class: np.ndarray, nmin: int) -> List[Bin]:
    mask = (p_class >= 0.5)
    if not np.any(mask): return []
    ps, ys = p_class[mask], y_class[mask]
    order = np.argsort(ps)
    ps, ys = ps[order], ys[order]

    out: List[Bin] = []
    i = 0
    while i < len(ps):
        j = min(len(ps), i + max(1, nmin))
        seg_y = ys[i:j]
        out.append(Bin(float(ps[i]), float(ps[j - 1]), j - i, int(seg_y.sum())))
        i = j
    return out


# ---------------- Scorer ----------------

class Scorer:
    """Implements Eq.(5): score = (p_hat * exp(lambda*width)) / (1 - p_hat)"""
    def __init__(self, lam: float, delta: float, laplace: Optional[Tuple[float, float]]):
        self.lam, self.delta, self.laplace = float(lam), float(delta), laplace
        self._eps, self._cap = 1e-12, 1.0 - 1e-12

    def _phat(self, succ: int, tot: int) -> float:
        if tot <= 0: return 0.0
        ph = succ/tot if self.laplace is None else beta_smooth(succ, tot, *self.laplace)
        return float(min(max(ph, self._eps), self._cap))

    def eval(self, succ: int, tot: int, lo: float, up: float, m_all: int):
        ph = self._phat(succ, tot)
        width = max(0.0, up - lo)
        risk  = max(self._eps, 1.0 - ph)
        score = (ph * math.exp(self.lam * width)) / risk
        return float(score), {"p_hat": ph, "width": width, "n": int(tot)}


# ---------------- Interval search ----------------

def search_interval_on_bins(bins: List[Bin], scorer: Scorer, m_all: int,
                            min_width: float, max_width: float,
                            bin_step: int=1, max_bin_span: int=0):
    if not bins: return None
    nB = len(bins)
    pref_n = np.zeros(nB+1, dtype=int)
    pref_s = np.zeros(nB+1, dtype=int)
    lows  = [b.lo for b in bins]
    highs = [b.hi for b in bins]
    for i, b in enumerate(bins, 1):
        pref_n[i] = pref_n[i-1] + b.n
        pref_s[i] = pref_s[i-1] + b.succ

    best, best_score, best_n = None, -1e300, -1
    step = max(1, bin_step)
    for i in range(nB):
        jmax = nB-1 if max_bin_span<=0 else min(nB-1, i+max_bin_span-1)
        j = i
        while j <= jmax:
            tot, succ = int(pref_n[j+1]-pref_n[i]), int(pref_s[j+1]-pref_s[i])
            lo, up = float(lows[i]), float(highs[j])
            width = up - lo
            if width < min_width or (max_width>0 and width > max_width):
                j += step; continue
            sc, info = scorer.eval(succ, tot, lo, up, m_all)
            if (sc > best_score) or (abs(sc-best_score)<1e-12 and info["n"]>best_n):
                best_score, best_n = sc, info["n"]
                best = {"q_low": lo, "q_up": up, "score": sc, "details": info}
            j += step
    return best


# ---------------- Per-class search ----------------

def per_class_search(P1: np.ndarray, y: np.ndarray, target_class: int,
                     nmin: int, scorer: Scorer, min_width: float, max_width: float,
                     delta: float, bin_step:int=1, max_bin_span:int=0):
    y_c = (y == target_class).astype(int)
    p_c = P1 if target_class==1 else (1.0-P1)
    m_all = int(np.sum(p_c>=0.5))
    bins = build_class_space_bins(p_c, y_c, nmin)
    best = search_interval_on_bins(bins, scorer, m_all or len(P1),
                                   min_width, max_width, bin_step, max_bin_span)
    if best is None:
        return {"class": target_class, "interval": None}

    d, n_in, p_hat = best["details"], best["details"]["n"], best["details"]["p_hat"]
    eps = hoeffding_epsilon(n_in, delta)
    lower, upper = max(0.0, p_hat-eps), min(1.0, p_hat+eps)

    return {
        "class": target_class,
        "interval": {
            "q_low": round(best["q_low"], 10),
            "q_up": round(best["q_up"], 10),
            "width": round(best["q_up"]-best["q_low"], 10),
            "n_in_interval": n_in,
            "p_hat": round(p_hat, 10),
            "epsilon": round(eps, 10),
            "precision_ci": [round(lower,10), round(upper,10)],
            "score": round(best["score"], 10),
        }
    }


# ---------------- Apply rejection ----------------

def apply_rejection(P1: np.ndarray, intervals: Dict[str, Any]) -> np.ndarray:
    """
    Reject samples outside class intervals.
    Return: array of predictions (1,0,-1) where -1 = reject.
    """
    pred = np.full(len(P1), -1, dtype=int)
    for cls_key in ["class_1","class_0"]:
        interval = intervals.get(cls_key, {}).get("interval", None)
        if not interval: continue
        lo, up = interval["q_low"], interval["q_up"]
        if "1" in cls_key:
            mask = (P1 >= lo) & (P1 <= up)
            pred[mask] = 1
        else:
            mask = ((1.0-P1) >= lo) & ((1.0-P1) <= up)
            pred[mask] = 0
    return pred


# ---------------- Main ----------------

def run_dtra(P1: np.ndarray, y: np.ndarray, nmin:int=50, lam:float=0.05,
             delta:float=0.05, laplace:Optional[Tuple[float,float]]=None,
             min_width:float=0.0, max_width:float=0.0,
             bin_step:int=1, max_bin_span:int=0) -> Dict[str,Any]:
    scorer = Scorer(lam, delta, laplace)
    out1 = per_class_search(P1,y,1,nmin,scorer,min_width,max_width,delta,bin_step,max_bin_span)
    out0 = per_class_search(P1,y,0,nmin,scorer,min_width,max_width,delta,bin_step,max_bin_span)
    return {
        "meta": {"n":len(P1),"nmin":nmin,"method":"paper Eq.(5)",
                 "lambda":lam,"delta":delta,"laplace":laplace,
                 "bin_step":bin_step,"max_bin_span":max_bin_span},
        "class_1": out1, "class_0": out0
    }


def parse_args(argv: Optional[Iterable[str]]=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--prob-col", default=None)
    ap.add_argument("--label-one-means", default="real", choices=["real","fake"])
    ap.add_argument("--nmin", type=int, default=150)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.10)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--laplace", type=str, default="")
    ap.add_argument("--min-width", type=float, default=0.0)
    ap.add_argument("--max-width", type=float, default=0.0)
    ap.add_argument("--bin-step", type=int, default=1)
    ap.add_argument("--max-bin-span", type=int, default=0)
    return ap.parse_args(argv)


def main(argv: Optional[Iterable[str]]=None):
    args = parse_args(argv)
    if args.input:
        df = load_any_table(args.input)
        lab, pr = pick_columns(df, args.label_col, args.prob_col)
        P1, y, _ = coerce_arrays(df, lab, pr, args.label_one_means)
    else:
        rng = np.random.default_rng(13)
        P1 = rng.uniform(0,1,5000)
        y  = rng.integers(0,2,5000)

    lap = None
    if args.laplace:
        a,b = args.laplace.split(","); lap=(float(a),float(b))

    summary = run_dtra(P1,y,args.nmin,args.lam,args.delta,lap,
                       args.min_width,args.max_width,args.bin_step,args.max_bin_span)

    # rejection example
    preds = apply_rejection(P1, summary)
    summary["rejection"] = {"accepted": int(np.sum(preds>=0)), "rejected": int(np.sum(preds<0))}

    js = json.dumps(summary, ensure_ascii=False, indent=2)
    print(js)


if __name__ == "__main__":
    main()
