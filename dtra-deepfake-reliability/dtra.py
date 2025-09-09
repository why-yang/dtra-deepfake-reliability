#!/usr/bin/env python3
# coding: utf-8

from __future__ import annotations
import os, sys, math, json, argparse
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import pandas as _pd
except Exception:
    _pd = None

# ---------- small helpers ----------

def _np(a)->np.ndarray: return np.asarray(a)

def _hoeff_lb(phat: float, n: int, delta: float)->float:
    if n <= 0: return 0.0
    eps = math.sqrt(max(0.0, math.log(2.0/max(delta,1e-12))) / (2.0*max(1,n)))
    return max(0.0, phat - eps)

def _beta_smooth(success:int, total:int, a:float, b:float)->float:
    return (success + a) / (total + a + b) if total >= 0 else 0.0

def _clip01(x: float)->float:
    return 0.0 if x<0 else (1.0 if x>1.0 else x)

def _as_float(x)->Optional[float]:
    try: return float(x)
    except Exception: return None

# ---------- IO ----------

def _load_any(path: str):
    if _pd is None:
        raise RuntimeError("pandas is required; pip install pandas")
    ext = os.path.splitext(path)[-1].lower()
    if ext in (".csv",".txt"): return _pd.read_csv(path)
    if ext in (".tsv",".tab"): return _pd.read_csv(path, sep="\t")
    if ext in (".xlsx",".xls"): return _pd.read_excel(path)
    if ext in (".parquet",".pq"): return _pd.read_parquet(path)
    if ext == ".npy":
        arr = np.load(path, allow_pickle=True)
        if arr.ndim==2 and arr.shape[1]>=2:
            return _pd.DataFrame(arr, columns=[f"col_{i}" for i in range(arr.shape[1])])
        raise ValueError("NPY must be 2D with >=2 cols (prob,label).")
    if ext == ".npz":
        z = np.load(path, allow_pickle=True)
        if {"probs","labels"}.issubset(z.keys()):
            return _pd.DataFrame({"prob":z["probs"], "label":z["labels"]})
        raise ValueError("NPZ must contain 'probs' and 'labels'.")
    # fallback try CSV
    return _pd.read_csv(path)

def _pick_cols(df, label_col: Optional[str], prob_col: Optional[str]):
    cands_lab = [label_col] if label_col else [
        "label","y","gt","target","真实标签","标签","真值","是否真实","class","类别","真实标签(1=real,0=fake)"
    ]
    cands_prob = [prob_col] if prob_col else [
        "prob","p","score","prob_real","P(real)","P_real","平均_预测为真实样本的概率",
        "pred_prob","probability","预测为真实样本的概率"
    ]
    lab = next((c for c in cands_lab if c in df.columns), None)
    if lab is None:
        # heuristic: last binary-ish col
        for c in df.columns[::-1]:
            s = df[c].dropna()
            if len(s)>0:
                vals = _np(s)
                try:
                    u = np.unique(vals.astype(int))
                    if set(u.tolist()).issubset({0,1}):
                        lab = c; break
                except Exception: pass
    if lab is None: raise KeyError("label column not found; use --label-col")
    pr = next((c for c in cands_prob if c in df.columns), None)
    if pr is None:
        for c in df.columns:
            v = df[c].dropna()
            if len(v)>0 and _as_float(v.iloc[0]) is not None:
                pr = c; break
    if pr is None: raise KeyError("probability column not found; use --prob-col")
    return lab, pr

def _coerce_arrays(df, lab, pr, label_one_means:str):
    y = _np(df[lab].values).astype(int)
    p_in = _np(df[pr].values).astype(float)
    p_in = np.clip(p_in, 0.0, 1.0)
    lom = label_one_means.strip().lower()
    if lom not in {"real","fake"}:
        raise ValueError("--label-one-means must be 'real' or 'fake'")
    # define P1 = P(label==1) according to semantics of the input prob column
    # if column is P(real) but label-one-means='fake', then P1 = 1 - P(real)
    P1 = p_in if lom=="real" else (1.0 - p_in)
    return P1, y, p_in

# ---------- bins in class-space [0.5,1] ----------

class _Bin:
    __slots__=("lo","hi","n","succ")
    def __init__(self, lo: float, hi: float, n:int, succ:int):
        self.lo=lo; self.hi=hi; self.n=n; self.succ=succ

def _build_bins_class_space(p_class: np.ndarray, y_class: np.ndarray, nmin:int)->List[_Bin]:
    # restrict to [0.5,1], sort ascending
    m = p_class >= 0.5
    if not np.any(m): return []
    ps = p_class[m]; ys = y_class[m]
    order = np.argsort(ps)
    ps = ps[order]; ys = ys[order]
    n = len(ps)
    out: List[_Bin] = []
    i=0
    while i<n:
        j = min(n, i+nmin)
        segy = ys[i:j]
        out.append(_Bin(float(ps[i]), float(ps[j-1]), j-i, int(segy.sum())))
        i = j
    if len(out)>=2 and out[-1].n < nmin:
        a,b = out[-2], out[-1]
        out[-2] = _Bin(a.lo, b.hi, a.n+b.n, a.succ+b.succ)
        out.pop()
    return out

# ---------- strategies ----------

class _Scorer:
    def __init__(self, name:str, lam:float, delta:float, laplace: Optional[Tuple[float,float]]):
        self.name=name; self.lam=float(lam); self.delta=float(delta); self.lap=laplace

    def _phat(self, succ:int, tot:int)->float:
        if self.lap is None:
            return (succ/tot) if tot>0 else 0.0
        a,b = self.lap
        return _beta_smooth(succ, tot, a, b)

    def eval(self, succ:int, tot:int, lo:float, up:float, m_all:int)->Tuple[float,Dict[str,Any]]:
        ph = self._phat(succ, tot)
        lb = _hoeff_lb(ph, tot, self.delta)
        width = max(0.0, up - lo)
        cov = (tot / max(1, m_all))
        if self.name=="width":
            sc = lb + self.lam * width
        elif self.name=="cover":
            sc = lb + self.lam * cov
        elif self.name=="countinv":
            sc = lb - (self.lam / max(1, tot))
        elif self.name=="none":
            sc = lb
        else:
            sc = lb
        return sc, {"p_hat":ph,"lb":lb,"width":width,"cov":cov,"n":tot}

# ---------- double-threshold search on consecutive bins ----------

def _search_interval_on_bins(bins: List[_Bin], scorer: _Scorer, m_all:int,
                             min_width: float, max_width: float)->Optional[Dict[str,Any]]:
    if not bins: return None
    nB = len(bins)
    pref_n = np.zeros(nB+1, dtype=int)
    pref_s = np.zeros(nB+1, dtype=int)
    lows = [b.lo for b in bins]
    highs= [b.hi for b in bins]
    for i,b in enumerate(bins, start=1):
        pref_n[i] = pref_n[i-1] + b.n
        pref_s[i] = pref_s[i-1] + b.succ
    best=None; best_score=-1e18; best_n=-1
    for i in range(nB):
        for j in range(i, nB):
            tot = int(pref_n[j+1]-pref_n[i])
            succ= int(pref_s[j+1]-pref_s[i])
            lo = float(lows[i]); up = float(highs[j])
            if up - lo < min_width - 1e-12: continue
            if max_width>0 and (up - lo) > max_width + 1e-12: continue
            sc, info = scorer.eval(succ, tot, lo, up, m_all)
            if (sc>best_score) or (abs(sc-best_score)<1e-12 and info["n"]>best_n):
                best_score=sc; best_n=info["n"]
                best={"q_low":lo,"q_up":up,"score":float(sc),"details":info}
    return best

# ---------- pipeline per class ----------

def _per_class_search(P1: np.ndarray, y: np.ndarray, target_class:int,
                      nmin:int, scorer:_Scorer,
                      min_width:float, max_width:float)->Dict[str,Any]:
    # class-space prob p_c
    y_c = (y == target_class).astype(int)
    p_c = P1 if target_class==1 else (1.0 - P1)  # both use [0.5,1]
    m_all = int(np.sum(p_c >= 0.5))
    bins = _build_bins_class_space(p_c, y_c, nmin=nmin)
    best = _search_interval_on_bins(bins, scorer, m_all if m_all>0 else len(P1),
                                    min_width=min_width, max_width=max_width)
    if best is None:
        return {"class": int(target_class), "interval_class_space": None, "mapped_to_input_prob": None}
    d = best["details"]
    # map interval back to original input probability space (column probability semantics unknown to user)
    # If class=1: p_input interval equals class-space interval iff input prob is P1; else convert.
    # We always output mapping under assumption input column is "p_in" which satisfied: p_in = P(real) if --label-one-means=real else P(fake)
    # But users care about *their input column*; we provide both mappings generically:
    if target_class==1:
        # p1 in [lo,up] -> input_prob interval for label-one-means == 'real' : [lo,up]; for 'fake' : [1-up, 1-lo]
        mapped_real = (best["q_low"], best["q_up"])
        mapped_fake = (1.0-best["q_up"], 1.0-best["q_low"])
    else:
        # p0 in [lo,up] means (1-p1) in [lo,up] => p1 in [1-up, 1-lo]
        mapped_real = (1.0-best["q_up"], 1.0-best["q_low"])
        mapped_fake = (best["q_low"], best["q_up"])
    return {
        "class": int(target_class),
        "interval_class_space": {
            "q_low": round(best["q_low"], 6),
            "q_up": round(best["q_up"], 6),
            "width": round(best["q_up"]-best["q_low"], 6),
            "n_in_interval": int(d["n"]),
            "coverage_ratio": round(d["cov"], 6),
            "p_hat": round(d["p_hat"], 6),
            "hoeffding_lb": round(d["lb"], 6),
            "score": round(best["score"], 6),
        },
        "mapped_to_input_prob": {
            # two possible interpretations depending on how the user’s prob column was defined
            "if_input_is_P(label=1)": {"low": round(mapped_real[0],6), "up": round(mapped_real[1],6)},
            "if_input_is_P(label=0)": {"low": round(mapped_fake[0],6), "up": round(mapped_fake[1],6)}
        }
    }

# ---------- DTRA core ----------

def run_dtra(P1: np.ndarray,
             y: np.ndarray,
             nmin:int=50,
             strategy:str="width",
             lam:float=0.05,
             delta:float=0.05,
             laplace: Optional[Tuple[float,float]]=None,
             min_width:float=0.0,
             max_width:float=0.0)->Dict[str,Any]:
    scorer = _Scorer(strategy, lam, delta, laplace)
    out1 = _per_class_search(P1, y, target_class=1, nmin=nmin, scorer=scorer,
                             min_width=min_width, max_width=max_width)
    out0 = _per_class_search(P1, y, target_class=0, nmin=nmin, scorer=scorer,
                             min_width=min_width, max_width=max_width)
    return {
        "meta": {
            "n": int(len(P1)),
            "nmin": int(nmin),
            "strategy": strategy,
            "lambda": float(lam),
            "delta": float(delta),
            "laplace": None if laplace is None else {"a": float(laplace[0]), "b": float(laplace[1])},
            "search_domain": "[0.5, 1] for both classes (class-space p_c)"
        },
        "class_1": out1,
        "class_0": out0
    }

# ---------- CLI ----------

def _parse():
    ap = argparse.ArgumentParser(
        description="DTRA (double-threshold, calibration-only). Identical [0.5,1] search for both classes in class-space."
    )
    ap.add_argument("--input", required=True, help="CSV/XLSX/Parquet/NPY/NPZ")
    ap.add_argument("--label-col", default=None, help="Binary labels column")
    ap.add_argument("--prob-col", default=None, help="Probability column")
    ap.add_argument("--label-one-means", default="real", choices=["real","fake"],
                    help="Meaning of label==1 relative to the probability column")
    ap.add_argument("--nmin", type=int, default=50, help="Minimal samples per bin (equal-frequency) in class-space")
    ap.add_argument("--strategy", default="width", choices=["width","cover","countinv","none"],
                    help="Scoring strategy: LB + λ*width | LB + λ*coverage | LB − λ/n | LB only")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.05, help="Trade-off λ")
    ap.add_argument("--delta", type=float, default=0.05, help="Hoeffding delta (1−δ confidence)")
    ap.add_argument("--laplace", type=str, default="", help="Optional 'a,b' smoothing for p̂ before LB (e.g., '1,1')")
    ap.add_argument("--min-width", type=float, default=0.0, help="Enforce min interval width in class-space")
    ap.add_argument("--max-width", type=float, default=0.0, help="Enforce max interval width (0=unbounded)")
    ap.add_argument("--out-json", default=None, help="Write summary JSON")
    ap.add_argument("--dump-bins", default=None, help="Dump per-class bins CSV (requires pandas)")
    return ap.parse_args()

def _dump_bins(path: str, P1: np.ndarray, y: np.ndarray, nmin:int):
    if _pd is None: return
    rows=[]
    for cls in (0,1):
        y_c = (y==cls).astype(int)
        p_c = P1 if cls==1 else (1.0 - P1)
        bins = _build_bins_class_space(p_c, y_c, nmin=nmin)
        for k,b in enumerate(bins):
            rows.append({"class": cls, "idx": k, "lo": b.lo, "hi": b.hi, "n": b.n, "succ": b.succ})
    _pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")

def main():
    args = _parse()
    df = _load_any(args.input)
    lab, pr = _pick_cols(df, args.label_col, args.prob_col)
    P1, y, p_input = _coerce_arrays(df, lab, pr, args.label_one_means)

    lap = None
    if args.laplace.strip():
        try:
            a,b = args.laplace.split(",")
            lap = (float(a), float(b))
        except Exception:
            raise ValueError("--laplace expects 'a,b', e.g., 1,1")

    out = run_dtra(
        P1=P1, y=y, nmin=args.nmin, strategy=args.strategy, lam=args.lam,
        delta=args.delta, laplace=lap, min_width=args.min_width, max_width=args.max_width
    )
    js = json.dumps(out, ensure_ascii=False, indent=2)
    print(js)
    if args.out_json:
        with open(args.out_json,"w",encoding="utf-8") as f: f.write(js)
    if args.dump_bins:
        _dump_bins(args.dump_bins, P1, y, args.nmin)

if __name__ == "__main__":
    main()
