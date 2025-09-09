# DTRA: Double-Threshold Reliable Algorithm (Calibration-Only)

Single-file Python implementation of the Double-Threshold Reliable Algorithm for binary reliability analysis
(e.g., deepfake detection). The method searches **within [0.5, 1]** in class space for both classes, builds
**equal-frequency bins** (each bin has at least `N_min` samples), and selects intervals that optimize a
**Hoeffding lower bound** on precision with optional width/coverage trade-offs.

## Features
- Calibration-only; no model training required.
- Equal-frequency probability confusion per class on `[0.5, 1]`.
- Double-threshold search over consecutive bins with multiple scoring strategies:
  - `width`: LB + λ * interval width
  - `cover`: LB + λ * coverage ratio
  - `countinv`: LB − λ / n_in_interval
  - `none`: LB only
- Hoeffding lower bound for distribution-free guarantees.
- Robust I/O: CSV/XLSX/Parquet/NPY/NPZ; Chinese/English header heuristics.
- Optional Laplace/Beta smoothing on p̂ before LB via `--laplace a,b`.

## Install
```bash
pip install -r requirements.txt
```

## Usage
```bash
python dtra.py --input calib.xlsx   --label-col "真实标签"   --prob-col "预测为真实样本的概率"   --label-one-means real   --nmin 200   --strategy width   --lambda 0.1   --delta 0.05   --out-json result.json   --dump-bins bins.csv
```

Key flags:
- `--strategy {width,cover,countinv,none}`
- `--laplace a,b` to apply Beta(a,b) smoothing on p̂ before the Hoeffding bound
- `--min-width` / `--max-width` to constrain interval width in class space

## Example
A tiny synthetic calibration file is provided in `examples/calib.csv`:
```bash
python dtra.py --input examples/calib.csv --nmin 50 --strategy cover --lambda 0.05 --out-json out.json
```

## License
Apache-2.0. See [LICENSE](LICENSE).
