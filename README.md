DTRA: Dual-Threshold Risk-Aware Reliability Estimation

This repository provides the implementation of the Dual-Threshold, Risk-Aware Reliability Estimation (DTRA) method, as introduced in our paper.
The algorithm searches for optimal probability intervals that maximize reliability and reduce misclassification risk, following the utility function in Eq. (5).

The method is implemented in dtra.py.

Rejection strategy is supported: samples outside the selected interval are rejected instead of forced prediction.

Optimizations such as Laplace smoothing and Hoeffding bounds are included for stability.

Run python dtra.py --help for usage instructions.
