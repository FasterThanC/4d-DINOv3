
## Baselines (test mIoU, fold 5)
| Single date (m07, chosen on val) | 30.8 ± 0.4 (n=3) |
| Mean of all dates | 37.5 ± 0.2 (n=3) |
| Mean of cloud-free dates | 37.6 ± 0.2 (n=3) |

## L-TAE grid (test mIoU)
| | none | position | date | both |
|---|---|---|---|---|
| all dates | 51.1 ± 0.4 (n=3) | 52.6 ± 0.2 (n=3) | 52.9 ± 0.2 (n=3) | 52.8 ± 0.3 (n=3) |
| clear dates | 51.3 ± 0.2 (n=3) | 51.9 ± 0.1 (n=3) | 52.8 ± 0.2 (n=3) | 52.6 ± 0.1 (n=3) |

Shuffled dates (all): 51.1 ± 0.4 (n=3)

## Key differences (test mIoU points)
- time series vs one date: +6.7
- removing clouds (mean): +0.0
- attention beyond cloud removal: +13.6
- order (all dates): +1.4
- calendar date beyond order (all): +0.3
- order (clear dates): +0.6
- calendar date beyond order (clear): +0.9

## Single date by month (val / test)
  single_m01: 15.4 / 16.4
  single_m02: 20.0 / 19.8
  single_m03: 19.5 / 19.8
  single_m04: 25.0 / 24.7
  single_m05: 20.5 / 20.9
  single_m06: 28.0 / 28.6
  single_m07: 31.1 / 30.8
  single_m08: 25.1 / 26.2
  single_m09: 15.1 / 14.7
  single_m10: 15.2 / 14.9
  single_m11: 15.3 / 15.4
  single_m12: 13.4 / 13.9
