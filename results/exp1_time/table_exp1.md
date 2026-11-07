# Experiment 1 (Time) in our code: dinov3_sat features, our L-TAE + decoder, Experiment 2 settings

Test mIoU on fold 5, mean ± std over seeds (n = number of seeds).

## Baselines (test mIoU, fold 5)
| Single date (calendar month 07, chosen on val) | 29.5 ± 0.1 (n=3) |
| Mean of all dates | 36.2 ± 0.4 (n=3) |
| Mean of cloud-free dates | 37.0 ± 0.1 (n=3) |

## L-TAE grid (test mIoU)
| | none | position | date | both |
|---|---|---|---|---|
| all dates | 49.8 ± 0.2 (n=3) | 51.0 ± 0.1 (n=3) | 51.6 ± 0.2 (n=3) | 51.4 ± 0.0 (n=3) |
| clear dates | 49.1 ± 0.1 (n=3) | 49.6 ± 0.5 (n=3) | 51.3 ± 0.4 (n=3) | 50.6 ± 0.5 (n=3) |

Shuffled dates (all): 49.8 ± 0.4 (n=3)

## Key differences (test mIoU points; seeds pooled, paired bootstrap 95% CI over test series)
- time series vs one date: +6.7  (pooled +6.65, 95% CI [+5.63, +7.71])
- removing clouds (mean): +0.8  (pooled +0.86, 95% CI [+0.41, +1.33])
- attention beyond cloud removal (L-TAE all, no time vs mean of cloud-free): +12.8  (pooled +12.81, 95% CI [+11.84, +13.79])
- order (all dates): +1.2  (pooled +1.20, 95% CI [+0.61, +1.77])
- calendar date beyond order (all): +0.6  (pooled +0.56, 95% CI [+0.11, +1.01])
- order (clear dates): +0.5  (pooled +0.46, 95% CI [-0.00, +0.90])
- calendar date beyond order (clear): +1.7  (pooled +1.70, 95% CI [+1.20, +2.20])
- real vs shuffled dates (all): +1.8  (pooled +1.76, 95% CI [+1.24, +2.27])
- removing clouds (L-TAE, date): -0.3  (pooled -0.28, 95% CI [-0.73, +0.20])
- L-TAE (all, date) vs one date: +22.1  (pooled +22.08, 95% CI [+20.71, +23.31])

## Single date by calendar month, seed 0 (val / test)
  month 01: 14.9 / 15.5
  month 02: 19.9 / 19.1
  month 03: 18.7 / 19.0
  month 04: 24.7 / 23.3
  month 05: 19.8 / 20.2
  month 06: 26.4 / 26.9
  month 07: 29.6 / 29.5
  month 08: 23.9 / 25.1
  month 09: 14.4 / 14.0
  month 10: 14.7 / 14.6
  month 11: 14.8 / 15.7
  month 12: 12.9 / 13.7
