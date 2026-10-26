# 1_same_recording_gaussian

Generated 2026-09-17 18:56 | data source: **physionet**


| | |
|---|---|
| Protocol | **same recording** - learn each person from 70% of their `rec_1` beats, then test on the other 30% of the same recording (plus an equal number of other people's beats). The easy case. |
| Wavelet | Gaussian derivative (gaus1) |
| People | 30 (Person_01-Person_30) |
| Seed | 42 |
| Figures | written to `paper/figures/` |

See [`../README.md`](../README.md) for how this run compares with the others
and for what each file in `metrics/`, `data/` and `logs/` contains.
