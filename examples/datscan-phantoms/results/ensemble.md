# Ensemble

| Member | Calibration | Single-model log loss |
| --- | --- | --- |
| resnet3d__S5 | temperature | 0.2244 |
| simple3d__S6 | temperature | 0.2718 |

Final temperature: **0.8250**

| Estimate | Log loss | AUROC / macro-AUROC | Brier | ECE |
| --- | --- | --- | --- | --- |
| Best single run (resnet3d__S5) | 0.2244 | — | — | — |
| Ensemble, cross-fitted (selected on the same data) | 0.2005 | 0.9752 | 0.0556 | 0.0230 |
| Ensemble, **nested** (selection inside outer folds) | 0.2397 | 0.9699 | 0.0590 | 0.0276 |

Selection frequency across outer folds: resnet3d__S5 100%, simple3d__S6 60%, simple3d__S4 40%, simple3d__S1 40%, slice2p5d__S4 20%, resnet3d__S2 20%
