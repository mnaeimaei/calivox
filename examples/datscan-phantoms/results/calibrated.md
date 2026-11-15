# DaT-SPECT phantoms — calibrated leaderboard

Out-of-fold predictions; calibration is cross-fitted. Sorted by calibrated log loss (lower is better).

| Run | Model | Inputs | Aug. | Log loss | AUROC | AUPRC | Brier | ECE | Bal. acc. | Calibration | Calibrated log loss | Calibrated ECE | Train (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| resnet3d__S5 | resnet3d | roi | yes | 0.2553 | 0.9675 | 0.9769 | 0.0616 | 0.0459 | 0.928 | temperature | 0.2244 | 0.0352 | 1653 |
| simple3d__S6 | simple3d | whole+roi | yes | 0.2802 | 0.9541 | 0.9675 | 0.0846 | 0.0467 | 0.885 | temperature | 0.2718 | 0.0344 | 183 |
| simple3d__S3 | simple3d | whole+roi | no | 0.3057 | 0.9496 | 0.9646 | 0.0916 | 0.0620 | 0.884 | temperature | 0.2864 | 0.0510 | 127 |
| simple3d__S4 | simple3d | whole | yes | 0.2962 | 0.9576 | 0.9702 | 0.0797 | 0.0353 | 0.889 | temperature | 0.2918 | 0.0352 | 208 |
| resnet3d__S2 | resnet3d | roi | no | 0.3203 | 0.9478 | 0.9604 | 0.0931 | 0.0657 | 0.859 | platt | 0.2996 | 0.0631 | 996 |
| simple3d__S1 | simple3d | whole | no | 0.3418 | 0.9405 | 0.9578 | 0.0984 | 0.0659 | 0.875 | temperature | 0.3171 | 0.0520 | 103 |
| r3d18__S5 | r3d18 | roi | yes | 0.3951 | 0.9355 | 0.9552 | 0.1045 | 0.0762 | 0.866 | temperature | 0.3522 | 0.0652 | 1067 |
| slice2p5d__S4 | slice2p5d | whole | yes | 0.3567 | 0.9331 | 0.9524 | 0.1047 | 0.0504 | 0.843 | none | 0.3567 | 0.0504 | 799 |
| simple3d__S2 | simple3d | roi | no | 0.4371 | 0.9088 | 0.9343 | 0.1229 | 0.0917 | 0.828 | temperature | 0.3951 | 0.0600 | 49 |
| simple3d__S5 | simple3d | roi | yes | 0.4324 | 0.9089 | 0.9396 | 0.1213 | 0.0671 | 0.839 | temperature | 0.3991 | 0.0666 | 70 |
| slice2p5d__S2 | slice2p5d | roi | no | 0.4824 | 0.8577 | 0.9038 | 0.1520 | 0.1045 | 0.819 | temperature | 0.4531 | 0.0739 | 354 |
| slice2p5d__S5 | slice2p5d | roi | yes | 0.4738 | 0.8637 | 0.9048 | 0.1466 | 0.0624 | 0.809 | none | 0.4738 | 0.0624 | 277 |
| r3d18__S2 | r3d18 | roi | no | 0.5867 | 0.8347 | 0.8606 | 0.1722 | 0.1272 | 0.772 | temperature | 0.5344 | 0.1074 | 812 |
| swin3d_t__S2 | swin3d_t | roi | no | 0.6455 | 0.6526 | 0.7130 | 0.2284 | 0.0758 | 0.562 | none | 0.6455 | 0.0758 | 508 |
