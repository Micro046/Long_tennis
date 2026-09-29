# Regression results

Generated 20260929-110624 at commit `7742144` by `eval/run.py`.

| Benchmark | Split | Kind | Result |
|---|---|---|---|
| amateur-indoor-4apx (21 min, end changes) | dev | identity | 94 correct / 0 wrong / 2 missed -> **97.9%** |
| amateur-indoor-low-camera-otJ (13 min) | dev | identity | 69 correct / 0 wrong / 4 missed -> **94.5%** |
| amateur-indoor-swingvision-view-MZZ (13.6 min) | heldout | identity | 79 correct / 0 wrong / 4 missed -> **95.2%** |
| 4apx exhaustive hit/bounce windows (2 x 12 s) | dev | events | hits P 0.75 R 0.923 (**F1 0.828**); bounces P 0.727 R 1.0 (**F1 0.842**); hitter 11/12; stroke 6/11 |
| MZZ exhaustive hit/bounce windows (3 x 12 s) | heldout | events | hits P 0.6 R 0.375 (**F1 0.462**); bounces P 0.667 R 1.0 (**F1 0.8**); hitter 6/6; stroke 4/6 |
