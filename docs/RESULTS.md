# Regression results

Generated 20260928-193905 at commit `db0c697` by `eval/run.py`.

| Benchmark | Split | Kind | Result |
|---|---|---|---|
| amateur-indoor-4apx (21 min, end changes) | dev | identity | 94 correct / 0 wrong / 2 missed -> **97.9%** |
| amateur-indoor-low-camera-otJ (13 min) | dev | identity | 69 correct / 0 wrong / 4 missed -> **94.5%** |
| amateur-indoor-swingvision-view-MZZ (13.6 min) | heldout | identity | 83 correct / 0 wrong / 0 missed -> **100.0%** |
| 4apx exhaustive hit/bounce windows (2 x 12 s) | dev | events | hits P 0.75 R 0.667 (**F1 0.706**); bounces P 0.714 R 0.833 (**F1 0.769**); hitter 6/6; stroke 3/6 |
| MZZ exhaustive hit/bounce windows (3 x 12 s) | heldout | events | hits P 0.6 R 0.375 (**F1 0.462**); bounces P 0.583 R 0.7 (**F1 0.636**); hitter 6/6; stroke 4/6 |
