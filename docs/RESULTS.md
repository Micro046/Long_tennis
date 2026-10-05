# Regression results

Generated 20261005-125333 at commit `5fbc935` by `eval/run.py`.

| Benchmark | Split | Kind | Result |
|---|---|---|---|
| amateur-indoor-4apx (21 min, end changes) | dev | identity | 94 correct / 0 wrong / 2 missed -> **97.9%** |
| amateur-indoor-low-camera-otJ (13 min) | dev | identity | 69 correct / 0 wrong / 4 missed -> **94.5%** |
| amateur-indoor-swingvision-view-MZZ (13.6 min) | heldout | identity | 79 correct / 0 wrong / 4 missed -> **95.2%** |
| 4apx exhaustive hit/bounce windows (2 x 12 s) | dev | events | hits P 0.857 R 0.923 (**F1 0.889**); bounces P 0.8 R 1.0 (**F1 0.889**); hitter 12/12; stroke 7/12 |
| MZZ exhaustive hit/bounce windows (3 x 12 s) | heldout | events | hits P 0.615 R 0.5 (**F1 0.552**); bounces P 0.75 R 0.9 (**F1 0.818**); hitter 8/8; stroke 4/8 |
| TrackNet broadcast, leave-one-game-out (10 games) | heldout | tracknet_events | hit F1 **0.843** (P 0.852, R 0.835); bounce F1 **0.905** (P 0.864, R 0.95) |
| test1 SwingVision view, 60 s (all events) | dev | clip_events | hits P 0.962 R 0.926 (**F1 0.943**); bounces P 1.0 R 0.88 (**F1 0.936**); hitter 25/25; stroke 13/17; dead-ball events 1 |
| test2 indoor behind baseline, 90 s, edited (all events) | dev | clip_events | hits P 0.955 R 0.808 (**F1 0.875**); bounces P 1.0 R 0.913 (**F1 0.955**); hitter 20/21; stroke 5/5; dead-ball events 0 |
| test3 short rally, 30 s, phone upload (all events) | dev | clip_events | hits P 1.0 R 1.0 (**F1 1.0**); bounces P 1.0 R 0.889 (**F1 0.941**); hitter 10/10; stroke 3/3; dead-ball events 0 |
