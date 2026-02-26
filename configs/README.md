# Configs

## deep_sort.yaml

Used by `detect_dual_tracking.py` when using the external **deep_sort_pytorch** package.

- **REID_CKPT**: Path to the person Re-ID checkpoint. Place a `.pth` (or `.t7`) file here, or set an absolute path in the YAML. You can obtain a checkpoint from the [deep_sort_pytorch](https://github.com/ZQPei/deep_sort_pytorch) repo (e.g. `deep_sort/deep/checkpoint/ckpt.t7` or their PyTorch equivalent).
- Other keys (MAX_DIST, MAX_AGE, etc.) can be tuned as needed.

`detect_dual_tracking.py` resolves paths in this file relative to the MAVBE repo root.
