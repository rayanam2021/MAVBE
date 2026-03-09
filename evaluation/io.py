"""
I/O helpers for evaluation: load MOTChallenge-format files and convert to frame lists.
"""
from __future__ import print_function

import numpy as np
from collections import defaultdict


def frames_from_boxes_and_ids(gt_boxes_per_frame, pred_boxes_ids_per_frame):
    """
    Build gt_frames and pred_frames from per-frame lists (e.g. from CARLA-MOT).

    gt_boxes_per_frame: list of list of bbox_xyxy. If no IDs, assign 0,1,2,... per frame.
    pred_boxes_ids_per_frame: list of list of (bbox_xyxy, track_id) or (x1,y1,x2,y2, track_id).
    Returns: (gt_frames, pred_frames) for compute_metrics().
    """
    gt_frames = []
    for boxes in gt_boxes_per_frame:
        if not boxes:
            gt_frames.append([])
            continue
        # Allow boxes as 4-tuple or 5-tuple (with id at end)
        with_id = []
        for i, b in enumerate(boxes):
            if len(b) >= 5:
                tid, box = b[-1], tuple(b[:4])
            else:
                tid, box = i, tuple(b[:4])
            with_id.append((tid, box))
        gt_frames.append(with_id)

    pred_frames = []
    for row in pred_boxes_ids_per_frame:
        if not row:
            pred_frames.append([])
            continue
        with_id = []
        for r in row:
            r = list(r) if hasattr(r, "__iter__") and not isinstance(r, (str, dict)) else [r]
            if len(r) >= 5:
                track_id, box = int(r[4]), tuple(float(x) for x in r[:4])
            elif len(r) == 2:
                box, track_id = r[0], int(r[1])
                box = tuple(box[:4])
            else:
                track_id, box = 0, tuple(r[:4])
            with_id.append((track_id, box))
        pred_frames.append(with_id)

    return gt_frames, pred_frames


def load_mot_file(path, bbox_format="xyxy"):
    """
    Load a MOT format file (e.g. gt/gt.txt or tracker output).

    Format: one line per detection: frame,id,bb_left,bb_top,bb_width,bb_height[,conf]
    Returns: list of list of (id, bbox). out[frame_idx] = [(id, (x1,y1,x2,y2)), ...]
    Frame indices are 1-based in file; we convert to 0-based.
    """
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    if data.size == 0:
        return []

    # Columns: frame, id, left, top, width, height [, conf]
    frame_ids = data[:, 0].astype(int)
    ids = data[:, 1].astype(int)
    left = data[:, 2]
    top = data[:, 3]
    w = data[:, 4]
    h = data[:, 5]

    # Convert to xyxy if needed
    x1 = left
    y1 = top
    x2 = left + w
    y2 = top + h
    if bbox_format == "xyxy":
        boxes = np.column_stack((x1, y1, x2, y2))
    else:
        boxes = np.column_stack((x1, y1, w, h))

    # Group by frame (convert to 0-based index)
    by_frame = defaultdict(list)
    for i in range(len(frame_ids)):
        f = int(frame_ids[i]) - 1  # 1-based -> 0-based
        by_frame[f].append((int(ids[i]), tuple(boxes[i].tolist())))

    max_frame = max(by_frame.keys()) if by_frame else -1
    return [by_frame.get(f, []) for f in range(max_frame + 1)]


def save_metrics_report(metrics, path):
    """Write metrics dict to a text file."""
    from .metrics import format_metrics
    with open(path, "w") as f:
        f.write(format_metrics(metrics))
        f.write("\n")
    print("Wrote %s" % path)
