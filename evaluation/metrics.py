"""
MOT and trajectory evaluation metrics.

Data format:
- gt_frames: list of list of (gt_id, bbox_xyxy). gt_frames[f] = [(id, (x1,y1,x2,y2)), ...]
- pred_frames: list of list of (track_id, bbox_xyxy). pred_frames[f] = [(id, (x1,y1,x2,y2)), ...]

Bbox is (x1, y1, x2, y2) in image coordinates.
"""
from __future__ import division

import numpy as np
from collections import defaultdict


def _bbox_xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return (x1, y1, x2 - x1, y2 - y1)


def iou_bbox(box1_xyxy, box2_xyxy):
    """IoU of two boxes in (x1, y1, x2, y2) format."""
    x1 = max(box1_xyxy[0], box2_xyxy[0])
    y1 = max(box1_xyxy[1], box2_xyxy[1])
    x2 = min(box1_xyxy[2], box2_xyxy[2])
    y2 = min(box1_xyxy[3], box2_xyxy[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1])
    a2 = (box2_xyxy[2] - box2_xyxy[0]) * (box2_xyxy[3] - box2_xyxy[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def center_distance(box1_xyxy, box2_xyxy):
    """Euclidean distance between box centers (e.g. in pixels)."""
    c1 = ((box1_xyxy[0] + box1_xyxy[2]) / 2, (box1_xyxy[1] + box1_xyxy[3]) / 2)
    c2 = ((box2_xyxy[0] + box2_xyxy[2]) / 2, (box2_xyxy[1] + box2_xyxy[3]) / 2)
    return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def _hungarian_match(cost_matrix):
    """Return (row_ind, col_ind) for minimum cost assignment. Uses scipy if available."""
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return row_ind, col_ind
    except ImportError:
        # Simple greedy: repeatedly pick smallest cost
        n, m = cost_matrix.shape
        used_row = [False] * n
        used_col = [False] * m
        row_ind, col_ind = [], []
        flat = np.argsort(cost_matrix.ravel())
        for idx in flat:
            i, j = idx // m, idx % m
            if not used_row[i] and not used_col[j]:
                row_ind.append(i)
                col_ind.append(j)
                used_row[i] = used_col[j] = True
        return np.array(row_ind), np.array(col_ind)


def _match_frame(gt_list, pred_list, min_iou=0.5):
    """
    Match GT and predictions for one frame by IoU (Hungarian).
    gt_list, pred_list: list of (id, bbox_xyxy).
    Returns: matches [(gt_idx, pred_idx)], fn_count, fp_count, list of (gt_id, pred_id, iou).
    """
    n_gt = len(gt_list)
    n_pred = len(pred_list)
    if n_gt == 0 and n_pred == 0:
        return [], 0, 0, []
    if n_gt == 0:
        return [], 0, n_pred, []
    if n_pred == 0:
        return [], n_gt, 0, []

    # Cost = 1 - IoU (maximize IoU = minimize cost)
    cost = np.ones((n_gt, n_pred))
    for i in range(n_gt):
        for j in range(n_pred):
            iou_val = iou_bbox(gt_list[i][1], pred_list[j][1])
            if iou_val >= min_iou:
                cost[i, j] = 1.0 - iou_val
            else:
                cost[i, j] = 1.0  # no match

    row_ind, col_ind = _hungarian_match(cost)
    match_pairs = []
    match_info = []
    for ri, ci in zip(row_ind, col_ind):
        if cost[ri, ci] < 1.0:  # valid match (IoU >= min_iou)
            match_pairs.append((ri, ci))
            iou_val = 1.0 - cost[ri, ci]
            match_info.append((gt_list[ri][0], pred_list[ci][0], iou_val))

    matched_gt = set(p[0] for p in match_pairs)
    matched_pred = set(p[1] for p in match_pairs)
    fn_count = n_gt - len(matched_gt)
    fp_count = n_pred - len(matched_pred)
    return match_pairs, fn_count, fp_count, match_info


def compute_mot_metrics(gt_frames, pred_frames, min_iou=0.5):
    """
    Compute MOTA, MOTP, IDSW, FP, FN, and IDF1.

    gt_frames: list of list of (gt_id, bbox_xyxy)
    pred_frames: list of list of (track_id, bbox_xyxy)
    """
    total_gt = 0
    total_fp = 0
    total_fn = 0
    idsw = 0
    sum_iou = 0.0
    num_matches = 0
    # For IDF1: count (gt_id, pred_id) match frames
    id_match_frames = defaultdict(lambda: 0)
    # Last assigned pred_id per gt_id (for ID switch)
    last_pred_per_gt = {}
    # Total pred and gt detection counts (for IDF1 denominator)
    total_gt_dets = 0
    total_pred_dets = 0

    num_frames = max(len(gt_frames), len(pred_frames))
    for f in range(num_frames):
        gt_list = gt_frames[f] if f < len(gt_frames) else []
        pred_list = pred_frames[f] if f < len(pred_frames) else []
        total_gt += len(gt_list)
        total_gt_dets += len(gt_list)
        total_pred_dets += len(pred_list)

        match_pairs, fn, fp, match_info = _match_frame(gt_list, pred_list, min_iou=min_iou)
        total_fn += fn
        total_fp += fp

        for gt_id, pred_id, iou_val in match_info:
            sum_iou += iou_val
            num_matches += 1
            id_match_frames[(gt_id, pred_id)] += 1
            # ID switch: same gt_id was matched to a different pred_id before
            if gt_id in last_pred_per_gt and last_pred_per_gt[gt_id] != pred_id:
                idsw += 1
            last_pred_per_gt[gt_id] = pred_id

    # MOTA: 1 - (FN + FP + IDSW) / total_gt (Bernardin & Stiefelhagen 2008)
    mota = 1.0 - (total_fn + total_fp + idsw) / total_gt if total_gt > 0 else 0.0

    # MOTP: average IoU of matched pairs (often reported as 1 - mean_distance; we use mean IoU)
    motp = sum_iou / num_matches if num_matches > 0 else 0.0

    # IDF1: 2*IDTP / (2*IDTP + IDFP + IDFN). IDTP = sum of min(gt_count, pred_count) per (gt_id, pred_id) identity.
    # Simplified: IDTP = number of "correct" ID matches. We take sum of match counts as IDTP (each frame match is one).
    idtp = sum(id_match_frames.values())
    idfn = total_gt_dets - idtp
    idfp = total_pred_dets - idtp
    idf1 = 2.0 * idtp / (2 * idtp + idfp + idfn) if (2 * idtp + idfp + idfn) > 0 else 0.0

    return {
        "MOTA": mota,
        "MOTP": motp,
        "IDF1": idf1,
        "IDSW": idsw,
        "FP": total_fp,
        "FN": total_fn,
        "num_matches": num_matches,
        "total_gt": total_gt,
    }


def compute_trajectory_metrics(gt_frames, pred_frames, min_iou=0.5):
    """
    Compute RMSE (center distance), ADE, FDE for matched trajectories.

    Uses per-frame matching by IoU; for matched (gt_id, pred_id) over time,
    computes center distance. RMSE = sqrt(mean(d^2)), ADE = mean(d), FDE = mean distance at last frame per trajectory.
    """
    # Build trajectories: (gt_id, pred_id) -> list of (gt_center, pred_center) per frame
    # Match frame by frame and record centers for same (gt_id, pred_id)
    trajectory_pairs = defaultdict(list)  # (gt_id, pred_id) -> [(gt_center, pred_center), ...]

    num_frames = max(len(gt_frames), len(pred_frames))
    for f in range(num_frames):
        gt_list = gt_frames[f] if f < len(gt_frames) else []
        pred_list = pred_frames[f] if f < len(pred_frames) else []
        _, _, _, match_info = _match_frame(gt_list, pred_list, min_iou=min_iou)
        for gt_id, pred_id, _ in match_info:
            gt_box = next(box for i, box in gt_list if i == gt_id)
            pred_box = next(box for i, box in pred_list if i == pred_id)
            gt_center = ((gt_box[0] + gt_box[2]) / 2, (gt_box[1] + gt_box[3]) / 2)
            pred_center = ((pred_box[0] + pred_box[2]) / 2, (pred_box[1] + pred_box[3]) / 2)
            trajectory_pairs[(gt_id, pred_id)].append((gt_center, pred_center))

    all_distances = []
    final_distances = []
    for key, points in trajectory_pairs.items():
        for gt_c, pred_c in points:
            d = np.sqrt((gt_c[0] - pred_c[0]) ** 2 + (gt_c[1] - pred_c[1]) ** 2)
            all_distances.append(d)
        if points:
            gt_c, pred_c = points[-1]
            d = np.sqrt((gt_c[0] - pred_c[0]) ** 2 + (gt_c[1] - pred_c[1]) ** 2)
            final_distances.append(d)

    rmse = np.sqrt(np.mean(np.array(all_distances) ** 2)) if all_distances else float("nan")
    ade = np.mean(all_distances) if all_distances else float("nan")
    fde = np.mean(final_distances) if final_distances else float("nan")

    return {
        "RMSE": rmse,
        "ADE": ade,
        "FDE": fde,
        "num_trajectory_points": len(all_distances),
    }


def compute_metrics(gt_frames, pred_frames, min_iou=0.5):
    """
    Compute all metrics: MOT (MOTA, MOTP, IDF1, IDSW, FP, FN) and trajectory (RMSE, ADE, FDE).

    gt_frames: list of list of (gt_id, bbox_xyxy)
    pred_frames: list of list of (track_id, bbox_xyxy)
    """
    mot = compute_mot_metrics(gt_frames, pred_frames, min_iou=min_iou)
    traj = compute_trajectory_metrics(gt_frames, pred_frames, min_iou=min_iou)
    return {**mot, **traj}


def format_metrics(metrics):
    """Return a human-readable string of the metrics dict."""
    lines = []
    for k in ["MOTA", "MOTP", "IDF1", "IDSW", "FP", "FN", "RMSE", "ADE", "FDE"]:
        if k in metrics:
            v = metrics[k]
            if isinstance(v, float) and (v != v or abs(v) > 1e6):  # nan or inf
                lines.append("%s: N/A" % k)
            else:
                lines.append("%s: %.4g" % (k, v))
    return "\n".join(lines)
