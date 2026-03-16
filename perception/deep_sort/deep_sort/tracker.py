# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import behavioral_imm as _imm
from . import iou_matching
from . import linear_assignment
from .track import Track


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : behavioral_ekf.BehavioralEKFFilter
        Filter for target trajectories (behavioral EKF with CT + social force).
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_age=3000, n_init=3,
                 fallback_mode="iou", fallback_3d_threshold=2.0,
                 max_iou_distance=0.7, lambda_=0.5):
        """
        Parameters
        ----------
        fallback_mode : str
            Stage-2 fallback matching strategy for unconfirmed / recently-missed
            tracks.  Options:
              "3d"  – Euclidean distance between last observed world positions
                      (no filter involved).  Threshold: fallback_3d_threshold m.
              "iou" – Bounding-box IOU on the cached last-detected bbox.
                      Threshold: max_iou_distance.
        fallback_3d_threshold : float
            Maximum Euclidean distance (metres) allowed in "3d" fallback mode.
        max_iou_distance : float
            Maximum 1-IOU cost allowed in "iou" fallback mode.
        lambda_ : float
            Weight for appearance vs motion cost (0=motion only, 1=appearance only).
        """
        self.metric = metric
        self.max_age = max_age
        self.n_init = n_init
        self.fallback_mode = fallback_mode
        self.fallback_3d_threshold = fallback_3d_threshold
        self.max_iou_distance = max_iou_distance
        self.lambda_ = lambda_

        self.kf = _imm.BehavioralIMMFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for i, track in enumerate(self.tracks):
            other_means = [self.tracks[j].mean for j in range(len(self.tracks)) if j != i]
            track.predict(self.kf, other_track_means=other_means)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):
        gate_threshold = _imm.chi2inv95[3]  # 3-D world-space chi2 gate

        def _world_positions(det_indices):
            """Return list of world_pos arrays; None entries get zeros (gated out)."""
            positions, valid = [], []
            for i in det_indices:
                wp = detections[i].world_pos
                positions.append(wp if wp is not None else np.zeros(3))
                valid.append(wp is not None)
            return np.asarray(positions), valid

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            appearance_cost = self.metric.distance(features, targets)  # ∈ [0, 1]

            # Normalised Mahalanobis: divide by gate_threshold → ∈ [0, 1] inside
            # gate, 1e5 outside.  No-depth detections get motion_cost=0 so
            # appearance is the sole criterion for those columns.
            world_pos_arr, valid = _world_positions(detection_indices)
            motion_cost = np.zeros_like(appearance_cost)
            for row, track_idx in enumerate(track_indices):
                gating_dist = self.kf.gating_distance(
                    tracks[track_idx].mean, tracks[track_idx].covariance,
                    world_pos_arr)
                for col, (dist, is_valid) in enumerate(zip(gating_dist, valid)):
                    if is_valid:
                        motion_cost[row, col] = (
                            dist / gate_threshold if dist <= gate_threshold
                            else 1e5)
                    # else: no depth → leave 0, appearance decides

            return self.lambda_ * appearance_cost + (1 - self.lambda_) * motion_cost

        def fallback_3d_metric(tracks, _dets, track_indices, detection_indices):
            """Euclidean distance between last observed world positions — no filter."""
            cost_matrix = np.full((len(track_indices), len(detection_indices)), 1e5)
            for row, track_idx in enumerate(track_indices):
                last_wp = tracks[track_idx]._last_world_pos
                if last_wp is None:
                    continue
                for col, det_idx in enumerate(detection_indices):
                    wp = detections[det_idx].world_pos
                    if wp is None:
                        continue
                    dist = float(np.linalg.norm(last_wp - wp))
                    if dist <= self.fallback_3d_threshold:
                        cost_matrix[row, col] = dist
            return cost_matrix

        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Stage 1: confirmed tracks — appearance + 3-D Mahalanobis gate.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Stage 2: unconfirmed + recently-missed tracks — fallback matching.
        fallback_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1]

        if self.fallback_mode == "iou":
            fallback_metric = iou_matching.iou_cost
            fallback_threshold = self.max_iou_distance
        else:  # "3d"
            fallback_metric = fallback_3d_metric
            fallback_threshold = self.fallback_3d_threshold

        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                fallback_metric, fallback_threshold, self.tracks,
                detections, fallback_candidates, unmatched_detections)

        matches = matches_a + matches_b
        # print("LEN MATCHES A AND B , ", len(matches_a), len(matches_b))
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        if detection.world_pos is not None:
            mean, covariance = self.kf.initiate(detection.world_pos)
        else:
            # No depth this frame — create a placeholder filter state with huge
            # uncertainty.  The track will match via appearance/IOU until depth
            # returns, at which point the filter will be corrected on update.
            mean, covariance = self.kf.initiate_no_depth()
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature, tlwh=detection.tlwh,
            world_pos=detection.world_pos))
        self._next_id += 1
