# vim: expandtab:ts=4:sw=4
"""
Tracker that uses the vanilla (constant-velocity) Kalman filter only.
No behavioral EKF, no other_track_means in predict.
"""
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    Multi-target tracker using vanilla Kalman filter (constant velocity model).
    Same interface as deep_sort.tracker.Tracker but predict() does not pass
    other_track_means, so the standard KalmanFilter is used as-is.
    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=3000, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward (vanilla KF only)."""
        for track in self.tracks:
            track.predict(self.kf, other_track_means=None)

    def update(self, detections):
        """Perform measurement update and track management."""
        matches, unmatched_tracks, unmatched_detections = self._match(detections)
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
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
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            appearance_cost = self.metric.distance(features, targets)
            measurements = np.asarray([dets[i].to_xyah() for i in detection_indices])
            gate_threshold = kalman_filter.chi2inv95[4]
            motion_cost = np.full_like(appearance_cost, 1e5)
            for row, track_idx in enumerate(track_indices):
                track = tracks[track_idx]
                gating_dist = self.kf.gating_distance(
                    track.mean, track.covariance, measurements, only_position=False
                )
                motion_cost[row, gating_dist > gate_threshold] = 1e5
                motion_cost[row, gating_dist <= gate_threshold] = gating_dist[gating_dist <= gate_threshold]
            lambda_ = 0.0
            cost_matrix = lambda_ * appearance_cost + (1 - lambda_) * motion_cost
            return cost_matrix

        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]
        matches_a, unmatched_tracks_a, unmatched_detections = linear_assignment.matching_cascade(
            gated_metric, self.metric.matching_threshold, self.max_age,
            self.tracks, detections, confirmed_tracks)
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = linear_assignment.min_cost_matching(
            iou_matching.iou_cost, self.max_iou_distance, self.tracks,
            detections, iou_track_candidates, unmatched_detections)
        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1
