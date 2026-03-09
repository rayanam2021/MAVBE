# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import behavioral_ekf_2 as behavioral_ekf
# from . import kalman_filter as behavioral_ekf
from . import linear_assignment
from . import iou_matching
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

    def __init__(self, metric, max_iou_distance=0.7, max_age=3000, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = 3000
        self.n_init = n_init

        self.kf = behavioral_ekf.BehavioralEKFFilter()
        # self.kf = behavioral_ekf.KalmanFilter()
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

        # def gated_metric(tracks, dets, track_indices, detection_indices):
        #     features = np.array([dets[i].feature for i in detection_indices])
        #     targets = np.array([tracks[i].track_id for i in track_indices])
        #     cost_matrix = self.metric.distance(features, targets)
        #     cost_matrix = linear_assignment.gate_cost_matrix(
        #         self.kf, cost_matrix, tracks, dets, track_indices,
        #         detection_indices)

        #     return cost_matrix

        # def gated_metric(tracks, dets, track_indices, detection_indices):

        #     features = np.array([dets[i].feature for i in detection_indices])
        #     targets = np.array([tracks[i].track_id for i in track_indices])

        #     # Appearance cost
        #     appearance_cost = self.metric.distance(features, targets)

        #     # Motion (Mahalanobis) cost
        #     measurements = np.asarray([dets[i].to_xyah() for i in detection_indices])

        #     motion_cost = np.zeros_like(appearance_cost)

        #     for row, track_idx in enumerate(track_indices):
        #         track = tracks[track_idx]

        #         # Mahalanobis distance for this track to all detections
        #         gating_dist = self.kf.gating_distance(
        #             track.mean, track.covariance, measurements, only_position=False
        #         )

        #         motion_cost[row, :] = gating_dist

        #     motion_cost = motion_cost / (motion_cost.max() + 1e-6)

        #     # Weighted fusion
        #     #current status on the pedestrian turning around video - lambda 1 works, lambda 0 doesnt work with a bad ekf
        #     lambda_ = 0.0   # appearance weight
        #     cost_matrix = lambda_ * appearance_cost + (1 - lambda_) * motion_cost

        #     return cost_matrix


        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            appearance_cost = self.metric.distance(features, targets)

            measurements = np.asarray([dets[i].to_xyah() for i in detection_indices])

            # Gate using chi2 threshold instead of relative normalization
            gate_threshold = behavioral_ekf.chi2inv95[4]  # 4D measurement space

            motion_cost = np.full_like(appearance_cost, 1e5)  # default: large cost
            for row, track_idx in enumerate(track_indices):
                track = tracks[track_idx]
                gating_dist = self.kf.gating_distance(
                    track.mean, track.covariance, measurements, only_position=False
                )
                # Only allow matches within the chi2 gate
                motion_cost[row, gating_dist > gate_threshold] = 1e5
                motion_cost[row, gating_dist <= gate_threshold] = gating_dist[gating_dist <= gate_threshold]

            lambda_ = 0.5
            cost_matrix = lambda_ * appearance_cost + (1 - lambda_) * motion_cost
            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
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
