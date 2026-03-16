# vim: expandtab:ts=4:sw=4


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with 24-D IMM state (3×7-D sub-models + 3 mode probs).
    State is maintained in 3-D world-frame coordinates [pX, pZ, pY] (metres).
    """

    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None, tlwh=None, world_pos=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age
        # Last observed bounding box — used for visualization only.
        # The filter state is 3-D world-frame; bboxes come from detections.
        self._last_tlwh = tlwh.copy() if tlwh is not None else None
        # Last observed 3-D world position — used for direct-3D fallback matching.
        import numpy as np
        self._last_world_pos = np.asarray(world_pos, dtype=np.float64) if world_pos is not None else None

    def predict(self, kf, other_track_means=None):
        """Propagate the state distribution to the current time step using the
        filter prediction step.

        Parameters
        ----------
        kf : filter instance (e.g. BehavioralEKFFilter or KalmanFilter)
            The motion filter.
        other_track_means : optional list of ndarray
            Other tracks' mean vectors (for behavioral/social force models).
        """
        if other_track_means is not None:
            self.mean, self.covariance = kf.predict(
                self.mean, self.covariance, other_track_means=other_track_means
            )
        else:
            self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
        # Inflate uncertainty during occlusion — capped to prevent numerical blowup.
        # Only inflate the 21×21 model-state block; mode-probability entries
        # ([21:24]) are mixing weights, not variances — scaling them is wrong.
        if self.time_since_update > 1:
            inflation = min(1.5 ** (self.time_since_update - 1), 4.0)
            self.covariance[:21, :21] *= inflation

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : filter instance (e.g. BehavioralEKFFilter or KalmanFilter)
            The motion filter.
        detection : Detection
            The associated detection.

        """
        import numpy as np
        if detection.world_pos is not None:
            self.mean, self.covariance = kf.update(
                self.mean, self.covariance, detection.world_pos)
            self._last_world_pos = np.asarray(detection.world_pos, dtype=np.float64)
        # If no depth this frame, skip filter update — appearance still matches
        # and the filter coasts on its prediction until depth returns.
        self._last_tlwh = detection.tlwh.copy()
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def to_tlwh(self):
        """Return last observed bounding box as (top-left x, top-left y, w, h).
        Returns zeros if the track has never been associated with a detection.
        """
        import numpy as np
        if self._last_tlwh is None:
            return np.zeros(4)
        return self._last_tlwh.copy()

    def to_tlbr(self):
        """Return last observed bounding box as (x1, y1, x2, y2)."""
        ret = self.to_tlwh()
        ret[2:] += ret[:2]
        return ret

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
