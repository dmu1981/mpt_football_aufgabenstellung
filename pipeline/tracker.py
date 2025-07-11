"""
# Simple Multi-Object Tracker using Kalman Filter and Hungarian Assignment    #
# This code implements two classes:                                           #
# 1. Filter  - A Kalman filter to estimate position and velocity of an        #
#              object over time.                                              #
# 2. Tracker - Manages multiple Filter instances, assigning new detections    #
#              to existing tracks using the Hungarian algorithm, creating     #
#              new tracks, and removing stale ones.                           #
#                                                                             #
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


class Filter:
    def __init__(self, z, cls, track_id):
        # I initialize the main properties of the filter here.
        self.id = track_id  # unique track identifier
        self.object_class = cls  # detected object class
        self.age = 1  # how many frames this track has existed
        self.missing_frames = 0  # count of consecutive frames without update

        # State vector [x, y, vx, vy]hochT; we start with zero velocity
        self.X = np.array([[z[0]], [z[1]], [0.0], [0.0]])
        # Covariance matrix: high uncertainty in velocity initially
        self.P = np.diag([10.0, 10.0, 50.0, 50.0])

        # Measurement noise covariance (pixel-level noise)
        self.R = np.diag([5.0, 5.0])
        # Measurement matrix: we only observe x and y
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        # Process noise covariance: small for position, larger for velocity
        self.Q = np.diag([0.01, 0.01, 0.5, 0.5])

        # Store width and height of bounding box
        self.width = z[2]
        self.height = z[3]
        # Keep last measurement to initialize velocity later
        self.last_z = np.array([z[0], z[1]])

    def predict(self, dt=1.0):
        # Predict next state using constant velocity model
        F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        # State prediction
        self.X = F @ self.X
        # Covariance prediction
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z_new, dt=1.0):
        # Incorporate a new measurement [x, y]
        z = np.array([[z_new[0]], [z_new[1]]])

        # On first update, estimate initial velocity
        if self.age == 1:
            vx = (z_new[0] - self.last_z[0]) / dt
            vy = (z_new[1] - self.last_z[1]) / dt
            self.X[2, 0], self.X[3, 0] = vx, vy

        # Innovation (measurement residual)
        y = z - self.H @ self.X
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.X = self.X + K @ y
        # Covariance update
        self.P = (np.eye(4) - K @ self.H) @ self.P

        # Update bbox size
        self.width = z_new[2]
        self.height = z_new[3]

        # Reset missing counter and age up
        self.missing_frames = 0
        self.age += 1
        self.last_z = np.array([z_new[0], z_new[1]])

    def no_update(self):
        # No detection matched: increase missing count and age
        self.missing_frames += 1
        self.age += 1

    def gating_distance(self, z_new):
        # Compute Mahalanobis distance to see if measurement fits this track
        z = np.array([[z_new[0]], [z_new[1]]])
        y = z - self.H @ self.X
        S = self.H @ self.P @ self.H.T + self.R
        return float(y.T @ np.linalg.inv(S) @ y)

    def should_delete(self, max_missing_frames=5):
        # Decide to delete track if too many frames without update
        return self.missing_frames > max_missing_frames

    @property
    def position(self):
        # Return current [x, y, width, height]
        return np.array(
            [self.X[0, 0], self.X[1, 0], self.width, self.height], dtype=np.float32
        )

    @property
    def velocity(self):
        # Return current [vx, vy]
        return np.array([self.X[2, 0], self.X[3, 0]], dtype=np.float32)


class Tracker:
    def __init__(self):
        # Initialize tracker with empty list of filters
        self.name = "Tracker"
        self.tracks = []
        self.next_id = 0


    def start(self, data=None):
        # Reset tracker state
        self.tracks = []
        self.next_id = 0

    def stop(self, data=None):
        # No special cleanup needed here

        pass

    def step(self, data, dt=1.0):
        """
        Process one frame:
        1. Predict all existing tracks forward.
        2. Compute cost matrix between tracks and detections.
        3. Solve assignment via Hungarian algorithm.
        4. Update matched tracks, mark unmatched.
        5. Delete stale tracks, create new ones for unmatched detections.
        """
        detections = data.get("detections", np.empty((0, 4), np.float32))
        det_classes = data.get("classes", np.empty((0,), np.int32))
        N = len(self.tracks)
        M = len(detections)
        gating_threshold = 9.21  # chi-square 2DOF 95% limit
        large_cost = 1e6  # cost for impossible assignments

        # 1. Predict stage
        for f in self.tracks:
            f.predict(dt)

        # 2. Build cost matrix
        cost = np.full((N, M), large_cost, dtype=np.float32)
        for i, f in enumerate(self.tracks):
            for j, z in enumerate(detections):
                md2 = f.gating_distance(z)
                if md2 <= gating_threshold:
                    cost[i, j] = np.sqrt(md2)

        # 3. Solve assignment
        if N and M:
            row_ind, col_ind = linear_sum_assignment(cost)
        else:
            row_ind, col_ind = np.array([], dtype=int), np.array([], dtype=int)

        matched_tracks = set()
        matched_detections = set()
        # 4. Update matched
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= large_cost:
                continue
            self.tracks[r].update(detections[c], dt)
            self.tracks[r].object_class = int(det_classes[c])
            matched_tracks.add(r)
            matched_detections.add(c)

        # Handle unmatched: increase missing count or delete
        new_tracks = []
        for idx, f in enumerate(self.tracks):
            if idx not in matched_tracks:
                f.no_update()
            if not f.should_delete():
                new_tracks.append(f)
        self.tracks = new_tracks

        # 5. Create new tracks for unmatched detections
        for j, z in enumerate(detections):
            if j in matched_detections:
                continue
            f = Filter(z, int(det_classes[j]), self.next_id)
            self.next_id += 1
            self.tracks.append(f)

        # Prepare output
        positions = np.array([f.position for f in self.tracks], dtype=np.float32)
        velocities = np.array([f.velocity for f in self.tracks], dtype=np.float32)
        ages = [f.age for f in self.tracks]
        classes_out = [f.object_class for f in self.tracks]
        ids = [f.id for f in self.tracks]

        return {

            "tracks": positions,
            "trackVelocities": velocities,
            "trackAge": ages,
            "trackClasses": classes_out,
            "trackIds": ids,

        }
