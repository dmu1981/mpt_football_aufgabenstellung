import numpy as np


class Filter:
    _next_id = 1

    def __init__(self, z, cls):
        self.x = np.zeros((6,), dtype=np.float32)
        self.x[:4] = z
        self.cls = int(cls)

        self.P = np.diag([10, 10, 10, 10, 1000, 1000]).astype(np.float32)

        self.F = np.eye(6, dtype=np.float32)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0

        self.H = np.zeros((4, 6), dtype=np.float32)
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0

        self.Q = np.eye(6, dtype=np.float32) * 0.01
        self.R = np.eye(4, dtype=np.float32) * 1.0

        self.id = Filter._next_id
        Filter._next_id += 1

        self.age = 1
        self.misses = 0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.misses += 1

    def update(self, z):
        z = np.asarray(z, dtype=np.float32)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(6, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P
        self.misses = 0

    def get_state(self):
        return self.x[:4].copy()

    def get_velocity(self):
        return self.x[4:].copy()

    def is_deleted(self, max_misses=5):
        return self.misses >= max_misses


class Tracker:
    def __init__(self):
        self.name = "Tracker"
        self.filters = []
        self.max_misses = 5
        self.dist_thresh = 50.0

    def start(self, data):
        self.filters = []
        Filter._next_id = 1

    def stop(self, data):
        self.filters = []

    def step(self, data):
        dets = np.asarray(data.get("detections", []), dtype=np.float32)
        classes = list(data.get("classes", []))

        for f in self.filters:
            f.predict()

        unmatched_dets = list(range(len(dets)))

        for f in self.filters:
            best_idx = -1
            best_dist = float("inf")
            for i in unmatched_dets:
                dist = np.linalg.norm(f.get_state()[:2] - dets[i][:2])
                if dist < best_dist and dist < self.dist_thresh:
                    best_idx = i
                    best_dist = dist
            if best_idx >= 0:
                f.update(dets[best_idx])
                f.cls = classes[best_idx]
                unmatched_dets.remove(best_idx)

        for i in unmatched_dets:
            nf = Filter(dets[i], classes[i])
            self.filters.append(nf)

        self.filters = [f for f in self.filters if not f.is_deleted(self.max_misses)]

        return {
            "tracks": np.array([f.get_state() for f in self.filters], dtype=np.float32),
            "trackVelocities": np.array(
                [f.get_velocity() for f in self.filters], dtype=np.float32
            ),
            "trackAge": [f.age for f in self.filters],
            "trackClasses": [f.cls for f in self.filters],
            "trackIds": [f.id for f in self.filters],
        }