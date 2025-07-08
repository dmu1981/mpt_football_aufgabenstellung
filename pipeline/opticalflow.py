import cv2
import numpy as np

class OpticalFlow:
    def __init__(self):
        self.name = "Optical Flow"
        self.prev_gray = None
        self.prev_pts = None
        self.scaling = 0.6  # Downscale for performance
        self.retrack_threshold = 10  # Redetect Shi-Tomasi features if fewer remain

        # Shi-Tomasi corner detection parameters
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )

        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

    def start(self, data):
        print("[OpticalFlow] Module started.")
        self.prev_gray = None
        self.prev_pts = None

    def stop(self, data):
        print("[OpticalFlow] Module stopped.")
        self.prev_gray = None
        self.prev_pts = None

    def step(self, data):
        frame = data.get("image")
        if frame is None:
            return {"opticalFlow": np.array([0, 0], dtype=np.float32)}

        # Resize frame
        if self.scaling < 1.0:
            frame = cv2.resize(frame, (0, 0), fx=self.scaling, fy=self.scaling)

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # First frame: detect initial features
        if self.prev_gray is None or self.prev_pts is None:
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            return {"opticalFlow": np.array([0, 0], dtype=np.float32)}

        # Track features using Lucas-Kanade
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None, **self.lk_params)

        # Filter good points
        if next_pts is not None and status is not None:
            good_new = next_pts[status == 1]
            good_old = self.prev_pts[status == 1]

            if len(good_new) > 0:
                flow_vectors = good_new - good_old
                avg_flow = np.mean(flow_vectors, axis=0)
            else:
                avg_flow = np.array([0, 0], dtype=np.float32)

            # Redetect if too few points
            if len(good_new) < self.retrack_threshold:
                self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            else:
                self.prev_pts = good_new.reshape(-1, 1, 2)

        else:
            avg_flow = np.array([0, 0], dtype=np.float32)
            self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)

        self.prev_gray = gray
        return {"opticalFlow": avg_flow.astype(np.float32) / self.scaling}
