
import cv2
import numpy as np


class OpticalFlow:
    def __init__(self):
        self.name = "Optical Flow"  # Do not change the name of the module as otherwise recording replay would break!
        self.prev_gray = None  # Previous grayscale image
        self.prev_features = None  # Previous features for tracking


    def start(self, data):
        """Initialize the module at the beginning - unnecessary for this module, but required for interface consistency."""
        self.prev_gray = None
        self.prev_features = None

    def stop(self, data):
        """Clean up the module at the end - unnecessary for this module, but required for interface consistency."""
        self.prev_gray = None
        self.prev_features = None

    def step(self, data):

        """Process the current frame and calculate optical flow."""

        # Get current image
        current_image = data["image"]

        # Convert current image to grayscale
        current_gray = cv2.cvtColor(current_image, cv2.COLOR_RGB2GRAY)

        # Initialize optical flow to zero for first frame
        if self.prev_gray is None:
            self.prev_gray = current_gray.copy()
            return {"opticalFlow": np.array([0.0, 0.0], dtype=np.float32)}

        # Detect features to track in the previous frame if not already done
        if self.prev_features is None or len(self.prev_features) < 10:
            # Detect good features to track in previous frame
            corners = cv2.goodFeaturesToTrack(
                self.prev_gray,
                maxCorners=100,  # Maximum number of features
                qualityLevel=0.01,  # Quality threshold
                minDistance=10,  # Minimum distance between features
                blockSize=3,  # Size of averaging block
            )

            if corners is not None:
                self.prev_features = corners
            else:
                self.prev_gray = current_gray.copy()
                return {"opticalFlow": np.array([0.0, 0.0], dtype=np.float32)}

        # Calculate optical flow using Lucas-Kanade method
        new_features, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,  # Previous frame
            current_gray,  # Current frame
            self.prev_features,  # Points from previous frame
            None,  # Output array for new points
            winSize=(15, 15),  # Window size for search
            maxLevel=2,  # Maximum pyramid level
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        # Filter out points that were lost during tracking
        # status = 1 means the point was successfully tracked, 0 means it was lost
        successfully_tracked_old = self.prev_features[status == 1]
        successfully_tracked_new = new_features[status == 1]

        # If no points were successfully tracked, return zero motion
        if len(successfully_tracked_old) == 0:
            self.previous_frame = current_gray.copy()
            return {"opticalFlow": np.array([0.0, 0.0], dtype=np.float32)}

        # Calculate average motion (represents overall camera movement)
        motion_vectors = successfully_tracked_new - successfully_tracked_old
        mean_motion = -np.mean(motion_vectors.reshape(-1, 2), axis=0)

        # Update features and frame for next step
        self.prev_features = successfully_tracked_new.reshape(-1, 1, 2)

        if len(self.prev_features) < 5:
            self.prev_features = None

        self.prev_gray = current_gray.copy()

        # The "opticalFlow" signal must contain a 1x2 NumPy Array with the X and Y shift (delta values in pixels) of the image motion vector
        return {"opticalFlow": mean_motion.astype(np.float32)}
