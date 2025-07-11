import numpy as np

import cv2
from sklearn.decomposition import PCA


class ShirtClassifier:
    def __init__(self, update_rate=0.1, use_lab=True, apply_pca=True, pca_components=2):
        # Name identifier for the classifier
        self.name = "Shirt Classifier"
        # Flag indicating whether initial team colors have been determined
        self.initialized = False
        # Cached team colors (as HSV or BGR arrays) for team A and B
        self.teamA_color = None
        self.teamB_color = None
        # Rate at which to update the running average of team colors
        self.update_rate = update_rate
        # Use LAB color space if True, else HSV
        self.use_lab = use_lab
        # Whether to apply PCA denoising to color vectors
        self.apply_pca = apply_pca
        # Number of PCA components to use
        self.pca_components = pca_components
        # Background subtractor for isolating jersey region
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=25, detectShadows=False
        )
        # Kernel for morphological operations
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def start(self, data):
        """
        Called at the beginning of a new video/run: resets internal state.
        """
        self.initialized = False
        self.teamA_color = None
        self.teamB_color = None
        self.bg_subtractor.clear()

    def stop(self, data):
        """
        Called at the end of processing. Currently unused.
        """
        pass


    def step(self, data):

        """
        Main per-frame processing method. Returns team colors and assignments.

        Args:
            data (dict): contains keys:
                - 'image': current frame (H x W x 3 BGR image)
                - 'tracks': list of bounding boxes [x, y, w, h]
                - 'trackClasses': list of class labels per track (2 = player)

        Returns:
            dict with:
                - 'teamAColor', 'teamBColor': output colors as BGR tuples
                - 'teamClasses': list of 0/1/2 assignments per track
        """
        # Retrieve input data
        image = data.get("image")
        tracks = data.get("tracks", [])
        track_classes = data.get("trackClasses", [])

        # If no players detected, return default zeros
        if len(tracks) == 0 or len(track_classes) == 0:
            return {"teamAColor": (0, 0, 0), "teamBColor": (0, 0, 0), "teamClasses": []}

        # Extract mean colors from player crops
        colors = []  # list of mean color vectors
        indices = []  # corresponding indices in original track list
        for idx, (track, cls) in enumerate(zip(tracks, track_classes)):
            # Only process if this track is classified as a player (class 2)
            if cls != 2:
                continue
            # Extract bounding box coordinates
            x, y, w, h = track.astype(int)
            # Calculate crop coordinates, ensuring they are within image bounds
            x1, y1 = max(x - w // 2, 0), max(y - h // 2, 0)
            x2, y2 = min(x + w // 2, image.shape[1] - 1), min(
                y + h // 2, image.shape[0] - 1
            )
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            # Background subtraction + morphology to isolate jersey
            mask = self.bg_subtractor.apply(crop)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel)
            masked = cv2.bitwise_and(crop, crop, mask=mask)

            # Convert to chosen color space (LAB or HSV)
            conv = cv2.COLOR_BGR2LAB if self.use_lab else cv2.COLOR_BGR2HSV
            cs = cv2.cvtColor(masked, conv)

            # Flatten pixels to a list of color vectors
            pixels = cs.reshape(-1, 3).astype(np.float32)
            if pixels.shape[0] == 0:
                continue

            # Compute mean color, with optional PCA denoising
            if self.apply_pca and pixels.shape[0] > self.pca_components:
                pca = PCA(n_components=self.pca_components)
                comps = pca.fit_transform(pixels)
                # Reconstruct to original space, then mean
                recon = pca.inverse_transform(comps)
                mean = np.mean(recon, axis=0)
            else:
                mean = np.mean(pixels, axis=0)

            colors.append(mean)
            indices.append(idx)

        # If not enough players, return default or previous colors
        if len(colors) < 2:
            team_classes = [0 if c != 2 else 1 for c in track_classes]
            outA = tuple(int(c) for c in (self.teamA_color or (0, 0, 0)))
            outB = tuple(int(c) for c in (self.teamB_color or (0, 0, 0)))
            return {"teamAColor": outA, "teamBColor": outB, "teamClasses": team_classes}

        colors = np.array(colors)

        # Initial clustering on first frame
        if not self.initialized:
            # Use first two detected players as initial cluster centers
            c0, c1 = colors[0], colors[1]
            # Simple k-means clustering for 10 iterations
            for _ in range(10):
                d0 = np.linalg.norm(colors - c0, axis=1)
                d1 = np.linalg.norm(colors - c1, axis=1)
                lbls = (d1 < d0).astype(int)
                if np.any(lbls == 0):
                    c0 = np.mean(colors[lbls == 0], axis=0)
                if np.any(lbls == 1):
                    c1 = np.mean(colors[lbls == 1], axis=0)
            self.teamA_color = c0
            self.teamB_color = c1
            self.initialized = True
            labels = lbls
        else:
            # Assign each player to the closest team color
            dA = np.linalg.norm(colors - self.teamA_color, axis=1)
            dB = np.linalg.norm(colors - self.teamB_color, axis=1)
            labels = (dB < dA).astype(int)
            # Update team colors using running average
            newA = np.mean(colors[labels == 0], axis=0) if np.any(labels == 0) else None
            newB = np.mean(colors[labels == 1], axis=0) if np.any(labels == 1) else None
            if newA is not None:
                self.teamA_color = (
                    1 - self.update_rate
                ) * self.teamA_color + self.update_rate * newA
            if newB is not None:
                self.teamB_color = (
                    1 - self.update_rate
                ) * self.teamB_color + self.update_rate * newB

        # Assign team class labels to original track indices
        team_classes = [0] * len(track_classes)
        for ci, orig in enumerate(indices):
            team_classes[orig] = 1 if labels[ci] == 0 else 2

        # Convert cluster centers back to BGR for output
        centerA = np.zeros((1, 1, 3), dtype=np.uint8)
        centerB = np.zeros((1, 1, 3), dtype=np.uint8)
        # Clip into valid range
        cA = np.clip(self.teamA_color, 0, 255).astype(np.uint8)
        cB = np.clip(self.teamB_color, 0, 255).astype(np.uint8)
        if self.use_lab:
            centerA[0, 0] = cA
            centerB[0, 0] = cB
            outA = cv2.cvtColor(centerA, cv2.COLOR_LAB2BGR)[0, 0]
            outB = cv2.cvtColor(centerB, cv2.COLOR_LAB2BGR)[0, 0]
        else:
            centerA[0, 0] = cA
            centerB[0, 0] = cB
            outA = cv2.cvtColor(centerA, cv2.COLOR_HSV2BGR)[0, 0]
            outB = cv2.cvtColor(centerB, cv2.COLOR_HSV2BGR)[0, 0]
        # Return the final results
        return {
            "teamAColor": tuple(int(c) for c in outA),
            "teamBColor": tuple(int(c) for c in outB),
            "teamClasses": team_classes,
        }

