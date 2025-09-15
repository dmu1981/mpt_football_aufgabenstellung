import cv2 as cv
import numpy as np

class OpticalFlow:
    """
    Optischer Fluss , berechnet die durchschnittliche Pixelverschiebung
    zwischen aufeinanderfolgenden Bildern.
    """

    def __init__(self, scale_factor=0.5, mirror=True, use_gpu=False):
        """
        :param scale_factor: Faktor zur Verkleinerung der Bilder für mehr Geschwindigkeit.
        :param mirror: Ob das Bild horizontal gespiegelt werden soll.
        :param use_gpu: Ob die CUDA-Version verwendet werden soll, falls verfügbar.
        """
        self.name = "Optical Flow"
        self.prev_gray = None
        self.scale_factor = scale_factor
        self.mirror = mirror
        self.use_gpu = use_gpu  

        if self.use_gpu:
            try:
                self.gpu_flow = cv.cuda_FarnebackOpticalFlow.create(
                    numLevels=5,
                    pyrScale=0.5,
                    fastPyramids=False,
                    winSize=15,
                    numIters=3,
                    polyN=5,
                    polySigma=1.2,
                    flags=0,
                )
            except AttributeError:
                self.use_gpu = False
        
    def start(self, data):
        self.prev_gray = None

    def stop(self, data):
        self.prev_gray = None
    
    def step(self, data):
        """
        Verarbeitet ein Bild und gibt den durchschnittlichen Bewegungsvektor zurück.
        :param data: Dictionary mit dem Schlüssel 'image' für das BGR-Bild.
        :return: dict {'opticalFlow': np.ndarray der Form (2,) oder None}
        """
        frame = data.get("image")
        if frame is None:
            return {"opticalFlow": None}

        if self.scale_factor != 1.0:
            frame = cv.resize(frame, (0, 0),
                              fx=self.scale_factor,
                              fy=self.scale_factor,
                              interpolation=cv.INTER_LINEAR)

        if self.mirror:
            frame = cv.flip(frame, 1)

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return {"opticalFlow": np.zeros((2,), dtype=np.float32)}

        if self.use_gpu:
            try:
                d_prev = cv.cuda_GpuMat(); d_prev.upload(self.prev_gray)
                d_curr = cv.cuda_GpuMat(); d_curr.upload(gray)
                d_flow = self.gpu_flow.calc(d_prev, d_curr, None)
                flow = d_flow.download()
            except Exception:
                self.use_gpu = False
                flow = cv.calcOpticalFlowFarneback(
                    self.prev_gray, gray,
                    None, 0.5, 5, 15, 3, 5, 1.2, 0
                )
        else:
            flow = cv.calcOpticalFlowFarneback(
                self.prev_gray, gray,
                None, 0.5, 5, 15, 3, 5, 1.2, 0
            )
        
        avg_flow = np.mean(flow, axis=(0, 1)) / self.scale_factor
        self.prev_gray = gray
        return {"opticalFlow": avg_flow}