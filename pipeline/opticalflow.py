import cv2 as cv
import numpy as np

class OpticalFlow:
    def __init__(self, scale_factor=0.5, mirror=True, use_gpu=False):
        """
        :param scale_factor: Skalierungsfaktor für schnellere Verarbeitung (0.5 für 50% der Originalgröße)
        :param mirror: True = Bild wird horizontal gespiegelt
        :param use_gpu: True = Verwendung von CUDA (nur bei NVIDIA-GPU möglich)
        """

        self.name = "Optical Flow"
        self.prev_gray = None
        self.scale_factor = scale_factor
        self.mirror = mirror
        self.use_gpu = use_gpu    

    def start(self, data):
        self.prev_gray = None

    def stop(self, data):
        self.prev_gray = None

    def step(self, data):
        # TODO: Implement processing of a single frame
        # The task of the optical flow module is to determine the overall avergae pixel shift between this and the previous image. 
        # You 
#
        # Note: You can access data["image"] to receive the current image
        # Return a dictionary with the motion vector between this and the last frame
     #   #
        # The "opticalFlow" signal must contain a 1x2 NumPy Array with the X and Y shift (delta values in pixels) of the image motion vector
        return {
           "opticalFlow": None
        }

