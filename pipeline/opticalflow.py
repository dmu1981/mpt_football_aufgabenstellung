import numpy as np
import cv2

class OpticalFlow:
    def __init__(self):
        self.name = "Optical Flow" # Do not change the name of the module as otherwise recording replay would break!
        self.old_frame_gray = None


    def start(self, data):
        # cap = cv2.VideoCapture(data["video"])
        # ret, old_frame = cap.read()
        # self.old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        pass

    def stop(self, data):
        # TODO: Implement shut down procedure of the module
        pass

    def step(self, data):
        # TODO: Implement processing of a single frame
        # The task of the optical flow module is to determine the overall avergae pixel shift between this and the previous image. 
        # You 

        # Note: You can access data["image"] to receive the current image
        # Return a dictionary with the motion vector between this and the last frame
        #
        # The "opticalFlow" signal must contain a 1x2 NumPy Array with the X and Y shift (delta values in pixels) of the image motion vector
        # Parameters for lucas kanade optical flow

        current_frame_gray = cv2.cvtColor(data["image"], cv2.COLOR_BGR2GRAY)


        if self.old_frame_gray is None:
            self.old_frame_gray = current_frame_gray
            return {
               "opticalFlow": np.zeros(2,)
            }
        else:
            current_frame_gray = cv2.cvtColor(data["image"], cv2.COLOR_BGR2GRAY)
            # x y flow for each pixel as a (540, 960, 2) numpy
            flow = cv2.calcOpticalFlowFarneback(self.old_frame_gray, current_frame_gray, flow=None,
                                            pyr_scale=0.5, levels=1, winsize=15,
                                            iterations=2,
                                            poly_n=5, poly_sigma=1.1, flags=0)
            self.old_frame_gray = current_frame_gray.copy()
            
            average_flow = flow.mean(axis=(0,1)) # falsches vorzeichen?


            return {
               "opticalFlow": average_flow*(-1)
            }