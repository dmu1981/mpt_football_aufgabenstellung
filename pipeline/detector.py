from ultralytics import YOLO
import torch


class Detector:
    def __init__(self):
        self.name = "Detector"  # Do not change the name of the module, otherwise recording and replay will break!
        self.model = YOLO("yolov8n-football.pt")  # loads YOLOv8 model

    def start(self, data):
        '''
        Attempts to run YOLO on GPU for better performance.
        If GPU is unavailable, falls back to CPU (much slower but still functional).
        Enables half precision on CUDA devices to reduce memory usage.
        '''

        print(f"{self.name}: Detector module started.")

        if torch.cuda.is_available():
            self.model.to('cuda')           # moves model to GPU
            self.model.half()               # enables FP16 inference (faster/less memory)
            self.device = 'cuda'
        else:
            print(f"{self.name}: CUDA not available. Falling back to CPU.")
            self.device = 'cpu'

        print(f"{self.name}: Model ready on {self.model.device}.")

    def stop(self, data):
        # placeholder
        print(f"{self.name}: Detector module stopped.")

    def step(self, data):
        # TODO: Implement processing of a single frame
        # The task of the detector is to detect the ball, the goal keepers, the players and the referees if visible.
        # A bounding box needs to be defined for each detected object including the objects center position (X,Y) and its width and height (W, H) 
        # You can return an arbitrary number of objects 
        
        # Note: You can access data["image"] to receive the current image
        # Return a dictionary with detections and classes
        #
        # Detections must be a Nx4 NumPy Tensor, one 4-dimensional vector per detection
        # The detection vector itself is encoded as (X, Y, W, H), so X and Y coordinate first, then width and height of each detection box.
        # X and Y coordinates are the center point of the object, so the bounding box is drawn from (X - W/2, Y - H/2) to (X + W/2, Y + H/2)
        #
        # Classes must be Nx1 NumPy Tensor, one scalar entryx per detection
        # For each corresponding detection, the following mapping must be used
        #   0: Ball
        #   1: GoalKeeper
        #   2: Player
        #   3: Referee

        return {
            "detections": None,
            "classes": None
        }
