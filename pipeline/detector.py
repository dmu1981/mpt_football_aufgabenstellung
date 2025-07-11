from ultralytics import YOLO
import numpy as np    

class Detector:
    def __init__(self):
        self.name = "Detector" # Do not change the name of the module as otherwise recording replay would break!

        self.model = None


    def start(self, data):
        model_path = data.get("yolo_model_path")
        if model_path is None:
            raise ValueError("YOLO model path not specified in data dictionary.")
        self.model = YOLO(model_path)

    def stop(self, data):
        pass

    
    def step(self, data):
        image = data["image"]  # get current image 

        # use YOLO on the image
        results = self.model(image)[0]  #self.model(image) returns a list, we take the first one (current)

        bboxes = []
        classes = []

        for box in results.boxes: #iterate through detected objects 
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # extract Bounding Box 
            cls = int(box.cls.cpu().numpy())            # assign objekt to a class

            #calculate center(as required), width and hight for each object so that .display can be able to draw a box around the objects 
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1

            bboxes.append([x, y, w, h]) #add to the list in a bounding box format (as required)
            classes.append(cls)         #add to object classification the list 

        return { #return in the right format
            "detections": np.array(bboxes, dtype=np.float32),
            "classes": np.array(classes, dtype=np.int32)
         }