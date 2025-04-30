from ultralytics import YOLO
import torch
import numpy as np


class Detector:
    def __init__(self):
        self.name = 'Detector'  # Do not change the name of the module, otherwise recording and replay will break!
        self.model = YOLO('yolov8n-football.pt')  # loads YOLOv8 model

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
        '''
        Runs YOLO detection on the current frame and returns bounding boxes and class IDs.
        Called once per frame.
        '''

        image = data['image']    # gets current frame

        results = self.model.predict(
            source=image,
            device=self.device,
            imgsz=864,           # high resolution improves ball detection
            conf=0.2,            # confidence threshold
            verbose=False,
            retina_masks=False   # not needed
        )[0]                     # only first result in the list needed

        detections = []
        classes = []

        VALID_CLASSES = {0, 1, 2, 3}  # specifies relevant classes in case model changes in future

        for box in results.boxes:
            # each box contains one detection result from YOLO (coordinates, class, confidence)

            cls_id = int(box.cls.item())  # predicted class id (0 = ball, 1 = goalkeeper, 2 = player, 3 = referee)

            if cls_id not in VALID_CLASSES:
                continue  # skip other classes, if there are any

            score = float(box.conf.item())  # confidence score

            # class specific thresholds
            if cls_id == 0:         # ball
                if score < 0.2:     # lower confidence, because detection is harder
                    continue
            else:                   # player, goalkeeper, referee
                if score < 0.4:
                    continue

            x_center, y_center, width, height = box.xywh[0].tolist()  # bounding box (center coordinates width, height)
            
            if cls_id == 0:     # shrinks bounding box of ball class by 30%
                width *= 0.7
                height *= 0.7

            detections.append([x_center, y_center, width, height])  # adds bounding box to list
            classes.append([cls_id])  # adds corresponding class id

        # converts valid detections to np arrays
        if detections:
            detections = np.array(detections).astype(np.float32)  # x_center, y_center, width, height
            classes = np.array(classes).astype(np.int32)          # class_id
        else:
            # returns empty arrays  if no detections are found
            detections = np.zeros((0, 4), dtype=np.float32)
            classes = np.zeros((0, 1), dtype=np.int32)

        return {
            "detections": detections,
            "classes": classes
        }

