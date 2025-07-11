from engine import Engine, npTensor, rgbImage, lst
from modules import VideoReader, Display,  recordReplayMultiplex, RRPlexMode
from pipeline.detector import Detector
from pipeline.opticalflow import OpticalFlow
from pipeline.tracker import Tracker
from pipeline.shirtClassifier import ShirtClassifier
import signal
import sys

def signal_handler(sig, frame):
    print("\n⇨ Abbruch durch Benutzer, beende Programm.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
        
recordMode = RRPlexMode.BYPASS

shape = (960, 540)
engine = Engine(
  modules=[
    VideoReader(targetSize=shape),
    recordReplayMultiplex(Detector(), RRPlexMode.REPLAY),
    recordReplayMultiplex(OpticalFlow(), RRPlexMode.REPLAY),
    recordReplayMultiplex(Tracker(), RRPlexMode.REPLAY),
    recordReplayMultiplex(ShirtClassifier(), RRPlexMode.BYPASS),
    Display(historyBufferSize=1000)
    ],
  signals={
    "image": rgbImage(shape[0], shape[1]),
    "opticalFlow": npTensor((2,)),
    "detections": npTensor((-1, 4)),
    "classes": npTensor((-1,)),
    "tracks": npTensor((-1, 4)),
    "trackVelocities": npTensor((-1, 2)),
    "trackAge": lst(),
    "trackClasses": lst(),
    "trackIds": lst(),
    "teamClasses": lst(),
    "terminate": bool,
    "stopped": bool,
    "testout": int
  })

data = { "video": 'videos/1.mp4',
        "yolo_model_path": "modules/yolov8m-football.pt" }

signals = engine.run(data)
