# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 18:48:46 2026

@author: k4927
"""

# Fold-1(raw) dataset load
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="MY_API_KEY")
project = rf.workspace("yolo-v8-tutorial").project("fold1_raw")
version = project.version(1)
dataset = version.download("yolov8")

# Fold-2(raw) dataset load
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="MY_API_KEY")
project = rf.workspace("yolo-v8-tutorial").project("fold2_raw")
version = project.version(2)
dataset = version.download("yolov8")

# Fold-3(raw) dataset load
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="MY_API_KEY")
project = rf.workspace("yolo-v8-tutorial").project("fold3_raw")
version = project.version(1)
dataset = version.download("yolov8")

# Fold-4(raw) dataset load
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="MY_API_KEY")
project = rf.workspace("yolo-v8-tutorial").project("fold4_raw")
version = project.version(2)
dataset = version.download("yolov8")

# Fold-5(raw) dataset load
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="MY_API_KEY")
project = rf.workspace("yolo-v8-tutorial").project("fold5_raw")
version = project.version(1)
dataset = version.download("yolov8")

!pip install ultralytics

from IPython import display
display.clear_output()
import ultralytics
ultralytics.checks()
from ultralytics import YOLO
from IPython.display import display, Image

# Data train(Model-C fruped k-fold cross validation [fold 1])
%cd {HOME}
!yolo task=detect mode=train model=/content/yolov8x.pt data=/content/Fold1_raw-1/data.yaml epochs=300 imgsz=640 plots=True patience=50
%cd {HOME}
!yolo task=detect mode=val model=/content/runs/detect/train/weights/best.pt data=/content/Fold1_raw-1/data.yaml

# Data train(Model-C fruped k-fold cross validation [fold 2])
%cd {HOME}
!yolo task=detect mode=train model=/content/yolov8x.pt data=/content/Fold2_raw-2/data.yaml epochs=300 imgsz=640 plots=True patience=50
%cd {HOME}
!yolo task=detect mode=val model=/content/runs/detect/train-2/weights/best.pt data=/content/Fold2_raw-2/data.yaml
# Data train(Model-C fruped k-fold cross validation [fold 3])
%cd {HOME}
!yolo task=detect mode=train model=/content/yolov8x.pt data=/content/Fold3_raw-1/data.yaml epochs=300 imgsz=640 plots=True patience=50
%cd {HOME}
!yolo task=detect mode=val model=/content/runs/detect/train-3/weights/best.pt data=/content/Fold3_raw-1/data.yaml
# Data train(Model-C fruped k-fold cross validation [fold 4])
%cd {HOME}
!yolo task=detect mode=train model=/content/yolov8x.pt data=/content/Fold4_raw-2/data.yaml epochs=300 imgsz=640 plots=True patience=50
%cd {HOME}
!yolo task=detect mode=val model=/content/runs/detect/train-4/weights/best.pt data=/content/Fold4_raw-2/data.yaml
# Data train(Model-C fruped k-fold cross validation [fold 5])
%cd {HOME}
!yolo task=detect mode=train model=/content/yolov8x.pt data=/content/Fold5_raw-1/data.yaml epochs=300 imgsz=640 plots=True patience=50
%cd {HOME}
!yolo task=detect mode=val model=/content/runs/detect/train-5/weights/best.pt data=/content/Fold5_raw-1/data.yaml