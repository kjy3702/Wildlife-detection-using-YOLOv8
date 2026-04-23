# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 22:13:41 2026

@author: k4927
"""

# Google Colab GPU Setting
!nvidia-smi

# GoogleDrive Mount
from google.colab import drive
drive.mount('/content/drive')

# Ask YOLOv8 Function 및 Tools 

!pip install ultralytics
!pip install roboflow

from IPython import display
display.clear_output()
import ultralytics
ultralytics.checks()
from ultralytics import YOLO
from IPython.display import display, Image

# bring my dataset
import shutil
shutil.copytree("/content/drive/MyDrive/WFdata_Boar,Deer(2208_2212)/Total_Test_120", "/content/Test_dataset_120") # Test dataset(Roedeer : 20 imgz | Wildboar : 20 imgz | Waterdeer : 20 imgz | Badger : 20 imgz | Racoondog : 20 imgz | Background : 20 imgz)


# Bounding Box Dataset (Original(No augmentation), Auto-Orient)
from roboflow import Roboflow
rf = Roboflow(api_key="MY_API_KEY")
project = rf.workspace("yolo-v8-tutorial").project("boundingbox_species5_200")
dataset = project.version(2).download("yolov8")

# Bounding Box Dataset (Horizonal Flip(Data augmentation), Auto-Orient)
from roboflow import Roboflow
rf = Roboflow(api_key="MY_API_KEY")
project = rf.workspace("yolo-v8-tutorial").project("boundingbox_species5_200")
dataset = project.version(3).download("yolov8")

# Polygon Dataset (Original(No augmentation), Auto-Orient)

from roboflow import Roboflow
rf = Roboflow(api_key="MY_API_KEY")
project = rf.workspace("yolo-v8-tutorial").project("speicies5_Merge")
dataset = project.version(1).download("yolov8")

# Polygon Dataset (Horizonal Flip(Data augmentation), Auto-Orient)
from roboflow import Roboflow
rf = Roboflow(api_key="MY_API_KEY")
project = rf.workspace("yolo-v8-tutorial").project("speicies5_Merge")
dataset = project.version(10).download("yolov8")

# Dataset move

import os
os.mkdir("datasets")
!mv /content/boundingbox_species5_200-2 /content/datasets
!mv /content/boundingbox_species5_200-3 /content/datasets
!mv /content/speicies5_Merge-1 /content/datasets
!mv /content/speicies5_Merge-10 /content/datasets

# Data Train
# Default setting Bounding box - Original
%cd {HOME}
!yolo task=detect mode=train model=/content/yolov8x.pt data=/content/datasets/boundingbox_species5_200-2/data.yaml epochs=300 imgsz=640 plots=True
# Default setting Bounding box - Flip
%cd {HOME}
!yolo task=detect mode=train model=/content/yolov8x.pt data=/content/datasets/boundingbox_species5_200-3/data.yaml epochs=300 imgsz=640 plots=True
# Default setting Polygon - Original
%cd {HOME}
!yolo task=detect mode=train model=/content/yolov8x.pt data=/content/datasets/speicies5_Merge-1/data.yaml epochs=300 imgsz=640 plots=True
# Default setting Polygon - Flip
%cd {HOME}
!yolo task=detect mode=train model=/content/yolov8x.pt data=/content/datasets/speicies5_Merge-10/data.yaml epochs=300 imgsz=640 plots=True

# Data Validation
# Bounding box Original(Model-A)
%cd {HOME}
!yolo task=detect mode=val model=/content/Validation_factor/Localization/Localization_v8x_Boundingbox/weights/best.pt data=/content/datasets/Boundingbox_species5_200-2/data.yaml
# Bounding box Extension(Model-B)
%cd {HOME}
!yolo task=detect mode=val model=/content/Validation_factor/Localization/Flip_Boundingbox_v8x/weights/best.pt data=/content/datasets/Boundingbox_species5_200-3/data.yaml
# Polygon Original(Model-C)
%cd {HOME}
!yolo task=detect mode=val model=/content/Validation_factor/Localization/Localization_v8x_Polygon/weights/best.pt data=/content/datasets/speicies5_Merge-1/data.yaml
# Polygon Extension(Model-D)
%cd {HOME}
!yolo task=detect mode=val model=/content/Validation_factor/Localization/Flip_Polygon_v8x/weights/best.pt data=/content/datasets/speicies5_Merge-10/data.yaml

# Model-C test
# Data test (Polygon - Original)
import pandas as pd
%cd {HOME}
Eachs = []
Each = !yolo task=detect mode=predict model=/content/runs/detect/train3/weights/best.pt conf=0.571 source=/content/Test_dataset_120 save=True # iou = 0.7

# Test results
for i in range(len(Each)):
  Eachs.append(Each[i])

df=pd.DataFrame(Eachs)
df.to_csv("/content/drive/MyDrive/WFresult_Boar,Deer(2208_2212)/Results/Poly_Ori.csv")