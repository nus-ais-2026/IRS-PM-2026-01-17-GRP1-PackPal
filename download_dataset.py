# download_dataset.py

#!pip install roboflow

from roboflow import Roboflow
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.environ["ROBOFLOW_API_KEY"]

rf = Roboflow(api_key=api_key)
project = rf.workspace("df2").project("deepfashion2-m-10k")
version = project.version(2)
dataset = version.download("yolov8-obb")
                