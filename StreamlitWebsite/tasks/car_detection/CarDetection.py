from . import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import io

class CarDetection():
    def __init__(self):
        self.model = YOLO()
    
    def predict(self, image_path):
        return self.model.predict(image_path)