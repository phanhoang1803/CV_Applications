import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import io

class DogClassification:
    def __init__(self):
        # Load the model once when the class is instantiated
        print("Loading model...")
        self.model = tf.keras.models.load_model("models/finetuned_mobilenetv2_dog_classifier_model.h5")
        print("Loading model... Done!")
    
    # Define preprocessing function
    def preprocess_image(self, file):
        img = image.load_img(io.BytesIO(file), target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = tf.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)  # Ensure proper preprocessing
        return img_array

    # Define prediction function
    def predict(self, image_data):
        prediction = self.model.predict(image_data)
        if prediction > 0.5:
            result = 'This is a Dog'
        else:
            result = 'This is a Cat'
        return result
