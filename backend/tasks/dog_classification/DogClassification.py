import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io

# Define preprocessing function
def preprocess_image(file):
    img = image.load_img(io.BytesIO(file.stream.read()), target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    return img_array

# Define prediction function
def predict(image_data):
    # Load the model
    model = tf.keras.models.load_model("models/finetuned_mobilenetv2_dog_classifier_model.h5")
    prediction = model.predict(image_data)
    if prediction > 0.5:
        result = 'This is a Dog'
    else:
        result = 'This is a Cat'
    return result