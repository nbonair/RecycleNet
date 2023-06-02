import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import vit_keras as vit
MODEL = "/Users/nbonair/Documents/SIT319 Deep Learning/Assignment 2/recycle_vit.tf"
class RecycleNet():
    def __init__(self,model_name_or_path = MODEL) -> None:
        self.model = load_model(MODEL)

        self.labels = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}
    def pre_processing(self,image):
        transformed_image = np.array(image) / 255.0
        transformed_image = np.array([transformed_image])
        return transformed_image
    def predict(self, image):
        image = self.pre_processing(image)
        logits = self.model.predict(image, verbose=0)
        predictions = np.argmax(logits, axis=1)
        return self.labels[predictions[0]]