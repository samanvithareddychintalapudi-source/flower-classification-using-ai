import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

print("Loading model...")

# Load trained model
model = tf.keras.models.load_model("flower_model.h5")

# Flower classes
class_names = ['daisy','dandelion','rose','sunflower','tulip']

print("Loading image...")

# Image path
img_path = "test.jpg"

img = image.load_img(img_path, target_size=(150,150))

# Convert image to array
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

print("Predicting...")

# Prediction
prediction = model.predict(img_array)

index = np.argmax(prediction)

print("Predicted Flower:", class_names[index])
