import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

print("Loading model...")

model = tf.keras.models.load_model("flower_model.h5")

class_names = ['daisy','dandelion','rose','sunflower','tulip']

img_path = "test1.jpg"

print("Loading image...")

img = image.load_img(img_path, target_size=(150,150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

print("Predicting...")

prediction = model.predict(img_array)

index = np.argmax(prediction)
confidence = prediction[0][index]

flower = class_names[index]

print("🌸 Predicted Flower:", flower)
print("📊 Confidence:", round(confidence*100,2), "%")

# -------------------------------
# 🧠 SIMPLE REASONING (TEXT)
# -------------------------------

if flower == "sunflower":
    reason = "because the image has bright yellow petals and a large circular center"

elif flower == "rose":
    reason = "because of layered petals and typical rose shape"

elif flower == "tulip":
    reason = "because of smooth petals and cup-like shape"

elif flower == "daisy":
    reason = "because of white petals with a yellow center"

elif flower == "dandelion":
    reason = "because of small thin yellow petals forming a round shape"

else:
    reason = "based on learned patterns of color, shape, and texture"

print("🧠 Reason:", reason)
