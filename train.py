
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Data Augmentation
data = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True
)

# Training dataset
train_data = data.flow_from_directory(
    "flowers",
    target_size=(150,150),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

# Validation dataset
val_data = data.flow_from_directory(
    "flowers",
    target_size=(150,150),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# CNN Model
model = models.Sequential([
    layers.Input(shape=(150,150,3)),
    
    layers.Conv2D(32,(3,3),activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128,(3,3),activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dense(train_data.num_classes,activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# Save model
model.save("flower_model.h5")

print("Training completed and model saved!")
