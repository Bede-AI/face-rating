import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import base64
from PIL import Image
import io

RATINGS_FOLDER = 'ratings_male'

def preprocess_data(folder_path):
    images, labels = [], []
    json_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.json')]
    for file_name in json_files:
        with open(file_name, 'r') as file:
            data = json.load(file)
            for item in data:
                image_data = item['imageBase64'].split(",")[1].strip()
                try:
                    decoded_data = base64.b64decode(image_data, validate=True)
                    image = Image.open(io.BytesIO(decoded_data))
                    if image.format == 'WEBP':
                        with io.BytesIO() as png_io:
                            image.save(png_io, format="PNG")
                            png_io.seek(0)
                            image = tf.io.decode_png(png_io.read(), channels=3)
                    else:
                        image = tf.io.decode_image(decoded_data, channels=3, expand_animations=False)

                    image = tf.image.resize(image, (224, 224))
                    rating = item['rating']
                    if rating == 0:
                        rating = 10
                    images.append(image.numpy())
                    labels.append(rating - 1)  # make labels 0-based for to_categorical
                except (base64.binascii.Error, Exception) as e:
                    print(f"Error processing image in file {file_name}, item with data: {image_data[:30]}... Error: {str(e)}")
                    continue
    return np.array(images), to_categorical(labels, num_classes=10)

# Building the model
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load and preprocess the data
images, labels = preprocess_data(RATINGS_FOLDER)

# Fit the model
model.fit(images, labels, epochs=10, validation_split=0.1)

# Save the model
model.save('path_to_save/male1.h5')  # Save as HDF5 file
