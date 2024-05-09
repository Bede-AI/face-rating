import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import base64
from PIL import Image
import io

class ImageDataProcessor:
    def __init__(self, ratings_folder):
        self.ratings_folder = ratings_folder

    def load_data(self):
        images, labels = [], []
        for root, dirs, files in os.walk(self.ratings_folder):
            for file_name in files:
                if file_name.endswith('.json'):
                    json_file_path = os.path.join(root, file_name)
                    with open(json_file_path, 'r') as file:
                        data = json.load(file)
                        for item in data:
                            image_data = item['imageBase64'].split(",")[1].strip()
                            try:
                                image = self.decode_image(image_data)
                                rating = item['rating']
                                if rating == 0:
                                    rating = 10
                                images.append(image)
                                labels.append(rating - 1)  # make labels 0-based for to_categorical
                            except (base64.binascii.Error, Exception) as e:
                                print(f"Error processing image in file {json_file_path}, item with data: {image_data[:30]}... Error: {str(e)}")
                                continue
        images = np.array(images)  # Convert images list to numpy array
        labels = to_categorical(labels, num_classes=10)  # Convert labels to one-hot encoding
        return images, labels

    def decode_image(self, image_data):
        decoded_data = base64.b64decode(image_data, validate=True)
        image = Image.open(io.BytesIO(decoded_data))
        if image.format == 'WEBP':
            with io.BytesIO() as png_io:
                image.save(png_io, format="PNG")
                png_io.seek(0)
                image = tf.io.decode_png(png_io.read(), channels=3)
        else:
            image = tf.io.decode_image(decoded_data, channels=3, expand_animations=False)
        return image.numpy()

    def resize_images(self, images):
            resized_images = []
            for image in images:
                # Resize with crop or pad to 150x150
                resized_image = tf.image.resize_with_crop_or_pad(image, target_height=150, target_width=150)
                resized_images.append(resized_image)
            return np.array(resized_images)  # Convert resized images list to numpy array


class ImageClassifier:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(512, (3, 3), activation='relu'),
        layers.Conv2D(512, (3, 3), activation='relu'),
        layers.Conv2D(512, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(512, (3, 3), activation='relu'),
        layers.Conv2D(512, (3, 3), activation='relu'),
        layers.Conv2D(512, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dense(4096, activation='relu'),
        layers.Dense(1000, activation='relu'),
        layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, images, labels, epochs=10, validation_split=0.1):
        self.model.fit(images, labels, epochs=epochs, validation_split=validation_split)

    def save_model(self, path):
        self.model.save(path)

if __name__ == "__main__":
    RATINGS_FOLDER = 'ratings_female'

    data_processor = ImageDataProcessor(RATINGS_FOLDER)
    images, labels = data_processor.load_data()
    images = data_processor.resize_images(images)

    classifier = ImageClassifier()
    classifier.train(images, labels)

    classifier.save_model('path_to_save/female1.h5')
