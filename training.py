import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import base64

class ImageDataProcessor:
    def __init__(self, ratings_folder, test_ratings_folder):
        self.ratings_folder = ratings_folder
        self.test_ratings_folder = test_ratings_folder

    def list_training_files(self):
        return self._list_files(self.ratings_folder)

    def list_test_files(self):
        return self._list_files(self.test_ratings_folder)

    def _list_files(self, folder):
        return [os.path.join(root, file)
                for root, _, files in os.walk(folder)
                for file in files if file.endswith('.json')]

    def total_images(self, folder):
        total = 0
        for json_file_path in self._list_files(folder):
            with open(json_file_path, 'r') as file:
                data = json.load(file)
            total += len(data)
        return total

    def data_generator(self, json_files, batch_size=256):
        while True:
            for json_file_path in json_files:
                with open(json_file_path, 'r') as file:
                    data = json.load(file)
                images, labels = [], []
                for item in data:
                    image_data = item['imageBase64'].split(",")[1].strip()
                    try:
                        image = self.decode_image(image_data)
                        rating = item['rating']
                        if rating in [4, 5, 6]:
                            images.append(image)
                            labels.append(rating - 4)
                        if len(images) == batch_size:
                            yield self.prepare_batch(images, labels)
                            images, labels = [], []
                    except Exception as e:
                        continue
                if images:
                    yield self.prepare_batch(images, labels)

    def prepare_batch(self, images, labels):
        images = np.array(images)
        labels = to_categorical(labels, num_classes=3)
        return images, labels

    def decode_image(self, image_data):
        image_bytes = base64.b64decode(image_data)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.resize(image, [150, 150])
        return image / 255.0

class ImageClassifier:
    def __init__(self, load_checkpoint=None):
        if load_checkpoint:
            self.model = load_model(load_checkpoint)
        else:
            self.model = self.build_model()

    def build_model(self):
        model = models.Sequential([
            layers.Input(shape=(150, 150, 3)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, generator, test_generator, epochs=250, steps_per_epoch=10, steps_for_test=10):
        checkpoint_path = 'path_to_save/female1_augm_456n_epoch_{epoch:02d}.keras'
        checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, save_freq=5*steps_per_epoch)
        test_callback = TestEvaluation(test_generator, steps_for_test)
        self.model.fit(generator, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback, test_callback])

class TestEvaluation(tf.keras.callbacks.Callback):
    def __init__(self, test_generator, steps):
        super().__init__()
        self.test_generator = test_generator
        self.steps = steps

    def on_epoch_end(self, epoch, logs=None):
        test_loss, test_accuracy = self.model.evaluate(self.test_generator, steps=self.steps, verbose=0)
        logs['test_loss'] = test_loss
        logs['test_accuracy'] = test_accuracy
        print(f" - test_loss: {test_loss:.4f} - test_accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    ratings_folder = 'ratings_female_augm_456'
    test_ratings_folder = 'ratings_female_augm_456_test'
    data_processor = ImageDataProcessor(ratings_folder, test_ratings_folder)
    
    total_images_train = data_processor.total_images(ratings_folder)
    batch_size = 512
    steps_per_epoch = (total_images_train + batch_size - 1) // batch_size
    
    generator = data_processor.data_generator(data_processor.list_training_files(), batch_size)
    test_generator = data_processor.data_generator(data_processor.list_test_files(), batch_size)
    steps_for_test = (data_processor.total_images(test_ratings_folder) + batch_size - 1) // batch_size

    load_checkpoint_path = 'path_to_save/female1_augm_456n_epoch_30.keras'
    classifier = ImageClassifier(load_checkpoint=load_checkpoint_path)
    classifier.train(generator, test_generator, epochs=250, steps_per_epoch=steps_per_epoch, steps_for_test=steps_for_test)
