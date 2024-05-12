import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
import io
from PIL import Image
import base64

# Path to your original data
data_dir = './ratings_female'
output_dir = './ratings_female_augm_456'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Data augmentation generator
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Function to determine if a file is a JSON file
def is_json_file(filename):
    return filename.endswith('.json')

# Count images per rating
rating_counts = {}

def process_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if is_json_file(file):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    for item in data:
                        rating = item['rating']
                        rating_counts[rating] = rating_counts.get(rating, 0) + 1

process_directory(data_dir)
print("Initial data class counts:", rating_counts)

# Calculate the target count for augmentation based on the highest count among ratings 4, 5, and 6
target_count = max(rating_counts[r] for r in [4, 5, 6]) * 4

def augment_images(image_data_base64, rating, num_augmented_images):
    image_data = base64.b64decode(image_data_base64.split(',')[1])
    image = Image.open(io.BytesIO(image_data))
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    x = img_to_array(image)  # Convert image to numpy array
    x = x.reshape((1,) + x.shape)  # Reshape to (1, height, width, channels)

    augmented_data = []

    for _ in range(num_augmented_images):
        batch = next(datagen.flow(x, batch_size=1))
        image_array = batch[0]
        img = Image.fromarray(image_array.astype('uint8'), 'RGB')
        img = img.convert('RGBA')  # Convert back to RGBA

        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        augmented_data.append({"imageBase64": f"data:image/png;base64,{img_str}", "rating": rating})

    return augmented_data

# Augment images based on calculated target counts
augmented_images = []

def augment_directory(directory):
    global augmented_images
    file_index = 1
    total_augmented_count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if is_json_file(file):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    for item in data:
                        rating = item['rating']
                        if rating in [4, 5, 6]:
                            current_count = rating_counts[rating]
                            additional_needed = max(0, target_count - current_count)
                            augmentation_factor = additional_needed // current_count + 1
                            new_data = augment_images(item['imageBase64'], rating, augmentation_factor)
                            augmented_images.extend(new_data)
                            total_augmented_count += len(new_data)

                            # Save in chunks of 1000
                            if len(augmented_images) >= 1000:
                                augmented_file_path = os.path.join(output_dir, f'augmented_data_{file_index}.json')
                                with open(augmented_file_path, 'w') as f:
                                    json.dump(augmented_images, f)
                                print(f"Saved {len(augmented_images)} images to {augmented_file_path}")
                                file_index += 1
                                augmented_images = []

    # Save any remaining data
    if augmented_images:
        augmented_file_path = os.path.join(output_dir, f'augmented_data_{file_index}.json')
        with open(augmented_file_path, 'w') as f:
            json.dump(augmented_images, f)
        print(f"Saved remaining {len(augmented_images)} images to {augmented_file_path}")

    print("Total augmented images:", total_augmented_count)
    print("Final augmented data class counts:", rating_counts)

augment_directory(data_dir)
