import os
import json
import random
import matplotlib.pyplot as plt
import base64
from PIL import Image
import io

# Define the folder containing the JSON files
JSON_FOLDER = 'ratings_female'

# Function to read and parse JSON files
def read_json_files(folder_path):
    data = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file), 'r') as f:
                    data.extend(json.load(f))
    return data

# Function to search for images by rating category
def search_by_rating(data, rating):
    return [item['imageBase64'] for item in data if item['rating'] == rating]

# Function to count ratings in each category
def count_ratings(data):
    rating_counts = {}
    for item in data:
        rating = item['rating']
        if rating not in rating_counts:
            rating_counts[rating] = 0
        rating_counts[rating] += 1
    return rating_counts

# Read and parse JSON files
json_data = read_json_files(JSON_FOLDER)

# Count and display the number of ratings in each category
rating_counts = count_ratings(json_data)
print("\nNumber of images in each rating category:")
for rating, count in sorted(rating_counts.items()):
    print(f"Rating {rating}: {count} images")

# Allow user to search for images by rating category
while True:
    search_rating = input("\nEnter the rating category (0-9) to search for images (or 'q' to quit): ")
    if search_rating.lower() == 'q':
        break
    try:
        search_rating = int(search_rating)
        if 0 <= search_rating <= 9:
            images = search_by_rating(json_data, search_rating)
            if images:
                img_data = random.choice(images)
                decoded_data = base64.b64decode(img_data.split(",")[1].strip())
                img = Image.open(io.BytesIO(decoded_data))
                plt.imshow(img)
                plt.title(f"Rating: {search_rating}")
                plt.axis('off')
                plt.show()
            else:
                print(f"No images found for rating {search_rating}.")
        else:
            print("Invalid rating category. Please enter a number between 0 and 9.")
    except ValueError:
        print("Invalid input. Please enter a number between 0 and 9 or 'q' to quit.")
