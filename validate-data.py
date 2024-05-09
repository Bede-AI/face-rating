import os
import json
import base64
import sys

def validate_base64_data(folder_path):
    json_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.json')]
    for file_name in json_files:
        with open(file_name, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file {file_name}: {str(e)}")
                continue  # Skip this file and move to the next
            
            for item in data:
                image_data = item['imageBase64'].split(",")[1].strip()
                # Validate base64 using Python's base64 module
                try:
                    base64.b64decode(image_data, validate=True)
                except base64.binascii.Error as e:
                    print(f"Invalid base64 character found in file {file_name}: {str(e)}")
                    print(f"Problematic data: {image_data[:50]}")  # Print the first 50 characters of the problematic data
                    sys.exit(1)  # Stops the program if an invalid character is found

if __name__ == "__main__":
    folder_path = 'Ratings ~1000'  # Specify the folder where the JSON files are stored
    validate_base64_data(folder_path)
