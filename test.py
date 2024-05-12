import cv2
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import json
import io

# Load your trained model
model = tf.keras.models.load_model('path_to_save/female1_augm_456_epoch_35.keras')

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_frame(frame):
    """Converts color from BGR to RGB and resize to (150, 150)."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    image = image.resize((150, 150))
    image_array = np.array(image)
    return np.expand_dims(image_array, axis=0)

def decode_image(image_base64):
    """Decodes a base64 image string into a numpy array."""
    image_data = base64.b64decode(image_base64.split(',')[1])
    image = Image.open(io.BytesIO(image_data))
    return np.array(image)

def predict_image(image):
    """Predicts class, confidence, and probabilities for an image."""
    processed_image = preprocess_frame(image)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    probabilities = predictions[0].tolist()
    return predicted_class, confidence, probabilities

def draw_predictions(frame, predictions, x, y):
    """Draws predictions on the frame."""
    predicted_class, confidence, probabilities = predictions
    print(predictions)
    cv2.putText(frame, f'Class: {predicted_class}, Conf: {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    for i, prob in enumerate(probabilities, start=1):
        cv2.putText(frame, f'Class {i}: {prob:.2f}', (x, y - 10 - 20 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

def display_from_json(file_path):
    """Displays images from a JSON file and makes predictions."""
    with open(file_path, 'r') as f:
        images_data = json.load(f)
    cv2.namedWindow('Test Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Test Image', 500, 500)
    for image_info in images_data:
        image = decode_image(image_info['imageBase64'])
        predictions = predict_image(image)
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Resize frame for better display
        frame = cv2.resize(frame, (500, 500))
        draw_predictions(frame, predictions, 10, 30)
        cv2.imshow('Test Image', frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def main(use_camera=True):
    """Main function to handle camera input or image tests."""
    if use_camera:
        cap = cv2.VideoCapture(0)
        cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Camera', 500, 500)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                frame = process_frame(frame)
                cv2.imshow('Camera', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
    else:
        display_from_json("test_images/ratings.json")

if __name__ == '__main__':
    main(use_camera=False)
