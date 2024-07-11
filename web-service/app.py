from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import logging
import sys

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# Load model
model = tf.keras.models.load_model('../covid19_model.h5')

# Load class names
with open('../classes.txt', 'r', encoding='utf-8') as f:
    class_names = f.read().splitlines()

def preprocess_image(image):
    try:
        image = image.resize((128, 128))  # Resize to match your model's input shape
        image = np.array(image)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = image / 255.0  # Normalize if necessary
        return image
    except Exception as e:
        app.logger.error(f"Error during image preprocessing: {e}")
        raise

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'url' not in data:
        return jsonify({'error': 'No URL provided'}), 400

    image_url = data['url']

    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Ensure we notice bad responses
        image = Image.open(BytesIO(response.content))
    except requests.RequestException as e:
        app.logger.error(f"Error fetching image: {e}")
        return jsonify({'error': 'Error fetching image'}), 400
    except Exception as e:
        app.logger.error(f"Error processing image: {e}")
        return jsonify({'error': 'Error processing image'}), 400

    try:
        app.logger.info("Preprocessing image...")
        image = preprocess_image(image)
        app.logger.info(f"Image preprocessed successfully. Shape: {image.shape}, Type: {image.dtype}")

        app.logger.info("Making prediction...")
        predictions = model.predict(image, vebrose=0)
        app.logger.info(f"Prediction made successfully. Predictions: {predictions}")

        predicted_class_index = np.argmax(predictions)
        app.logger.info(f"Predicted class index: {predicted_class_index}")

        predicted_class_name = class_names[predicted_class_index]
        app.logger.info(f"Predicted class name: {predicted_class_name}")

        probability = float(np.max(predictions))
        app.logger.info(f"Probability: {probability}")

    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e).encode('utf-8', 'ignore').decode('utf-8')}")
        return jsonify({'error loc 2': 'Error during prediction'}), 500

    return jsonify({'class': predicted_class_name, 'probability': probability})

if __name__ == '__main__':
    app.run(debug=True)
