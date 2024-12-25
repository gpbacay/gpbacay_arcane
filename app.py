from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import cv2
import logging
import socket
from contextlib import closing

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_free_port(start_port=5001, max_port=5999):
    """Find a free port to use for the server."""
    for port in range(start_port, max_port):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            try:
                sock.bind(('', port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free ports found in range")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model
try:
    MODEL_FILEPATH = 'Models/dstsmgser_retrained_model.keras'
    model = tf.keras.models.load_model(MODEL_FILEPATH)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

def preprocess_image_data(image_data):
    """
    Preprocess the image data to match the model's expected input format.
    """
    try:
        # Reshape the flat array to 28x28
        image = np.array(image_data, dtype=np.float32).reshape(28, 28)
        
        # Add batch and channel dimensions
        image = np.expand_dims(image, axis=[0, -1])  # Shape: (1, 28, 28, 1)
        
        return image
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to receive image data and return predictions.
    """
    try:
        # Get the image data from the request
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        image_data = data['image']
        
        # Validate input
        if len(image_data) != 784:  # 28x28 = 784
            return jsonify({'error': 'Invalid image dimensions'}), 400
        
        # Preprocess the image
        processed_image = preprocess_image_data(image_data)
        
        # Make prediction
        predictions = model.predict(processed_image)
        
        # Get the classification output (first output of the model)
        classification_output = predictions[0]
        predicted_digit = int(np.argmax(classification_output[0]))
        confidence = float(np.max(classification_output[0]))
        
        # Return the prediction
        return jsonify({
            'digit': predicted_digit,
            'confidence': confidence
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Simple health check endpoint to verify the API is running.
    """
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

def start_server():
    """
    Start the Flask server with the appropriate configurations.
    """
    try:
        # Find an available port
        port = find_free_port()
        logger.info(f"Starting server on port {port}")
        
        # Print instructions for updating the frontend
        print("\n" + "="*80)
        print(f"Server starting on port {port}")
        print(f"Please update the fetch URL in your HTML file to: http://localhost:{port}/predict")
        print("="*80 + "\n")
        
        # Start the Flask app
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        logger.error(f"Server startup failed: {str(e)}")
        raise

if __name__ == "__main__":
    start_server()

# python app.py