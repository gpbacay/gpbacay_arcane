import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the trained model
try:
    MODEL_FILEPATH = 'Models/dstsmgser_model.keras'
    model = tf.keras.models.load_model(MODEL_FILEPATH)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: The {MODEL_FILEPATH} model file was not found.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the model: {str(e)}")
    exit()

def preprocess_image(image_path):
    try:
        # Load and preprocess an image using OpenCV
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Error: Image at path '{image_path}' could not be loaded.")
            return None
        image = cv2.resize(image, (28, 28))
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=[0, -1])  # Shape to (1, 28, 28, 1)
        return image
    except Exception as e:
        print(f"An error occurred during image preprocessing: {str(e)}")
        return None

def predict(image_path):
    image = preprocess_image(image_path)
    if image is None:
        print("Skipping prediction due to preprocessing error.")
        return None
    
    try:
        predictions = model.predict(image)
        # The first output is the classification result
        classification_output = predictions[0]
        predicted_digit = np.argmax(classification_output)
        confidence = np.max(classification_output)
        
        print(f"Prediction shape: {[p.shape for p in predictions]}")
        print(f"Prediction types: {[type(p) for p in predictions]}")
        
        return predicted_digit, confidence
    except Exception as e:
        print(f"An error occurred during prediction: {str(e)}")
        return None

def display_image_and_prediction(image_path, predicted_digit, confidence):
    # Display the image along with its predicted digit and confidence
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Image at path '{image_path}' could not be loaded.")
        return
    
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.title(f'Predicted digit: {predicted_digit}\nConfidence: {confidence:.2f}')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Example usage
    test_image_path = 'Test/img_4.png'  # Replace with your image path
    result = predict(test_image_path)
    if result is not None:
        predicted_digit, confidence = result
        print(f'Predicted digit: {predicted_digit}')
        print(f'Confidence: {confidence:.2f}')
        display_image_and_prediction(test_image_path, predicted_digit, confidence)
    else:
        print("Prediction failed.")


# python test_model.py
# Test Accuracy: 0.9695, Loss: 0.1832
# Retrained Model Test Accuracy: 0.9736, Loss: 0.1852
# 40 epochs: Test Accuracy: 0.9808, Loss: 0.1511