from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import tensorflow as tf
import json
import logging
from gpbacay_arcane.layers import GSER, HebbianHomeostaticNeuroplasticity, DenseGSER, RelationalConceptModeling, RelationalGraphAttentionReasoning
from django.conf import settings
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define custom objects for model loading
custom_objects = {
    'GSER': GSER,
    'HebbianHomeostaticNeuroplasticity': HebbianHomeostaticNeuroplasticity,
    'DenseGSER': DenseGSER,
    'RelationalConceptModeling': RelationalConceptModeling,
    'RelationalGraphAttentionReasoning': RelationalGraphAttentionReasoning,
}

# Load the model
try:
    # Go one level up from BASE_DIR to find the 'Models' directory
    models_dir = os.path.join(settings.BASE_DIR.parent, 'Models')
    MODEL_FILEPATH = os.path.join(models_dir, 'dstsmgser_model.keras')
    model = tf.keras.models.load_model(MODEL_FILEPATH, custom_objects=custom_objects)
    logger.info("Model loaded successfully.")
    model_loaded_successfully = True
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    print(e) # Print the full exception
    model = None
    model_loaded_successfully = False

def preprocess_image_data(image_data):
    """
    Preprocess the image data to match the model's expected input format.
    """
    # Reshape the flat array to 28x28
    image = np.array(image_data, dtype=np.float32).reshape(28, 28)
    # Add batch and channel dimensions
    image = np.expand_dims(image, axis=[0, -1])  # Shape: (1, 28, 28, 1)
    return image

# Create your views here.
def index(request):
    return render(request, 'recognizer/index.html')

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_data = data.get('image')

            if not image_data:
                return JsonResponse({'error': 'No image data provided'}, status=400)
            
            if len(image_data) != 784:  # 28x28 = 784
                return JsonResponse({'error': 'Invalid image dimensions'}, status=400)

            if not model:
                 return JsonResponse({'error': 'Model not loaded'}, status=500)

            # Preprocess the image
            processed_image = preprocess_image_data(image_data)
            
            # Make prediction
            predictions = model.predict(processed_image)
            
            # Get the classification output (first output of the model)
            classification_output = predictions[0]
            predicted_digit = int(np.argmax(classification_output[0]))
            confidence = float(np.max(classification_output[0]))
            
            # Return the prediction
            return JsonResponse({
                'digit': predicted_digit,
                'confidence': confidence
            })
        
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

def health_check(request):
    """
    Simple health check endpoint to verify the API is running.
    """
    return JsonResponse({'status': 'healthy', 'model_loaded': model_loaded_successfully})

# python manage.py runserver