from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import tensorflow as tf
import json
import logging
import os
import pickle
from django.conf import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import custom layers for model loading
from gpbacay_arcane.layers import DenseGSER, BioplasticDenseLayer

# Define custom objects for model loading
custom_objects = {
    'DenseGSER': DenseGSER,
    'BioplasticDenseLayer': BioplasticDenseLayer,
}

# Global variables for model and tokenizer
model = None
tokenizer = None
model_loaded_successfully = False
tokenizer_loaded_successfully = False

def load_neuromimetic_model():
    """Load the neuromimetic language model and tokenizer."""
    global model, tokenizer, model_loaded_successfully, tokenizer_loaded_successfully
    
    try:
        # Model path
        models_dir = os.path.join(settings.BASE_DIR.parent, 'Models')
        model_path = os.path.join(models_dir, 'neuromimetic_lm.keras')
        tokenizer_path = os.path.join(models_dir, 'neuromimetic_lm_tokenizer.pickle')
        
        # Load model
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            logger.info("Neuromimetic language model loaded successfully.")
            model_loaded_successfully = True
        else:
            logger.warning(f"Model file not found: {model_path}")
            model_loaded_successfully = False
        
        # Load tokenizer
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)
            logger.info("Tokenizer loaded successfully.")
            tokenizer_loaded_successfully = True
        else:
            logger.warning(f"Tokenizer file not found: {tokenizer_path}")
            tokenizer_loaded_successfully = False
            
    except Exception as e:
        logger.error(f"Failed to load model or tokenizer: {str(e)}")
        model_loaded_successfully = False
        tokenizer_loaded_successfully = False

# Load model and tokenizer on startup
load_neuromimetic_model()

def index(request):
    """Render the main neuromimetic language model interface."""
    context = {
        'model_loaded': model_loaded_successfully,
        'tokenizer_loaded': tokenizer_loaded_successfully,
    }
    return render(request, 'recognizer/index.html', context)

@csrf_exempt
def generate_text(request):
    """Generate text using the neuromimetic language model."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            seed_text = data.get('seed_text', '').strip()
            max_length = min(int(data.get('max_length', 50)), 100)  # Cap at 100
            temperature = float(data.get('temperature', 0.8))
            temperature = max(0.1, min(2.0, temperature))  # Clamp between 0.1 and 2.0

            if not seed_text:
                return JsonResponse({'error': 'No seed text provided'}, status=400)
            
            if len(seed_text) > 200:
                return JsonResponse({'error': 'Seed text too long (max 200 characters)'}, status=400)

            if not model_loaded_successfully:
                return JsonResponse({'error': 'Model not loaded'}, status=500)
            
            if not tokenizer_loaded_successfully:
                return JsonResponse({'error': 'Tokenizer not loaded'}, status=500)

            # Generate text using the neuromimetic model
            generated_text = generate_neuromimetic_text(
                seed_text=seed_text,
                max_length=max_length,
                temperature=temperature
            )
            
            return JsonResponse({
                'generated_text': generated_text,
                'seed_text': seed_text,
                'temperature': temperature,
                'max_length': max_length
            })
        
        except Exception as e:
            logger.error(f"Text generation error: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

def generate_neuromimetic_text(seed_text, max_length=50, temperature=0.8, seq_len=16):
    """
    Generate text using the neuromimetic language model.
    
    Args:
        seed_text (str): Initial text to start generation
        max_length (int): Maximum number of tokens to generate
        temperature (float): Sampling temperature for creativity control
        seq_len (int): Sequence length used during training
        
    Returns:
        str: Generated text
    """
    # Create reverse mapping
    reverse_tokenizer = {v: k for k, v in tokenizer.word_index.items()}
    
    # Convert seed to tokens
    seed_tokens = tokenizer.texts_to_sequences([seed_text.lower()])[0]
    if not seed_tokens:
        seed_tokens = [1]  # fallback
    
    # Pad to sequence length
    if len(seed_tokens) < seq_len:
        seed_tokens = [0] * (seq_len - len(seed_tokens)) + seed_tokens
    else:
        seed_tokens = seed_tokens[-seq_len:]
    
    # Generate text
    current_seq = seed_tokens.copy()
    generated_words = []
    
    for _ in range(max_length):
        pred = model.predict(np.array([current_seq]), verbose=0)[0]
        
        # Temperature sampling with different strategies
        if temperature < 0.7:
            # Conservative: top-k sampling
            k = 8
            top_indices = np.argsort(pred)[-k:]
            top_probs = pred[top_indices]
            top_probs = top_probs / top_probs.sum()
            next_token = np.random.choice(top_indices, p=top_probs)
        elif temperature < 1.2:
            # Balanced: temperature sampling
            pred = pred / temperature
            pred = tf.nn.softmax(pred).numpy()
            next_token = np.random.choice(len(pred), p=pred)
        else:
            # Creative: nucleus sampling
            pred = pred / temperature
            pred = tf.nn.softmax(pred).numpy()
            
            # Nucleus sampling (top-p = 0.9)
            sorted_indices = np.argsort(pred)[::-1]
            cumsum_probs = np.cumsum(pred[sorted_indices])
            cutoff_idx = np.where(cumsum_probs > 0.9)[0]
            if len(cutoff_idx) > 0:
                cutoff_idx = cutoff_idx[0] + 1
            else:
                cutoff_idx = 15
            
            nucleus_indices = sorted_indices[:cutoff_idx]
            nucleus_probs = pred[nucleus_indices]
            nucleus_probs = nucleus_probs / nucleus_probs.sum()
            
            next_token = np.random.choice(nucleus_indices, p=nucleus_probs)
        
        # Convert token to word
        word = reverse_tokenizer.get(next_token, "")
        
        if word and word != "<UNK>" and word.strip():
            generated_words.append(word)
        
        # Update sequence
        current_seq = current_seq[1:] + [next_token]
        
        # Natural stopping points
        if word in [".", "!", "?"] and len(generated_words) > 5:
            break
    
    return " ".join(generated_words)

@csrf_exempt
def model_info(request):
    """Get information about the loaded neuromimetic model."""
    if request.method == 'GET':
        try:
            info = {
                'model_loaded': model_loaded_successfully,
                'tokenizer_loaded': tokenizer_loaded_successfully,
                'model_name': 'Neuromimetic Language Foundation Model',
                'description': 'Bio-inspired language model with spiking neural dynamics',
                'features': [
                    'Dual DenseGSER spiking neural layers',
                    'BioplasticDenseLayer Hebbian learning', 
                    'LSTM temporal processing',
                    'Homeostatic plasticity regulation',
                    'Advanced text generation capabilities'
                ]
            }
            
            if model_loaded_successfully and model:
                # Get model parameters
                total_params = model.count_params()
                trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
                
                info['parameters'] = {
                    'total_parameters': int(total_params),
                    'trainable_parameters': int(trainable_params),
                    'layers': len(model.layers)
                }
            
            return JsonResponse(info)
        
        except Exception as e:
            logger.error(f"Model info error: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

def health_check(request):
    """Simple health check endpoint to verify the API is running."""
    return JsonResponse({
        'status': 'healthy',
        'model_loaded': model_loaded_successfully,
        'tokenizer_loaded': tokenizer_loaded_successfully,
        'service': 'A.R.C.A.N.E. Neuromimetic Language Model'
    })

@csrf_exempt
def reload_model(request):
    """Reload the neuromimetic model and tokenizer."""
    if request.method == 'POST':
        try:
            load_neuromimetic_model()
            return JsonResponse({
                'message': 'Model reload attempted',
                'model_loaded': model_loaded_successfully,
                'tokenizer_loaded': tokenizer_loaded_successfully
            })
        except Exception as e:
            logger.error(f"Model reload error: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)