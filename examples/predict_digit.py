import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Add current dir to path to allow importing from mnist_arcane
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def predict_digit(image_path):
    """
    Inference script for the ARCANE MNIST Classifier.
    Demonstrates the model's ability to classify handwritten digits 
    using Resonant Gated Spiking Elastic Reservoirs.
    """
    print("\n--- ARCANE MNIST Inference ---")
    
    # 1. Rebuild the ARCANE architecture
    from mnist_arcane import build_mnist_arcane_model
    print("Rebuilding model architecture...")
    model = build_mnist_arcane_model()
    
    # 2. Locate Weights
    # We prioritize the full .keras file, fallback to .weights.h5
    weights_path = 'mnist_arcane_model.keras'
    if not os.path.exists(weights_path):
        weights_path = 'mnist_arcane_model.weights.h5'
        
    if not os.path.exists(weights_path):
        print(f"Error: Trained model weights not found ('{weights_path}').")
        print("Please run 'examples/mnist_arcane.py' first to train and save the model.")
        return

    # 3. Initialize and Load Weights
    # We perform a dummy pass to ensure Keras instantiates all internal variables 
    # for the custom neuromimetic layers.
    print(f"Initializing layers and loading weights from {weights_path}...")
    model(np.zeros((1, 28, 28)))
    
    try:
        model.load_weights(weights_path)
    except Exception as e:
        print(f"Standard weight load failed, attempting custom mapping...")
        model.load_weights(weights_path, skip_mismatch=True, by_name=True)

    # 4. Image Preprocessing
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found.")
        return

    print(f"Analyzing image: {image_path}")
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img_data = np.array(img).astype('float32') / 255.0
    
    # ARCANE/MNIST Expectation: White digit on black background (high activation on pixels)
    # If the image provided has a light background (mean > 0.5), we automatically invert it.
    if np.mean(img_data) > 0.5:
        print("Detected light background, inverting image for model compatibility...")
        img_data = 1.0 - img_data
        
    batch_data = np.expand_dims(img_data, axis=0)
    
    # 5. Perform Resonant Inference
    # The model deliberates across hierarchical levels to resolve the digit identity.
    preds = model.predict(batch_data, verbose=0)
    digit = np.argmax(preds[0])
    confidence = preds[0][digit]

    print(f"\n--- Prediction Results ---")
    print(f"Predicted Digit: {digit}")
    print(f"Confidence Level: {confidence * 100:.2f}%")
    
    # 6. Visualization (Premium Dark Mode Style)
    plt.figure(figsize=(6, 6))
    plt.imshow(img_data, cmap='gray')
    plt.title(f"ARCANE Forecast: {digit} ({confidence*100:.1f}%)", 
              color='#00ffcc', fontsize=18, fontweight='bold', pad=20)
    plt.axis('off')
    plt.gcf().set_facecolor('#1a1a1a')
    plt.gca().set_facecolor('#1a1a1a')
    
    # Save a visual report
    report_path = 'mnist_prediction_report.png'
    plt.savefig(report_path, facecolor='#1a1a1a', bbox_inches='tight')
    print(f"Visual report saved to: {report_path}")
    plt.show()

if __name__ == "__main__":
    # Allow command line argument for image path
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    else:
        # Default to the sample image created during test
        target_path = r'c:\Users\Gianne Bacay\Desktop\project test\gpbacay_arcane\examples\2test.png'
        
        # Create a sample "4" if nothing exists to test
        if not os.path.exists(target_path):
            print("Creating a synthetic test digit (Digit: 4)...")
            data = np.zeros((28,28), dtype=np.uint8)
            # Simple "4" geometry
            data[5:15, 8] = 255 # Vertical left
            data[15, 8:20] = 255 # Horizontal
            data[5:23, 20] = 255 # Vertical right
            Image.fromarray(data).save(target_path)
            
    predict_digit(target_path)

# python examples/predict_digit.py
