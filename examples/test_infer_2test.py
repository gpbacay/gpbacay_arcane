import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image


def main():
    # Ensure we can import build_mnist_arcane_model from mnist_arcane.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.append(script_dir)

    from mnist_arcane import build_mnist_arcane_model

    print("Rebuilding ARCANE MNIST model and loading weights...")
    model = build_mnist_arcane_model()

    # Dummy forward pass to initialize custom layers
    model(np.zeros((1, 28, 28), dtype=np.float32))

    weights_path = os.path.join(os.path.dirname(script_dir), "mnist_arcane_model.weights.h5")
    if not os.path.exists(weights_path):
        print(f"Weights file not found: {weights_path}")
        return

    # Try standard weight loading. If it fails, explain and exit gracefully.
    try:
        model.load_weights(weights_path)
        print("Weights loaded successfully (standard load).")
    except Exception as e:
        print("Standard weight load failed.")
        print(f"Error details: {e}")
        print(
            "\nThe saved weights are incompatible with the current ARCANE MNIST "
            "architecture.\n"
            "Please re-train the model with `python examples/mnist_arcane.py` to "
            "generate fresh weights, then run this script again."
        )
        return

    # Test image path (2test.png)
    img_path = os.path.join(script_dir, "2test.png")
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return

    # Preprocess image (same logic as predict_digit.py)
    img = Image.open(img_path).convert("L")
    img = img.resize((28, 28))
    img_data = np.array(img).astype("float32") / 255.0

    # Invert if background is light
    if np.mean(img_data) > 0.5:
        print("Detected light background, inverting image for model compatibility...")
        img_data = 1.0 - img_data

    batch_data = np.expand_dims(img_data, axis=0)

    preds = model.predict(batch_data, verbose=0)
    digit = int(np.argmax(preds[0]))
    confidence = float(preds[0][digit])

    print("\n--- Resonant ARCANE MNIST Inference on 2test.png ---")
    print(f"Predicted Digit: {digit}")
    print(f"Confidence: {confidence * 100:.2f}%")


if __name__ == "__main__":
    main()

