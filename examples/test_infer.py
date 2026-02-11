import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image


def load_model(script_dir: str):
    """
    Rebuild the ARCANE MNIST model and load trained weights.
    Assumes `examples/mnist_arcane.py` has already been run.
    """
    if script_dir not in sys.path:
        sys.path.append(script_dir)

    from mnist_arcane import build_mnist_arcane_model

    print("Rebuilding ARCANE MNIST model and loading weights...")
    # Enable persistent predictive resonance and inference-time plasticity
    # so that the model can adapt slightly during repeated inference calls.
    model = build_mnist_arcane_model(
        persistent_predictive=True,
        bioplastic_inference_plasticity=True,
    )

    # Dummy forward pass to initialize custom layers
    model(np.zeros((1, 28, 28), dtype=np.float32))

    weights_path = os.path.join(os.path.dirname(script_dir), "mnist_arcane_model.weights.h5")
    if not os.path.exists(weights_path):
        print(f"Weights file not found: {weights_path}")
        print("Run `python examples/mnist_arcane.py` first to train and save the model.")
        return None

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
        return None

    return model


def preprocess_image(img_path: str):
    """
    Load and preprocess a digit image for the ARCANE MNIST model.

    Important: During training, MNIST was passed to the model in raw [0, 255]
    and the model itself applies a Rescaling(1./255) layer. To match that,
    we keep the 0–255 convention here and do NOT divide by 255 again.
    """
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return None

    img = Image.open(img_path).convert("L")
    img = img.resize((28, 28))
    img_data = np.array(img).astype("float32")  # keep 0–255 range

    # Invert if background is light (match 0–255 convention)
    if np.mean(img_data) > 127.5:
        print("Detected light background, inverting image for model compatibility...")
        img_data = 255.0 - img_data

    batch_data = np.expand_dims(img_data, axis=0)
    return batch_data


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Determine image path (CLI arg or default)
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        if not os.path.isabs(img_path):
            img_path = os.path.join(script_dir, img_path)
    else:
        # Default to a sample digit image in the examples folder
        img_path = os.path.join(script_dir, "5test.png")

    print(f"Using test image: {img_path}")

    # 2. Load model
    model = load_model(script_dir)
    if model is None:
        return

    # 3. Preprocess image
    batch_data = preprocess_image(img_path)
    if batch_data is None:
        return

    # 4. Run prediction
    preds = model.predict(batch_data, verbose=0)
    digit = int(np.argmax(preds[0]))
    confidence = float(preds[0][digit])

    print("\n--- ARCANE MNIST Inference ---")
    print(f"Image: {os.path.basename(img_path)}")
    print(f"Predicted Digit: {digit}")
    print(f"Confidence: {confidence * 100:.2f}%")

    # 5. Visualization: show processed digit and class probabilities
    probabilities = preds[0]

    plt.figure(figsize=(10, 4))

    # Left: processed input image as seen by the model
    plt.subplot(1, 2, 1)
    plt.imshow(batch_data[0], cmap="gray")
    plt.axis("off")
    plt.title(f"Predicted: {digit}\nConf: {confidence * 100:.2f}%", fontsize=10)

    # Right: bar chart of class probabilities
    plt.subplot(1, 2, 2)
    plt.bar(range(10), probabilities, color="#00ffcc")
    plt.xticks(range(10))
    plt.ylim(0, 1)
    plt.xlabel("Digit")
    plt.ylabel("Probability")
    plt.title("Prediction Distribution")

    plt.tight_layout()

    # Save visualization next to this script
    viz_path = os.path.join(script_dir, "mnist_arcane_infer_result.png")
    plt.savefig(viz_path)
    print(f"Inference visualization saved to: {viz_path}")

    # 6. Optionally persist inference-time plasticity updates so that
    # repeated calls to this script can gradually adapt the model.
    weights_path = os.path.join(os.path.dirname(script_dir), "mnist_arcane_model.weights.h5")
    try:
        model.save_weights(weights_path)
        print(f"Updated weights (including plasticity state) saved to: {weights_path}")
    except Exception as e:
        print(f"Warning: could not save updated weights: {e}")


if __name__ == "__main__":
    main()


# python examples/test_infer.py
# Results: with just 98.95% accuracy, the model is able to predict the digit correctly.