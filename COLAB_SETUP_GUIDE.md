# Google Colab Training Guide for Neuromimetic Language Model

## Quick Setup (5 minutes)

### Step 1: Prepare Your Files
1. **Zip your project**: Create a zip file containing your `gpbacay_arcane` folder
   ```
   gpbacay_arcane.zip
   ├── gpbacay_arcane/
   │   ├── __init__.py
   │   ├── models.py
   │   ├── layers.py
   │   ├── callbacks.py
   │   └── cli_commands.py
   └── colab_train_language_model.py
   ```

### Step 2: Open Google Colab
1. Go to https://colab.research.google.com/
2. Sign in with your Google account
3. Create a new notebook

### Step 3: Enable GPU
1. Click **Runtime** → **Change runtime type**
2. Set **Hardware accelerator** to **GPU**
3. Click **Save**

### Step 4: Upload and Setup
Copy and paste this code into the first cell:

```python
# Cell 1: Install packages and upload files
!pip install requests matplotlib

# Upload your zip file
from google.colab import files
uploaded = files.upload()  # Select your gpbacay_arcane.zip file

# Extract files
!unzip gpbacay_arcane.zip

# Verify GPU
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))
```

### Step 5: Run Training
Copy the entire `colab_train_language_model.py` content into a new cell and run it.

## Alternative: Using Google Drive

If you prefer using Google Drive:

```python
# Cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy your project folder to Colab
!cp -r "/content/drive/My Drive/your_project_folder/gpbacay_arcane" /content/

# Install packages
!pip install requests matplotlib

# Add to Python path
import sys
sys.path.append('/content')
```

## Expected Training Time
- **With GPU**: 15-30 minutes
- **Model Size**: ~15MB (much smaller than original)
- **Memory Usage**: ~2GB GPU memory

## After Training
The model and tokenizer will be automatically downloaded to your local machine. You can then use them with your local `test_language_model.py` script.

## Troubleshooting

### Out of Memory Error
Reduce these parameters in the training script:
```python
batch_size = 32  # Reduce from 64
d_model = 64     # Reduce from 96
reservoir_dim = 128  # Reduce from 192
```

### Runtime Disconnection
- Colab sessions last ~12 hours
- Your progress is saved automatically
- Models are checkpointed during training

### Import Errors
Make sure your `gpbacay_arcane` folder structure is correct:
```
gpbacay_arcane/
├── __init__.py
├── models.py      # Contains NeuromimeticLanguageModel
├── layers.py      # Contains all custom layers
├── callbacks.py   # Contains DynamicSelfModelingReservoirCallback
└── cli_commands.py
```

## Tips for Success

1. **Start Small**: Use the reduced parameters in `colab_train_language_model.py`
2. **Monitor GPU Usage**: Click the RAM/Disk indicator to see GPU usage
3. **Save Frequently**: Download checkpoints periodically
4. **Use Early Stopping**: The model will stop training when performance plateaus

## Free Alternatives to Colab

1. **Kaggle Notebooks**: 30 hours/week GPU
2. **Paperspace Gradient**: Limited free GPU hours
3. **Hugging Face Spaces**: Good for smaller models

Your neuromimetic language model should train successfully in Colab with GPU acceleration!
