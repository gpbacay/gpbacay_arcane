# Neuromimetic Language Model Training Script
# This is the main training script for the gpbacay_arcane neuromimetic language foundation model

import numpy as np
import tensorflow as tf
import requests
import os
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import re

# Set policy for stability
tf.keras.mixed_precision.set_global_policy('float32')

# Import stable custom layers
from gpbacay_arcane.layers import DenseGSER, BioplasticDenseLayer

def download_shakespeare():
    """Download Shakespeare text dataset."""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    
    if not os.path.exists("shakespeare.txt"):
        print("Downloading Shakespeare dataset...")
        response = requests.get(url)
        with open("shakespeare.txt", "w", encoding="utf-8") as f:
            f.write(response.text)
        print("Dataset downloaded!")
    
    with open("shakespeare.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
    return text

def prepare_data(text, seq_len=16, vocab_size=1000):
    """Prepare data with optimal settings."""
    # Simple preprocessing
    text = text.lower()
    text = re.sub(r'[^\w\s\.\,\!\?\:\;]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Create tokenizer
    tokenizer = Tokenizer(
        num_words=vocab_size,
        oov_token="<UNK>",
        lower=True
    )
    tokenizer.fit_on_texts([text])
    
    sequences = tokenizer.texts_to_sequences([text])[0]
    
    # Create sequences
    X, y = [], []
    step = 3
    
    for i in range(0, len(sequences) - seq_len, step):
        X.append(sequences[i:i + seq_len])
        y.append(sequences[i + seq_len])
    
    print(f"Created {len(X)} training sequences")
    print(f"Vocabulary size: {len(tokenizer.word_index) + 1}")
    print(f"Sequence length: {seq_len}")
    
    return np.array(X), np.array(y), tokenizer

class NeuromimeticLanguageFoundationModel:
    """Neuromimetic Language Foundation Model - Advanced Bio-Inspired Architecture."""
    
    def __init__(self):
        self.seq_len = 16
        self.vocab_size = 1000
        self.embed_dim = 32
        self.hidden_dim = 64
        
    def build_model(self, actual_vocab_size):
        """Build neuromimetic language foundation model."""
        from tensorflow.keras.layers import (
            Input, Embedding, Dense, GlobalAveragePooling1D, 
            Dropout, LayerNormalization, LSTM, Concatenate
        )
        from tensorflow.keras import Model
        
        # Force CPU device for variables to avoid GPU/CPU conflicts
        with tf.device('/CPU:0'):
            inputs = Input(shape=(self.seq_len,), name='text_input')
            
            # Embedding layer
            embedded = Embedding(
                actual_vocab_size, 
                self.embed_dim,
                name='embedding'
            )(inputs)
            
            # First DenseGSER layer - Primary spiking neural processing
            gser1 = DenseGSER(
                units=self.hidden_dim,
                spectral_radius=0.9,
                leak_rate=0.1,
                spike_threshold=0.35,
                activation='gelu',
                name='dense_gser_1'
            )(embedded)
            
            # Layer normalization and dropout
            gser1_norm = LayerNormalization(name='layer_norm_1')(gser1)
            gser1_drop = Dropout(0.15, name='dropout_1')(gser1_norm)
            
            # Second DenseGSER layer - Secondary spiking neural processing
            gser2 = DenseGSER(
                units=self.hidden_dim,
                spectral_radius=0.8,
                leak_rate=0.12,
                spike_threshold=0.3,
                activation='gelu',
                name='dense_gser_2'
            )(gser1_drop)
            
            # LSTM for sequential temporal processing
            lstm_out = LSTM(
                self.hidden_dim,
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.1,
                name='lstm_temporal'
            )(gser2)
            
            # Multiple pooling strategies for feature extraction
            avg_pool = GlobalAveragePooling1D(name='avg_pool')(lstm_out)
            gser2_pool = GlobalAveragePooling1D(name='gser2_pool')(gser2)
            
            # Feature fusion from multiple neural pathways
            combined = Concatenate(name='feature_fusion')([avg_pool, gser2_pool])
            
            # BioplasticDenseLayer - Hebbian learning and homeostatic plasticity
            bioplastic = BioplasticDenseLayer(
                units=self.hidden_dim * 2,  # Match combined features dimension
                learning_rate=1.5e-3,
                target_avg=0.11,
                homeostatic_rate=8e-5,
                activation='gelu',
                dropout_rate=0.12,
                name='bioplastic_main'
            )(combined)
            
            # Additional dense processing layer
            dense_hidden = Dense(
                self.hidden_dim,
                activation='gelu',
                name='dense_processing'
            )(bioplastic)
            
            dense_dropout = Dropout(0.1, name='dense_dropout')(dense_hidden)
            
            # Output layer for language modeling
            outputs = Dense(
                actual_vocab_size,
                activation='softmax',
                name='language_output'
            )(dense_dropout)
            
            model = Model(
                inputs=inputs,
                outputs=outputs,
                name='neuromimetic_language_foundation_model'
            )
        
        return model
    
    def train_model(self, save_path="Models/neuromimetic_lm.keras"):
        """Train the neuromimetic language foundation model."""
        print("🧠 Training Neuromimetic Language Foundation Model")
        print("=" * 60)
        
        # Download and prepare data
        text = download_shakespeare()
        X, y, tokenizer = prepare_data(text, self.seq_len, self.vocab_size)
        
        # Use optimal subset size
        subset_size = min(25000, len(X))
        X = X[:subset_size]
        y = y[:subset_size]
        
        # Split data
        split_idx = int(0.88 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Get actual vocab size
        actual_vocab_size = len(tokenizer.word_index) + 1
        print(f"Vocabulary size: {actual_vocab_size}")
        
        # Build model
        print("\n🏗️ Building neuromimetic architecture...")
        
        # Use strategy for proper device coordination
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = self.build_model(actual_vocab_size)
            
            # Stable optimizer with gradient clipping
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=1e-3,
                clipnorm=1.0,
                beta_1=0.9,
                beta_2=0.999
            )
            
            # Compile model
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        
        print("\n🏛️ Neuromimetic Model Architecture:")
        model.summary()
        
        # Training callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=4,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.6,
                patience=2,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                save_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Create Models directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Train model
        print("🎯 Training neuromimetic language model...")
        try:
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=12,
                batch_size=96,
                callbacks=callbacks,
                verbose=1
            )
        except Exception as e:
            print(f"Training failed: {e}")
            return None, None
        
        # Evaluate
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        print(f"\n📊 Final Results:")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Perplexity: {np.exp(val_loss):.2f}")
        
        # Plot results
        self.plot_training_results(history)
        
        # Save tokenizer
        tokenizer_path = save_path.replace('.keras', '_tokenizer.pickle')
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(tokenizer, f)
        
        print(f"💾 Model saved to: {save_path}")
        print(f"💾 Tokenizer saved to: {tokenizer_path}")
        
        return model, tokenizer
    
    def plot_training_results(self, history):
        """Plot training results."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Loss
        axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2, color='blue')
        axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='red')
        axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, color='green')
        axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='orange')
        axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Perplexity
        train_perplexity = [np.exp(loss) for loss in history.history['loss']]
        val_perplexity = [np.exp(loss) for loss in history.history['val_loss']]
        axes[2].plot(train_perplexity, label='Training Perplexity', linewidth=2, color='purple')
        axes[2].plot(val_perplexity, label='Validation Perplexity', linewidth=2, color='brown')
        axes[2].set_title('Model Perplexity', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Perplexity')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print improvement
        initial_loss = history.history['loss'][0]
        final_loss = history.history['loss'][-1]
        improvement = ((initial_loss - final_loss) / initial_loss) * 100
        print(f"📈 Training improvement: {improvement:.1f}% loss reduction")

def main():
    """Main training function."""
    print("🧠 A.R.C.A.N.E. Neuromimetic Language Foundation Model")
    print("Augmented Reconstruction of Consciousness through Artificial Neural Evolution")
    print("=" * 80)
    print("🚀 Features:")
    print("   🔬 Dual DenseGSER spiking neural layers")
    print("   🧬 BioplasticDenseLayer Hebbian learning")
    print("   🔄 LSTM temporal processing")
    print("   🎯 Advanced language modeling")
    print("   📊 Comprehensive training monitoring")
    print()
    
    trainer = NeuromimeticLanguageFoundationModel()
    model, tokenizer = trainer.train_model()
    
    if model is not None:
        print("\n🎉 Neuromimetic Language Foundation Model training completed!")
        print("🏆 Model features:")
        print("✓ Bio-inspired spiking neural dynamics")
        print("✓ Hebbian learning and homeostatic plasticity")
        print("✓ Advanced temporal sequence processing")
        print("✓ Ready for deployment and inference")
    else:
        print("❌ Training failed - please check the error logs")

if __name__ == "__main__":
    main()
