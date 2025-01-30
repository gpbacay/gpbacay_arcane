import numpy as np
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from gpbacay_arcane.models import DSTSMGSER
import matplotlib.pyplot as plt
import pandas as pd

# Load stock data
symbol = "AAPL"
start_date = "2015-01-01"
end_date = "2025-01-01"
df = yf.download(symbol, start=start_date, end=end_date)

# Use 'Close' prices for prediction
data = df[['Close']].values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Prepare sequences for training
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(scaled_data, seq_length)

# Reshape X to be 3D for the model
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define the DSTSMGSER model
input_shape = X_train.shape[1:]
reservoir_dim = 100
output_dim = 1

model = DSTSMGSER(
    input_shape=input_shape,
    reservoir_dim=reservoir_dim,
    spectral_radius=1.0,
    leak_rate=0.5,
    spike_threshold=0.5,
    max_dynamic_reservoir_dim=200,
    output_dim=output_dim,
    d_model=128,
    num_heads=8
)

# Build and compile the model
model.build_model()
model.compile_model()

# Train the model
history = model.model.fit(
    X_train,
    {
        'clf_out': tf.keras.utils.to_categorical(np.zeros_like(y_train), num_classes=1),
        'sm_out': X_train.reshape(X_train.shape[0], -1)
    },
    epochs=50,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# Predict on the test set
predictions = model.model.predict(X_test)
clf_predictions, _ = predictions

# Rescale predictions back to the original scale
predicted_prices = scaler.inverse_transform(clf_predictions.reshape(-1, 1))
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Forecast future prices
future_days = 365
last_sequence = scaled_data[-seq_length:]
future_predictions = []
current_input = last_sequence.reshape(1, seq_length, 1)

for _ in range(future_days):
    next_pred = model.model.predict(current_input, verbose=0)
    next_value = next_pred[0][0, 0]
    future_predictions.append(next_value)
    current_input = np.append(current_input[:, 1:, :], [[[next_value]]], axis=1)

# Rescale future predictions
future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Create a time axis for plotting using actual dates
train_dates = df.index[:train_size]
test_dates = df.index[train_size:train_size + len(y_test)]
future_dates = pd.date_range(start=df.index[-1], periods=future_days + 1, freq='B')[1:]  # Business days only

# Plot the results
plt.figure(figsize=(14, 7))

# Plot training data
plt.plot(train_dates, scaler.inverse_transform(y_train.reshape(-1, 1)), label="Training Data", color="blue")

# Plot actual test prices
plt.plot(test_dates, actual_prices, label="Actual Prices (Test Set)", color="green")

# Plot predicted test prices
plt.plot(test_dates, predicted_prices, label="Predicted Prices (Test Set)", color="red")

# Plot forecasted prices (2026)
plt.plot(future_dates, future_prices, label="Forecast (2026)", color="purple", linestyle="--")

# Format the plot
plt.title(f"{symbol} Stock Price Prediction and Forecast")
plt.xlabel("Time (Years)")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid()
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent label overlap
plt.show()

# Evaluate model performance
mse = mean_squared_error(actual_prices, predicted_prices)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual_prices, predicted_prices)

print("\nModel Performance Metrics:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")




# python main.py

