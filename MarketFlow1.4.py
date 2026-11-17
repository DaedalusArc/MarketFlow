import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# 1. Feature Engineering
# ------------------------------
def compute_features(df):
    """
    Compute 12 features from OHLCV data.
    Features: returns, log returns, volatility, volume change,
              RSI, MACD (2 features), Bollinger %B, ATR,
              price relative to SMA, momentum, volume ratio
    """
    df = df.copy()
    
    # 1. Simple returns
    df['ret'] = df['Close'].pct_change()
    
    # 2. Log returns
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 3. Realized volatility (20-day rolling std of returns)
    df['vol'] = df['ret'].rolling(20).std()
    
    # 4. Volume change
    df['vol_chg'] = df['Volume'].pct_change()
    
    # 5. RSI (14)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 6. MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # 7. Bollinger %B
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    df['bb_pct_b'] = (df['Close'] - sma20) / (2 * std20)
    
    # 8. ATR (14)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    # 9. Price / SMA(50)
    df['price_sma50'] = df['Close'] / df['Close'].rolling(50).mean()
    
    # 10. Momentum (10-day return)
    df['momentum'] = df['Close'].pct_change(10)
    
    # 11. Volume / average volume (20-day)
    df['vol_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    # 12. Close price (will be normalized later)
    df['close_norm'] = df['Close']
    
    # Drop NaN rows
    df.dropna(inplace=True)
    
    # Select the 12 features
    feature_cols = ['ret', 'log_ret', 'vol', 'vol_chg', 'rsi', 'macd', 
                    'macd_signal', 'bb_pct_b', 'atr', 'price_sma50', 
                    'momentum', 'vol_ratio']
    
    return df[feature_cols]

# ------------------------------
# 2. Create Windowed Dataset (FIXED)
# ------------------------------
def create_sequences(features, prices, window_size=64):
    """
    features: DataFrame of shape (n_samples, n_features)
    prices: Series of close prices aligned with features
    Returns: X (n_windows, window_size, n_features), y (binary up/down)
    """
    X, y = [], []
    
    # Convert to numpy arrays for faster and cleaner indexing
    features_array = features.values
    prices_array = prices.values
    
    for i in range(len(features_array) - window_size):
        X.append(features_array[i:i+window_size])
        
        # Next bar direction: 1 if price increased, else 0
        next_price = prices_array[i+window_size]
        current_price = prices_array[i+window_size-1]
        y.append(1 if next_price > current_price else 0)
    
    return np.array(X), np.array(y)

# ------------------------------
# 3. Causal TCN Model
# ------------------------------
def causal_tcn_model(input_shape, num_filters=64, kernel_size=3, dilations=[1, 2, 4, 8]):
    """
    Build a causal TCN with residual blocks.
    input_shape: (window_size, n_features)
    """
    inputs = layers.Input(shape=input_shape)
    x = inputs
    
    for dilation in dilations:
        # Causal conv with padding='causal' ensures no future leakage
        x = layers.Conv1D(filters=num_filters, kernel_size=kernel_size,
                           padding='causal', dilation_rate=dilation,
                           activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
    
    # Global pooling to produce a fixed-size vector
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ------------------------------
# 4. Volatility Forecast (EWMA)
# ------------------------------
def forecast_volatility(returns, lambda_=0.94):
    """Exponentially weighted moving average of squared returns."""
    vol = np.zeros_like(returns)
    vol[0] = returns[0]**2
    for t in range(1, len(returns)):
        vol[t] = lambda_ * vol[t-1] + (1 - lambda_) * returns[t]**2
    return np.sqrt(vol)

# ------------------------------
# 5. Risk Management Functions
# ------------------------------
def apply_deadband(signal, threshold=0.1):
    """Set signal to zero if |signal| < threshold."""
    return np.where(np.abs(signal) < threshold, 0.0, signal)

def inverse_vol_scaling(signal, vol_forecast, max_scale=10.0):
    """Scale signal inversely with forecasted volatility."""
    vol_forecast = np.maximum(vol_forecast, 1e-6)
    scaled = signal / vol_forecast
    scaled = np.clip(scaled, -max_scale, max_scale)
    return scaled

def cap_exposure(position, max_exposure=1.0):
    """Cap position size to max_exposure."""
    return np.clip(position, -max_exposure, max_exposure)

# ------------------------------
# 6. Main Function
# ------------------------------
def main():
    print("Downloading stock data...")
    # Download data (AAPL from 2018-2023 for faster training)
    ticker = 'AAPL'
    data = yf.download(ticker, start='2018-01-01', end='2023-12-31', progress=False)
    print(f"Downloaded {len(data)} days of data")
    
    # Compute features
    print("Computing features...")
    features = compute_features(data)
    prices = data['Close'].loc[features.index]
    
    print(f"Features shape: {features.shape}")
    print(f"Prices shape: {prices.shape}")
    
    # Create sequences
    print("Creating sequences...")
    X, y = create_sequences(features, prices, window_size=64)
    print(f"Created {len(X)} sequences")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Class balance: {np.sum(y)} up / {len(y)-np.sum(y)} down")
    
    # Train/validation split
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Normalize features
    print("\nNormalizing features...")
    scaler = StandardScaler()
    
    # Reshape for scaling
    original_shape_train = X_train.shape
    original_shape_val = X_val.shape
    
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_val_flat = X_val.reshape(-1, X_val.shape[-1])
    
    scaler.fit(X_train_flat)
    X_train_scaled = scaler.transform(X_train_flat).reshape(original_shape_train)
    X_val_scaled = scaler.transform(X_val_flat).reshape(original_shape_val)
    
    # Build and train model
    print("\nBuilding TCN model...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = causal_tcn_model(input_shape)
    model.summary()
    
    print("\nTraining model...")
    history = model.fit(X_train_scaled, y_train,
                        validation_data=(X_val_scaled, y_val),
                        epochs=15, batch_size=32, verbose=1)
    
    # Inference and Risk Management
    print("\nRunning inference with risk management...")
    probs = model.predict(X_val_scaled).flatten()
    raw_signal = 2 * probs - 1
    
    # Get returns for volatility forecast
    returns_series = data['Close'].pct_change().loc[features.index]
    
    # Get returns corresponding to each prediction
    # Each prediction corresponds to the bar after each window
    prediction_indices = []
    for i in range(split_idx, len(features) - 64):
        # Index of the bar we're predicting
        prediction_indices.append(features.index[i+64])
    
    # Get returns at those indices for volatility calculation
    ret_window = returns_series.loc[prediction_indices].values
    
    # Ensure we have matching lengths
    min_len = min(len(raw_signal), len(ret_window))
    raw_signal = raw_signal[:min_len]
    ret_window = ret_window[:min_len]
    
    # Calculate volatility forecast
    vol_forecast = forecast_volatility(ret_window)
    
    # Apply risk management
    signal_deadband = apply_deadband(raw_signal, threshold=0.1)
    signal_scaled = inverse_vol_scaling(signal_deadband, vol_forecast, max_scale=2.0)
    final_position = cap_exposure(signal_scaled, max_exposure=1.0)
    
    # Show results
    print("\n" + "="*70)
    print("Risk Management Example (first 15 predictions):")
    print("="*70)
    print(f"{'#':>3} | {'Raw Signal':>12} | {'Deadband':>12} | {'Vol Scaled':>12} | {'Final Pos':>12}")
    print("-"*70)
    
    for i in range(min(15, len(raw_signal))):
        print(f"{i+1:3d} | {raw_signal[i]:12.4f} | {signal_deadband[i]:12.4f} | "
              f"{signal_scaled[i]:12.4f} | {final_position[i]:12.4f}")
    
    # Calculate some statistics
    print("\n" + "="*70)
    print("Signal Statistics:")
    print("="*70)
    print(f"Mean raw signal: {np.mean(raw_signal):.4f}")
    print(f"Std raw signal: {np.std(raw_signal):.4f}")
    print(f"Mean final position: {np.mean(final_position):.4f}")
    print(f"Std final position: {np.std(final_position):.4f}")
    print(f"Zero signals after deadband: {np.sum(signal_deadband == 0) / len(signal_deadband)*100:.1f}%")
    print(f"Max absolute position: {np.max(np.abs(final_position)):.4f}")
    
    # Model accuracy
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    print(f"\nModel Accuracy:")
    print(f"Training: {train_acc:.4f}")
    print(f"Validation: {val_acc:.4f}")
    
    # Save model (optional)
    # model.save('trading_tcn_model.h5')
    # print("\nModel saved as 'trading_tcn_model.h5'")

if __name__ == "__main__":
    print("Starting Trading Bot with TCN and Risk Management...")
    print("="*70)
    main()
    print("\n" + "="*70)
    print("Done!")