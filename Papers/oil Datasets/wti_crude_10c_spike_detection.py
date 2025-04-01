
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = "synthetic_wti_crude_5min_data.csv"
df = pd.read_csv(file_path, parse_dates=["Time"])

# Define a smaller spike threshold (10 cents)
small_spike_threshold = 0.10  

# Identify potential smaller spikes based on price changes
df["Price_Change"] = df["Price"].diff()
df["Small_Spike_Up"] = (df["Price_Change"] >= small_spike_threshold)
df["Small_Spike_Down"] = (df["Price_Change"] <= -small_spike_threshold)

# Define lookback periods for early detection
lookback_range = [5, 10, 15]  

# Detect early signals
for lookback in lookback_range:
    df[f"Early_Buy_Signal_Small_{lookback}"] = (
        (df["EMA_3"].shift(lookback) > df["EMA_15"].shift(lookback)) &  
        (df["MACD"].shift(lookback) > df["Signal_Line"].shift(lookback)) &  
        (df["Volume"].shift(lookback) > df["Volume"].rolling(10).mean().shift(lookback) * 1.5)  
    )

    df[f"Early_Sell_Signal_Small_{lookback}"] = (
        (df["EMA_3"].shift(lookback) < df["EMA_15"].shift(lookback)) &  
        (df["MACD"].shift(lookback) < df["Signal_Line"].shift(lookback)) &  
        (df["Volume"].shift(lookback) > df["Volume"].rolling(10).mean().shift(lookback) * 1.5)  
    )

# Plot price movements and detected spikes
plt.figure(figsize=(14, 6))
plt.plot(df["Time"], df["Price"], label="Price", color="blue")
plt.scatter(df["Time"][df["Small_Spike_Up"]], df["Price"][df["Small_Spike_Up"]], color="green", label="Spike Up (10c)", marker="^", alpha=1)
plt.scatter(df["Time"][df["Small_Spike_Down"]], df["Price"][df["Small_Spike_Down"]], color="red", label="Spike Down (10c)", marker="v", alpha=1)
plt.title("Early 10-Cent Spike Detection in WTI Crude Oil")
plt.xlabel("Time")
plt.ylabel("Price ($)")
plt.legend()
plt.xticks(rotation=45)
plt.grid()

# Show plot
plt.show()
