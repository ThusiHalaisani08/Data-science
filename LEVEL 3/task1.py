import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

df = yf.download("AAPL", start="2020-01-01", end="2024-01-01")
df.columns = ["Close", "High", "Low", "Open", "Volume"]
df = df[["Close"]]  # we only need closing price

print(f"Total trading days: {len(df)}")
print(df.head())

print("\nPlotting & Decomposin")

# Raw price plot
plt.figure(figsize=(12, 5))
plt.plot(df.index, df["Close"], color="#3498db", linewidth=1)
plt.title("Apple (AAPL) Stock Price 2020-2024")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.tight_layout()
plt.savefig("stock_price.png", dpi=150, bbox_inches="tight")
plt.show()

# Decomposition
decomposition = seasonal_decompose(df["Close"],
                                   model="additive",
                                   period=252)

fig, axes = plt.subplots(4, 1, figsize=(12, 10))
fig.suptitle("Time Series Decomposition — AAPL", fontsize=14)
decomposition.observed.plot(ax=axes[0], color="#3498db")
axes[0].set_ylabel("Observed")
decomposition.trend.plot(ax=axes[1], color="#2ecc71")
axes[1].set_ylabel("Trend")
decomposition.seasonal.plot(ax=axes[2], color="#e74c3c")
axes[2].set_ylabel("Seasonality")
decomposition.resid.plot(ax=axes[3], color="#f39c12")
axes[3].set_ylabel("Residual")
plt.tight_layout()
plt.savefig("decomposition.png", dpi=150, bbox_inches="tight")
plt.show()



print("\nSmoothing Techniques ")

# Moving averages
df["MA_30"]  = df["Close"].rolling(window=30).mean()   # 30 day
df["MA_90"]  = df["Close"].rolling(window=90).mean()   # 90 day
df["EMA_30"] = df["Close"].ewm(span=30, adjust=False).mean()  # exponential

plt.figure(figsize=(12, 6))
plt.plot(df.index, df["Close"],  color="#bdc3c7", linewidth=1,  label="Actual Price")
plt.plot(df.index, df["MA_30"],  color="#3498db", linewidth=1.5, label="30-Day MA")
plt.plot(df.index, df["MA_90"],  color="#e74c3c", linewidth=1.5, label="90-Day MA")
plt.plot(df.index, df["EMA_30"], color="#2ecc71", linewidth=1.5, label="30-Day EMA")
plt.title("AAPL — Moving Averages & Exponential Smoothing")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.tight_layout()
plt.savefig("smoothing.png", dpi=150, bbox_inches="tight")
plt.show()


print("\n\t\t\t  ARIMA Model ")

# Split — use last 60 days as test
train = df["Close"].iloc[:-60]
test  = df["Close"].iloc[-60:]

print(f"Train size: {len(train)} days")
print(f"Test size:  {len(test)} days")

# Fit ARIMA(5,1,0)
# p=5 → use last 5 days to predict next
# d=1 → difference once to make stationary
# q=0 → no moving average component
print("Fitting ARIMA model...")
model = ARIMA(train, order=(5, 1, 0))
fitted = model.fit()

# Forecast 60 days ahead
forecast = fitted.forecast(steps=60)
forecast_index = test.index

# Evaluate
rmse = np.sqrt(mean_squared_error(test, forecast))
print(f"ARIMA RMSE: ${rmse:.2f}")

print("\n\t\t\t Forecast Plot ")

plt.figure(figsize=(12, 6))
plt.plot(train.index, train,           color="#3498db", linewidth=1,   label="Training Data")
plt.plot(test.index,  test,            color="#2ecc71", linewidth=1.5, label="Actual Price")
plt.plot(forecast_index, forecast,     color="#e74c3c", linewidth=1.5,
         linestyle="--",               label=f"ARIMA Forecast (RMSE=${rmse:.2f})")
plt.axvline(x=test.index[0], color="gray", linestyle=":", linewidth=1)
plt.title("AAPL Stock Price — ARIMA Forecast vs Actual")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.tight_layout()
plt.savefig("arima_forecast.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n All Time Series plots saved!")
print(f"\n Final Results")
print(f"ARIMA RMSE: ${rmse:.2f}")
print(f"Mean actual price in test period: ${test.mean():.2f}")
print(f"Error as % of price: {(rmse/test.mean())*100:.1f}%")