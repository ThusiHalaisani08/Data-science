import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

#Step 1: Download Apple Stock Data 
df = yf.download("AAPL", start="2020-01-01", end="2024-01-01")

print(df.shape)
print(df.head())
print(df.info())
print(f"\nDate range: {df.index[0]} to {df.index[-1]}")
print(f"Total trading days: {len(df)}")
