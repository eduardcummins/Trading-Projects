import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Get Data & Calculate Indicators ---
ticker = 'SPY'
# We'll use the new default (auto_adjust=True)
data = yf.download(ticker, start='2015-01-01', end='2025-01-01')

# Calculate SMAs using 'Close' (which is now the adjusted close)
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()

# Drop missing values (at the start)
data = data.dropna()


# --- 2. Generate Trading Signals ---
# Create a 'position' column: 1 if 50-day > 200-day, 0 otherwise
data['position'] = np.where(data['SMA_50'] > data['SMA_200'], 1, 0)

# The actual signal is the *change* in position (shifted)
data['signal'] = data['position'].diff().shift(1)


# --- 3. Calculate Returns ---

# Calculate daily market returns using 'Close'
data['market_return'] = data['Close'].pct_change()

# Calculate strategy returns
data['strategy_return'] = data['market_return'] * data['position'].shift(1)


# --- 4. Analyze Results ---

# Calculate cumulative returns for both
data['cumulative_market'] = (1 + data['market_return']).cumprod()
data['cumulative_strategy'] = (1 + data['strategy_return']).cumprod()

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(data['cumulative_market'], label='Buy and Hold (SPY)')
plt.plot(data['cumulative_strategy'], label='Golden Cross Strategy')

# Plot the buy/sell signals
buy_signals = data[data['signal'] == 1]
plt.plot(buy_signals.index, data.loc[buy_signals.index]['cumulative_market'],
         '^', markersize=10, color='g', label='Buy Signal', alpha=0.7)

sell_signals = data[data['signal'] == -1]
plt.plot(sell_signals.index, data.loc[sell_signals.index]['cumulative_market'],
         'v', markersize=10, color='r', label='Sell Signal', alpha=0.7)

# Also plot the 'Close' price itself for context (optional, but good to see)
# Note: We'll plot this on a secondary y-axis if we uncomment it
# data['Close'].plot(ax=plt.gca(), secondary_y=True, label='SPY Price (RHS)',
#                    style='--', color='grey', alpha=0.3)

plt.title(f'{ticker} Golden Cross vs. Buy and Hold')
plt.legend()
plt.show()

# --- 5. Print Final Performance Metrics ---
total_return_market = data['cumulative_market'].iloc[-1] - 1
total_return_strategy = data['cumulative_strategy'].iloc[-1] - 1

print(f"--- {ticker} Performance ---")
print(f"Buy and Hold Total Return: {total_return_market:.2%}")
print(f"Golden Cross Total Return: {total_return_strategy:.2%}")