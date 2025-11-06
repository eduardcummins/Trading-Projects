import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Configuration ---
# CHANGE THESE TWO TICKERS TO TEST DIFFERENT PAIRS
# Ideally, choose two assets that are historically correlated (e.g., KO & PEP, XOM & CVX)
tickers = ['KO', 'PEP']

asset1 = tickers[0]
asset2 = tickers[1]

# Date Range
start_date = '2015-01-01'
end_date = '2025-01-01'

# Parameters
window = 60           # Rolling window for Z-score
entry_threshold = 2.0 # Z-score to open a trade
exit_threshold = 0.5  # Z-score to close a trade

# --- 2. Get Data ---
print(f"Downloading data for {asset1} and {asset2}...")
data = yf.download(tickers, start=start_date, end=end_date)['Close']

# Drop any rows where data is missing for either asset
data = data.dropna()

# --- 3. Calculate Ratio and Z-Score ---
# We are betting on the ratio of Asset1 / Asset2
data['ratio'] = data[asset1] / data[asset2]

# Calculate rolling mean and standard deviation of the ratio
data['ratio_mean'] = data['ratio'].rolling(window=window).mean()
data['ratio_std'] = data['ratio'].rolling(window=window).std()

# Calculate Z-Score
data['z_score'] = (data['ratio'] - data['ratio_mean']) / data['ratio_std']
data = data.dropna()

# --- 4. Generate Signals ---
data['position'] = 0

# Entry Signals
# Z-score > 2.0: Ratio is too high. Short Asset1, Buy Asset2.
data.loc[data['z_score'] > entry_threshold, 'position'] = -1
# Z-score < -2.0: Ratio is too low. Buy Asset1, Short Asset2.
data.loc[data['z_score'] < -entry_threshold, 'position'] = 1

# Hold positions (forward fill)
data['position'] = data['position'].replace(0, method='ffill')

# Exit Signals
data.loc[(data['position'] == -1) & (data['z_score'] < exit_threshold), 'position'] = 0
data.loc[(data['position'] == 1) & (data['z_score'] > -exit_threshold), 'position'] = 0

# Ensure we are flat if we don't have a position, and SHIFT to avoid lookahead
data['position'] = data['position'].replace(0, method='ffill').fillna(0)
data['position'] = data['position'].shift(1).fillna(0)

# --- 5. Calculate Returns ---
daily_returns = data[[asset1, asset2]].pct_change()

# Calculate strategy returns based on position
# Pos 1: Long Asset1, Short Asset2
# Pos -1: Short Asset1, Long Asset2
data['strategy_return'] = np.where(
    data['position'] == 1, (daily_returns[asset1] - daily_returns[asset2]) / 2,
    np.where(
        data['position'] == -1, (daily_returns[asset2] - daily_returns[asset1]) / 2,
        0
    )
)

data['cumulative_strategy'] = (1 + data['strategy_return']).cumprod()

# --- 6. Plotting ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Z-Score
ax1.plot(data.index, data['z_score'], label='Ratio Z-Score', color='blue', alpha=0.7)
ax1.axhline(0, color='black', lw=1, linestyle='--')
ax1.axhline(entry_threshold, color='red', linestyle='--', label=f'Entry (+/- {entry_threshold})')
ax1.axhline(-entry_threshold, color='red', linestyle='--')
ax1.axhline(exit_threshold, color='green', linestyle='--', label=f'Exit (+/- {exit_threshold})')
ax1.axhline(-exit_threshold, color='green', linestyle='--')
ax1.set_title(f'{asset1} / {asset2} Ratio Z-Score')
ax1.legend(loc='upper left')

# Plot 2: Performance
ax2.plot(data['cumulative_strategy'], label='Pairs Trading Strategy', color='purple', lw=2)
ax2.plot((1 + daily_returns[asset1]).cumprod(), label=f'Buy & Hold {asset1}', color='grey', alpha=0.5, linestyle='--')
ax2.plot((1 + daily_returns[asset2]).cumprod(), label=f'Buy & Hold {asset2}', color='orange', alpha=0.5, linestyle='--')
ax2.set_title(f'Performance: Strategy vs {asset1} & {asset2}')
ax2.legend(loc='upper left')

plt.tight_layout()
plt.show()

# --- 7. Summary Stats ---
total_return = data['cumulative_strategy'].iloc[-1] - 1
annualized_return = (1 + total_return)**(252 / len(data)) - 1
volatility = data['strategy_return'].std() * np.sqrt(252)
sharpe_ratio = annualized_return / volatility

print(f"--- Pair: {asset1} & {asset2} ---")
print(f"Total Return: {total_return:.2%}")
print(f"Annualized Return: {annualized_return:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")