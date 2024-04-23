import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

stock = ['BTC-USD','ETH-USD','SOL-USD']
data = yf.download(stock, start = '2024-01-01')['Close']
data.sort_index(inplace=True)
# convert daily stock prices into daily returns
returns = data.pct_change()
mean_daily_returns = returns.mean()
cov_matrix = returns.cov()
# set number of runs of random portfolio weights
num_portfolios = 25000
# set up array to hold results
results = np.zeros((3, num_portfolios))

for i in range(num_portfolios):
    # select random weights for portfolio holdings
    weights = np.random.random(3)
    # rebalance weights to sum to 1
    weights /= np.sum(weights)

    # calculate portfolio return and volatility
    portfolio_return = np.sum(mean_daily_returns * weights) * 365
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(365)
    # store results in results array
    results[0, i] = portfolio_return
    results[1, i] = portfolio_std_dev
    # store Sharpe Ratio (return / volatility) - risk free rate element excluded for simplicity
    results[2, i] = results[0, i] / results[1, i]
# convert results array to Pandas DataFrame
results_frame = pd.DataFrame(results.T, columns=['ret', 'stdev', 'sharpe'])
# create scatter plot coloured by Sharpe Ratio
plt.scatter(results_frame.stdev, results_frame.ret, s=1, c=results_frame.sharpe, cmap='RdYlBu')
plt.colorbar()
plt.show()