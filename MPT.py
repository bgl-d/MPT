import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

stocks = ['AMZN','TSLA','META']
data = yf.download(stocks, start = '2024-01-01')['Close']
data.sort_index(inplace=True)
# convert daily stock prices into daily returns
returns = data.pct_change()
mean_daily_returns = returns.mean()
cov_matrix = returns.cov()
# set number of runs of random portfolio weights
num_portfolios = 25000
# set up array to hold results
results = np.zeros((4+len(stocks)-1,num_portfolios))

for i in range(num_portfolios):
    # select random weights for portfolio holdings
    weights = np.random.random(3)
    # rebalance weights to sum to 1
    weights /= np.sum(weights)

    # calculate portfolio return and volatility
    portfolio_return = np.sum(mean_daily_returns * weights) * 61
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(365)
    # store results in results array
    results[0, i] = portfolio_return
    results[1, i] = portfolio_std_dev
    # store Sharpe Ratio (return / volatility) - risk free rate element excluded for simplicity
    results[2, i] = results[0, i] / results[1, i]
    for j in range(len(weights)):
            results[j+3,i] = weights[j]
#convert results array to Pandas DataFrame
results_frame = pd.DataFrame(results.T,columns=['ret','stdev','sharpe',stocks[0],stocks[1],stocks[2]])
#locate position of portfolio with highest Sharpe Ratio
max_sharpe_port = results_frame.iloc[results_frame['sharpe'].idxmax()]
#locate positon of portfolio with minimum standard deviation
min_vol_port = results_frame.iloc[results_frame['stdev'].idxmin()]



#create scatter plot coloured by Sharpe Ratio
plt.scatter(results_frame.stdev,results_frame.ret,c=results_frame.sharpe,cmap='RdYlBu')
plt.xlabel('Volatility')
plt.ylabel('Returns')
plt.colorbar()
#plot red star to highlight position of portfolio with highest Sharpe Ratio
#plt.scatter(max_sharpe_port[1],max_sharpe_port[0],marker=(5,1,0),color='r',s=1000)
#plot green star to highlight position of minimum variance portfolio
#plt.scatter(min_vol_port[1],min_vol_port[0],marker=(5,1,0),color='g',s=1000)
plt.show()
print(max_sharpe_port)