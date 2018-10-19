import numpy as np
import pandas as pd
import pandas_datareader as web
import matplotlib.pyplot as plt
import quandl
from scipy import stats
from scipy.optimize import minimize

start, end = pd.to_datetime('2012-01-01'), pd.to_datetime('2017-01-01')

aapl = quandl.get('WIKI/AAPL.11', start_date=start, end_date=end)
cisco = quandl.get('WIKI/CSCO.11', start_date=start, end_date=end)
ibm = quandl.get('WIKI/IBM.11', start_date=start, end_date=end)
amzn = quandl.get('WIKI/AMZN.11', start_date=start, end_date=end)

print(aapl.iloc[0]['Adj. Close'])
for df in (aapl, cisco, ibm, amzn):
    df['Normed Return'] = df['Adj. Close'] / df.iloc[0]['Adj. Close']

# PORTFOLIO ALLOCATION (eg. 30% - apple, 20% - cisco, 40% - amazon, 10% - ibm
for df, alloc, in zip((aapl, cisco, ibm, amzn), [.3, .2, .4, .1]):
    df['Allocation'] = df['Normed Return'] * alloc
for df in (aapl, cisco, ibm, amzn):
    df['Position Values'] = df['Allocation'] * 1000000  # $1 MILLION
# NOW, ALL DFs HAVE 5 COLUMNS (Date, Adj. Close, Normed Return, Allocation, Position Values

all_pos_vals = [aapl['Position Values'], cisco['Position Values'],
                ibm['Position Values'], amzn['Position Values']]
portfolio_val = pd.concat(all_pos_vals, axis=1)
portfolio_val.columns = ['AAPL', 'CSCO', 'IBM', 'AMZN']
portfolio_val['Total Position'] = portfolio_val.sum(axis=1)
portfolio_val['Total Position'].plot(figsize=(10, 8))
plt.title('Total Portfolio Value')
portfolio_val.drop('Total Position', axis=1).plot(figsize=(10, 8))

""" DESCRIPTION ON ABOVE CODE
So after setting up portfolio of stocks, added some other columns 
(Normed Return - cumulative data return, Allocation, Position Values - then calc 'Total Position', then 'Total Portfolio Value ) 
"""

portfolio_val['Daily Return'] = portfolio_val['Total Position'].pct_change(1)
portfolio_val['Daily Return'].mean()  # AVG DAILY RETURN
portfolio_val['Daily Return'].std()  # VOLATILITY OF DAILY RETURN
portfolio_val['Daily Return'].plot(kind='hist', bins=100, figsize=(8, 5))  # OR kind='kde'

cumulative_return = 100 * (portfolio_val['Total Position'][-1] / portfolio_val['Total Position'][0] - 1)
#   NOW, CALC'ING THE SHARPE RATIO = (mean_portfolio_return - riskfree rate) / std
SR = portfolio_val['Daily Return'].mean() / portfolio_val['Daily Return'].std()
# YOU CAN ALSO MULTIPLY THE AVG RETURN BY RISK-FREE RATE, IF IT ISN'T ZERO (0)
ASR = (252 ** 0.5) * SR  # ANNUALIZED SHARPE RATIO, WITH k-factor = sqrt(252 trading days)
# k-factor WOULD'VE BEEN sqrt(52 trading weeks) IF THIS WAS WEEKLY RETURNS :)
# k-factor WOULD'VE BEEN sqrt(12 trading months) IF THIS WAS MONTHLY RETURNS :)


#   PORTFORLIO OPTIMATION - MONTE-CARLO SIMULATION & EFFICIENT FRONTIER
aapl = pd.read_csv('AAPL_CLOSE', index_col='Date', parse_dates=True)
cisco = pd.read_csv('CSCO_CLOSE', index_col='Date', parse_dates=True)
ibm = pd.read_csv('IBM_CLOSE', index_col='Date', parse_dates=True)
amzn = pd.read_csv('AMZN_CLOSE', index_col='Date', parse_dates=True)

stocks = pd.concat([aapl, cisco, ibm, amzn], axis=1)
stocks.columns = ['aapl', 'csco', 'ibm', 'amzn']
#   CALC DAILY RETURNS & THEIR CORRELATIONS WITH EACH OTHER
daily_returns = stocks.pct_change(1).mean()
correlation_of_returns = stocks.pct_change(1).corr()
#   COVERTING DAILY ARITHMETIC RETURNS TO LOG RETURNS (COZ IT'S BETTER TO NORMALIZE THE RETURNS, TO FIND TREND)
log_returns = np.log(stocks / stocks.shift(1))
log_returns.hist(bins=100, figsize=(12, 8))
plt.tight_layout()
avg_log_return = log_returns.mean()
log_returns.corr();
log_returns.cov()

np.random.seed(101)
num_ports = 5000
all_weights = np.zeros((num_ports, len(stocks.columns)))
ret_arr, vol_arr, sharpe_arr = np.zeros(num_ports), np.zeros(num_ports), np.zeros(num_ports)


def monte_carlo():
    for i in range(num_ports):
        # WEIGHTS
        weights = np.array(np.random.random(4))
        print("Random Weights -> {}".format(weights))
        normalized_weights = weights / np.sum(weights)
        print("Rebalanced Weights -> {}".format(normalized_weights))

        # SAVE WEIGHTS NOW
        weights = normalized_weights
        all_weights[i, :] = weights

        # NOW, CALC'ING THE EXPECTED (MEAN) PORTFOLIO RETURN FOR 252 DAYS (1 YEAR)
        # NB: MULTIPLE BY 252 TRADING DAYS, COZ YOU'RE WORKING WITH LOG RETURNS (NOT %)
        expected_return = np.sum((log_returns.mean() * weights) * 252)
        ret_arr[i] = expected_return

        # NOW, EXPECTED VARIANCE / VOLATILITY (MORE COMPLEX LINEAR ALGEBRA FORMULA, BUT FASTER)
        expected_volatility = np.sqrt(
            np.dot(weights.T, np.dot(log_returns.cov() * 252, weights))
        )
        vol_arr[i] = expected_volatility

        # SHARPE RATIO ..
        SR = expected_return / expected_volatility
        sharpe_arr[i] = SR

        print("Expected Portfolio Return -> {}".format(expected_return))
        print("Expected Portfolio Volatility -> {}".format(expected_volatility))
        print("Sharpe Ratio -> {}".format(SR))

    max_value, max_index = sharpe_arr.max(), sharpe_arr.argmax()
    print("MAX SHARPE RATIO: INDEX '{}' -> VALUE '{}'".format(max_value, max_index))
    #   NOW, PLOT OUT THE GRAPH
    plt.figure(figsize=(12, 8))  # COLOR GRAPH BY c=sharpe_arr WITH 'plasma' COLOR MAP
    plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='plasma')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    # OPTIMUM WEIGHTS, RETURNS, VOLATILITY
    optimal_weights, optimal_return, optimal_volatility = all_weights[max_index, :], ret_arr[max_index], vol_arr[
        max_index]
    print("OPTIMAL VALUES (WEIGHTS, VOLATILITY, RETURNS):\n{}\n{}\n{}".format(optimal_weights, optimal_return,
                                                                              optimal_volatility))
    #   NOW, PLOT THE OPTIMUM WEIGHTS WITH THE MAX SHARPE RATIO
    plt.scatter(optimal_volatility, optimal_return, c='red', s=50, edgecolors='black')


""" 
So basically, Monte-Carlo Simulation does above function, but in iterations,
Until it finds the most optimum set of weights for minimun -ve Sharpe Ratio (or maximum +ve Sharpe Ratio) the portfolio
"""
print("RUNNING MONTE-CARLO SIMULATION NOW....")
monte_carlo()
#

# #
#   ANOTHER WAY OF RUNNING THE MONTE-CARLO SIMULATION, WITH SciPy OPTIMIZATION (MINIMIZATION :)
# #

def negative_sharpe(weights):
    return get_ret_vol_sr(weights)[2] * (-1)


def check_sum(weights):  # WILL RETURN 0 IF SUM == 1 (NOT RETURN BOOL COZ WE'LL NEED THE DIFFERENCE VALUE ITSELF :)
    return np.sum(weights) - 1


def get_ret_vol_sr(weights):  # BASICALLY THE SAME CODE AS ABOVE
    weights = np.array(weights)
    expected_return = np.sum((log_returns.mean() * weights) * 252)
    expected_volatility = np.sqrt(
        np.dot(weights.T, np.dot(log_returns.cov() * 252, weights))
    )
    SR = expected_return / expected_volatility
    return np.array([expected_return, expected_volatility, SR])


def minimum_volatility(weights):
    return get_ret_vol_sr(weights)[1]


constraints = ({'type': 'eq', 'fun': check_sum})
bounds = ((0, 1), (0, 1), (0, 1), (0, 1))
init_guess = [0.25, 0.25, 0.25, 0.25]
opt_results = minimize(negative_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
print("OPITMIZATION RESULTS ...")
print(opt_results)
ret_vol_sr = get_ret_vol_sr(opt_results.x)

#   NOW, CALCULATE THE EFFICIENT FRONTIER, AND PLOT IT OUT ON THE GRAPH
frontier_y = np.linspace(0, 0.3, 100)
frontier_volatility = []
for possible_return in frontier_y:
    constraints_in_loop = ({'type': 'eq', 'fun': check_sum},
                           {'type': 'eq', 'fun': lambda w: get_ret_vol_sr(w)[0] - possible_return})
    result = minimize(minimum_volatility, init_guess, method='SLSQP', bounds=bounds, constraints=constraints_in_loop)
    # NOW APPEND result TO fronteir_volatility LIST
    frontier_volatility.append(result['fun'])

# PLOTTING THESE ARRAY VALUES WILL STILL WORK (COZ THEY'VE NOT BEEN EMPTIED AFTER USING THEM ABOVE :)
plt.figure(figsize=(12, 8))  # COLOR GRAPH BY c=sharpe_arr WITH 'plasma' COLOR MAP
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='plasma')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
# NOW, PLOT OUT THE EFFICIENT FRONTIER CURVE ON THE GRAPH
plt.plot(frontier_volatility, frontier_y, 'g--', linewidth=3)



#
# CAPITAL ASSET PRICING MODEL (CAPM)
#

spy_etf = web.DataReader('SPY', 'google')
print("OVERALL SPY ETF MARKET PERFORMANCE -> {}".format(spy_etf.info()))
start, end = pd.to_datetime('2010-01-04'), pd.to_datetime('2017-07-25')
aapl = web.DataReader('AAPL', 'google', start, end)
print("APPLE STOCK PERFORMANCE -> {}".format(aapl.head()))
#
aapl['Close'].plot(label='AAPL', figsize=(10, 8))
spy_etf['Close'].plot(label='SPY Index')
plt.legend()
#
aapl['Cumulative'] = aapl['Close'] / aapl['Close'].iloc[0]
spy_etf['Cumulative'] = spy_etf['Close'] / spy_etf['Close'].iloc[0]
#
aapl['Cumulative'].plot(label='AAPL', figsize=(10, 8))
spy_etf['Cumulative'].plot(label='SPY ETF', figsize=(10, 8))
plt.legend()
#
aapl['Daily Return'] = aapl['Close'].pct_change(1)
spy_etf['Daily Return'] = spy_etf['Close'].pct_change(1)
plt.scatter(aapl['Daily Return'], spy_etf['Daily Return'], alpha=0.25)
# DOESN'T SHOW ANY CORRELATION BETWEEN SPY-ETF & APPLE STOCK AT ALL#
beta, alpha, r_val, p_val, std_err = stats.linregress(aapl['Daily Return'].iloc[1:],
                                                      spy_etf['Daily Return'].iloc[1:])
# beta ISN'T CLOSE TO 1 COZ THERE'S NO CORRELATION AT ALL BETWEEN THE 2 ASSETS
#
noise = np.random.normal(0, 0.001, len(spy_etf['Daily Return'].iloc[1:]))
fake_stock = spy_etf['Daily Return'].iloc[1:] + noise  # ADD THE NOISE DATA
plt.scatter(fake_stock, spy_etf['Daily Return'], alpha=0.25)
# SHOWS A +ve CORRELATION BETWEEN THE 2 COZ fake_stock IS BASICALLY SPY-ETF WITH SOME noise
beta, alpha, r_val, p_val, std_err = stats.linregress(fake_stock,
                                                      spy_etf['Daily Return'].iloc[1:])
# NOW, beta VALUE IS CLOSE TO 1 (O.98) COZ OF THE +VE CORRELATION







