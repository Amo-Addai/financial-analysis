import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quandl



start, end = pd.to_datetime('2012-01-01'), pd.to_datetime('2017-01-01')

aapl = quandl.get('WIKI/AAPL.11', start_date=start, end_date=end)
cisco = quandl.get('WIKI/CSCO.11', start_date=start, end_date=end)
ibm = quandl.get('WIKI/IBM.11', start_date=start, end_date=end)
amzn = quandl.get('WIKI/AMZN.11', start_date=start, end_date=end)

print(aapl.iloc[0]['Adj. Close'])
for df in (aapl, cisco, ibm, amzn):
    df['Normed Return'] = df['Adj. Close'] / df.iloc[0]['Adj. Close']

#   PORTFOLIO ALLOCATION (eg. 30% - apple, 20% - cisco, 40% - amazon, 10% - ibm
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
daily_return = stocks.pct_change(1).mean()
correlation_of_returns = stocks.pct_change(1).corr()
#   COVERTING DAILY ARITHMETIC RETURNS TO LOG RETURNS (COZ IT'S BETTER TO NORMALIZE THE RETURNS, TO FIND TREND)









