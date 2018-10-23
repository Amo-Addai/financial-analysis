import numpy as np
import pandas as pd
import pandas_datareader as web
import matplotlib.pyplot as plt
import quandl
from scipy import stats
from scipy.optimize import minimize

import quantopian
import quantopian.algorithm as algo
import quantopian.optimize as opt


#   RETURNS A PANDAS DATAFRAME (frequency="daily" / "monthly" / "yearly" / etc
mcdon = get_pricing('MCD', start_date="2017-01-01", end_date="2017-02-01", frequency="minute")
print(mcdon.head()); print(mcdon.info())
#   ALL DATA get_pricing() RETURNS HAVE BEEN ADJUSTED FOR STOCK SPLITS, DIVIDENDS, etc ALREADY
mcdon['close_price'].plot()
#
daily_returns = mcdon['close_price'].pct_change(1)
daily_returns.hist(bins=100, figsize=(6,4))  # PLOT DAILY RETURNS (NORMAL DISTRIBUTION - hist)
#
mcdon_eq_info = symbols('MCD')  # RETURNS ZIPLINE EQUITY OBJECT
mcdon_eq_info = mcdon_eq_info.to_dict()  # NOW IN DICT FORMAT
#
fund = init_fundamentals()  # INITIALIZE FUNDAMENTALS OBJECT
query = query(fund.valuation.market_cap)  # SETUP QUERY ON FUNDAMENTALS OBJECT
funds = get_fundamentals(query, '2017-01-01')  # EXECUTE QUERY
print(funds.info())
# USING .filter() TO SPLICE THE DATA RETURNED FROM THE QUERY
big_comps = ( query(fund.valuation.market_cap).filter(fund.valuation.market_cap > 1000000) )
df = get_fundamentals(big_comps, '2017-07-19')
df.head()
# 





