import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quantopian

from scipy import stats
from scipy.optimize import minimize

mcdon = get_pricing('MCD', start_date="2017-01-01", end_date="2017-02-01", frequency="daily")
print(mcdon.head()); print(mcdon.info())



mcdon.head()

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


# In[3]:


def initialize(context):  # context CONTAINS STATE OF ALGO'
    context.aapl = sid(24)  # 24 SECURITY ID (FOR AAPL STOCK)
    context.csco = sid(1900)  # FOR CISCO
    context.amzn - sid(16841)  # FOR AMAZON
    
def handle_data(context, data):  # data STORES API FUNCTIONS
    # THIS FUNCT IS CALLED ONCE EVERY MINUTE
    order_target_percent(context.aapl, 0.27)
    # THIS FUNCTION PLACES AN order TO ADJUST TO 27% OF CURRENT PORTFOLIO VALUE OF .aapl ASSET
    order_target_percent(context.csco, 0.20)
    order_target_percent(context.amzn, 0.53)
    


# In[ ]:


# #  OR, YOU CAN DEFINE THE FUNCTIONS LIKE THIS

def initialize(context):  # context CONTAINS STATE OF ALGO'
    context.techies = [sid(24), sid(1900), sid(16841)]
    conext.aapl = context.techies[0]
    
    
def handle_data(context, data):  # data STORES API FUNCTIONS
    # THIS FUNCT IS CALLED ONCE EVERY MINUTE
    tech_close = data.current(context.techies, 'close')
    print(type(tech_close))  # PRINTS IT AS A DATAFRAME
    print(tech_close)
    # THEREFORE, EVERY MINUTE, THIS FUNCTION GRABS CURRENT 'close' PRICE OF ALL THE 
    # ASSETS WITHIN THE .techies ARRAY, & print()s THEM ALL OUT
    
    if data.is_stale(context.aapl):  # RETURNS TRUE IF THIS DATA IS CURRENT FOR THE PARAM ASSET
        print("APPLE DATA IS STALE")
        
    if data.can_trade(context.aapl):  # RETURNS TRUE IF THIS ASSET IS LISTED ON THE CONNECTED STOCK EXCHANGE
        print("APPLE CAN BE TRADED ON ..")
        order_target_percent(context.aapl, 0.25)
    
    # GET THE PRICE HISTORY OF AN ASSET  frequency='1d' (daily) or '1m' (minutely)
    price_history = data.history(context.techies, fields='price', bar_count=5, frequency='1d')
    print(price_history)  # DATAFRAME DATATYPE
    
    
    
# SCHEDULING FUNCTIONS
def initialize(context):  
    # date_rules. every_day/week_start/week_end/month_start/month_end
    # time_rules. market_open/market_close
    schedule_function(open_positions, date_rules.week_start(), time_rules.market_open())
    schedule_function(close_positions, date_rules.week_end(), time_rules.market_close(minutes=30))
    
def open_positions(context, data):
    order_target_percent(context.aapl, 0.10)
    
def close_positions(context, data):
    order_target_percent(context.aapl, 0)?
def rebalance(context, data):
    order_target_percent(context.amzn, 0.5)
    order_target_percent(context.ibm, -0.5)
    
# FUNCTION TO RECORD DATA INTO AN ALIAS    
def record_vars(context, data):
    # ALIAS CAN BE UR CUSTOM NAME eg. 'amzn_close' / 'xyz'
    record(amzn_close=data.current(context.amzn, 'close'))
    record(ibm_close=data.current(context.ibm, 'close'))



#############  BASIC TRADING STRATEGIES

start, end = '2015-01-01', '2017-01-01'

""" 
PRACTISING THE PAIRS TRADING STRATEGY
WHERE U TAKE 2 ASSETS WHICH ARE CLOSELY CORRELATED (eg. 2 AIRLINE COMPANIES)
& THEN TRADE BASED ON THEIR PERCULIAR SIMILARITIES
"""
# GET DATA FOR united & american AIRLINE COMPANIES
united = get_pricing('UAL', start_date=start, end_date=end)
american = get_pricing('AAL', start_date=start, end_date=end)
#
united.head()
american.head()
united['close_price'].plot(label='UU', figsize=(12, 8))
american['close_price'].plot(label='AA')
plt.legend()
#
np.corrcoef(american['close_price'], united['close_price'])

# WE CAN NOW ESTABLISH FROM THE CORRELATION MATRIX THAT THE 2 ASSETS ARE HIGHLY CORRELATED :)
spread = american['close_price'] - united['close_price']
spread.plot(label='Spread', figsize=(12, 8))
plt.axhline(spread.mean(), c='r')

def zscore(stocks):  # CALC'ING THE Z-SCORE (OR MAYBE VaR) OF A PORTFOLIO
    return (stocks - stocks.mean()) / np.std(stocks)

zscore(spread).plot(figsize=(14, 8))  # NOW, CALC THE Z-SCORE OF THE SPREAD
plt.axhline(zscore(spread).mean(), color='black')
plt.axhline(1.0, c='g', ls='--')
plt.axhline(-1.0, c='r', ls='--')
plt.legend()

spread_mavgq = spread.rolling(1).mean()
spread_mavg30 = spread.rolling(30).mean()
std_30 = spread.rolling(30).std()
zscore_30_1 = (spread_mavg1 - spread_mavg30) / std_30
zscore_30_1.plot(figsize=(12, 8), label='Rolling 30 Day Z score')
plt.axhline(0, color='black')
plt.axhline(1.0, color='red', ls='--')


# ANOTHER BASIC STRATEGY

# initialize - schedule function
def initialize(context):  # RUNS check_pairs() 60mins BEFORE MARKET CLOSES
    schedule_function(check_pairs, date_rules.every_day(), time_rules.market_close(minutes=60))
    #
    context.aal = sid(45971)  # AAL - AMERICAN CONTINENTAL HOLDINGS
    context.ual = sid(28051)  # UAL - UNITED CONTINENTAL HOLDINGS
    #
    context.long_on_spread = False
    context.shorting_spread = False


# check_pairs
def check_pairs(context, data):
    aal, ual = context.aal, context.ual
    prices = data.history([aal, ual], 'price', 30, '1d')
    short_prices = prices.iloc[-1:]  # TODAY'S PRICES
    # MEAN (SPREAD) & STANDARD DEVIATION OF SPREAD OVER 30 DAYS
    mavg_30 = np.mean(prices[aal] - prices[ual])
    std_30 = np.std(prices[aal] - prices[ual])
    # CURRENT SPREAD
    mavg_1 = np.mean(short_prices[aal] - short_prices[ual])
    #     std_1 = np.std(short_prices[aal] - short_prices[ual])

    if std_30 > 0:  # CALCULATE Z-SCORE 1ST ..
        zscore = (mavg_1 - mavg_30) / std_30
        if zscore > 0.5 and not context.shorting_spread:
            # SPREAD = AMERICAN AIRLINES PRICE - UNITED AIRLINES PRICE
            order_target_percent(aal, -0.5)  # SHORT
            order_target_percent(ual, 0.5)  # LONG
            context.shorting_spread = True
            context.long_on_spread = False
        elif zscore < 1.0 and not context.long_on_spread:
            order_target_percent(aal, 0.5)  # LONG
            order_target_percent(ual, -0.5)  # SHORT
            context.shorting_spread = False
            context.long_on_spread = True
        elif abs(zscore) < 0.1:
            order_target_percent(aal, 0)
            order_target_percent(ual, 0)
            context.shorting_spread = False
            context.long_on_spread = False
        else:
            pass

        record(Z_score=zscore)


########
# BOLLINGER BANDS TRADING STRATEGY

start, end = '2015-01-01', '2017-01-01'

def initialize(context):
    context.jj = sid(
        4151)  # OR MAYBE THIS? -> get_pricing('J&J', start_date=start, end_date=end, frequency="daily")
    schedule_function(check_bands, date_rules.every_day())

def handle_data(context, data):
    pass

def check_bands(context, data):
    current_price = data.current(context.jj, 'price')
    prices = data.history(context.jj, 'price', 20, frequency='1d')
    avg, std = prices.mean(), prices.std()
    lower_band, upper_band = (avg - (2 * std)), (avg + (2 * std))

    if current_price <= lower_band:
        order_target_percent(context.jj, 1.0)
        print("BUYING JJ STOCK")
    elif current_price >= upper_band:
        order_target_percent(context.jj, -1.0)
        print("SHORTING JJ STOCK")
    else:
        pass
    #
    record(upper=upper_band, lower=lower_band, mavg_20=avg, price=current_price)



######################################
# USING QUANTOPIAN PIPELINES (GOOD FOR ALGORITHMS THAT FOLLOW A SET STRUCTURE)
# YOU CAN USE "CLASSIFIERS" & "FACTORS" TO FURTHER FILTER OUT LARGE DATASETS COMING IN THROUGH A PIPELINE

from quantopian.pipeline import Pipeline
from quantopian.research import run_pipeline

from quantopian.pipeline.data.builtin import USEquityPricing
# CONTAINS PRICE INFORMATION FOR A LOT OF EQUITIES
from quantopian.pipeline.factors import SimpleMovingAverage

# SAMPLE FACTOR
mean_close_30 = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=30)
mean_close_10 = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=10)
latest_close = USEquityPricing.close.latest
# COMBINING SOME 2 FACTORS
percent_diff = (mean_close_10 - mean_close_30) / mean_close_30

# NOW, USING FILTERS (RETURN BOOL) & SCREENS (EXECS FILTERS) TO SLICE DATA FROM PIPELINE, BASED ON FACTORS
percent_diff_filter = percent_diff > 0
small_price = latest_close < 5
# COMBINING FILTERS NOW ..
combo_filter = perc_filter & small_price  # OR '|' NOT 'and' / 'or' KEYWORDS

def make_pipeline():
    return Pipeline(columns={
        '30 Day Mean Close': mean_close_30,
        'Percent Diff': percent_diff,
        'Latest Close': latest_close,
        'Percent Diff Filter': percent_diff_filter
    },  # OR USE COMBINATION FILTER combo_filter
        screen=percent_diff_filter)  # OR '~percent_diff_filter' FOR OPPOSITE OF THE FILTER

pipe = make_pipeline()  # start_date == end_date, THEREFORE THE SAME DAY
result = run_pipeline(pipe, start_date='2017-01-03', end_date='2017-01-03')
result.head()

# NOW, USING MASKS
latest_close = USEquityPricing.close.latest
small_price = latest_close < 5
mean_close_30 = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=30, mask=small_price)
mean_close_10 = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=10, mask=small_price)
# mask=val APPLIES A FILTER 1ST BEFORE CALC'ING THIS SMA FACTOR (SAVES COMPUTATIONAL COMPLEXITY)
# NOW USE FACTORS (mean_close_30, mean_close_10) HOWEVER


# NOW, USING CLASSIFIERS
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.classifiers.morningstar import Sector

ms_sector = Sector()  # A LOT OF PROPERTIES OF ms_sector ARE CLASSIFIERS
exchange_classifier = ms_sector.share_class_reference.exchange_id.latest
nyse_filter = exchange_classifier.eq('NYS')  # OR .isnull() / startswith()
# NOW, USE nyse_filter WITHIN pipe = Pipeline(..., screen=nyse_filter)


