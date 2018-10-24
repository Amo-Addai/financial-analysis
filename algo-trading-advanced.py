from quantopian.pipeline import Pipeline
from quantopian.research import run_pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.filters.morningstar import Q1500US
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.classifiers.morningstar import Sector
from quantopian.pipeline.factors import SimpleMovingAverage, AverageDollarVolume
from quantopian.algorithm import attach_pipeline, pipeline_output

import numpy as np
import matplotlib.pyplot as plt
from statsmodels import regression
from statsmodels.api import sm

universe = Q1500US()  # HAS ALMOST ALL STOCKS AVAILABLE IN THE WORLD
sector = morningstar.asset_classification.morningstar_sector_code.latest
# OR: sector = Sector()
#
energy_sector = sector.eq(309)  # Sector 309 for 'Energy' Sector
dollar_volume = AverageDollarVolume(window_length=30)  # 30-day window length
high_dollar_volume = dollar_volume.percentile_between(90, 100)  # TOP 10% (OR: .top(5) / .bottom(8(
#  MANIPULATING WITH FACTORS, FILTERS, MASKS, etc
open_prices, close_prices = USEquityPricing.open.latest, USEquityPricing.close.latest
top_open_prices = open_prices.top(50, mask=high_dollar_volume)  # USING A MASK
high_close_prices = close_prices.percentile_between(90, 100, mask=top_open_prices)


def initialize(context):
    schedule_function(rebalance, date_rules.week_start(), time_rules.market_open(hours=1))
    my_pipe = make_pipeline()
    attach_pipeline(my_pipe, 'my_pipeline')
    # LONG
    print(context.output)
    # SHORT


def rebalance(context, data):  #
    for security in context.portfolio.position:
        if (security not in context.longs) and (security not in context.shorts) \
                and (data.can_trade(security)):  # security ISN'T IN ANY OF OUR LONG / SHORT ASSET LISTS
            order_target_percent(security, 0)  # EXIT OUT OF THIS SECURITY'S POSITION
    for security in context.longs:
        if data.can_trade(security):
            order_target_percent(context, context.long_weight)
    for security in context.shorts:
        if data.can_trade(security):
            order_target_percent(context, context.short_weight)


def compute_weights(context):  # COMPUTE WEIGHTS (%s) FOR THE ASSETS IN THE PORTFOLIO
    long_weight = 0 if (len(context.longs) == 0) else (0.5 / len(context.longs))
    short_weight = 0 if (len(context.shorts) == 0) else (0.5 / len(context.shorts))
    return (long_weight, short_weight)


def before_trading_start(context, data):
    # CONNECT THE CREATED PIPELINE TO A PROP IN CONTEXT
    context.output = pipeline_output('my_pipeline')
    # LONG & SHORT
    context.longs = context.output[context.output['longs']].index.tolist()
    context.shorts = context.output[context.output['shorts']].index.tolist()
    # NOW, COMPUTE THE (LONG & SHORT) WEIGHTS FOR context OBJECT
    context.long_weight, context.short_weight = compute_weights(context)


def make_pipeline():
    # UNIVERSE Q1500US
    base_universe = Q1500US()
    # ENERGY SECTOR  (OR: sector = Sector())
    sector = morningstar.asset_classification.morningstar_sector_code.latest
    energy_sector = sector.eq(309)
    # MAKE MASK OF 1500US & ENERGY
    base_energy = base_universe & energy_sector
    # DOLLAR VOLUME (30 Days) GRAB THE INFO
    dollar_volume = AverageDollarVolume(window_length=30)
    # GRAB THE TOP 5% IN AVERAGE DOLLAR VOLUME
    high_dollar_volume = dollar_volume.percentile_between(95, 100)
    # COMBINE THE FILTERS
    top_five_base_energy = base_energy & high_dollar_volume
    # GRAB THE 10-day & 30-day MEAN CLOSE
    mean_10 = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=10, mast=top_five_base_energy)
    mean_30 = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=30, mast=top_five_base_energy)
    # PERCENT DIFFERNENCE
    percent_difference = (mean_10 - mean_30) / mean_30
    # LIST OF SHORTS & LONGS
    shorts, longs = (percent_difference < 0), (percent_difference > 0)
    # FINAL MASK/FILTER FOR ANYTHING IN SHORTS / LONGS
    securities_to_trade = shorts & longs
    # RETURN THE PIPELINE
    return Pipeline(columns={
        'longs': longs, 'shorts': shorts, 'percent_diff': percent_difference
    }, screen=securities_to_trade)


##  WORKING WITH LEVERAGE - BORROWING MONEY FROM BROKER TO INVEST FOR MORE RETURNS
set_max_leverage(1.05)  # SET MAXIMUM LEVERAGE
"""
THEREFORE -> order_target_percent(asset, 2.0) WILL FAIL COZ 2.0 (OR -2.0 SEF) > max. leverage 1.05
"""

def initialize(context):
    record(Leverage=context.account.leverage)  # Leverage = Gross Exposure / Net Liquidation
    record(Exposure=context.account.net_leverage)  # Exposure = Net Exposure / Net Liquidation
    #

    bt = get_backtest("hashcode")  # ALL BACKTESTS HAVE UNIQUE HASHCODES
    bt.recorded_vars['Leverage'].plot()  # RETRIEVING A RECORDED VARIABLE


##  HEDGING
"""
USING CAPM TO CALC BETA (MARKET RISK) & ALPHA (COEFFICIENT) OF A STRA'
THEN HEDGING AGAINST THE BETA, TO REDUCE RISK/EXPOSURE TO THE MARKET
STRA -  1. CHOOSE A STOCK eg. AAPL, & OBTAIN ITS ALPHA & BETA VALUES, 
        2. CALC A SHORT POSITION ON THE MARKET eg. SPY INDEX TO ELIMINATE
           THE ASSET'S EXPOSURE TO THE MARKET & TRADE ONLY ON THE ASSET'S ALPHA
"""

start, end = '2016-01-01', '2017-01-01'
asset = get_pricing('AAPL', fields='price', start_date=start, end_date=end)
benchmark = get_pricing('SPY', fields='price', start_date=start, end_date=end)
#
asset_ret = asset.pct_change(1)[1:]
benchmark_ret = benchmark.pct_change(1)[1:]
#   PLOT
asset_ret.plot();
benchmark_ret.plot();
plt.legend()
plt.scatter(benchmark_ret, asset_ret, alpha=0.6, s=50)
plt.xlabel('SPY Return');
plt.ylabel('AAPL Return')
# JUST GET THE VALUES (ACTUAL NUMBERS) WITHOUT THE DATETIME INDEX
AAPL, SPY = asset_ret.values, benchmark_ret.values
#
spy_constant = sm.add_constant(SPY)
model = regression.linear_model.OLS(AAPL, spy_constant).fit()
print("MODEL PARAMS (ALPHA & BETA) -> {}".format(model.params))
alpha, beta = model.params
#   GET MINIMUM & MAXIMUM NUMBERS OF THE BENCHMARK (SPY)
min_spy, max_spy = benchmark_ret.values.min(), benchmark_ret.values.max()
spy_line = np.linspace(min_spy, max_spy, 100)  # LINEARLY SPACED AMOUNT OF NUMBERS FROM min_spy TO max_spy
y = (spy_line * beta) / alpha
#   PLOT
plt.plot(spy_line, y, 'r')
plt.scatter(benchmark_ret, asset_ret, alpha=0.6, s=50)
plt.xlabel('SPY Return');
plt.ylabel('AAPL Return')

##  NOW, TO USE THE BETA & ALPHA VALUES HEDGE OUR STRATEGY
""" Negating the beta value to eventually cancel out any relationship with the market (SPY) """
hedged = (-1 * (beta * benchmark_ret)) + asset_ret
hedged.plot('AAPL WITH HEDGE')
asset_ret.plot(alpha=0.5); benchmark_ret.plot(alpha=0.5);
plt.xlim(['2016-06-01', '2016-08-01']); plt.legend()


def alpha_beta(benchmark_return, asset):
    benchmark = sm.add_constant(benchmark_return)
    model = regression.linear_model.OLS(asset, benchmark).fit()
    return (model.params[0], model.params[1])  # RETURNS alpha & beta VALUES


#   2016 DATA
start, end = '2016-01-01', '2017-01-01'
asset_2016 = get_pricing('AAPL', fields='price', start_date=start, end_date=end)
benchmark_2016 = get_pricing('SPY', fields='price', start_date=start, end_date=end)
#
asset_ret_2016 = asset_2016.pct_change(1)[1:]
benchmark_ret_2016 = benchmark_2016.pct_change(1)[1:]
#
asset_ret_values = asset_ret_2016.values
benchmark_ret_values = benchmark_ret_2016.values
#
alpha_2016, beta_2016 = alpha_beta(benchmark_ret_values, asset_ret_values)
print('2016 VALUES -> ALPHA ({}) & BETA ({})'.format(alpha_2016, beta_2016))
#   CREATE A HEDGED PORTFOLIO & COMPUTE ALPHA & BETA FROM THERE
portfolio = (-1 * (beta_2016 * benchmark_ret_2016)) + asset_ret_2016
#   THIS NEW beta WILL BE GREATLY REDUCED (CLOSE TO 0) COZ beta_2016 WAS NEGATED
alpha, beta = alpha_beta(benchmark_2016, portfolio)
print("PORTFOLIO ALPHA & BETA")
print("ALPHA ({}) & BETA ({})".format(alpha, beta))
portfolio.plot(alpha=0.9, label='AAPL WITH HEDGE')
asset_ret_2016.plot(alpha=0.5)
benchmark_ret_2016.plot(alpha=0.5)
plt.ylabel('DAILY RETURN'); plt.legend()
#
portfolio.mean(); asset_ret_2016.mean()
portfolio.std(); asset_ret_2016.std()
#








