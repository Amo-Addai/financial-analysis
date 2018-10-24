import pyfolio as pf
import empyrical
import matplotlib.pyplot as plt
#
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.factos import AverageDollarVolume
from quantopian.pipeline.data.accern import alphaone_free   # FREE VERSION OF SENTIMENT ANALYSIS


###     PORTFOLIO ANALYSIS WITH quantopian's PyFolio

#   ACTUAL ASSET'S BACKTEST
bt = get_backtest("hashcode")
bt_returns = bt.daily_performance['returns']
bt_positions = bt.pyfolio_positions
bt_transactions = bt.pyfolio_transactions
#   BENCHMARK BACKTEST TOO
benchmark = get_backtest("another hashcode")
bn_returns = benchmark.daily_performance['returns']
bn_positions = benchmark.pyfolio_positions
bn_transactions = benchmark.pyfolio_transactions

# CALC'ING THE SHARPE RATIO
bt_sr = empyrical.sharpe_ratio(bt_returns)
bn_sr = empyrical.sharpe_ratio(bn_returns)
# CALC'ING BETA
beta = empyrical.beta(bt_returns, bn_returns)

#   PLOTTING WITH PyFolio (ALL TYPES OF PLOTS AVAILABLE - pf.plotting.type_of_plot())
# CUMULATIVE RETURNS
plt.subplot(2, 1, 1)
pf.plotting.plot_rolling_returns(bt_returns, bn_returns)
# DAILY (NON-CUMULATIVE) RETURNS
plt.subplot(2, 1, 2)
pf.plotting.plot_returns(bt_returns)
plt.tight_layout()

#
#   ACTUAL ALGORITHM TO BE BACKTESTED (WHILE THE CODE ABOVE DOES ANALYSIS, USING THE BACKTESTING RESULTS)
#   COMES WITH initialize() & shit


#
#   SENTIMENT ANALYSIS (USING NLP - NATURAL LANGUAGE PROCESSING)
#




def initialize(context):
    schedule_function(rebalance, date_rules.every_day())
    attach_pipeline(make_pipeline(), 'pipeline')


def make_pipeline():
    dollar_volume = AverageDollarVolume(window_length=20)
    # dollar_volume.rank(ascending=False) < 0
    is_liquid = dollar_volume.top(1000)
    #
    impact = alphaone_free.impact_score.latest  # FROM 0% TO 100%
    sentiment = alphaone_free.article_sentiment.latest  # FROM -1 TO +1
    #
    return Pipeline(columns={  # THEREFORE, SET THEM (impact & sentiment) AS COLUMNS OF THE df RETURNED BY THE PIPELINE
        'impact': impact, 'sentiment': sentiment
    }, screen=is_liquid)


def before_trading_start(context, data):
    port = pipeline_output('pipeline')
    context.longs = port[(port['impact'] == 100) and (port['sentiment'] > 0.5)].index.tolist()
    context.shorts = port[(port['impact'] == 100) and (port['sentiment'] < -0.5)].index.tolist()
    # 
    context.long_weight, context.short_weight = compute_weights(context)


def compute_weights(context):
    long_weight, short_weight = (0.5 / len(context.longs)), (-0.5 / len(context.shorts))
    return (long_weight, short_weight)


def rebalance(context, data):
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
