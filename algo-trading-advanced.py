from quantopian.pipeline import Pipeline
from quantopian.research import run_pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.filters.morningstar import Q1500US
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.classifiers.morningstar import Sector
from quantopian.pipeline.factors import SimpleMovingAverage, AverageDollarVolume
from quantopian.algorithm import attach_pipeline, pipeline_output



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



