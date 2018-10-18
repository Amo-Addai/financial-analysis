import datetime
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as plt
from matplotlib.finance import candlestick_ohlc
from matplotlib.dates import DateFormatter, date2num, WeekdayLocator, DayLocator, MONDAY
import pandas_datareader
import pandas_datareader.data as web
from pandas.plotting import scatter_matrix
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.graphics.api as smg
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import autocorrelation_plot

start = datetime.datetime(2012, 1, 1)
end = datetime.datetime(2017, 1, 1)

tesla = web.DataReader('TSLA', 'google', start, end)
tesla.head()  # VIEW TESLA FINANCIAL DATA IN THE CMD LINE

ford = web.DataReader('F', 'google', start, end)
ford.head()  # VIEW FORD FINANCIAL DATA IN THE CMD LINE

gm = web.DataReader('GM', 'google', start, end)
gm.head()  # VIEW GENERAL MOTORS FINANCIAL DATA IN THE CMD LINE

#   FIRST, PLOT ALL THE OPENING PRICES
tesla['Open'].plot(label='Tesla', figsize=(12,8), title='Opening Prices')
gm['Open'].plot(label='GM'); ford['Open'].plot(label='Ford')
plt.legend()

#   NOW, YOU CAN ALSO PLOT THE VOLUME OF THE ASSET TRADED
tesla['Volume'].plot(label='Tesla', figsize=(12,8), title='Volumeing Traded')
gm['Volume'].plot(label='GM'); ford['Volume'].plot(label='Ford')
plt.legend()

ford['Volume'].max()  # RETURNS THE MAX AMOUNT OF VOLUME TRADED
ford['Volume'].argmax()  # RETURNS THE DATE (INDEX COLUMN) ON WHICH MAX AMOUNT OF VOLUME WAS TRADED

tesla['Total Traded'] = tesla['Open']*tesla['Volume']
tesla['Total Traded'].plot(label='Tesla', figsize=(16, 8))
plt.lengend()
tesla['Total Traded'].argmax()

#   NOW, PLOTTING OUT SOME TRADE INDICATORS (eg. SMA, BOLLINGER BANDS, etc)
gm['MA50'] = gm['Open'].rolling(50).mean()
gm['UPPER-BB50'] = 2*gm['MA50'] + gm['Open'].rolling(50).std()
gm['LOWER-BB50'] = 2*gm['MA50'] - gm['Open'].rolling(50).std()
#   NOW, PLOT THEM ALL
gm[['Open', 'MA50', 'UPPER-BB50', 'LOWER-BB50']].plot(figsize=(16,8))

#   NOW, TRY CONCATENATING ALL THE OPEN PRICES OF THE 3 DIFFERENT ASSETS
car_comp = pd.concat([tesla["Open"], gm["Open"], ford["Open"]], axis=1)
car_comp.columns = ['Tesla Open', 'GM Open', 'Ford Open']
#   DON'T FORGET TO REST THE COLUMNS NAMES, COZ ALL THE CONCAT'D COLUMNS ARE CALLED 'Open'
car_comp.head()

#   NOW, CREATE A SCATTER PLOT MATRIX OF THE CONCAT'D DATAFRAME (THEREFORE, ALL THE OPEN PRICES OF THE 3 DIFFERENT ASSETS)
scatter_matrix(car_comp, figsize=(8,8), alpha=0.2, hist_kwds={'bins':50})

#   SINCE THE DATE COLUMN IS THE ID, THIS GRABS ALL ITEMS ON DATE 'O1/2012' (JAN. 2012)
#   THEN RESETS THEIR INDICES
ford_reset = ford.loc['2012-01'].reset_index()
ford_reset.info()

#   SETUP ANOTHER COLUMN FOR DATE-AXIS AS A NUMERICAL VALUE (NOT A DATETIME OBJECT LIKE 'Date' COLUMN)
#   THIS 'date_ax' COLUMN IS GIVEN TO matplotlib (COZ IT'S NOT REALLY GOOD WITH WORKING WITH DATETIME INDICES
ford_reset['date_ax'] = ford_reset['Date'].apply(lambda date: date2num(date))
ford_reset.head()

list_of_cols = ['date_ax', 'Open', 'High', 'Low', 'Close']
ford_values = [ tuple(vals) for vals in ford_reset[list_of_cols].values ]
#   SO NOW, ford_values WILL BE A LIST OF TUPLES OF THE .values OF THE 5 COLUMNS SPECIFIED


def showCandleStickChart(values):
    mondays = WeekdayLocator(MONDAY)
    alldays = DayLocator()
    weekFormatter = DateFormatter('%b %d')
    dayFormatter = DateFormatter('%d')
    #
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    ax.xaxis.set_major_locator(mondays)
    ax.xaxis.set_minor_locator(alldays)
    ax.xaxis.set_major_formatter(weekFormatter)
    #
    candlestick_ohlc(ax, values, width=0.6, colorup='g', colordown='r')
    print("CANDLESICK PLOTTED :)")

#   NOW, SHOW A CANDLE STICK CHART FOR AN ASSET
showCandleStickChart(ford_values)


##################################################################################
###     NOW, TIME FOR SOME BASIC FINANCIAL ANALYSIS CALCULATIONS

#   DAILY % CHANGE (RETURNS) : rt = (pt/(ptminus1)) - 1
#   YOU CAN PERFORM THIS CALC WITH .shift(1) METH FOR ptminus1,
#   OR YOU CAN USE THE IN-BUILT PANDAS %-CHANGE METHOD pct_change()
tesla['returns'] = (tesla['Close']/(tesla['Close'].shift(1))) - 1
tesla['returns'] = tesla['Close'].pct_change(1)  # OR, JUST CALL THIS :)
ford['returns'] = ford['Close'].pct_change(1)
gm['returns'] = gm['Close'].pct_change(1)

#   NOW, LET'S PLOT A HISTOGRAM TO COMPARE RETURNS OF THE ASSETS
ford['returns'].hist(bins=100, label='Ford', figsize=(10,8), alpha=0.5)  # LOOKS LIKE A NORMAL DISTRIBUTION
gm['returns'].hist(bins=100, label='GM', figsize=(10,8), alpha=0.5)
tesla['returns'].hist(bins=100, label='Tesla', figsize=(10,8), alpha=0.5)
plt.legend()

#   NOW, PLOT A kde (KERNEL DENSITY ESTIMATION) GRAPH
ford['returns'].plot(kind='kde', label='Ford', figsize=(10,8), alpha=0.5)  # LOOKS LIKE A NORMAL DISTRIBUTION
gm['returns'].plot(kind='kde', label='GM', figsize=(10,8), alpha=0.5)
tesla['returns'].plot(kind='kde', label='Tesla', figsize=(10,8), alpha=0.5)
plt.legend()


#   NOW, TO USE BOX-PLOTS TO COMPARE THE RETURNS OF THE 3 ASSETS
#   WITH BOXPLOTS, YOU MUST PUT ALL THE RETURN COLUMNS INTO 1 DATAFRAME
box_df = pd.concat([tesla['returns'], ford['returns'], gm['returns']], axis=1)
box_df.columns = ['Tesla Returns', 'Ford Returns', 'GM Returns']
box_df.plot(kind='box', figsize=(8, 11))

#   NOW, WITH THE SCATTER MATIX FUNCTION
scatter_matrix(box_df, figsize=(8,8), alpha=0.2, hist_kwds={'bins:100'})
#   THIS IS JUST TO WRITE THE CORRECT COLUMN NAMES WITHIN THE SCATTER PLOT
box_df.plot(kind='scatter', x='Ford Returns', y='GM Returns', alpha=0.5, figsize=(8,8))


##  NOW, CALC'ING THE CUMULATIVE RETURNS
tesla['Cumulative Return'] = ( 1 + tesla['returns'] ).cumprod()
ford['Cumulative Return'] = ( 1 + ford['returns'] ).cumprod()
gm['Cumulative Return'] = ( 1 + gm['returns'] ).cumprod()

#   NOW, PLOT :)
tesla['Cumulative Return'].plot(label='Tesla', figsize=(16,8))
ford['Cumulative Return'].plot(label='ford', figsize=(16,8))
gm['Cumulative Return'].plot(label='GM', figsize=(16,8))
plt.legend()

#   WORKING WITH TIME-SERIES DATA
df = sm.datasets.macrodata.load_pandas().data  # LOAD A SAMPLE DATASET AS A DATAFRAME
print(sm.datasets.macrodata.NOTE)  # PRINTS OUT DESCRIPTION OF THIS DATASET
i = sm.tsa.datetools.dates_from_range('1959Q1', '2009Q3')  # 1ST QUARTER OF 1959 - 3RD QUARTER OF 2009
df.index = pd.Index(i)  # COZ i IS A DATE-RANGE, THIS WILL BE A DATETIME INDEX
gdp_cycle, gdp_trend = sm.tsa.filters.hpfilter(df['realgdp']) # RETURNS A TUPLE OF PANDAS SERIES OBJECTS
# hpfilter() - Hodrick-Prescott filter, which separates Timeseries y(t) into Trend T(t) and Cyclical Component E(t)

# THE CODE BELOW IS JUST SOME PREPROCESSING STUFF
airline = pd.read_csv('airline_passengers.csv', index_col="Month")
airline.dropna(inplace=True)
airline.index = pd.to_datetime(airline.index)

# EWMA MODELS: Exponential Weighted Moving Average
airline['6-month-SMA'] = airline['passengers'].rolling(window=6).mean()
airline['12-month-SMA'] = airline['passengers'].rolling(window=12).mean()
airline[['passengers', '6-month-SMA', '12-month-SMA']].plot(figsize=(10, 8)) # PLOTTING OUT A SIMPLE MOVING AVERAGE

# BUT SMA ALWAYS LAGS AT THE BEGINNING WITH THE SIZE OF THE WINDOW
# IT ALSO NEVER REACHES THE FULL PEAK/VALLEY DUE TO AVERAGING
# HOWEVER, THE EWMA LETS THE MOST RECENT POINTS HAVE MORE WEIGHT & FIXES THESE ISSUES
airline['EWMA-12'] = airline['passengers'].ewm(span=12).mean()
airline[['passengers', 'EWMA-12']].plot(figsize=(10, 8)) # PLOTTING OUT A EXPO-WEIGHTED MOVING AVERAGE

# ETS MODELS: Error-Trend-Seasonality
result = seasonal_decompose(airline['passengers'], model="multiplicative")  # ETS DECOMPOSES THIS SERIES
# model="additive" IF THE TREND OF THE SERIES LOOKS MORE LINEAR, THAN EXPONENTIAL
print(result.seasonal); result.seasonal.plot(); result.trend.plot()
result.plot()  # THIS PLOTS ALL THE E-T-S COMPONENTS (ABOVE) SEPARATELY


# ARIMA MODELS: AUTOREGRESSIVE INTEGRATED MOVING AVERAGES (GENERALIZATION OF ARMA MODEL)

# 1. Visualize the Time Series data
df = pd.read_csv('monthly-milk-production-pounds-p.csv')
df.head(); df.columns = ['Month', 'Milk in Pounds per Cow']; df.drop(168, axis=0, inplace=True); df.tail()
df['Month'] = pd.to_datetime(df['Month']); df.set_index('Month', inplace=True)  # CONVERT INDEX TO DATTIME INDEX
df.describe().transpose(); df.plot()
ts = df['Milk in Pounds per Cow']  # AFTER PLOTTING, WE REALIZE THAT DATA IS SEASONAL
# JUST BASIC VISUALIZATION WITHIN THE CONSOLE (REGULAR SMA PLOTS FOR NOW)
ts.rolling(window=12).mean().plot(label='12 Month Rolling Mean');
ts.rolling(window=12).std().plot(label='12 Month Rolling Std');
ts.plot(); plt.legend()  # DATA WAS SEASONAL IN YEARS (THEREFORE 12 ROLLING FOR 12 MONTHS)
# NOW, TESTING THE EWMA PLOTS - MOST RECENT POINTS HAVE MORE WEIGHT & FIXES THESE ISSUES
ts_ewma = ts.ewm(span=12).mean().plot(figsize=(10, 8))
# NOW ALSO, ETS DECOMPOSITION (WITH frequency=12 MONTHS)
ts_ets = seasonal_decompose(ts, model="additive", frequency=12)  # OR: model="multiplicative"
print(ts_ets); fig = ts_ets.plot(); fig.set_size_inches(15, 8)  # THIS PLOTS ALL THE E-T-S COMPONENTS SEPARATELY

# 2. Make the Time Series data stationary
# USING THE AUGMENTED DICKEY-FULLER TEST TO TEST STATIONARITY
def adf_check(ts):  # TESTS FOR STATIONARITY OF THE TIME SERIES
    result = adfuller(ts)
    print("Augmented Dicker-Fuller Test")
    labels = ['ADF Test Statistic', 'p-value', '# of Lags', '# of Observations used']
    for value, label in zip(result, labels):
        print("{} : {}".format(label, str(value)))
    test = result[1] <= 0.05  # result[1] IS 2nd ELEM -> 'p-value' (i think :)
    if test:
        print("Strong evidence against null hypothesis")
        print("Reject null hypothesis")
        print("Data has no unit root, and is stationary")
    else:
        print("Weak evidence against null hypothesis")
        print("Fail to reject null hypothesis")
        print("Data has a unit root, it is non-stationary")
    return test
# NOW, CALL YOUR FUNCTION :)
result = adf_check(ts); diff1, diff2 = None, None
if not result:  # DATA IS NON-STATIONARY, SO WE CAN START WITH DIFFERENCING, UNTIL IT BECOMES STATIONARY
    diff1 = ts - ts.shift(1); diff1.plot()
    res = adf_check(diff1.dropna())  # NOW, TEST THE 1ST DIFFERENCE (BUT DROP NAN VALUES COZ OF THE SHIFTING)
    if not res:  # IN CASE, IT'S STILL NOT STATIONARY, YOU CAN TAKE A 2ND DIFFERENCE
        diff2 = diff1 - diff1.shift(1); adf_check(diff2.dropna())
# YOU CAN ENHANCE THE ALGO' ABOVE A LOT MORE
# OR YOU CAN ALSO TAKE THE SEASONAL DIFFERECNE INSTEAD (FOR SEASONAL ARIMA)
seasonal_diff = ts - ts.shift(12); seasonal_diff.plot(); result = adf_check(seasonal_diff.dropna())
# NOW, TRY CREATING  A SEASONAL 1ST DIFFERENCE (1ST DIFF, THEN SEASONAL DIFF OF 1ST DIFF)
seasonal_diff1 = diff1 - diff1.shift(12); seasonal_diff1.plot(); adf_check(seasonal_diff1.dropna())

# 3. Plot the Correlation and AutoCorrelation Charts (ACF & PACF charts)
# YOU CAN EITHER GET THE 'Gradual Decline' OR 'Sharp Drop-off' PLOTS HERE
acf_diff1_plot = plot_acf(diff1.dropna())  # GRADUAL DECLINE
acf_seasonal_diff1_plot = plot_acf(seasonal_diff1.dropna())  # SHARP DROP-OFF
# PANDAS ALSO HAS THESE PLOTS IN-BUILT IN AUTO-CORRELATIONS (NOT PARTIAL AUTO-CORRELATION DOE :)
autocorrelation_plot(seasonal_diff1.dropna())  # LIKE THIS
# NOW, FOR THE PARTIAL AUTO-CORRELATION PLOT
pacf_seasonal_diff1_plot = plot_pacf(seasonal_diff1.dropna())  # SHARP DROP-OFF

# 4. Construct the ARIMA (or Seasonal ARIMA) Model (using p, d, q, P, D, Q parameters)



# 5. Use ARIMA model to make predictions / forecasts




