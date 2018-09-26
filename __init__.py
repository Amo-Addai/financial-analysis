import numpy as np
import pandas as pd
import matplotlib as plt
from matplotlib.finance import candlestick_ohlc
from matplotlib.dates import DateFormatter, date2num, WeekdayLocator, DayLocator, MONDAY
import pandas_datareader
import pandas_datareader.data as web
from pandas.plotting import scatter_matrix
import datetime

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


