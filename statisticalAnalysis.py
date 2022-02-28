# Written and designed by Aaron Tucker 
#
#
#
# ----------------------------------------------------------------------------
"""
Statistical Analysis File
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from scipy import stats
import statsmodels as sm
import statsmodels.graphics.tsaplots as smgraphs
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import matplotlib.font_manager as font_manager
from matplotlib.collections import PolyCollection, LineCollection

import patient as pat
import customLayers as cLayers
import customModels as cModels
import training as trn
import customPlots as cPlots
import customStats as cStats

with open('results\\patient1_analysis.pickle', 'rb') as f:
    l1, r1 = pickle.load(f)

with open('results\\patient2_analysis.pickle', 'rb') as f:
    l2, r2 = pickle.load(f)
    
with open('results\\patient3_analysis.pickle', 'rb') as f:
    l3, r3 = pickle.load(f)
    
with open('results\\patient4_analysis.pickle', 'rb') as f:
    l4, r4 = pickle.load(f)

with open('results\\patient5_analysis.pickle', 'rb') as f:
    l5, r5 = pickle.load(f)

with open('results\\patient6_analysis.pickle', 'rb') as f:
    l6, r6 = pickle.load(f)

with open('results\\patient7_analysis.pickle', 'rb') as f:
    l7, r7 = pickle.load(f)
    
with open('results\\patient8_analysis.pickle', 'rb') as f:
    l8, r8 = pickle.load(f)
    
with open('results\\patient9_analysis.pickle', 'rb') as f:
    l9, r9 = pickle.load(f)
    
with open('results\\patient10_analysis.pickle', 'rb') as f:
    l10, r10 = pickle.load(f)
    
with open('results\\patient11_analysis.pickle', 'rb') as f:
    l11, r11 = pickle.load(f)
    
with open('results\\patient12_analysis.pickle', 'rb') as f:
    l12, r12 = pickle.load(f)

with open('results\\patient13_analysis.pickle', 'rb') as f:
    l13, r13 = pickle.load(f)
    
lPats = [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13]
rPats = [r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13]
# patNames = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
patNames = ['Patient' f' {i}' for i in range(1, 14)]

# %% Example plots
from random import gauss
from random import seed
from pandas import Series

series = [gauss(0.0, 1.0)+0.01*i for i in range(1000)]
series = Series(series)
axWN = series.plot(color='k')
axWN.set_xlabel('Generated Time')
axWN.set_ylabel('Generated Data with Trend')
axWN.set_title('Non-stationry Generated Data')

plt.savefig('ts_plots\\ExampleTrendPlot.pdf')

fig, ax = plt.subplots(1,1)
smgraphs.plot_acf(series, ax, color='k')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation Function')
plt.title('Autocorrelation of Generated Data with Trend')

for item in ax.collections:
    #change the color of the CI 
        if type(item)==PolyCollection:
            item.set_facecolor('black')
        #change the color of the vertical lines
        if type(item)==LineCollection:
            item.set_color('black')    
            
plt.savefig('ts_plots\\ExampleTrendACF.pdf')

cPlots.ccfPlot(l1.GenData.loc["2018-10-14"],
               r1.GenData.loc["2018-10-14"],
               20,
               1,
               'ts_plots\\Pat1CCF.pdf',
               True)

cPlots.ccfPlot(l2.GenData.loc["2018-10-22"],
               r2.GenData.loc["2018-10-22"],
               20,
               2,
               'ts_plots\\Pat2CCF.pdf',
               True)

cPlots.ccfPlot(l3.GenData.loc["2018-11-10"],
               r3.GenData.loc["2018-11-10"],
               20,
               3,
               'ts_plots\\Pat3CCF.pdf',
               True)

cPlots.ccfPlot(l4.GenData.loc["2018-11-15"],
               r4.GenData.loc["2018-11-15"],
               20,
               4,
               'ts_plots\\Pat4CCF.pdf',
               True)

cPlots.ccfPlot(l5.GenData.loc["2018-12-10"],
               r5.GenData.loc["2018-12-10"],
               20,
               5,
               'ts_plots\\Pat5CCF.pdf',
               True)

cPlots.ccfPlot(l6.GenData.loc["2019-01-07"],
               r6.GenData.loc["2019-01-07"],
               20,
               6,
               'ts_plots\\Pat6CCF.pdf',
               True)

cPlots.ccfPlot(l7.GenData.loc["2019-05-27"],
               r7.GenData.loc["2019-05-27"],
               20,
               7,
               'ts_plots\\Pat7CCF.pdf',
               True)

cPlots.ccfPlot(l8.GenData.loc["2019-05-27"],
               r8.GenData.loc["2019-05-27"],
               20,
               8,
               'ts_plots\\Pat8CCF.pdf',
               True)

cPlots.ccfPlot(l9.GenData.loc["2019-06-11"],
               r9.GenData.loc["2019-06-11"],
               20,
               9,
               'ts_plots\\Pat9CCF.pdf',
               True)

cPlots.ccfPlot(l10.GenData.loc["2019-06-18"],
               r10.GenData.loc["2019-06-18"],
               20,
               10,
               'ts_plots\\Pat10CCF.pdf',
               True)

cPlots.ccfPlot(l11.GenData.loc["2019-07-08"],
               r11.GenData.loc["2019-07-08"],
               20,
               11,
               'ts_plots\\Pat11CCF.pdf',
               True)

cPlots.ccfPlot(l12.GenData.loc["2019-07-18"],
               r12.GenData.loc["2019-07-18"],
               20,
               12,
               'ts_plots\\Pat12CCF.pdf',
               True)

cPlots.ccfPlot(l13.GenData.loc["2019-11-03"],
               r13.GenData.loc["2019-11-03"],
               20,
               13,
               'ts_plots\\Pat13CCF.pdf',
               True)
# %% Mean, ACF, PACF Plots

cPlots.statisticalEvalPlot(l1.GenData,
                           r1.GenData,
                           l1.DayData[1].iloc[1:],
                           r1.DayData[1].iloc[1:],
                           1,
                           50,
                           True,
                           False,
                           'ts_plots\\Pat1TSAnalysis.pdf')
cPlots.statisticalEvalPlot(l1.GenData.diff(), 
                           r1.GenData.diff(), 
                           l1.DayData[1].diff().iloc[1:], 
                           r1.DayData[1].diff().iloc[1:], 
                           1, 
                           50, 
                           True,
                           True,
                           'ts_plots\\Pat1TSAnalysisDiff.pdf')

cPlots.statisticalEvalPlot(l6.GenData,
                           r6.GenData,
                           l6.DayData[23].iloc[1:],
                           r6.DayData[23].iloc[1:],
                           6,
                           50,
                           True,
                           False,
                           'ts_plots\\Pat6TSAnalysis.pdf')
cPlots.statisticalEvalPlot(l6.GenData.diff(), 
                           r6.GenData.diff(), 
                           l6.DayData[23].diff().iloc[1:], 
                           r6.DayData[23].diff().iloc[1:], 
                           6, 
                           50, 
                           True,
                           True,
                           'ts_plots\\Pat6TSAnalysisDiff.pdf')

# for i in range(len(lPats)):
#     cPlots.statisticalEvalPlot(lPats[i].DayData[1].diff().iloc[1:], rPats[i].DayData[1].diff().iloc[1:], i+1)
for i in range(len(lPats)):
    cPlots.statisticalEvalPlot(lPats[i].GenData,
                               rPats[i].GenData,
                               lPats[i].DayData[1].iloc[1:], 
                               rPats[i].DayData[1].iloc[1:], 
                               i+1, 
                               50, 
                               True, 
                               False,
                               'ts_plots\\Pat'f'{i+1}TSAnalysis.pdf')
    
    cPlots.statisticalEvalPlot(lPats[i].GenData.diff(),
                               rPats[i].GenData.diff(),
                               lPats[i].DayData[1].diff().iloc[1:], 
                               rPats[i].DayData[1].diff().iloc[1:], 
                               i+1, 
                               50, 
                               True, 
                               True,
                               'ts_plots\\Pat'f'{i+1}TSAnalysisDiff.pdf')
    
    
# %% ARMA Forecasting
# l1.GenData.iloc[1:200].plot()

# model = ARIMA(l1.GenData.iloc[1:200], order=(3, 0, 4))
# model_fit = model.fit()
# print(model_fit.summary())

# resids = pd.DataFrame(model_fit.resid)
# resids.plot()

# resids.plot(kind='kde')
# print(resids.describe())

    
X = l1.GenData.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
# walk-forward validation

for t in range(len(test)-4):
    # model = ARIMA(history, order=(3,0,4))
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit()
    # output = model_fit.forecast()
    # yhat = output[0]
    output = model_fit.forecast(steps=4)
    yhat = output
    predictions.append(yhat)
    obs = [test[t], test[t+1], test[t+2], test[t+3]]
    history.append(obs[0])
    
    if t % 10 == 0:
        print('predicted=%f, expected=%f' % (yhat[0], obs[0]))

predictions15 = list()
predictions30 = list()
predictions45 = list()
predictions60 = list()


for i in range(len(predictions)):
    predictions15.append(predictions[i][0])
    predictions30.append(predictions[i][1])
    predictions45.append(predictions[i][2])
    predictions60.append(predictions[i][3])

from math import sqrt
from sklearn.metrics import mean_squared_error
rmse15 = sqrt(mean_squared_error(test[0:len(predictions)], predictions15))
rmse30 = sqrt(mean_squared_error(test[1:len(predictions)+1], predictions30))
rmse45 = sqrt(mean_squared_error(test[2:len(predictions)+2], predictions45))
rmse60 = sqrt(mean_squared_error(test[3:len(predictions)+3], predictions60))
print('Test RMSE15: %.3f' % rmse15)
print('Test RMSE30: %.3f' % rmse30)
print('Test RMSE45: %.3f' % rmse45)
print('Test RMSE60: %.3f' % rmse60)