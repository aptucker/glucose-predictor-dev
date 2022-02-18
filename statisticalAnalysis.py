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
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import matplotlib.font_manager as font_manager

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

# %% ARMA Forecasting
l1.GenData.iloc[1:200].plot()

model = ARIMA(l1.GenData.iloc[1:200], order=(3, 0, 4))
model_fit = model.fit()
print(model_fit.summary())

resids = pd.DataFrame(model_fit.resid)
resids.plot()

resids.plot(kind='kde')
print(resids.describe())

for i in range(200):
    model
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