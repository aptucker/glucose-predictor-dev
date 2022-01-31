# Written and designed by Aaron Tucker 
#
#
#
# ----------------------------------------------------------------------------
"""
Comparison analysis file pulls results from patient trainings; includes plotting
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from scipy import stats
import statsmodels as sm
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
    
# %% Time Series Analysis

# cPlots.statisticalEvalPlot(l1.DayData[1], r1.DayData[1], 1)

# for i in range(len(lPats)):
#     cPlots.statisticalEvalPlot(lPats[i].DayData[1].diff().iloc[1:], rPats[i].DayData[1].diff().iloc[1:], i+1)
for i in range(len(lPats)):
    cPlots.statisticalEvalPlot(lPats[i].DayData[1].iloc[1:], rPats[i].DayData[1].iloc[1:], i+1)
# %% Single patient error analysis
# Model Names
# JDST
# Sequential H=2
# Circadian 1
# Parallel
# Parallel H2
# Parallel Circadian

modelNames = ['JDST', 'Sequential H=2', 'Circadian 1', 'Parallel', 'Parallel H2', 'Parallel Circadian', 'GRU H=1']
index = [1,2,3,4]
labels = ['Left-Left', 'Right-Left', 'Right-Right', 'Left-Right']
modelDrops = ['Parallel H2']
# modelDrops = []

cPlots.singlePatientError(l1, r1, modelNames, labels, index, modelDrops, 1)
cPlots.singlePatientError(l2, r2, modelNames, labels, index, modelDrops, 2)
cPlots.singlePatientError(l3, r3, modelNames, labels, index, modelDrops, 3)
cPlots.singlePatientError(l4, r4, modelNames, labels, index, modelDrops, 4)
cPlots.singlePatientError(l5, r5, modelNames, labels, index, modelDrops, 5)
cPlots.singlePatientError(l6, r6, modelNames, labels, index, modelDrops, 6)
cPlots.singlePatientError(l7, r7, modelNames, labels, index, modelDrops, 7)
cPlots.singlePatientError(l8, r8, modelNames, labels, index, modelDrops, 8)
cPlots.singlePatientError(l9, r9, modelNames, labels, index, modelDrops, 9)
cPlots.singlePatientError(l10, r10, modelNames, labels, index, modelDrops, 10)
cPlots.singlePatientError(l11, r11, modelNames, labels, index, modelDrops, 11)
cPlots.singlePatientError(l12, r12, modelNames, labels, index, modelDrops, 12)
cPlots.singlePatientError(l13, r13, modelNames, labels, index, modelDrops, 13)

# %% Single Model Plots


# [e15, e30, e45, e60] = 
cPlots.modelEvalPlot(lPats, rPats, 'JDST', labels, index, patNames, False, "")
cPlots.modelEvalPlot(lPats, rPats, 'Sequential H=2', labels, index, patNames, False, "")
cPlots.modelEvalPlot(lPats, rPats, 'Circadian 1', labels, index, patNames, False, "")
cPlots.modelEvalPlot(lPats, rPats, 'Parallel', labels, index, patNames, False, "")
cPlots.modelEvalPlot(lPats, rPats, 'Parallel Circadian', labels, index, patNames, False, "")
cPlots.modelEvalPlot(lPats, rPats, 'GRU H=1', labels, index, patNames, True, 'C:\Code\glucose-predictor-dev\GRUError.pdf')

# %% Export to Excel

phLabels = [15, None, None, None,
            30, None, None, None,
            45, None, None, None,
            60, None, None, None]
compLabels = ['Left-Left', 'Right-Left', 'Right-Right', 'Left-Right',
              'Left-Left', 'Right-Left', 'Right-Right', 'Left-Right',
              'Left-Left', 'Right-Left', 'Right-Right', 'Left-Right',
              'Left-Left', 'Right-Left', 'Right-Right', 'Left-Right']
plusMinusLabels = ['+/-', '+/-', '+/-', '+/-',
                   '+/-', '+/-', '+/-', '+/-',
                   '+/-', '+/-', '+/-', '+/-',
                   '+/-', '+/-', '+/-', '+/-']

rmseDF = pd.DataFrame(phLabels, columns=['Prediction Horizon (min)'])
rmseDF['Algorithm Setup (trained-tested)'] = compLabels


for e in range(len(lPats)):
    tempDFllRMSE = pd.DataFrame(np.mean(lPats[e].rmseStorage['GRU H=1']['llRMSE'], axis=0))
    tempDFrlRMSE = pd.DataFrame(np.mean(lPats[e].rmseStorage['GRU H=1']['rlRMSE'], axis=0))
    tempDFrrRMSE = pd.DataFrame(np.mean(rPats[e].rmseStorage['GRU H=1']['rrRMSE'], axis=0))
    tempDFlrRMSE = pd.DataFrame(np.mean(rPats[e].rmseStorage['GRU H=1']['lrRMSE'], axis=0))
    
    tempDFllRMSEstd = pd.DataFrame(np.std(lPats[e].rmseStorage['GRU H=1']['llRMSE'], axis=0))
    tempDFrlRMSEstd = pd.DataFrame(np.std(lPats[e].rmseStorage['GRU H=1']['rlRMSE'], axis=0))
    tempDFrrRMSEstd = pd.DataFrame(np.std(rPats[e].rmseStorage['GRU H=1']['rrRMSE'], axis=0))
    tempDFlrRMSEstd = pd.DataFrame(np.std(rPats[e].rmseStorage['GRU H=1']['lrRMSE'], axis=0))
    
    patDFmean = pd.DataFrame()
    patDFstd = pd.DataFrame()
    for i in range(3, -1, -1):
        # patDF = pd.DataFrame([tempDFllRMSE.iloc[i],
        #                       tempDFrlRMSE.iloc[i],
        #                       tempDFrrRMSE.iloc[i],
        #                       tempDFlrRMSE.iloc[i]])
        patDFmean = patDFmean.append([tempDFllRMSE.iloc[i],
                              tempDFrlRMSE.iloc[i],
                              tempDFrrRMSE.iloc[i],
                              tempDFlrRMSE.iloc[i]])
        patDFstd = patDFstd.append([tempDFllRMSEstd.iloc[i],
                              tempDFrlRMSEstd.iloc[i],
                              tempDFrrRMSEstd.iloc[i],
                              tempDFlrRMSEstd.iloc[i]])
    
    patDFmean = patDFmean.reset_index(drop=True)
    patDFstd = patDFstd.reset_index(drop=True)
    
    rmseDF['Patient ' f'{e+1} ' 'RMSE (mg/dL)'] = patDFmean
    rmseDF['Patient ' f'{e+1}'] = plusMinusLabels
    rmseDF['Patient ' f'{e+1} ' 'STD'] = patDFstd
    
rmseDF.to_excel("G:\My Drive\Minnesota Files\Erdman Research\Final Paper\python_tables.xlsx", sheet_name='Raw_Python_Data', index=False)    
    

# %% Plot Testing
