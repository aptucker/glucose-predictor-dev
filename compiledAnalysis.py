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
import pickle
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import matplotlib.font_manager as font_manager

import patient as pat
import customLayers as cLayers
import customModels as cModels
import training as trn

with open('results\\patient1_analysis.pickle', 'rb') as f:
    l1, r1 = pickle.load(f)


# %% Single patient error analysis
# Model Names
# JDST
# Sequential H=2
# Circadian 1
# Parallel
# Parallel H2
# Parallel Circadian

labels = ['Left-Left', 'Right-Left', 'Right-Right', 'Left-Right']
llMeans = []
rlMeans = []
rrMeans = []
lrMeans = []
for i in l1.rmseStorage:
    llMeans.append(np.mean(l1.rmseStorage[i]['llRMSE'], axis=0))
    rlMeans.append(np.mean(l1.rmseStorage[i]['rlRMSE'], axis=0))
    rrMeans.append(np.mean(r1.rmseStorage[i]['rrRMSE'], axis=0))
    lrMeans.append(np.mean(r1.rmseStorage[i]['lrRMSE'], axis=0))
    
llMeans = np.array(llMeans)
rlMeans = np.array(rlMeans)
rrMeans = np.array(rrMeans)
lrMeans = np.array(lrMeans)

# fig, ax = plt.subplots()
# jdstVals = ax.bar()

# %% JDST Dot Chart

#
# NEEDS ALL PATIENTS
#
#
#
#

# Comps = ["", "LEFT - LEFT", "RIGHT - LEFT", "RIGHT - RIGHT", "LEFT - RIGHT"]
# PatNum = 13

# PatLabs = ["", "PATIENT 1", "PATIENT 2", "PATIENT 3", "PATIENT 4", "PATIENT 5", "PATIENT 6", "PATIENT 7", "PATIENT 8", "PATIENT 9", "PATIENT 10", "PATIENT 11", "PATIENT 12", "PATIENT 13"]

# for i in range(PatNum):
#     Comps.extend(["LEFT - LEFT", "RIGHT - LEFT", "RIGHT - RIGHT", "LEFT - RIGHT"])

# PlotVals_15 = [0]*13*4
# PlotVals_30 = [0]*13*4
# PlotVals_45 = [0]*13*4
# PlotVals_60 = [0]*13*4

# for i in range(PatNum):
#     PlotVals_15[(((1+i)*4)-3)-1] = np.mean(l1.rmseStorage['JDST']['llRMSE'], axis = 0)[3]
#     PlotVals_15[(((1+i)*4)-2)-1] = np.mean(l1.rmseStorage['JDST']['rlRMSE'], axis = 0)[3]
#     PlotVals_15[(((1+i)*4)-1)-1] = np.mean(r1.rmseStorage['JDST']['rrRMSE'], axis = 0)[3]
#     PlotVals_15[(((1+i)*4)-0)-1] = np.mean(r1.rmseStorage['JDST']['lrRMSE'], axis = 0)[3]
    
# for i in range(PatNum):
#     PlotVals_30[(((1+i)*4)-3)-1] = np.mean(l1.rmseStorage['JDST']['llRMSE'], axis = 0)[2]
#     PlotVals_30[(((1+i)*4)-2)-1] = np.mean(l1.rmseStorage['JDST']['rlRMSE'], axis = 0)[2]
#     PlotVals_30[(((1+i)*4)-1)-1] = np.mean(r1.rmseStorage['JDST']['rrRMSE'], axis = 0)[2]
#     PlotVals_30[(((1+i)*4)-0)-1] = LRerrMean_30[i]

# for i in range(PatNum):
#     PlotVals_45[(((1+i)*4)-3)-1] = np.mean(l1.rmseStorage['JDST']['llRMSE'], axis = 0)[1]
#     PlotVals_45[(((1+i)*4)-2)-1] = np.mean(l1.rmseStorage['JDST']['rlRMSE'], axis = 0)[1]
#     PlotVals_45[(((1+i)*4)-1)-1] = np.mean(r1.rmseStorage['JDST']['rrRMSE'], axis = 0)[1]
#     PlotVals_45[(((1+i)*4)-0)-1] = LRerrMean_45[i]

# for i in range(PatNum):
#     PlotVals_60[(((1+i)*4)-3)-1] = np.mean(l1.rmseStorage['JDST']['llRMSE'], axis = 0)[0]
#     PlotVals_60[(((1+i)*4)-2)-1] = np.mean(l1.rmseStorage['JDST']['rlRMSE'], axis = 0)[0]
#     PlotVals_60[(((1+i)*4)-1)-1] = np.mean(r1.rmseStorage['JDST']['rrRMSE'], axis = 0)[0]
#     PlotVals_60[(((1+i)*4)-0)-1] = np.mean(r1.rmseStorage['JDST']['lrRMSE'], axis = 0)[0]

# PlotTicks = []

# for i in range(PatNum):
#     PlotTicks.append("PATIENT")
#     PlotTicks.extend(Comps)

# PlotYVals = [1.2, 1.4, 1.6, 1.8, 2.2, 2.4, 2.6, 2.8, 3.2, 3.4, 3.6, 3.8,
#              4.2, 4.4, 4.6, 4.8, 5.2, 5.4, 5.6, 5.8, 6.2, 6.4, 6.6, 6.8,
#              7.2, 7.4, 7.6, 7.8, 8.2, 8.4, 8.6, 8.8, 9.2, 9.4, 9.6, 9.8,
#              10.2, 10.4, 10.6, 10.8, 11.2, 11.4, 11.6, 11.8, 12.2, 12.4,
#              12.6, 12.8, 13.2, 13.4, 13.6, 13.8]

# dotchart = plt.figure(figsize=(9.5,11))
# ax = plt.axes()
# ax.set_xlim(0,80)
# ax.set_ylim(14,0.9)
# ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
# ax.xaxis.set_ticks_position("both")
# ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
# ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
# ax.tick_params(axis = 'x', which='major', width=1.00, length=5)
# ax.tick_params(axis = 'y', which='major', length=0, labelsize=10, pad=85)
# ax.tick_params(axis = 'y', which='minor', length=0, labelsize=7, pad=70)
# ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
# ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('Patient %d'))
# ax.yaxis.set_minor_formatter(ticker.FixedFormatter(Comps))
# ax.set_yticklabels(labels=Comps, minor=True, ha='left', fontname='Times New Roman')
# ax.set_yticklabels(labels=PatLabs, ha='left', fontname='Times New Roman')
# ax.set_xlabel('ROOT MEAN SQUARE (RMS) PREDICTION ERROR (mg/dL)', fontname='Times New Roman', size=14)
# #plt.title("BLOOD GLUCOSE NEURAL NETWORK PREDICTION ERRORS", fontname='Times New Roman', size=18)
# ax.title.set_position([0.5, 1.01])
# secax = ax.twinx()
# secax.set_ylim(14, 0.9)
# secax.tick_params(axis='y', which='both', length=0)
# secax.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))

# secax.set_yticklabels(labels=[])
# # secax.set_yticklabels(labels = sigLabs, minor=True)



# for i in range(len(PlotYVals)):
#     #ax.plot(np.linspace(0, PlotVals_15[i],50), PlotYVals[i]*np.ones(50), dashes=[1,5], color='k', lw=1)
#     ax.plot(np.linspace(0, max(PlotVals_15[i], PlotVals_30[i], PlotVals_45[i], 
#                                PlotVals_60[i]),50), PlotYVals[i]*np.ones(50), 
#             dashes=[1,5], color='#191A27', lw=0.8, label='_nolegend_')
#     ax.plot(PlotVals_15[i], PlotYVals[i], 'o', color='#3F7D6E', fillstyle='full', markersize=5)     
#     ax.plot(PlotVals_30[i], PlotYVals[i], 'o', color='#593560', fillstyle='full', markersize=5)
#     ax.plot(PlotVals_45[i], PlotYVals[i], 'o', color='#A25756', fillstyle='full', markersize=5)
#     ax.plot(PlotVals_60[i], PlotYVals[i], 'o', color='#E7B56D', fillstyle='full', markersize=5)
    
#     if ((i+2)%4==0):
#         if (Rfp_15[int((i+2)/4-1)] < 0.049):
#             ax.plot(76, PlotYVals[i+1], marker=(6,2,0), label='_nolegend_', color = '#3F7D6E', markersize=5)
#         if (Rfp_30[int((i+2)/4-1)] < 0.049):
#             ax.plot(77, PlotYVals[i+1], marker=(6,2,0), label='_nolegend_', color = '#593560', markersize=5)
#         if (Rfp_45[int((i+2)/4-1)] < 0.049):
#             ax.plot(78, PlotYVals[i+1], marker=(6,2,0), label='_nolegend_', color = '#A25756', markersize=5)
#         if (Rfp_60[int((i+2)/4-1)] < 0.049):
#             ax.plot(79, PlotYVals[i+1], marker=(6,2,0), label='_nolegend_', color = '#E7B56D', markersize=5)
        
#     if ((i+4)%4==0):
#         if (Lfp_15[int((i+4)/4-1)] < 0.049):
#             ax.plot(76, PlotYVals[i+1], marker=(6,2,0), label='_nolegend_', color = '#3F7D6E', markersize=5)
#         if (Lfp_30[int((i+4)/4-1)] < 0.049):
#             ax.plot(77, PlotYVals[i+1], marker=(6,2,0), label='_nolegend_', color = '#593560', markersize=5)
#         if (Lfp_45[int((i+4)/4-1)] < 0.049):
#             ax.plot(78, PlotYVals[i+1], marker=(6,2,0), label='_nolegend_', color = '#A25756', markersize=5)
#         if (Lfp_60[int((i+4)/4-1)] < 0.049):
#             ax.plot(79, PlotYVals[i+1], marker=(6,2,0), label='_nolegend_', color = '#E7B56D', markersize=5)
    
# ax.legend(["15 Minute PH", "30 Minute PH", "45 Minute PH", "60 Minute PH"], ncol=4, loc='center left', bbox_to_anchor=(0.07,1.03))
# plt.savefig('C:\Code\Keras_Glucose_Predictor_V1_0\ErrFullPHKeras.pdf', bbox_inches='tight')