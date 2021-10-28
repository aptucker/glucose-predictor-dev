# Written and designed by Aaron Tucker 
#
#
#
# ----------------------------------------------------------------------------
"""
Custom plotting functions
"""


import tensorflow as tf
import numpy as np
import pandas as pd
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


def singlePatientError(lPat, 
                       rPat,
                       modelNames,
                       labels,
                       index,
                       modelDrops,
                       patNumber):
    
    llMeans = []
    rlMeans = []
    rrMeans = []
    lrMeans = []
    for i in lPat.rmseStorage:
        llMeans.append(np.mean(lPat.rmseStorage[i]['llRMSE'], axis=0))
        rlMeans.append(np.mean(lPat.rmseStorage[i]['rlRMSE'], axis=0))
        rrMeans.append(np.mean(rPat.rmseStorage[i]['rrRMSE'], axis=0))
        lrMeans.append(np.mean(rPat.rmseStorage[i]['lrRMSE'], axis=0))
    
    llMeans = np.array(llMeans)
    rlMeans = np.array(rlMeans)
    rrMeans = np.array(rrMeans)
    lrMeans = np.array(lrMeans)
    
    llMeansDF = pd.DataFrame(np.flip(np.transpose(llMeans), axis=0), index = index, columns = modelNames)
    rlMeansDF = pd.DataFrame(np.flip(np.transpose(rlMeans), axis=0), index = index, columns = modelNames)
    rrMeansDF = pd.DataFrame(np.flip(np.transpose(rrMeans), axis=0), index = index, columns = modelNames)
    lrMeansDF = pd.DataFrame(np.flip(np.transpose(lrMeans), axis=0), index = index, columns = modelNames)
    
    errDF15 = pd.DataFrame([llMeansDF.iloc[0,:], rlMeansDF.iloc[0,:], rrMeansDF.iloc[0,:], lrMeansDF.iloc[0,:]], index = labels)
    errDF30 = pd.DataFrame([llMeansDF.iloc[1,:], rlMeansDF.iloc[1,:], rrMeansDF.iloc[1,:], lrMeansDF.iloc[1,:]], index = labels)
    errDF45 = pd.DataFrame([llMeansDF.iloc[2,:], rlMeansDF.iloc[2,:], rrMeansDF.iloc[2,:], lrMeansDF.iloc[2,:]], index = labels)
    errDF60 = pd.DataFrame([llMeansDF.iloc[3,:], rlMeansDF.iloc[3,:], rrMeansDF.iloc[3,:], lrMeansDF.iloc[3,:]], index = labels)
    
    for i in range(len(modelDrops)):
        errDF15.drop(modelDrops[i], axis=1, inplace=True)
        errDF30.drop(modelDrops[i], axis=1, inplace=True)
        errDF45.drop(modelDrops[i], axis=1, inplace=True)
        errDF60.drop(modelDrops[i], axis=1, inplace=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9,5))
    errDF15.plot(kind='bar', ax=ax1, legend=None)
    ax1.get_xaxis().set_visible(False)
    ax1.set_title('15-minute PH')
    ax1.set_ylabel('RMSE (mg/dL')
    
    errDF30.plot(kind='bar', ax=ax2, legend=None)
    ax2.get_xaxis().set_visible(False)
    ax2.set_title('30-minute PH')
    
    errDF45.plot(kind='bar', ax=ax3, legend=None)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation = 0)
    ax3.set_title('45-minute PH')
    ax3.set_ylabel('RMSE (mg/dL')
    
    errDF60.plot(kind='bar', ax=ax4, legend=None)
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation = 0)
    ax4.set_title('60-minute PH')
    
    fig.legend(errDF15.columns, loc='lower center', ncol=6)
    fig.suptitle('Patient' f' {patNumber}')
    
    rects1 = ax1.patches
    rects2 = ax2.patches
    rects3 = ax3.patches
    rects4 = ax4.patches
    tempModelNames = modelNames.copy()
    for modelDrop in modelDrops:
        tempModelNames.remove(modelDrop)
        
    # datLabels = [f"label{i}" for i in range(len(rects))]
    datLabels15 = ["" for i in range(len(rects1))]
    datLabels30 = ["" for i in range(len(rects2))]
    datLabels45 = ["" for i in range(len(rects3))]
    datLabels60 = ["" for i in range(len(rects4))]
    
   
    for i in range(len(rects1)):
        for e in range(len(tempModelNames)):
            if (i >= (e*4)) and (i < ((e*4)+4)) and (i%4==0): # and (i+1)%2==0:
                if (lPat.fStorage[tempModelNames[e]]['pValues'][-1] < 0.054):
                    datLabels15[i+1] = "*"
            if (i >= (e*4)) and (i < ((e*4)+4)) and (i%2==0) and (i%4!=0):
                if (rPat.fStorage[tempModelNames[e]]['pValues'][-1] < 0.054):
                    datLabels15[i+1] = "*"
    
    for i in range(len(rects2)):
       for e in range(len(tempModelNames)):
           if (i >= (e*4)) and (i < ((e*4)+4)) and (i%4==0): # and (i+1)%2==0:
               if (lPat.fStorage[tempModelNames[e]]['pValues'][-2] < 0.054):
                   datLabels30[i+1] = "*"
           if (i >= (e*4)) and (i < ((e*4)+4)) and (i%2==0) and (i%4!=0):
               if (rPat.fStorage[tempModelNames[e]]['pValues'][-2] < 0.054):
                   datLabels30[i+1] = "*"
    
    for i in range(len(rects3)):
       for e in range(len(tempModelNames)):
           if (i >= (e*4)) and (i < ((e*4)+4)) and (i%4==0): # and (i+1)%2==0:
               if (lPat.fStorage[tempModelNames[e]]['pValues'][-3] < 0.054):
                   datLabels45[i+1] = "*"
           if (i >= (e*4)) and (i < ((e*4)+4)) and (i%2==0) and (i%4!=0):
               if (rPat.fStorage[tempModelNames[e]]['pValues'][-3] < 0.054):
                   datLabels45[i+1] = "*"

    for i in range(len(rects4)):
       for e in range(len(tempModelNames)):
           if (i >= (e*4)) and (i < ((e*4)+4)) and (i%4==0): # and (i+1)%2==0:
               if (lPat.fStorage[tempModelNames[e]]['pValues'][-4] < 0.054):
                   datLabels60[i+1] = "*"
           if (i >= (e*4)) and (i < ((e*4)+4)) and (i%2==0) and (i%4!=0):
               if (rPat.fStorage[tempModelNames[e]]['pValues'][-4] < 0.054):
                   datLabels60[i+1] = "*"
    
    for rect, datLabel in zip(rects1, datLabels15):
        height = rect.get_height()
        ax1.text(
            rect.get_x() + rect.get_width() / 2, height + 0.1, datLabel, ha="center", va="bottom"
            )
    
    for rect, datLabel in zip(rects2, datLabels30):
        height = rect.get_height()
        ax2.text(
            rect.get_x() + rect.get_width() / 2, height + 0.1, datLabel, ha="center", va="bottom"
            )
        
    for rect, datLabel in zip(rects3, datLabels45):
        height = rect.get_height()
        ax3.text(
            rect.get_x() + rect.get_width() / 2, height + 0.1, datLabel, ha="center", va="bottom"
            )
        
    for rect, datLabel in zip(rects4, datLabels60):
        height = rect.get_height()
        ax4.text(
            rect.get_x() + rect.get_width() / 2, height + 0.1, datLabel, ha="center", va="bottom"
            )


def modelEvalPlot(lPats,
                  rPats,
                  modelName,
                  labels,
                  index,
                  patNames):
    
    llMeans = []
    rlMeans = []
    rrMeans = []
    lrMeans = []
    
    Comps = ["", "LEFT - LEFT", "RIGHT - LEFT", "RIGHT - RIGHT", "LEFT - RIGHT"]
    for i in range(len(lPats)):
        Comps.extend(["LEFT - LEFT", "RIGHT - LEFT", "RIGHT - RIGHT", "LEFT - RIGHT"])
   
    for lPat in lPats:
        llMeans.append(np.mean(lPat.rmseStorage[modelName]['llRMSE'], axis=0))
        rlMeans.append(np.mean(lPat.rmseStorage[modelName]['rlRMSE'], axis=0))
    for rPat in rPats:
        rrMeans.append(np.mean(rPat.rmseStorage[modelName]['rrRMSE'], axis=0))
        lrMeans.append(np.mean(rPat.rmseStorage[modelName]['lrRMSE'], axis=0))
    
    llMeans = np.array(llMeans)
    rlMeans = np.array(rlMeans)
    rrMeans = np.array(rrMeans)
    lrMeans = np.array(lrMeans)
    
    llMeansDF = pd.DataFrame(np.flip(np.transpose(llMeans), axis=0), index = index, columns = patNames)
    rlMeansDF = pd.DataFrame(np.flip(np.transpose(rlMeans), axis=0), index = index, columns = patNames)
    rrMeansDF = pd.DataFrame(np.flip(np.transpose(rrMeans), axis=0), index = index, columns = patNames)
    lrMeansDF = pd.DataFrame(np.flip(np.transpose(lrMeans), axis=0), index = index, columns = patNames)
    
    errDF15 = pd.DataFrame([llMeansDF.iloc[0,:],
                            rlMeansDF.iloc[0,:],
                            rrMeansDF.iloc[0,:],
                            lrMeansDF.iloc[0,:]],
                           index = labels)
    errDF30 = pd.DataFrame([llMeansDF.iloc[1,:],
                            rlMeansDF.iloc[1,:],
                            rrMeansDF.iloc[1,:],
                            lrMeansDF.iloc[1,:]],
                           index = labels)
    errDF45 = pd.DataFrame([llMeansDF.iloc[2,:],
                            rlMeansDF.iloc[2,:],
                            rrMeansDF.iloc[2,:],
                            lrMeansDF.iloc[2,:]],
                           index = labels)
    errDF60 = pd.DataFrame([llMeansDF.iloc[3,:],
                            rlMeansDF.iloc[3,:],
                            rrMeansDF.iloc[3,:],
                            lrMeansDF.iloc[3,:]],
                           index = labels)
    
    patLabs = ["",
               "PATIENT 1",
               "PATIENT 2",
               "PATIENT 3",
               "PATIENT 4",
               "PATIENT 5",
               "PATIENT 6",
               "PATIENT 7",
               "PATIENT 8",
               "PATIENT 9",
               "PATIENT 10",
               "PATIENT 11",
               "PATIENT 12",
               "PATIENT 13"]
    
    dotchart = plt.figure(figsize=(9.5,11))
    ax = plt.axes()
    ax.set_xlim(0,80)
    ax.set_ylim(14,0.9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax.tick_params(axis = 'x', which='major', width=1.00, length=5)
    ax.tick_params(axis = 'y', which='major', length=0, labelsize=10, pad=85)
    ax.tick_params(axis = 'y', which='minor', length=0, labelsize=7, pad=70)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('Patient %d'))
    ax.yaxis.set_minor_formatter(ticker.FixedFormatter(Comps))
    ax.set_yticklabels(labels=Comps, minor=True, ha='left', fontname='Times New Roman')
    ax.set_yticklabels(labels=patLabs, ha='left', fontname='Times New Roman')
    ax.set_xlabel(f'{modelName} ' 'ROOT MEAN SQUARE (RMS) PREDICTION ERROR (mg/dL)', fontname='Times New Roman', size=14)
    #plt.title("BLOOD GLUCOSE NEURAL NETWORK PREDICTION ERRORS", fontname='Times New Roman', size=18)
    ax.title.set_position([0.5, 1.01])
    secax = ax.twinx()
    secax.set_ylim(14, 0.9)
    secax.tick_params(axis='y', which='both', length=0)
    secax.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    
    secax.set_yticklabels(labels=[])
    # secax.set_yticklabels(labels = sigLabs, minor=True)
    
    for i in range(len(errDF15.columns)):
        for e in range(len(errDF15['Patient ' f'{i+1}'])):    
            ax.plot(np.linspace(0, 
                                max(errDF15['Patient ' f'{i+1}'].iloc[e],
                                    errDF30['Patient ' f'{i+1}'].iloc[e],
                                    errDF45['Patient ' f'{i+1}'].iloc[e],
                                    errDF60['Patient ' f'{i+1}'].iloc[e]),
                                50),
                    np.ones(50)*(i+1)+((e+1)*0.2),
                    dashes=[1,5],
                    color='#191A27',
                    lw=0.8,
                    label='_nolegend_')
        
        ax.plot(errDF15['Patient ' f'{i+1}'],
                (i+1)+np.array([0.2, 0.4, 0.6, 0.8]),
                'o', color='#3F7D6E',
                fillstyle='full',
                markersize=5)
        ax.plot(errDF30['Patient ' f'{i+1}'],
                (i+1)+np.array([0.2, 0.4, 0.6, 0.8]),
                'o', color='#593560',
                fillstyle='full',
                markersize=5)
        ax.plot(errDF45['Patient ' f'{i+1}'],
                (i+1)+np.array([0.2, 0.4, 0.6, 0.8]),
                'o', color='#A25756',
                fillstyle='full',
                markersize=5)
        ax.plot(errDF60['Patient ' f'{i+1}'],
                (i+1)+np.array([0.2, 0.4, 0.6, 0.8]),
                'o', color='#E7B56D',
                fillstyle='full',
                markersize=5)
        
        colors = ['#3F7D6E',
                  '#593560',
                  '#A25756',
                  '#E7B56D']
        
        for e in range(len(lPats[i].fStorage[modelName]['pValues'])):
            if (lPats[i].fStorage[modelName]['pValues'][e] < 0.054):
                ax.plot((79-e),
                        (i+1)+0.4,
                        marker=(6,2,0),
                        label='_nolegend_',
                        color = colors[-(e+1)],
                        markersize=5)
            
            if (rPats[i].fStorage[modelName]['pValues'][e] < 0.054):
                ax.plot((79-e),
                        (i+1)+0.8,
                        marker=(6,2,0),
                        label='_nolegend_',
                        color = colors[-(e+1)],
                        markersize=5)
                
            
        
    ax.legend(["15 Minute PH",
               "30 Minute PH",
               "45 Minute PH",
               "60 Minute PH"],
              ncol=4,
              loc='center left',
              bbox_to_anchor=(0.07,1.03))
    






        