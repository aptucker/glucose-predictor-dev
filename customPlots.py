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
import statsmodels.graphics.tsaplots as smgraphs
from statsmodels.tsa.stattools import ccf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import matplotlib.font_manager as font_manager
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.ticker import FormatStrFormatter

import patient as pat
import customLayers as cLayers
import customModels as cModels
import training as trn
import customStats as cStats


def singlePatientError(lPat, 
                       rPat,
                       modelNames,
                       labels,
                       index,
                       modelDrops,
                       patNumber):
    
    """Bar chart of all model errors for a single patient, automatically 
    updates regardless of how many models are stored.
    
    Inputs:
        lPat - left arm patient instance to plot
        rPat - right arm patient instance to plot
        modelNames - names of the models to be plotted for labelling
        labels - arm comparison lables (e.g., Left-Left)
        index - list 1-4
        modelDrops - remove a model from plotting (but not from patient instance)
        patNames - patient names for labels on plot
        
    """
    
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
                  patNames,
                  savePlot,
                  plotName):
    """Dot chart to evaluate individual model performances.
    
    Inputs:
        lPats - list of left arm patient instances 
        rPats - list of right arm patient instances
        modelName - the name of the model to plot
        labels - arm comparison lables (e.g., Left-Left)
        index - list 1-4
        patNames - patient names for labels on plot
    
    Returns:
        Dot chart with automatic statistical significance labels
        
    """
    
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
               "PARTICIPANT 1",
               "PARTICIPANT 2",
               "PARTICIPANT 3",
               "PARTICIPANT 4",
               "PARTICIPANT 5",
               "PARTICIPANT 6",
               "PARTICIPANT 7",
               "PARTICIPANT 8",
               "PARTICIPANT 9",
               "PARTICIPANT 10",
               "PARTICIPANT 11",
               "PARTICIPANT 12",
               "PARTICIPANT 13"]
    
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
    
    if savePlot==True:
        plt.savefig(plotName, bbox_inches='tight')
    
def statisticalEvalPlot(lPatDataMean,
                        rPatDataMean,
                        lPatData,
                        rPatData,
                        patNumber,
                        meanWindowSize,
                        savePlot,
                        diffData,
                        plotFileName):
    """Plot the autocorrelation and partial autocorrelation; report the ADF
    and KPSS test results for left and right arm data
    
    Inputs:
        lPatData - left arm dataset to be analyzed (not patient instance)
        rPatData - right arm dataset to be analyzed (not patient instance)
        patNumber - patient number for plot title
        
    Returns:
        4 tile subplot with automatic labels
        
    """
    
    lPatMeanWindow = lPatDataMean.rolling(meanWindowSize)
    rPatMeanWindow = rPatDataMean.rolling(meanWindowSize)
    
    lPatMean = lPatMeanWindow.mean().dropna()
    rPatMean = rPatMeanWindow.mean().dropna()
    
    ladfTest = cStats.adf_test(lPatData)
    radfTest = cStats.adf_test(rPatData)
    # lkpssTest = cStats.kpss_test(lPatData)
    # rkpssTest = cStats.kpss_test(rPatData)
    
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(9,8))
    ax1.plot(lPatDataMean.resample('D').mean(), 'k')
    ax2.plot(rPatDataMean.resample('D').mean(), 'k')
    smgraphs.plot_acf(lPatData, ax3, color='k')
    smgraphs.plot_pacf(lPatData, ax5, color='k', method='ols')
    smgraphs.plot_acf(rPatData, ax4, color='k')
    smgraphs.plot_pacf(rPatData, ax6, color='k', method='ols')
    
    ax1.set_xlabel("Time (Day)", fontsize=9)
    ax1.set_ylabel("Left Arm Glucose (mg/dL)", fontsize=9)
    ax1.tick_params(axis='x', rotation = 25, labelsize=7)
    ax1.tick_params(axis='y', labelsize=8)
    ax1.set_title("Glucose Mean over Time")
    ax2.set_xlabel("Time (Day)", fontsize=9)
    ax2.set_ylabel("Right ArmGlucose (mg/dL)", fontsize=9)
    ax2.tick_params(axis='x', rotation = 25, labelsize=7)
    ax2.tick_params(axis='y', labelsize=8)
    ax2.set_title("Glucose Mean over Time")
    ax3.set_xlabel("Lag", fontsize=9)
    ax3.set_ylabel("Left Arm", fontsize=9)
    ax5.set_xlabel("Lag", fontsize=9)
    ax5.set_ylabel("Left Arm", fontsize=9)
    ax4.set_xlabel("Lag", fontsize=9)
    ax4.set_ylabel("Right Arm", fontsize=9)
    ax6.set_xlabel("Lag", fontsize=9)
    ax6.set_ylabel("Right Arm", fontsize=9)
    
    fig.tight_layout(pad=3.0)
    if diffData == False:
        fig.suptitle("Patient" f" {patNumber} Statistical Analysis", fontsize=16)
    
    if diffData == True:
        fig.suptitle("Patient" f" {patNumber} Statistical Analysis - Differenced Data", fontsize=16)
    
    fig.text(0.5,0, "P-values for: $L_{ADF}$ = " f"{ladfTest['p-value']:.3f}; "\
                 "$R_{ADF}$ = " f"{radfTest['p-value']:.3f}; ", ha='center')
    
    for item in ax3.collections:
    #change the color of the CI 
        if type(item)==PolyCollection:
            item.set_facecolor('black')
        #change the color of the vertical lines
        if type(item)==LineCollection:
            item.set_color('black')    

    #change the color of the markers/horizontal line
    for item in ax3.lines:
        item.set_color('black')
        
        
    for item in ax4.collections:
    #change the color of the CI 
        if type(item)==PolyCollection:
            item.set_facecolor('black')
        #change the color of the vertical lines
        if type(item)==LineCollection:
            item.set_color('black')    

    #change the color of the markers/horizontal line
    for item in ax4.lines:
        item.set_color('black')
        
        
    for item in ax5.collections:
    #change the color of the CI 
        if type(item)==PolyCollection:
            item.set_facecolor('black')
        #change the color of the vertical lines
        if type(item)==LineCollection:
            item.set_color('black')    

    #change the color of the markers/horizontal line
    for item in ax5.lines:
        item.set_color('black')
        
        
    for item in ax6.collections:
    #change the color of the CI 
        if type(item)==PolyCollection:
            item.set_facecolor('black')
        #change the color of the vertical lines
        if type(item)==LineCollection:
            item.set_color('black')    

    #change the color of the markers/horizontal line
    for item in ax6.lines:
        item.set_color('black')
    
    
    if savePlot==True:
        plt.savefig(plotFileName, bbox_inches='tight')
    # fig.text(0.5,0, "P-values for: L_ADF = " f"{ladfTest['p-value']:.3f}; "\
    #          "L_KPSS = " f"{lkpssTest['p-value']:.3f}; "\
    #              "R_ADF = " f"{radfTest['p-value']:.3f}; "\
    #                  "R_KPSS = " f"{rkpssTest['p-value']:.3f}", ha='center')


def ccfPlot(lPatData,
            rPatData,
            lagsToPlot,
            patNumber,
            plotFileName,
            savePlot):
    
    """
    The cross-correlation plot between two sets of glucose data is used to 
    examine correlation between right and left arm CGM data. A whole day of 
    data should be used to calculate this metric.
    
    Inputs:
        lPatData - Left arm data
        rPatData - Right arm Data
        lagsToPlot - The number of lags to show on plot
        patNumber - Patient number for plot
        plotFileName - File name to save plot
        savePlot - Boolean to toggle file save
        
    Outputs:
        ccfPlot - saved if toggled on
        
    """
    
    if len(lPatData) < len(rPatData):
        rPatData = rPatData.iloc[0:len(lPatData)]
    if len(rPatData) < len(lPatData):
        lPatData = lPatData.iloc[0:len(rPatData)]
        
    lags = range(1, len(lPatData)+1)
    
    ccfResults = ccf(lPatData, rPatData)
    
    fig, ax = plt.subplots(1,1)
    ax.plot(lags[0:lagsToPlot],
            ccfResults[0:lagsToPlot],
            marker='o',
            markerfacecolor='k',
            markeredgecolor='k',
            linestyle='None')
    
    for i in range(0, lagsToPlot):
        ax.plot([lags[i], lags[i]], [0, ccfResults[i]], color='k')
        
    ax.fill_between(lags[0:lagsToPlot], 0, 2/np.sqrt(len(lPatData)), color='k', edgecolors='none', alpha=0.25)
    ax.fill_between(lags[0:lagsToPlot], 0, -2/np.sqrt(len(lPatData)), color='k', edgecolors='none', alpha=0.25)
    ax.axhline(0, color='k')
    
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    ax.set_xlim([0, 21])
    ax.set_ylim([-1,1])
    ax.set_title("Patient" f" {patNumber} Cross-Correlation", fontsize=14)
    ax.set_xlabel('Lag', fontsize=12)
    ax.set_ylabel('Cross-Correlation Left-Right', fontsize=12)
    
    if savePlot==True:
        plt.savefig(plotFileName, bbox_inches='tight')
    

def excelTableExport(lPats,
                     rPats,
                     phLabels,
                     compLabels,
                     plusMinusLabels,
                     dataNameToExport,
                     modelName,
                     fileName):
    
    """
    Export RMSE or MARD data to an excel table.
    
    Inputs:
        lPats - List of patients left arm CGM data
        rPats - List of patients right arm CGM data
        phLabels - Prediction horizon labels
        compLabels - Comparator (e.g., left-left) labels
        plusMinusLabels - Add a plus minus for visuals
        dataNameToExport - Toggle rmse or mard
        modelName - Name of algorithm to export
        fileName - Name of file to export to
        
    Outputs:
        excelTableExport - Excel table saved in location given by fileName
    
    """
    
    outputDF = pd.DataFrame(phLabels, columns=['Prediction Horizon (min)'])
    outputDF['Algorithm Setup (trained-tested)'] = compLabels
    
    if dataNameToExport == 'RMSE':
    
        for e in range(len(lPats)):
            
            tempDFllRMSE = pd.DataFrame(np.mean(lPats[e].rmseStorage[modelName]['llRMSE'], axis=0))
            tempDFrlRMSE = pd.DataFrame(np.mean(lPats[e].rmseStorage[modelName]['rlRMSE'], axis=0))
            tempDFrrRMSE = pd.DataFrame(np.mean(rPats[e].rmseStorage[modelName]['rrRMSE'], axis=0))
            tempDFlrRMSE = pd.DataFrame(np.mean(rPats[e].rmseStorage[modelName]['lrRMSE'], axis=0))
            
            tempDFllRMSEstd = pd.DataFrame(np.std(lPats[e].rmseStorage[modelName]['llRMSE'], axis=0))
            tempDFrlRMSEstd = pd.DataFrame(np.std(lPats[e].rmseStorage[modelName]['rlRMSE'], axis=0))
            tempDFrrRMSEstd = pd.DataFrame(np.std(rPats[e].rmseStorage[modelName]['rrRMSE'], axis=0))
            tempDFlrRMSEstd = pd.DataFrame(np.std(rPats[e].rmseStorage[modelName]['lrRMSE'], axis=0))
            
            patDFmean = pd.DataFrame()
            patDFstd = pd.DataFrame()
            for i in range(3, -1, -1):
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
        
            outputDF['Patient ' f'{e+1} ' 'RMSE (mg/dL)'] = patDFmean
            outputDF['Patient ' f'{e+1}'] = plusMinusLabels
            outputDF['Patient ' f'{e+1} ' 'STD'] = patDFstd
    
    
    if dataNameToExport == 'MARD':
        
        for e in range(len(lPats)):
            tempDFllMARD = pd.DataFrame(np.mean(lPats[e].mardStorage[modelName]['llMARD'], axis=0))
            tempDFrlMARD = pd.DataFrame(np.mean(lPats[e].mardStorage[modelName]['rlMARD'], axis=0))
            tempDFrrMARD = pd.DataFrame(np.mean(rPats[e].mardStorage[modelName]['rrMARD'], axis=0))
            tempDFlrMARD = pd.DataFrame(np.mean(rPats[e].mardStorage[modelName]['lrMARD'], axis=0))
            
            tempDFllMARDstd = pd.DataFrame(np.std(lPats[e].mardStorage[modelName]['llMARD'], axis=0))
            tempDFrlMARDstd = pd.DataFrame(np.std(lPats[e].mardStorage[modelName]['rlMARD'], axis=0))
            tempDFrrMARDstd = pd.DataFrame(np.std(rPats[e].mardStorage[modelName]['rrMARD'], axis=0))
            tempDFlrMARDstd = pd.DataFrame(np.std(rPats[e].mardStorage[modelName]['lrMARD'], axis=0))
            
            patDFmean = pd.DataFrame()
            patDFstd = pd.DataFrame()
            for i in range(3, -1, -1):
                patDFmean = patDFmean.append([tempDFllMARD.iloc[i],
                                      tempDFrlMARD.iloc[i],
                                      tempDFrrMARD.iloc[i],
                                      tempDFlrMARD.iloc[i]])
                patDFstd = patDFstd.append([tempDFllMARDstd.iloc[i],
                                      tempDFrlMARDstd.iloc[i],
                                      tempDFrrMARDstd.iloc[i],
                                      tempDFlrMARDstd.iloc[i]])
            
            patDFmean = patDFmean.reset_index(drop=True)
            patDFstd = patDFstd.reset_index(drop=True)
            
            outputDF['Patient ' f'{e+1} ' 'MARD (%)'] = patDFmean
            outputDF['Patient ' f'{e+1}'] = plusMinusLabels
            outputDF['Patient ' f'{e+1} ' 'STD'] = patDFstd
    
    
    outputDF.to_excel(fileName, sheet_name='Raw_Python_Data', index=False)
    

def dayPredictionPlot(lPatData,
                      rPatData,
                      model,
                      patNumber,
                      saveFile,
                      fileNames):
    
    
    lPatPlotDF = pd.DataFrame(lPatData.copy())
    rPatPlotDF = pd.DataFrame(rPatData.copy())
    
    pat.createLagData(lPatPlotDF, lag=3)
    pat.createLagData(rPatPlotDF, lag=3)
    
    lPatPlot = lPatPlotDF[['Lag: 1', 'Lag: 2', 'Lag: 3']].values
    rPatPlot = rPatPlotDF[['Lag: 1', 'Lag: 2', 'Lag: 3']].values
    
    [lPatZScore, lPatMean, lPatStd] = trn.zscoreData(lPatPlot)
    [rPatZScore, rPatMean, rPatStd] = trn.zscoreData(rPatPlot)
    
    lPatPredNorm = model.predict(lPatZScore, batch_size=1)
    rPatPredNorm = model.predict(rPatZScore, batch_size=1)
    
    lPatPred = trn.deNormData(lPatPredNorm, lPatMean, lPatStd)
    rPatPred = trn.deNormData(rPatPredNorm, rPatMean, rPatStd)
    
    lPatPredDF = pd.DataFrame(lPatPlotDF['Historic Glucose(mg/dL)'].iloc[3:].copy())
    rPatPredDF = pd.DataFrame(rPatPlotDF['Historic Glucose(mg/dL)'].iloc[3:].copy())
    
    lPatPredDF['PH = 15MIN'] = lPatPred[3:, 3]
    rPatPredDF['PH = 15MIN'] = rPatPred[3:, 3]
    
    lPatPredDF['PH = 30MIN'] = lPatPred[3:, 2]
    rPatPredDF['PH = 30MIN'] = rPatPred[3:, 2]
    
    lPatPredDF['PH = 45MIN'] = lPatPred[3:, 1]
    rPatPredDF['PH = 45MIN'] = rPatPred[3:, 1]
    
    lPatPredDF['PH = 60MIN'] = lPatPred[3:, 0]
    rPatPredDF['PH = 60MIN'] = rPatPred[3:, 0]
    
    lPatPredDF.rename(columns={'Historic Glucose(mg/dL)': 'MEASURED'}, inplace=True)
    rPatPredDF.rename(columns={'Historic Glucose(mg/dL)': 'MEASURED'}, inplace=True)
    
    
    
    # ax.plot(lPatPredDF.index, lPatPredDF['Historic Glucose(mg/dL)'])
    colors=['k',
           '#3F7D6E',
           '#593560',
           '#A25756',
           '#E7B56D']
    linestyles = ['--',
                  '-',
                  '-',
                  '-',
                  '-']
    
    figL, axL = plt.subplots(1,1)
    axL.set_prop_cycle(c=colors,
                      ls=linestyles)
    
    lPatPredDF.plot(kind='line', ax=axL)
    
    axL.set_xlabel('TIME (H:M)')
    axL.set_ylabel('PARTICIPANT ' f'{patNumber} ' 'LEFT CGM\n GLUCOSE CONCENTRATION (mg/dL)')
    axL.legend(fontsize='small', frameon=False)
    
    if saveFile == True:
        plt.savefig(fileNames[0], bbox_inches='tight')
    
    figR, axR = plt.subplots(1,1)
    axR.set_prop_cycle(c=colors,
                      ls=linestyles)
    
    rPatPredDF.plot(kind='line', ax=axR)
    
    axR.set_xlabel('TIME (H:M)')
    axR.set_ylabel('PARTICIPANT ' f'{patNumber} ' 'RIGHT CGM\n GLUCOSE CONCENTRATION (mg/dL)')
    axR.legend(fontsize='small', frameon=False)
    
    if saveFile == True:
        plt.savefig(fileNames[1], bbox_inches='tight')
    

def multiModelEvalPlot(lPats,
                  rPats,
                  modelNames,
                  labels,
                  index,
                  patNames,
                  savePlot,
                  plotName):
    
    
    # outputDF = pd.DataFrame(phLabels, columns=['Prediction Horizon (min)'])
    # # outputDF['Algorithm Setup (trained-tested)'] = compLabels
    
    # for e in range(len(lPats)):
    #     tempDFllMARD = pd.DataFrame(np.mean(lPats[e].mardStorage[modelName]['llMARD'], axis=0))
    #     tempDFrlMARD = pd.DataFrame(np.mean(lPats[e].mardStorage[modelName]['rlMARD'], axis=0))
    #     tempDFrrMARD = pd.DataFrame(np.mean(rPats[e].mardStorage[modelName]['rrMARD'], axis=0))
    #     tempDFlrMARD = pd.DataFrame(np.mean(rPats[e].mardStorage[modelName]['lrMARD'], axis=0))
        
    #     tempDFllMARDstd = pd.DataFrame(np.std(lPats[e].mardStorage[modelName]['llMARD'], axis=0))
    #     tempDFrlMARDstd = pd.DataFrame(np.std(lPats[e].mardStorage[modelName]['rlMARD'], axis=0))
    #     tempDFrrMARDstd = pd.DataFrame(np.std(rPats[e].mardStorage[modelName]['rrMARD'], axis=0))
    #     tempDFlrMARDstd = pd.DataFrame(np.std(rPats[e].mardStorage[modelName]['lrMARD'], axis=0))
        
    #     patDFmean = pd.DataFrame()
    #     patDFstd = pd.DataFrame()
    #     for i in range(3, -1, -1):
    #         patDFmean = patDFmean.append([tempDFllMARD.iloc[i],
    #                               tempDFrlMARD.iloc[i],
    #                               tempDFrrMARD.iloc[i],
    #                               tempDFlrMARD.iloc[i]])
    #         patDFstd = patDFstd.append([tempDFllMARDstd.iloc[i],
    #                               tempDFrlMARDstd.iloc[i],
    #                               tempDFrrMARDstd.iloc[i],
    #                               tempDFlrMARDstd.iloc[i]])
        
    #     patDFmean = patDFmean.reset_index(drop=True)
    #     patDFstd = patDFstd.reset_index(drop=True)
        
    #     outputDF['Patient ' f'{e+1} ' 'MARD (%)'] = patDFmean
    #     # outputDF['Patient ' f'{e+1}'] = plusMinusLabels
    #     outputDF['Patient ' f'{e+1} ' 'STD'] = patDFstd
    
    llMeans = []
    rlMeans = []
    rrMeans = []
    lrMeans = []
    
    plotModelNames = modelNames.copy()
    
    for i in range(len(modelNames)):
        if modelNames[i] == 'JDST':
            plotModelNames[i] = 'FF NN'
    
        if modelNames[i] == 'GRU H=1':
            plotModelNames[i] = 'GRU NN'
    
    
    Comps = ["",
             f"{plotModelNames[0]}" " LEFT ARM",
             f"{plotModelNames[1]}" " LEFT ARM",
             f"{plotModelNames[0]}" " RIGHT ARM",
             f"{plotModelNames[1]}" " RIGHT ARM"]
    
    for i in range(len(lPats)):
        Comps.extend([f"{plotModelNames[0]}" " LEFT ARM",
             f"{plotModelNames[1]}" " LEFT ARM",
             f"{plotModelNames[0]}" " RIGHT ARM",
             f"{plotModelNames[1]}" " RIGHT ARM"])
   
    for lPat in lPats:
        llMeans.append(np.mean(lPat.rmseStorage[modelNames[0]]['llRMSE'], axis=0))
        rlMeans.append(np.mean(lPat.rmseStorage[modelNames[1]]['llRMSE'], axis=0))
    for rPat in rPats:
        rrMeans.append(np.mean(rPat.rmseStorage[modelNames[0]]['rrRMSE'], axis=0))
        lrMeans.append(np.mean(rPat.rmseStorage[modelNames[1]]['rrRMSE'], axis=0))
    
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
               "PARTICIPANT 1",
               "PARTICIPANT 2",
               "PARTICIPANT 3",
               "PARTICIPANT 4",
               "PARTICIPANT 5",
               "PARTICIPANT 6",
               "PARTICIPANT 7",
               "PARTICIPANT 8",
               "PARTICIPANT 9",
               "PARTICIPANT 10",
               "PARTICIPANT 11",
               "PARTICIPANT 12",
               "PARTICIPANT 13"]
    
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
    ax.set_xlabel(f'{plotModelNames[0]} ' 'AND ' f'{plotModelNames[1]}' ' ROOT MEAN SQUARE ERROR (mg/dL)', fontname='Times New Roman', size=14)
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
        
        for e in range(len(lPats[i].fStorage[f'{modelNames[0]}' 'v' f'{modelNames[1]}']['pValues'])):
            if (lPats[i].fStorage[f'{modelNames[0]}' 'v' f'{modelNames[1]}']['pValues'][e] < 0.054):
                ax.plot((79-e),
                        (i+1)+0.4,
                        marker=(6,2,0),
                        label='_nolegend_',
                        color = colors[-(e+1)],
                        markersize=5)
            
            if (rPats[i].fStorage[f'{modelNames[0]}' 'v' f'{modelNames[1]}']['pValues'][e] < 0.054):
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
    
    if savePlot==True:
        plt.savefig(plotName, bbox_inches='tight')
    
    

def timeTrialPlot(data, modelsToPlot, iterationsToPlot, axToPlot):
    
    legendNames = []
    
    for i in range(len(modelsToPlot)):
        for e in iterationsToPlot:
            data[modelsToPlot[i]]['Loss It. ' f'{e+1}'].plot(ax=axToPlot)
            
            if len(iterationsToPlot) > 1:
                legendNames.append(f'{modelsToPlot[i]}' ' It. ' f'{e+1}')
            
            else:
                legendNames.append(f'{modelsToPlot[i]}')
        
    
    return legendNames
    

