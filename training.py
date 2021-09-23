# Written and designed by Aaron Tucker 
#
#
#
# ----------------------------------------------------------------------------vvvv
"""
Classes and functions related to algorithm training
"""

import numpy as np
import tensorflow as tf

import patient as pat

def zscoreData(x):
    """Normalization function which returns normalized data, mean, and std.
    
    Arguments:
        x - data to be normalized
    """
    
    return [(x - np.mean(x))/np.std(x), np.mean(x), np.std(x)]

def deNormData(x, mean, std):
    """DeNormalization function which maps normalized data to original state
    
    Arguments:
        x = data to be denormalized
        mean = mean to denormalize about
        std = standard deviation to denormalize the range
    
    returns:
        denormalized data
    """
    
    return (x*std) + mean


def MSError(Y, y_trn):
    """Calculate Mean Square Error from prediction and labeled data.
    
    Arguments: 
        Y - predicted data
        y_trn - labeled training data
    """
    
    MSE = (1/len(Y)) * np.sum(np.square(Y-y_trn), axis=0)
    return MSE


def cvTraining(lPatient, rPatient, outSize, nFoldIter, kFold, lag, batch_size, epochs, modelName):
    llMSE1 = np.zeros(nFoldIter, outSize)
    llMSE2 = np.zeros(nFoldIter, outSize)
    rlMSE1 = np.zeros(nFoldIter, outSize)
    rlMSE2 = np.zeros(nFoldIter, outSize)
    rrMSE1 = np.zeros(nFoldIter, outSize)
    rrMSE2 = np.zeros(nFoldIter, outSize)
    lrMSE1 = np.zeros(nFoldIter, outSize)
    lrMSE2 = np.zeros(nFoldIter, outSize)
    
    if (len(lPatient.trainData.colums) < 2):
        pat.createLagData(lPatient.trainData, lag = lag, skip=0, dropNaN=True)
        pat.createLagData(rPatient.trainData, lag = lag, skip=0, dropNaN=True)
    
    for i in range(nFoldIter):
        lPatient.randomizeTrainingData(kFold, seed=i)
        rPatient.randomizeTrainingData(kFold, seed=i)
    
        [lTempTrainNorm, lTempTrainMean, lTempTrainStd] = zscoreData(lPatient.tempTrain)
        [lTempValNorm, lTempValMean, lTempValStd] = zscoreData(lPatient.tempVal)
        
        [rTempTrainNorm, rTempTrainMean, rTempTrainStd] = zscoreData(rPatient.tempTrain)
        [rTempValNorm, rTempValMean, rTempValStd] = zscoreData(rPatient.tempVal)
        
        # Consider tying index to variable
        llTrnTrain = lPatient.models[modelName].fit(lTempTrainNorm[:, outSize:-1], 
                                                   lTempTrainNorm[:, 0:outSize], 
                                                   batch_size = batch_size, 
                                                   epochs = epochs)
        llTrnTrain = lPatient.models[modelName].fit(lTempTrainNorm[:, 4:7], 
                                                   lTempTrainNorm[:, 0:4], 
                                                   batch_size = batch_size, 
                                                   epochs = epochs)
        llValPred = lPatient.models[modelName].predict(lTempValNorm[:, 4:7], 
                                                      batch_size = batch_size)
        lrValPred = lPatient.models[modelName].predict(rTempValNorm[:, 4:7], 
                                                       batch_size = batch_size)
        
        llValTrain = lPatient.models[modelName].fit(lTempValNorm[:, 4:7],
                                                   lTempValNorm[:, 0:4],
                                                   batch_size = batch_size,
                                                   epochs = epochs)
        llTrnPred = lPatient.models[modelName].predict(lTempTrainNorm[:, 4:7],
                                                      batch_size = batch_size,
                                                      epochs = epochs)
        lrTrnPred = lPatient.models[modelName].predict(rTempTrainNorm[:, 4:7],
                                                       batch_size = batch_size,
                                                       epochs = epochs)
        
        rrTrnTrain = lPatient.models[modelName].fit(rTempTrainNorm[:, 4:7], 
                                                   rTempTrainNorm[:, 0:4], 
                                                   batch_size = batch_size, 
                                                   epochs = epochs)
        lValPred = lPatient.models[modelName].predict(rTempValNorm[:, 4:7], 
                                                      batch_size = batch_size)
        lrValPred = lPatient.models[modelName].predict(lTempValNorm[:, 4:7], 
                                                       batch_size = batch_size)
        
        lValTrain = lPatient.models[modelName].fit(rTempValNorm[:, 4:7],
                                                   rTempValNorm[:, 0:4],
                                                   batch_size = batch_size,
                                                   epochs = epochs)
        lTrnPred = lPatient.models[modelName].predict(rTempTrainNorm[:, 4:7],
                                                      batch_size = batch_size,
                                                      epochs = epochs)
        lrTrnPred = lPatient.models[modelName].predict(lTempTrainNorm[:, 4:7],
                                                       batch_size = batch_size,
                                                       epochs = epochs)
        
        
        
    
    
    
    
    
    
    