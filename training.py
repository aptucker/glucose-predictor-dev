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
    llMSE1 = np.zeros([nFoldIter, outSize])
    llMSE2 = np.zeros([nFoldIter, outSize])
    rlMSE1 = np.zeros([nFoldIter, outSize])
    rlMSE2 = np.zeros([nFoldIter, outSize])
    rrMSE1 = np.zeros([nFoldIter, outSize])
    rrMSE2 = np.zeros([nFoldIter, outSize])
    lrMSE1 = np.zeros([nFoldIter, outSize])
    lrMSE2 = np.zeros([nFoldIter, outSize])
    
    if (len(lPatient.trainData.columns) < 2):
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
        llTrnTrain = lPatient.models[modelName].fit(lTempTrainNorm[:, outSize:], 
                                                    lTempTrainNorm[:, 0:outSize], 
                                                    batch_size = batch_size, 
                                                    epochs = epochs)
        llValPred = lPatient.models[modelName].predict(lTempValNorm[:, outSize:], 
                                                      batch_size = batch_size)
        lrValPred = lPatient.models[modelName].predict(rTempValNorm[:, outSize:], 
                                                       batch_size = batch_size)
        
        llValTrain = lPatient.models[modelName].fit(lTempValNorm[:, outSize:],
                                                   lTempValNorm[:, 0:outSize],
                                                   batch_size = batch_size,
                                                   epochs = epochs)
        llTrnPred = lPatient.models[modelName].predict(lTempTrainNorm[:, outSize:],
                                                      batch_size = batch_size)
        lrTrnPred = lPatient.models[modelName].predict(rTempTrainNorm[:, outSize:],
                                                       batch_size = batch_size)
        
        rrTrnTrain = lPatient.models[modelName].fit(rTempTrainNorm[:, outSize:], 
                                                   rTempTrainNorm[:, 0:outSize], 
                                                   batch_size = batch_size, 
                                                   epochs = epochs)
        rrValPred = lPatient.models[modelName].predict(rTempValNorm[:, outSize:], 
                                                      batch_size = batch_size)
        rlValPred = lPatient.models[modelName].predict(lTempValNorm[:, outSize:], 
                                                       batch_size = batch_size)
        
        rrValTrain = lPatient.models[modelName].fit(rTempValNorm[:, outSize:],
                                                   rTempValNorm[:, 0:outSize],
                                                   batch_size = batch_size,
                                                   epochs = epochs)
        rrTrnPred = lPatient.models[modelName].predict(rTempTrainNorm[:, outSize:],
                                                      batch_size = batch_size)
        rlTrnPred = lPatient.models[modelName].predict(lTempTrainNorm[:, outSize:],
                                                       batch_size = batch_size)
        
        llValPredDeNorm = deNormData(llValPred, lTempValMean, lTempValStd)
        rrValPredDeNorm = deNormData(rrValPred, rTempValMean, rTempValStd)
        lrValPredDeNorm = deNormData(lrValPred, rTempValMean, rTempValStd)
        rlValPredDeNorm = deNormData(rlValPred, lTempValMean, lTempValStd)
        
        llTrnPredDeNorm = deNormData(llTrnPred, lTempTrainMean, lTempTrainStd)
        rrTrnPredDeNorm = deNormData(rrTrnPred, rTempTrainMean, rTempTrainStd)
        lrTrnPredDeNorm = deNormData(lrTrnPred, rTempTrainMean, rTempTrainStd)
        rlTrnPredDeNorm = deNormData(rlTrnPred, lTempTrainMean, lTempTrainStd)
        
        llMSE1[i, :] = MSError(llValPredDeNorm, lPatient.tempVal[:, 0:4])
        llMSE2[i, :] = MSError(llTrnPredDeNorm, lPatient.tempTrain[:, 0:4])
        rrMSE1[i, :] = MSError(rrValPredDeNorm, rPatient.tempVal[:, 0:4])
        rrMSE2[i, :] = MSError(rrTrnPredDeNorm, rPatient.tempTrain[:, 0:4])
        lrMSE1[i, :] = MSError(lrValPredDeNorm, rPatient.tempVal[:, 0:4])
        lrMSE2[i, :] = MSError(lrTrnPredDeNorm, rPatient.tempTrain[:, 0:4])
        rlMSE1[i, :] = MSError(rlValPredDeNorm, lPatient.tempVal[:, 0:4])
        rlMSE2[i, :] = MSError(rlTrnPredDeNorm, lPatient.tempTrain[:, 0:4])
    
        
    llMSE = np.append(llMSE1, llMSE2, axis = 0)
    rrMSE = np.append(rrMSE1, rrMSE2, axis = 0)
    lrMSE = np.append(lrMSE1, lrMSE2, axis = 0)
    rlMSE = np.append(rlMSE1, rlMSE2, axis = 0)
    
    lFinalErrors = {
        "llMSE": llMSE,
        "lrMSE": lrMSE}
    rFinalErrors = {
        "rrMSE": rrMSE,
        "rlMSE": rlMSE}
    
    lPatient.mseStorage[modelName] = lFinalErrors
    rPatient.mseStorage[modelName] = rFinalErrors
    
    
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    