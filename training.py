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
from scipy import stats

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

def fStatistic(error1, error2, nFoldIter):
    """Calculate the f-statistic for cross validation
    
    Arguments:
        error1 = matrix of errors to compare
        error2 = matrix of errors to compare
        nFoldIter = matrix split point
    
    Returns:
        fp = p-value from f-statistic calculation
    """
        
    fp1 = error1[0:nFoldIter, :] - error2[0:nFoldIter, :]
    fp2 = error1[nFoldIter:, :] - error2[nFoldIter:, :]
    
    fpbar = (fp1 + fp1)/2
    fs = np.square(fp1 - fpbar) + np.square(fp2 - fpbar)
    f = np.divide(np.sum(np.square(fp1) + np.square(fp2), axis=0), 2*np.sum(fs, axis=0))
    fp = 1 - stats.f.cdf(f, 10, 5)
    return fp
    

def cvTraining(lPatient, rPatient, outSize, nFoldIter, kFold, lag, batch_size, epochs, models, modelName):
    """Cross validation training function uses model stored in patient object
    
    Arguments:
        lPatient = left arm patient object
        rPatient = right arm patient object
        outSize = size of output (K)
        nFoldIter = number of fold iterations
        kFold = number of folds 
        lag = amount to lag dataset
        batch_size = batch size for model training
        epochs = number of training epochs
        models = dictionary of models
        modelName = name of model stored in patient models dictionary
        
    Returns:
        None, updates error variables in patient object
    """
    
    # Initialize error variables
    llMSE = np.zeros([nFoldIter*2, outSize])
    rrMSE = np.zeros([nFoldIter*2, outSize])
    lrMSE = np.zeros([nFoldIter*2, outSize])
    rlMSE = np.zeros([nFoldIter*2, outSize])
    
    # Lag data if not already lagged
    if (len(lPatient.trainData.columns) < 2):
        pat.createLagData(lPatient.trainData, lag = lag, skip=0, dropNaN=True)
        pat.createLagData(rPatient.trainData, lag = lag, skip=0, dropNaN=True)
    
    for i in range(nFoldIter):
        
        # Randomize the training data with every iteration
        lPatient.randomizeTrainingData(kFold, seed=i)
        rPatient.randomizeTrainingData(kFold, seed=i)
        
        # Normalize the data
        [lTempTrainNorm, lTempTrainMean, lTempTrainStd] = zscoreData(lPatient.tempTrain)
        [lTempValNorm, lTempValMean, lTempValStd] = zscoreData(lPatient.tempVal)
        
        [rTempTrainNorm, rTempTrainMean, rTempTrainStd] = zscoreData(rPatient.tempTrain)
        [rTempValNorm, rTempValMean, rTempValStd] = zscoreData(rPatient.tempVal)
        
        # LEFT-LEFT Training->Validation
        llTrnTrain = models[modelName].fit(lTempTrainNorm[:, outSize:], 
                                                    lTempTrainNorm[:, 0:outSize], 
                                                    batch_size = batch_size, 
                                                    epochs = epochs)
        llValPred = models[modelName].predict(lTempValNorm[:, outSize:], 
                                                      batch_size = batch_size)
        # LEFT-RIGHT Validation
        lrValPred = models[modelName].predict(rTempValNorm[:, outSize:], 
                                                       batch_size = batch_size)
        
        # LEFT-LEFT Validation->Training
        llValTrain = models[modelName].fit(lTempValNorm[:, outSize:],
                                                   lTempValNorm[:, 0:outSize],
                                                   batch_size = batch_size,
                                                   epochs = epochs)
        llTrnPred = models[modelName].predict(lTempTrainNorm[:, outSize:],
                                                      batch_size = batch_size)
        # LEFT-RIGHT Training
        lrTrnPred = models[modelName].predict(rTempTrainNorm[:, outSize:],
                                                       batch_size = batch_size)
        
        # RIGHT-RIGHT Training->Validation
        rrTrnTrain = models[modelName].fit(rTempTrainNorm[:, outSize:], 
                                                   rTempTrainNorm[:, 0:outSize], 
                                                   batch_size = batch_size, 
                                                   epochs = epochs)
        rrValPred = models[modelName].predict(rTempValNorm[:, outSize:], 
                                                      batch_size = batch_size)
        # RIGHT-LEFT Validation
        rlValPred = models[modelName].predict(lTempValNorm[:, outSize:], 
                                                       batch_size = batch_size)
        
        # RIGHT-RIGHT Validation->Training
        rrValTrain = models[modelName].fit(rTempValNorm[:, outSize:],
                                                   rTempValNorm[:, 0:outSize],
                                                   batch_size = batch_size,
                                                   epochs = epochs)
        rrTrnPred = models[modelName].predict(rTempTrainNorm[:, outSize:],
                                                      batch_size = batch_size)
        # RIGHT-LEFT Training
        rlTrnPred = models[modelName].predict(lTempTrainNorm[:, outSize:],
                                                       batch_size = batch_size)
        
        # DeNormalize the predictions
        llValPredDeNorm = deNormData(llValPred, lTempValMean, lTempValStd)
        rrValPredDeNorm = deNormData(rrValPred, rTempValMean, rTempValStd)
        lrValPredDeNorm = deNormData(lrValPred, rTempValMean, rTempValStd)
        rlValPredDeNorm = deNormData(rlValPred, lTempValMean, lTempValStd)
        
        llTrnPredDeNorm = deNormData(llTrnPred, lTempTrainMean, lTempTrainStd)
        rrTrnPredDeNorm = deNormData(rrTrnPred, rTempTrainMean, rTempTrainStd)
        lrTrnPredDeNorm = deNormData(lrTrnPred, rTempTrainMean, rTempTrainStd)
        rlTrnPredDeNorm = deNormData(rlTrnPred, lTempTrainMean, lTempTrainStd)
        
        # Calculate the mean square error
        llMSE[i, :] = MSError(llValPredDeNorm, lPatient.tempVal[:, 0:outSize])
        llMSE[i+nFoldIter, :] = MSError(llTrnPredDeNorm, lPatient.tempTrain[:, 0:outSize])
        rrMSE[i, :] = MSError(rrValPredDeNorm, rPatient.tempVal[:, 0:outSize])
        rrMSE[i+nFoldIter, :] = MSError(rrTrnPredDeNorm, rPatient.tempTrain[:, 0:outSize])
        lrMSE[i, :] = MSError(lrValPredDeNorm, rPatient.tempVal[:, 0:outSize])
        lrMSE[i+nFoldIter, :] = MSError(lrTrnPredDeNorm, rPatient.tempTrain[:, 0:outSize])
        rlMSE[i, :] = MSError(rlValPredDeNorm, lPatient.tempVal[:, 0:outSize])
        rlMSE[i+nFoldIter, :] = MSError(rlTrnPredDeNorm, lPatient.tempTrain[:, 0:outSize])
    
    
    # Store the mean square error
    lFinalMSErrors = {
        "llMSE": llMSE,
        "lrMSE": lrMSE}
    rFinalMSErrors = {
        "rrMSE": rrMSE,
        "rlMSE": rlMSE}
    
    lPatient.mseStorage[modelName] = lFinalMSErrors
    rPatient.mseStorage[modelName] = rFinalMSErrors
    
    # Calculate and store the root mean square error
    llRMSE = np.sqrt(llMSE)
    rrRMSE = np.sqrt(rrMSE)
    lrRMSE = np.sqrt(lrMSE)
    rlRMSE = np.sqrt(rlMSE)
    
    lFinalRMSErrors = {
        "llRMSE": llRMSE,
        "lrRMSE": lrRMSE}
    rFinalRMSErrors = {
        "rrRMSE": rrRMSE,
        "rlRMSE": rlRMSE}
    
    lPatient.rmseStorage[modelName] = lFinalRMSErrors
    rPatient.rmseStorage[modelName] = rFinalRMSErrors
    
    lPatientfStatistic = fStatistic(llRMSE, lrRMSE, nFoldIter)
    rPatientfStatistic = fStatistic(rrRMSE, rlRMSE, nFoldIter)
    
    lPatient.fStorage[modelName] = {"pValues": lPatientfStatistic}
    rPatient.fStorage[modelName] = {"pValues": rPatientfStatistic}
        
        
        
        
def cvTrainingParallel(lPatient, rPatient, outSize, nFoldIter, kFold, lag, batch_size, epochs, models, modelName):
    """Cross validation training function uses model stored in patient object
    
    Arguments:
        lPatient = left arm patient object
        rPatient = right arm patient object
        outSize = size of output (K)
        nFoldIter = number of fold iterations
        kFold = number of folds 
        lag = amount to lag dataset
        batch_size = batch size for model training
        epochs = number of training epochs
        models = dictionary of models
        modelName = name of model stored in patient models dictionary
        
    Returns:
        None, updates error variables in patient object
    """
    
    # Initialize error variables
    llMSE = np.zeros([nFoldIter*2, outSize])
    rrMSE = np.zeros([nFoldIter*2, outSize])
    lrMSE = np.zeros([nFoldIter*2, outSize])
    rlMSE = np.zeros([nFoldIter*2, outSize])
    
    # Lag data if not already lagged
    if (len(lPatient.trainData.columns) < 2):
        pat.createLagData(lPatient.trainData, lag = lag, skip=0, dropNaN=True)
        pat.createLagData(rPatient.trainData, lag = lag, skip=0, dropNaN=True)
    
    for i in range(nFoldIter):
        
        # Randomize the training data with every iteration
        lPatient.randomizeTrainingData(kFold, seed=i)
        rPatient.randomizeTrainingData(kFold, seed=i)
        
        # Normalize the data
        [lTempTrainNorm, lTempTrainMean, lTempTrainStd] = zscoreData(lPatient.tempTrain)
        [lTempValNorm, lTempValMean, lTempValStd] = zscoreData(lPatient.tempVal)
        
        [rTempTrainNorm, rTempTrainMean, rTempTrainStd] = zscoreData(rPatient.tempTrain)
        [rTempValNorm, rTempValMean, rTempValStd] = zscoreData(rPatient.tempVal)
        
        if (len(lTempTrainNorm) < len(rTempTrainNorm)):
            normTrainInputs = np.append(lTempTrainNorm[:, outSize:], rTempTrainNorm[0:len(lTempTrainNorm), outSize:], axis=1)
            normValInputs = np.append(lTempValNorm[:, outSize:], rTempValNorm[0:len(lTempValNorm), outSize:], axis=1)
            lTempTrainNormComp = lTempTrainNorm
            lTempValNormComp = lTempValNorm
            rTempTrainNormComp = rTempTrainNorm[0:len(lTempTrainNorm), :]
            rTempValNormComp = rTempValNorm[0:len(lTempTrainNorm), :]
        elif (len(rTempTrainNorm) < len(lTempTrainNorm)):
            normTrainInputs = np.append(rTempTrainNorm[:, outSize:], lTempTrainNorm[0:len(rTempTrainNorm), outSize:], axis=1)
            normValInputs = np.append(rTempValNorm[:, outSize:], lTempValNorm[0:len(rTempValNorm), outSize:], axis=1)
            lTempTrainNormComp = lTempTrainNorm[0:len(rTempTrainNorm), :]
            lTempValNormComp = lTempValNorm[0:len(rTempTrainNorm), :]
            rTempTrainNormComp = rTempTrainNorm
            rTempValNormComp = rTempValNorm
        
        doubleLTempTrainNorm = np.append(lTempTrainNorm[:, outSize:], lTempTrainNorm[:, outSize:], axis=1)
        doubleLTempValNorm = np.append(lTempValNorm[:, outSize:], lTempValNorm[:, outSize:], axis=1)
        doubleRTempTrainNorm = np.append(rTempTrainNorm[:, outSize:], rTempTrainNorm[:, outSize:], axis=1)
        doubleRTempValNorm = np.append(rTempValNorm[:, outSize:], rTempValNorm[:, outSize:], axis=1)
        # ---------------------------------------------------
        #
        # HOW DO WE COMPARE WHEN TRAINING A PARALLEL NETWORK? 
        #
        # ---------------------------------------------------
        # LEFT-LEFT Training->Validation
        llTrnTrain = models[modelName].fit(normTrainInputs, 
                                                    lTempTrainNormComp[:, 0:outSize], 
                                                    batch_size = batch_size, 
                                                    epochs = epochs)
        llValPred = models[modelName].predict(doubleLTempValNorm, 
                                                      batch_size = batch_size)
        # LEFT-RIGHT Validation
        lrValPred = models[modelName].predict(doubleRTempValNorm, 
                                                       batch_size = batch_size)
        
        # LEFT-LEFT Validation->Training
        llValTrain = models[modelName].fit(normValInputs,
                                                   lTempValNormComp[:, 0:outSize],
                                                   batch_size = batch_size,
                                                   epochs = epochs)
        llTrnPred = models[modelName].predict(doubleLTempTrainNorm,
                                                      batch_size = batch_size)
        # LEFT-RIGHT Training
        lrTrnPred = models[modelName].predict(doubleRTempTrainNorm,
                                                       batch_size = batch_size)
        
        # RIGHT-RIGHT Training->Validation
        rrTrnTrain = models[modelName].fit(normTrainInputs, 
                                                   rTempTrainNormComp[:, 0:outSize], 
                                                   batch_size = batch_size, 
                                                   epochs = epochs)
        rrValPred = models[modelName].predict(doubleRTempValNorm, 
                                                      batch_size = batch_size)
        # RIGHT-LEFT Validation
        rlValPred = models[modelName].predict(doubleLTempValNorm, 
                                                       batch_size = batch_size)
        
        # RIGHT-RIGHT Validation->Training
        rrValTrain = models[modelName].fit(normValInputs,
                                                   rTempValNormComp[:, 0:outSize],
                                                   batch_size = batch_size,
                                                   epochs = epochs)
        rrTrnPred = models[modelName].predict(doubleRTempTrainNorm,
                                                      batch_size = batch_size)
        # RIGHT-LEFT Training
        rlTrnPred = models[modelName].predict(doubleLTempTrainNorm,
                                                       batch_size = batch_size)
        
        # DeNormalize the predictions
        llValPredDeNorm = deNormData(llValPred, lTempValMean, lTempValStd)
        rrValPredDeNorm = deNormData(rrValPred, rTempValMean, rTempValStd)
        lrValPredDeNorm = deNormData(lrValPred, rTempValMean, rTempValStd)
        rlValPredDeNorm = deNormData(rlValPred, lTempValMean, lTempValStd)
        
        llTrnPredDeNorm = deNormData(llTrnPred, lTempTrainMean, lTempTrainStd)
        rrTrnPredDeNorm = deNormData(rrTrnPred, rTempTrainMean, rTempTrainStd)
        lrTrnPredDeNorm = deNormData(lrTrnPred, rTempTrainMean, rTempTrainStd)
        rlTrnPredDeNorm = deNormData(rlTrnPred, lTempTrainMean, lTempTrainStd)
        
        # Calculate the mean square error
        llMSE[i, :] = MSError(llValPredDeNorm, lPatient.tempVal[:, 0:outSize])
        llMSE[i+nFoldIter, :] = MSError(llTrnPredDeNorm, lPatient.tempTrain[:, 0:outSize])
        rrMSE[i, :] = MSError(rrValPredDeNorm, rPatient.tempVal[:, 0:outSize])
        rrMSE[i+nFoldIter, :] = MSError(rrTrnPredDeNorm, rPatient.tempTrain[:, 0:outSize])
        lrMSE[i, :] = MSError(lrValPredDeNorm, rPatient.tempVal[:, 0:outSize])
        lrMSE[i+nFoldIter, :] = MSError(lrTrnPredDeNorm, rPatient.tempTrain[:, 0:outSize])
        rlMSE[i, :] = MSError(rlValPredDeNorm, lPatient.tempVal[:, 0:outSize])
        rlMSE[i+nFoldIter, :] = MSError(rlTrnPredDeNorm, lPatient.tempTrain[:, 0:outSize])
    
    
    # Store the mean square error
    lFinalMSErrors = {
        "llMSE": llMSE,
        "lrMSE": lrMSE}
    rFinalMSErrors = {
        "rrMSE": rrMSE,
        "rlMSE": rlMSE}
    
    lPatient.mseStorage[modelName] = lFinalMSErrors
    rPatient.mseStorage[modelName] = rFinalMSErrors
    
    # Calculate and store the root mean square error
    llRMSE = np.sqrt(llMSE)
    rrRMSE = np.sqrt(rrMSE)
    lrRMSE = np.sqrt(lrMSE)
    rlRMSE = np.sqrt(rlMSE)
    
    lFinalRMSErrors = {
        "llRMSE": llRMSE,
        "lrRMSE": lrRMSE}
    rFinalRMSErrors = {
        "rrRMSE": rrRMSE,
        "rlRMSE": rlRMSE}
    
    lPatient.rmseStorage[modelName] = lFinalRMSErrors
    rPatient.rmseStorage[modelName] = rFinalRMSErrors
    
    lPatientfStatistic = fStatistic(llRMSE, lrRMSE, nFoldIter)
    rPatientfStatistic = fStatistic(rrRMSE, rlRMSE, nFoldIter)
    
    lPatient.fStorage[modelName] = {"pValues": lPatientfStatistic}
    rPatient.fStorage[modelName] = {"pValues": rPatientfStatistic}
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    