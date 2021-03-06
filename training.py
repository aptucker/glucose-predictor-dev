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

def MARD(Y, y_trn):
    """Calculate Mean Absolute Relative difference from prediction and labeled data.
    
    Arguments: 
        Y - predicted data
        y_trn - labeled training data
    """
    MARD = (1/len(Y)) * np.sum(np.divide(np.absolute(y_trn-Y), y_trn), axis=0)
    return MARD

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
    
    fpbar = (fp1 + fp2)/2
    fs = np.square(fp1 - fpbar) + np.square(fp2 - fpbar)
    f = np.divide(np.sum(np.square(fp1) + np.square(fp2), axis=0), 2*np.sum(fs, axis=0))
    fp = 1 - stats.f.cdf(f, 10, 5)
    return fp
    

def cvTraining(lPatient,
               rPatient,
               outSize,
               nFoldIter,
               kFold,
               lag,
               skip,
               batch_size,
               epochs,
               models,
               modelName,
               callbacks,
               lossWeights,
               reComp=False):
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
    
    llMARD = np.zeros([nFoldIter*2, outSize])
    rrMARD = np.zeros([nFoldIter*2, outSize])
    lrMARD = np.zeros([nFoldIter*2, outSize])
    rlMARD = np.zeros([nFoldIter*2, outSize])
    
    if modelName == "GRU H=1":
        models[modelName].save_weights('model.start')
        
    else:
        modelWeightsStart = models[modelName].get_weights()
        
    
    # Lag data if not already lagged
    if (len(lPatient.trainData.columns) < 2):
        pat.createLagData(lPatient.trainData, lag = lag, skip=skip, dropNaN=True)
        pat.createLagData(rPatient.trainData, lag = lag, skip=skip, dropNaN=True)
    
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
        if modelName == "GRU H=1":
            # if reComp == False:
                # models[modelName].load_weights('model.start')
            if reComp == True:
                models[modelName].compile(optimizer= 'SGD', #tf.keras.optimizers.SGD(learning_rate=0.0001)
                  loss=tf.keras.losses.MeanSquaredError(), 
                  metrics=tf.keras.metrics.RootMeanSquaredError(),
                  loss_weights=lossWeights[0])
                models[modelName].load_weights('model.start')
        
        else:
            models[modelName].set_weights(modelWeightsStart)
        
        llTrnTrain = models[modelName].fit(lTempTrainNorm[:, outSize:], 
                                                    lTempTrainNorm[:, 0:outSize], 
                                                    batch_size = batch_size, 
                                                    epochs = epochs,
                                                    callbacks = callbacks[0])
        llValPred = models[modelName].predict(lTempValNorm[:, outSize:], 
                                                      batch_size = batch_size)
        # LEFT-RIGHT Validation
        lrValPred = models[modelName].predict(rTempValNorm[:, outSize:], 
                                                       batch_size = batch_size)
        
        models[modelName].reset_metrics()
        models[modelName].reset_states()
        
        # LEFT-LEFT Validation->Training
        if modelName == "GRU H=1":
            # if reComp == False:
                # models[modelName].load_weights('model.start')
            if reComp == True:
                models[modelName].compile(optimizer= 'SGD', #tf.keras.optimizers.SGD(learning_rate=0.0001)
                  loss=tf.keras.losses.MeanSquaredError(), 
                  metrics=tf.keras.metrics.RootMeanSquaredError(),
                  loss_weights=lossWeights[0])
                models[modelName].load_weights('model.start')
                
        else:
            models[modelName].set_weights(modelWeightsStart)
        
        llValTrain = models[modelName].fit(lTempValNorm[:, outSize:],
                                                   lTempValNorm[:, 0:outSize],
                                                   batch_size = batch_size,
                                                   epochs = epochs,
                                                   callbacks = callbacks[1])
        llTrnPred = models[modelName].predict(lTempTrainNorm[:, outSize:],
                                                      batch_size = batch_size)
        # LEFT-RIGHT Training
        lrTrnPred = models[modelName].predict(rTempTrainNorm[:, outSize:],
                                                       batch_size = batch_size)
        
        models[modelName].reset_metrics()
        models[modelName].reset_states()
        
        # RIGHT-RIGHT Training->Validation
        if modelName == "GRU H=1":
            # if reComp == False:
                # models[modelName].load_weights('model.start')
            if reComp == True:
                models[modelName].compile(optimizer= 'SGD', #tf.keras.optimizers.SGD(learning_rate=0.0001)
                  loss=tf.keras.losses.MeanSquaredError(), 
                  metrics=tf.keras.metrics.RootMeanSquaredError(),
                  loss_weights=lossWeights[0])
                models[modelName].load_weights('model.start')
                
        else:
            models[modelName].set_weights(modelWeightsStart)
        
        rrTrnTrain = models[modelName].fit(rTempTrainNorm[:, outSize:], 
                                                   rTempTrainNorm[:, 0:outSize], 
                                                   batch_size = batch_size, 
                                                   epochs = epochs,
                                                   callbacks = callbacks[2])
        rrValPred = models[modelName].predict(rTempValNorm[:, outSize:], 
                                                      batch_size = batch_size)
        # RIGHT-LEFT Validation
        rlValPred = models[modelName].predict(lTempValNorm[:, outSize:], 
                                                       batch_size = batch_size)
        
        models[modelName].reset_metrics()
        models[modelName].reset_states()
        
        # RIGHT-RIGHT Validation->Training
        if modelName == "GRU H=1":
            # if reComp == False:
                # models[modelName].load_weights('model.start')
            if reComp == True:
                models[modelName].compile(optimizer= 'SGD', #tf.keras.optimizers.SGD(learning_rate=0.0001)
                  loss=tf.keras.losses.MeanSquaredError(), 
                  metrics=tf.keras.metrics.RootMeanSquaredError(),
                  loss_weights=lossWeights[0])
                models[modelName].load_weights('model.start')
                
        else:
            models[modelName].set_weights(modelWeightsStart)
        
        rrValTrain = models[modelName].fit(rTempValNorm[:, outSize:],
                                                   rTempValNorm[:, 0:outSize],
                                                   batch_size = batch_size,
                                                   epochs = epochs,
                                                   callbacks = callbacks[3])
        rrTrnPred = models[modelName].predict(rTempTrainNorm[:, outSize:],
                                                      batch_size = batch_size)
        # RIGHT-LEFT Training
        rlTrnPred = models[modelName].predict(lTempTrainNorm[:, outSize:],
                                                       batch_size = batch_size)
        
        models[modelName].reset_metrics()
        models[modelName].reset_states()
        
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
    
        # Calculate the mean absolute relative difference
        llMARD[i, :] = MARD(llValPredDeNorm, lPatient.tempVal[:, 0:outSize])
        llMARD[i+nFoldIter, :] = MARD(llTrnPredDeNorm, lPatient.tempTrain[:, 0:outSize])
        rrMARD[i, :] = MARD(rrValPredDeNorm, rPatient.tempVal[:, 0:outSize])
        rrMARD[i+nFoldIter, :] = MARD(rrTrnPredDeNorm, rPatient.tempTrain[:, 0:outSize])
        lrMARD[i, :] = MARD(lrValPredDeNorm, rPatient.tempVal[:, 0:outSize])
        lrMARD[i+nFoldIter, :] = MARD(lrTrnPredDeNorm, rPatient.tempTrain[:, 0:outSize])
        rlMARD[i, :] = MARD(rlValPredDeNorm, lPatient.tempVal[:, 0:outSize])
        rlMARD[i+nFoldIter, :] = MARD(rlTrnPredDeNorm, lPatient.tempTrain[:, 0:outSize])
    
    # Store the mean absolute relative difference
    lFinalMARD = {
        'llMARD': llMARD,
        'rlMARD': rlMARD}
    rFinalMARD = {
        'rrMARD': rrMARD,
        'lrMARD': lrMARD}
    
    lPatient.mardStorage[modelName] = lFinalMARD
    rPatient.mardStorage[modelName] = rFinalMARD
    
    # Store the mean square error
    lFinalMSErrors = {
        "llMSE": llMSE,
        "rlMSE": rlMSE}
    rFinalMSErrors = {
        "rrMSE": rrMSE,
        "lrMSE": lrMSE}
    
    lPatient.mseStorage[modelName] = lFinalMSErrors
    rPatient.mseStorage[modelName] = rFinalMSErrors
    
    # Calculate and store the root mean square error
    llRMSE = np.sqrt(llMSE)
    rrRMSE = np.sqrt(rrMSE)
    lrRMSE = np.sqrt(lrMSE)
    rlRMSE = np.sqrt(rlMSE)
    
    lFinalRMSErrors = {
        "llRMSE": llRMSE,
        "rlRMSE": rlRMSE}
    rFinalRMSErrors = {
        "rrRMSE": rrRMSE,
        "lrRMSE": lrRMSE}
    
    lPatient.rmseStorage[modelName] = lFinalRMSErrors
    rPatient.rmseStorage[modelName] = rFinalRMSErrors
    
    # Calculate and store the f statistic p-value
    lPatientfStatistic = fStatistic(llRMSE, rlRMSE, nFoldIter)
    rPatientfStatistic = fStatistic(rrRMSE, lrRMSE, nFoldIter)
    
    lPatient.fStorage[modelName] = {"pValues": lPatientfStatistic}
    rPatient.fStorage[modelName] = {"pValues": rPatientfStatistic}
        
        
        
        
def cvTrainingParallel(lPatient,
                       rPatient,
                       outSize,
                       nFoldIter,
                       kFold,
                       lag,
                       skip,
                       batch_size,
                       epochs,
                       models,
                       modelName,
                       callbacks):
    """Cross validation training function for parallel models
    
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
    
    llMARD = np.zeros([nFoldIter*2, outSize])
    rrMARD = np.zeros([nFoldIter*2, outSize])
    lrMARD = np.zeros([nFoldIter*2, outSize])
    rlMARD = np.zeros([nFoldIter*2, outSize])
    
    # Lag data if not already lagged
    if (len(lPatient.trainData.columns) < 2):
        pat.createLagData(lPatient.trainData, lag = lag, skip=skip, dropNaN=True)
        pat.createLagData(rPatient.trainData, lag = lag, skip=skip, dropNaN=True)
    
    for i in range(nFoldIter):
        
        # Randomize the training data with every iteration
        lPatient.randomizeTrainingData(kFold, seed=i)
        rPatient.randomizeTrainingData(kFold, seed=i)
        
        # Normalize the data
        [lTempTrainNorm, lTempTrainMean, lTempTrainStd] = zscoreData(lPatient.tempTrain)
        [lTempValNorm, lTempValMean, lTempValStd] = zscoreData(lPatient.tempVal)
        
        [rTempTrainNorm, rTempTrainMean, rTempTrainStd] = zscoreData(rPatient.tempTrain)
        [rTempValNorm, rTempValMean, rTempValStd] = zscoreData(rPatient.tempVal)
        
        # Set the input data to be the same size based on whichever patient has more data        
        if (len(lTempTrainNorm) <= len(rTempTrainNorm)):
            normTrainInputs = np.append(lTempTrainNorm[:, outSize:], rTempTrainNorm[0:len(lTempTrainNorm), outSize:], axis=1)
            normValInputs = np.append(lTempValNorm[:, outSize:], rTempValNorm[0:len(lTempValNorm), outSize:], axis=1)
            lTempTrainNormComp = lTempTrainNorm
            lTempValNormComp = lTempValNorm
            # rTempTrainNormComp = rTempTrainNorm[0:len(lTempTrainNorm), :]
            rTempTrainNormComp = rTempTrainNorm[0:len(normTrainInputs), :]
            # rTempValNormComp = rTempValNorm[0:len(lTempTrainNorm), :]
            rTempValNormComp = rTempValNorm[0:len(normValInputs), :]
        elif (len(rTempTrainNorm) <= len(lTempTrainNorm)):
            normTrainInputs = np.append(rTempTrainNorm[:, outSize:], lTempTrainNorm[0:len(rTempTrainNorm), outSize:], axis=1)
            normValInputs = np.append(rTempValNorm[:, outSize:], lTempValNorm[0:len(rTempValNorm), outSize:], axis=1)
            # lTempTrainNormComp = lTempTrainNorm[0:len(rTempTrainNorm), :]
            lTempTrainNormComp = lTempTrainNorm[0:len(normTrainInputs), :]
            # lTempValNormComp = lTempValNorm[0:len(rTempTrainNorm), :]
            lTempValNormComp = lTempValNorm[0:len(normValInputs), :]
            rTempTrainNormComp = rTempTrainNorm
            rTempValNormComp = rTempValNorm
        
        # Duplicate the datasets within one array for prediction function
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
                                                    epochs = epochs,
                                                    callbacks = callbacks)
        llValPred = models[modelName].predict(doubleLTempValNorm, 
                                                      batch_size = batch_size)
        # LEFT-RIGHT Validation
        lrValPred = models[modelName].predict(doubleRTempValNorm, 
                                                       batch_size = batch_size)
        
        # LEFT-LEFT Validation->Training
        llValTrain = models[modelName].fit(normValInputs,
                                                   lTempValNormComp[:, 0:outSize],
                                                   batch_size = batch_size,
                                                   epochs = epochs,
                                                   callbacks = callbacks)
        llTrnPred = models[modelName].predict(doubleLTempTrainNorm,
                                                      batch_size = batch_size)
        # LEFT-RIGHT Training
        lrTrnPred = models[modelName].predict(doubleRTempTrainNorm,
                                                       batch_size = batch_size)
        
        # RIGHT-RIGHT Training->Validation
        rrTrnTrain = models[modelName].fit(normTrainInputs, 
                                                   rTempTrainNormComp[:, 0:outSize], 
                                                   batch_size = batch_size, 
                                                   epochs = epochs,
                                                   callbacks = callbacks)
        rrValPred = models[modelName].predict(doubleRTempValNorm, 
                                                      batch_size = batch_size)
        # RIGHT-LEFT Validation
        rlValPred = models[modelName].predict(doubleLTempValNorm, 
                                                       batch_size = batch_size)
        
        # RIGHT-RIGHT Validation->Training
        rrValTrain = models[modelName].fit(normValInputs,
                                                   rTempValNormComp[:, 0:outSize],
                                                   batch_size = batch_size,
                                                   epochs = epochs,
                                                   callbacks = callbacks)
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
        
        # Calculate the mean absolute relative difference
        llMARD[i, :] = MARD(llValPredDeNorm, lPatient.tempVal[:, 0:outSize])
        llMARD[i+nFoldIter, :] = MARD(llTrnPredDeNorm, lPatient.tempTrain[:, 0:outSize])
        rrMARD[i, :] = MARD(rrValPredDeNorm, rPatient.tempVal[:, 0:outSize])
        rrMARD[i+nFoldIter, :] = MARD(rrTrnPredDeNorm, rPatient.tempTrain[:, 0:outSize])
        lrMARD[i, :] = MARD(lrValPredDeNorm, rPatient.tempVal[:, 0:outSize])
        lrMARD[i+nFoldIter, :] = MARD(lrTrnPredDeNorm, rPatient.tempTrain[:, 0:outSize])
        rlMARD[i, :] = MARD(rlValPredDeNorm, lPatient.tempVal[:, 0:outSize])
        rlMARD[i+nFoldIter, :] = MARD(rlTrnPredDeNorm, lPatient.tempTrain[:, 0:outSize])
    
    # Store the mean absolute relative difference
    lFinalMARD = {
        'llMARD': llMARD,
        'rlMARD': rlMARD}
    rFinalMARD = {
        'rrMARD': rrMARD,
        'lrMARD': lrMARD}
    
    lPatient.mardStorage[modelName] = lFinalMARD
    rPatient.mardStorage[modelName] = rFinalMARD
    
    
     # Store the mean square error
    lFinalMSErrors = {
        "llMSE": llMSE,
        "lrMSE": rlMSE}
    rFinalMSErrors = {
        "rrMSE": rrMSE,
        "rlMSE": lrMSE}
    
    lPatient.mseStorage[modelName] = lFinalMSErrors
    rPatient.mseStorage[modelName] = rFinalMSErrors
    
    # Calculate and store the root mean square error
    llRMSE = np.sqrt(llMSE)
    rrRMSE = np.sqrt(rrMSE)
    lrRMSE = np.sqrt(lrMSE)
    rlRMSE = np.sqrt(rlMSE)
    
    lFinalRMSErrors = {
        "llRMSE": llRMSE,
        "rlRMSE": rlRMSE}
    rFinalRMSErrors = {
        "rrRMSE": rrRMSE,
        "lrRMSE": lrRMSE}
    
    lPatient.rmseStorage[modelName] = lFinalRMSErrors
    rPatient.rmseStorage[modelName] = rFinalRMSErrors
    
    # Calculate and store the f statistic p-value
    lPatientfStatistic = fStatistic(llRMSE, rlRMSE, nFoldIter)
    rPatientfStatistic = fStatistic(rrRMSE, lrRMSE, nFoldIter)
    
    lPatient.fStorage[modelName] = {"pValues": lPatientfStatistic}
    rPatient.fStorage[modelName] = {"pValues": rPatientfStatistic}
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    