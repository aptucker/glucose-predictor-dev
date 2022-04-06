# Written and designed by Aaron Tucker 
#
#
#
# ----------------------------------------------------------------------------
"""
Optimization functions 
"""

import numpy as np
import pandas as pd
import tensorflow as tf

import patient as pat
import training as trn
import customModels as cModels
import customCallbacks as cBacks



def dataCombiner(lPats,
                 rPats,
                 partNum,
                 partSize,
                 lag):
    
    """Function to combine and lag multiple patients worth of data
    
    Inputs:
        lPats - list of left-arm patient data to combine
        rPats - list of right-arm patient data to combine
        partNum - number of splits for the data 
        partSize - size of data splits
        lag - number of data points to lag
        
    Outputs:
        lTrainDataOut - dataframe of combined left-arm training data
        rTrainDataOut - dataframe of combined right-arm training data
        lTestDataOut - dataframe of combined left-arm test data
        rTestDataOut - dataframe of combined right-arm test data
    
    """
    
    for i in range(len(lPats)):
        
        lPats[i].resetData()
        rPats[i].resetData()
        
        lPats[i].partitionData(partNum, partSize)
        rPats[i].partitionData(partNum, partSize)

        pat.createLagData(lPats[i].trainData, lag, skip = None, dropNaN=True)
        pat.createLagData(lPats[i].testData, lag, skip = None, dropNaN=True)
        pat.createLagData(rPats[i].trainData, lag, skip = None, dropNaN=True)
        pat.createLagData(rPats[i].testData, lag, skip = None, dropNaN=True)
        
        if  i == 0:
            lTrainDataOut = lPats[i].trainData.copy()
            rTrainDataOut = rPats[i].trainData.copy()
            lTestDataOut = lPats[i].testData.copy()
            rTestDataOut = rPats[i].testData.copy()
        else:
            lTrainDataOut = pd.concat([lTrainDataOut, lPats[i].trainData.copy()], ignore_index=True)
            rTrainDataOut = pd.concat([rTrainDataOut, rPats[i].trainData.copy()], ignore_index=True)
            lTestDataOut = pd.concat([lTestDataOut, lPats[i].testData.copy()], ignore_index=True)
            rTestDataOut = pd.concat([rTestDataOut, rPats[i].testData.copy()], ignore_index=True)
        
    return [lTrainDataOut, rTrainDataOut, lTestDataOut, rTestDataOut]
        
    

def timeTester(lPats,
               rPats,
               partNum,
               partSize,
               lag, 
               models,
               modelNames,
               b_size,
               epochs,
               trialsToRun,
               maxDataSize=None):
    
    """Function to run a full time trial of multiple models
    
    Inputs:
        lPats - left arm data to test with
        rPats - right arm data to test with
        partNum - number of splits to make for train/validation
        partSize - train/validation partition fraction 
        lag - time delay value (should be 6)
        models - models to test
        modelNames - list of model names to test
        b_size - batch size
        epochs - number of epochs to run
        trialsToRun - number of repeated trials to run
        maxDataSize - maximum amount of data to test -> should be divisble by b_size if running JDST model
        
    Outputs:
        lossDict - dictionary of losses and times
    """
    
    [lTrainLarge,
     rTrainLarge,
     lTestLarge,
     rTestLarge] = dataCombiner(lPats,
                                rPats,
                                partNum,
                                partSize,
                                lag)
                                
    [lTrainNorm, lTrainMean, lTrainSTD] = trn.zscoreData(lTrainLarge.copy().sample(frac=1).to_numpy())
    [rTrainNorm, rTrainMean, rTrainSTD] = trn.zscoreData(rTrainLarge.copy().sample(frac=1).to_numpy())
    [lTestNorm, lTestMean, lTestSTD] = trn.zscoreData(lTestLarge.copy().sample(frac=1).to_numpy())
    [rTestNorm, rTestMean, rTestSTD] = trn.zscoreData(rTestLarge.copy().sample(frac=1).to_numpy())
    
    if maxDataSize != None:
        lTrainNorm = lTrainNorm[0:maxDataSize]
        rTrainNorm = rTrainNorm[0:maxDataSize]
        lTestNorm = lTestNorm[0:int(maxDataSize*partSize[0])]
        rTestNorm = rTestNorm[0:int(maxDataSize*partSize[0])]
    
    lossDict = {}
    # if inDict == None:
    #     lossDict = {}
        
    # else:
    #     lossDict = inDict
    
    for i in range(len(modelNames)):                        
        outDict = runTimeTrials(lTrainNorm,
                              lTestNorm,
                              models, 
                              modelNames[i],
                              trialsToRun,
                              b_size,
                              epochs,
                              models[modelNames[i]].callbacks)
        lossDict[modelNames[i]] = outDict
    
    
    return lossDict
        
        
    


def runTimeTrials(trainTrialData,
                  testTrialData,
                  models,
                  modelName,
                  trialsToRun,
                  b_size,
                  epochs,
                  callbacks):
    
    """Function to run time trials on one model
    
    Inputs:
        trainTrialData - training data to use for fit
        testTrialData - test data to use for fit
        models - dictionary of models
        modelName - name of model to test
        trialsToRun - number of repeat fits to run
        b_size - batch size
        epochs - number of epochs to run
        callbacks - model callbacks
    
    Outputs:
        batchLossDict - dictionary of losses on a batch level
    """
    
    outSize = 4
    
    # optimizers.setLearningRate(optimizerName,
    #                            learningRate)
    
    batchLossDict = {}
    
    for i in range(trialsToRun):
        
        if modelName != 'jdst':
            models[modelName].load_weights('weights_storage\\' f'{modelName}' 'Model.start')
            
        elif modelName == 'jdst':
            lr = models[modelName].optimizer.lr
            
            np.random.seed(1)
            initializer1 = tf.keras.initializers.Constant(np.random.normal(0, 0.005, (3, 4)))

            np.random.seed(4)
            initializer2 = tf.keras.initializers.Constant(np.random.normal(0, 0.005, (3+1, 4)))
            
            initializers = [initializer1, initializer2]
            
            models[modelName] = cModels.sbSeqModel(3, 
                                                   4,
                                                   use_bias = True,
                                                   initializers = initializers,
                                                   bias_size = b_size,
                                                   activators = ["sigmoid", None])
            models[modelName](tf.keras.Input(shape=3))
            
            models[modelName].compile(optimizer= tf.keras.optimizers.SGD(learning_rate=lr), 
                                      loss=tf.keras.losses.MeanSquaredError(), 
                                      metrics=tf.keras.metrics.RootMeanSquaredError())
            
            models[modelName].callbacks = [cBacks.EarlyStoppingAtMinLoss(patience=20, baseLoss=0.25),
                                           cBacks.batchErrorModel()]
        
        history = models[modelName].fit(trainTrialData[:, outSize:],
                                        trainTrialData[:, 0:outSize],
                                        batch_size=b_size,
                                        epochs=epochs,
                                        validation_data = (testTrialData[:, outSize:],
                                                           testTrialData[:, 0:outSize]),
                                        callbacks=callbacks)
        
        if 'newLoss' in models[modelName].lossDict.keys():
            batchLossDict['Loss It. ' f'{i+1}'] = pd.DataFrame(models[modelName].lossDict['newLoss'],
                                                               index=np.linspace(0,
                                                                                 models[modelName].trainTime,
                                                                                 len(models[modelName].lossDict['newLoss'])),
                                                               columns=['Loss It. ' f'{i+1}'])
            
    return batchLossDict
        
        
        
        


def findConvergenceTime(dfIn, averageWindow, threshold):
    
    """Find the convergence time using moving average of model results
    
    Inputs:
        dfIn - batch loss data to find time
        averageWindow - moving average value
        threshold - convergence threshold
    
    Outputs:
        timeOut - convergence time
        
    """
    
    dfMean = dfIn.rolling(averageWindow).mean()
    
    timeOut = dfMean[dfMean < threshold].first_valid_index()
    
    return timeOut

def findTimeConstant(dfIn):
    
    """Find the time constant from model batch loss training results
    
    Inputs:
        dfIn - batch loss data
        
    Outputs:
        timeConstant - calculated time constant
        
    """
    
    # tcVal = dfIn.max()*0.37
    tcVal = dfIn.iloc[20]*0.37
    
    timeConstant = dfIn[dfIn < tcVal].first_valid_index()
    
    return timeConstant

def compileTimeTrialResults(dfIn, modelNames, averageWindow, threshold):
    
    """Combine convergence time results into a dataframe
    
    Inputs:
        dfIn - dictionary input of batch losses
        modelNames - names of models to compile
        averageWindow - moving average window for convergence function
        threshold - convergence threshold
    
    Outputs:
        timeDF - dataframe of convergence times
    
    """
    
    tempList = []
    timeDF = pd.DataFrame()
    
    for i in range(len(modelNames)):
        for e in range(len(dfIn[modelNames[i]])):
            convTime = findConvergenceTime(dfIn[modelNames[i]]['Loss It. ' f'{e+1}'], 
                                           averageWindow=averageWindow,
                                           threshold=threshold)
            
            tempList.append(convTime)
        
        timeDF[modelNames[i]] = tempList
        
        tempList = []
        
    
    return timeDF

class optimizerDict:
    
    def __init__(self):
        super(optimizerDict, self).__init__()
        self.optimizers = {}
    
    def storeOptimizer(self, optimizerIn, optimizerName):
        self.optimizers[optimizerName] = optimizerIn
        
    def getOptimizer(self, optimizerName):
        return self.optimizers[optimizerName]
    
    def setLearningRate(self, optimizerName, learningRate):
        self.optimizers[optimizerName].learning_rate = learningRate
    
        
class lrDict:
    
    def __init__(self):
        super(lrDict, self).__init__()
        self.lrs = {}
        
    def storeLR(self, lrIn, lrName):        
        self.lrs[lrName] = lrIn
        
    def getLR(self, lrName):        
        return self.lrs[lrName]
    
    