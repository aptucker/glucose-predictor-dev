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
               callbacks):
    
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
    
    lossDict = {}
    
    for i in range(len(modelNames)):                        
        outDict = runTimeTrials(lTrainNorm,
                              lTestNorm,
                              models, 
                              modelNames[i],
                              trialsToRun,
                              b_size,
                              epochs,
                              callbacks)
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
    
    outSize = 4
    
    # optimizers.setLearningRate(optimizerName,
    #                            learningRate)
    
    batchLossDict = {}
    
    for i in range(trialsToRun):
        
        models[modelName].load_weights(f'{modelName}' 'Model.start')
        
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
    
    