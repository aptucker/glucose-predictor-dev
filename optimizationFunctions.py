# Written and designed by Aaron Tucker 
#
#
#
# ----------------------------------------------------------------------------
"""
Optimization functions 
"""

import pandas as pd

import patient as pat



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
        
    
        
    
