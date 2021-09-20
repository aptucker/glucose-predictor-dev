# Written and designed by Aaron Tucker 
#
#
#
# ----------------------------------------------------------------------------
"""
Patient class and related functions
"""

import numpy as np
import pandas as pd

class Patient:
    """This is the main class Patient. For now, this has attributes for diabetes
    type (0=not diabetic) and the CGM data."""
    
    def __init__(self, diabType, GenData, DayData):
        self.diabType = diabType
        self.GenData = GenData
        self.DayData = DayData
        self.trainData = []
        self.valData = []
        self.testData = []
        
        
    def partitionData(self, n, per, seed=None):
        """Method for splitting the patient data into training, test, and
        validation sets. If different random values are wanted, set different seed
        number. Note that the first partition becomes the test set.

        Arguments: 
            n = Number of new partitions (i.e. 2 means 3 total sets)
            per = Percent of each new partition (i.e. [0.1 0.1] is two new partitions
            with 10% of the data in each) 
            
        Returns: 
            None - updates testData, trainData, valData
        """
        
        datSize = int(self.GenData.size)
        partData = []
        
        oldData = self.GenData.copy()
        
        if seed is None:
            np.random.seed(0)
        else:
            np.random.seed(seed)
        
        for i in range(n):
            start = np.random.randint(int(round(datSize - (per[i]*datSize))))
            end = start + int(round(per[i]*datSize))
            partData.append(oldData[start:end])
            
            oldData.drop(oldData.index[start:end], inplace=True)
        
        self.testData = pd.DataFrame(partData[0])
        self.trainData = oldData
        
        if (n>1):
            for i in range(n-1):
                self.valData.append(partData[i+1])
            
            if (len(self.valData) < 2):
                self.valData = pd.DataFrame(self.valData[0])
                
    def resetData(self):
        """Eliminates data in trainData, valData, and testData so you can 
        re-partition w/o deleting the main dataset stored in GenData"""
        
        self.trainData = []
        self.valData = []
        self.testData = []



def createPatient(file, diabType='0'):
    """This function creates a new instance of a patient from a .csv file. Data 
    from the file is collected and stored in the patient object. This function
    automatically drops any exact duplicates of data as a result of sensor 
    error.
    
    Arguments:
        file - csv file 
        diabType - defaults to 0 meaning non-diabetic, this has no functionality
            at the moment.
    
    Returns: 
        newPat - new instance of a Patient class
    """
    
    patData = pd.read_csv(file, header=1, 
                     usecols = ['Meter Timestamp', 'Historic Glucose(mg/dL)'],
                     parse_dates=['Meter Timestamp'])
    
    patData.drop_duplicates(subset='Meter Timestamp',
                            keep='first', inplace=True)
    
    patData.set_index('Meter Timestamp', inplace=True)
    
    DailyData = []
    for group in patData.groupby(patData.index.date):
        DailyData.append(group[1])
    
    newPat = Patient(diabType, patData, DailyData)
    return newPat