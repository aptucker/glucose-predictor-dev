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
        # Diabetic type of the patient (unused)
        self.diabType = diabType
        # Store general unmodified data
        self.GenData = GenData
        # Currently unused
        self.DayData = DayData
        # Store training data in pandas form
        self.trainData = []
        # Store validation data in pandas form
        self.valData = []
        # Store test data in pandas form
        self.testData = []
        # Store the models to test in a list
        self.models = dict()
        # Temporary training data in numpy form for cross validation training
        self.tempTrain = []
        # Temporary validation data in numpy form for cross validation training
        self.tempVal = []
        # MSE rates for models 
        self.mseStorage = {}
        # RMSE rates for models
        self.rmseStorage = {}
        # F-statistic storage for models
        self.fStorage = {}
        # MARD Storage
        self.mardStorage = {}
        # Timing Storage
        self.timeStorage = {}
        
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
    
    def randomizeTrainingData(self, Kfold, seed=None):
        """Function for randomly (repeatably if you pass a seed) splitting 
        the training data into temporary train and validation sets
        
        Arguments:
            Kfold = number of splits (Kfold - 1) e.g. Kfold = 2 will give 1 
            halfway split
            seed = the random seed initializer for repeatable splits
            
        Returns:
            Updates self.tempVal and self.tempTrain
        """
        
        splitPoint = int(len(self.trainData)/Kfold)
        
        np.random.seed(seed)
        randData = self.trainData.sample(frac=1)
        randData = randData.to_numpy()
        self.tempVal = randData[0:splitPoint]
        self.tempTrain = randData[splitPoint:-1]
    
    
    
    # CHANGE THIS 
    # def saveModelRMSE(self, nFoldIter, outSize, modelName):
        
        
                
    def resetData(self):
        """Eliminates data in trainData, valData, and testData so you can 
        re-partition w/o deleting the main dataset stored in GenData"""
        
        self.trainData = []
        self.valData = []
        self.testData = []
    
    def resetModels(self):
        self.models = {}



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


def createLagData(data, lag, skip=None, dropNaN=True):
    """Create a simple lagged data set.
    
    Arguments:
        data - data to be lagged
        lag - how many time steps to lag
        skip - remove columns at the front end of the lag
        dropNaN - drop rows which have NaN to avoid errors
    """
    
    firstData = data.copy()
    
    for i in range(lag):
        data.insert(i+1, "Lag: {}".format(i+1), firstData.shift(periods=i+1))
    
    # It's easiest to remove values which don't have a corresponding lag. Note
    # that the more values lagged, the less training data there is.
    
    if dropNaN:
        data.dropna(inplace=True)
        
    # for e in range(skip):
    if (skip != None):
        for e in skip:
            data.drop(columns=["Lag: {}".format(e+1)], inplace=True)
        
