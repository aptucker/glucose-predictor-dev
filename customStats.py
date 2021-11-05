# Written and designed by Aaron Tucker 
#
#
#
# ----------------------------------------------------------------------------
"""
Custom statistical analyses functions
"""

import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

def adf_test(timeseries, toPrint = False):
    """Augmented Dickey-Fuller Test determines the presence of a unit root in
    a time series.
    
    Null Hypothesis: The series has a unit root (non-stationary)
    
    Inputs: 
        timeseries - the time series to be examined
        toPrint - default False; whether or not to print results
    
    Returns: 
        Test Statistic
        p-value
        #Lags Used
        Number of Observations Used
    """
    
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    if toPrint == True:
        print("Results of Dickey-Fuller Test:")
        print(dfoutput)
    
    return dfoutput
    
    
def kpss_test(timeseries, toPrint = False):
    """Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test examines whether a series
    is trend stationary
    
    Null Hypothesis: The process is trend stationary
    
    Inputs: 
        timeseries - the time series to be examined
        toPrint - default False; whether or not to print results
    
    Returns: 
        Test Statistic
        p-value
        #Lags Used
        Number of Observations Used
    """        
    
    kpsstest = kpss(timeseries, regression="c", nlags="auto")
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    
    if toPrint == True:
        print("Results of KPSS Test:")
        print(kpss_output)
    
    return kpss_output
    
