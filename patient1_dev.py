# Written and designed by Aaron Tucker 
#
#
#
# ----------------------------------------------------------------------------
"""
Patient 1 dev file - for all initial testing 
"""
# %% Imports and Data Loading
import tensorflow as tf
import numpy as np
import pickle
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import matplotlib.font_manager as font_manager

import patient as pat
import customLayers as cLayers

with open("processed_data\\patient1.pickle", "rb") as f:
    Ldat, Rdat = pickle.load(f)
    
