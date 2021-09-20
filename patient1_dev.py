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
import customModels as cModels

with open("processed_data\\patient1.pickle", "rb") as f:
    L1, Rdat = pickle.load(f)
    
partNum = 1
partSize = [0.1]

lag = 6

L1.partitionData(partNum, partSize)

pat.createLagData(L1.trainData, lag, dropNaN=True)
pat.createLagData(L1.testData, lag, dropNaN=True)

# %%
Lsplit = int(len(L1.trainData)/2)
np.random.seed(1)
Lrand = L1.trainData.sample(frac=1)
Ltrn = tf.convert_to_tensor(Lrand[0:Lsplit].to_numpy(), dtype=tf.float32)
Lval = Lrand[Lsplit:-1].to_numpy()

H = 3
K = 4
D = lag+1

np.random.seed(1)
initializer1 = tf.keras.initializers.Constant(np.random.normal(0, 0.005, (H, 4)))
np.random.seed(1)
aInit = np.random.normal(0, 0.005, (H, D))

np.random.seed(4)
initializer2 = tf.keras.initializers.Constant(np.random.normal(0, 0.005, (H+1, K)))
np.random.seed(4)
bInit = np.random.normal(0, 0.005, (H+1, K))

[mlpNorm, mean, std] = pat.zscoreData(Ltrn)

b_size = 1

model = cModels.sbSeqModel(inShape = H, outShape = K, use_bias = True, initializers = [initializer1, initializer2], bias_size = b_size, activators = ["sigmoid", None])


