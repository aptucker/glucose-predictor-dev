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
import training as trn

with open("processed_data\\patient1.pickle", "rb") as f:
    L1, Rdat = pickle.load(f)
    
partNum = 1
partSize = [0.1]

lag = 6

L1.partitionData(partNum, partSize)

pat.createLagData(L1.trainData, lag, dropNaN=True)
pat.createLagData(L1.testData, lag, dropNaN=True)

# %%
Kfold = 2
nFoldIter = 5

H = 3
K = 4
D = lag+1

L1.randomizeTrainingData(Kfold, seed=1)
L1.initializeErrors(nFoldIter, K)


np.random.seed(1)
initializer1 = tf.keras.initializers.Constant(np.random.normal(0, 0.005, (H, 4)))
np.random.seed(1)
aInit = np.random.normal(0, 0.005, (H, D))

np.random.seed(4)
initializer2 = tf.keras.initializers.Constant(np.random.normal(0, 0.005, (H+1, K)))
np.random.seed(4)
bInit = np.random.normal(0, 0.005, (H+1, K))

[mlpNorm, mean, std] = trn.zscoreData(L1.tempTrain)

b_size = 1

# initializers = [initializer1, initializer2]
initializers = [tf.keras.initializers.RandomNormal(mean=0, stddev=0.005), tf.keras.initializers.RandomNormal(mean=0, stddev=0.005)]

model = cModels.sbSeqModel(H, K, use_bias = True, initializers = initializers, bias_size = b_size, activators = ["sigmoid", None])
model(tf.keras.Input(shape=H))

model.compile(optimizer= 'SGD', #tf.keras.optimizers.SGD(learning_rate=0.0001)
              loss=tf.keras.losses.MeanSquaredError(), 
              metrics=tf.keras.metrics.RootMeanSquaredError())

modelTest = model.fit(mlpNorm[:,4:7], mlpNorm[:,0:4], batch_size=b_size, epochs=5)

YNewTest = model.predict(mlpNorm[:,4:7], batch_size=b_size)

YReNorm = (YNewTest*std) + mean

# Output Mapping -> Column 3 = 15min, Column 0 = 60min

trnErrTest = trn.MSError(YReNorm[:,3], tf.reshape(Ltrn[:,3], [4671,1]))

L1.models.append(model)


# %%

# inputs = tf.keras.Input(shape=(3,))
# HLayer = cLayers.staticBiasLayer(units = 3, 
#                  activation = "sigmoid", 
#                  use_bias=True, 
#                  kernel_initializer = initializer1, 
#                  ones_size = b_size)(inputs)
# outLayer = cLayers.staticBiasLayer(units = K, 
#                    activation = None, 
#                    use_bias=True, 
#                    kernel_initializer = initializer2, 
#                    ones_size = b_size)(HLayer)

# model = tf.keras.Model(inputs = inputs, outputs = outLayer)

# model.compile(optimizer= 'SGD', #tf.keras.optimizers.SGD(learning_rate=0.0001)
#               loss=tf.keras.losses.MeanSquaredError(), 
#               metrics=tf.keras.metrics.RootMeanSquaredError())

# modelTest = model.fit(mlpNorm[:,4:7], mlpNorm[:,0:4], batch_size=b_size, epochs=5)

# YNewTest = model.predict(mlpNorm[:,4:7], batch_size=b_size)

# YReNorm = (YNewTest*std) + mean

# # Output Mapping -> Column 3 = 15min, Column 0 = 60min

# trnErrTest = pat.MSError(YReNorm[:,3], tf.reshape(Ltrn[:,3], [4671,1]))


# %%
# objects = []
# with (open("KerasVal.pickle", "rb")) as openfile:
#     while True:
#         try:
#             objects.append(pickle.load(openfile))
#         except EOFError:
#             break