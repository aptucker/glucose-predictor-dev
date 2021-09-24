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
    L1, R1 = pickle.load(f)

# L1 = pat.createPatient("..\\raw_data\\CGM1Left.csv", 0)
# R1 = pat.createPatient("..\\raw_data\\CGM1Right.csv", 0)
    
models = {}    

partNum = 1
partSize = [0.1]

lag = 6

Kfold = 2
nFoldIter = 5

H = 3
K = 4
D = lag+1

b_size = 1
epochs = 5

L1.partitionData(partNum, partSize)
R1.partitionData(partNum, partSize)

pat.createLagData(L1.trainData, lag, dropNaN=True)
pat.createLagData(L1.testData, lag, dropNaN=True)
pat.createLagData(R1.trainData, lag, dropNaN=True)
pat.createLagData(R1.testData, lag, dropNaN=True)

# %% JDST Model 

L1.randomizeTrainingData(Kfold, seed=1)

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

models["JDST"] = model

# trn.cvTraining(L1, R1, K, nFoldIter, Kfold, lag, b_size, 5, models, "JDST")

# modelTest = model.fit(mlpNorm[:,4:7], mlpNorm[:,0:4], batch_size=b_size, epochs=5)

# YNewTest = model.predict(mlpNorm[:,4:7], batch_size=b_size)

# YReNorm = (YNewTest*std) + mean

# # Output Mapping -> Column 3 = 15min, Column 0 = 60min

# trnErrTest = trn.MSError(YReNorm, L1.tempTrain[:,0:4])
# # trnErrTest = trn.MSError(np.reshape(YReNorm[:,3], [4671, 1]), tf.reshape(L1.tempTrain[:,3], [4671,1]))




# %% Parallel Network
L1.randomizeTrainingData(Kfold, seed=1)
[mlpNorm, mean, std] = trn.zscoreData(L1.tempTrain)


inputs = tf.keras.Input(shape=(H,))

tower1 = cLayers.staticBiasLayer(H,
                                 activation = 'sigmoid',
                                 use_bias = True,
                                 kernel_initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=0.005),
                                 ones_size = b_size)(inputs)
tower1 = cLayers.staticBiasLayer(K,
                                 activation = None,
                                 use_bias=True,
                                 kernel_initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=0.005),
                                 ones_size = b_size)(tower1)

tower2 = cLayers.staticBiasLayer(H,
                                 activation = 'sigmoid',
                                 use_bias = True,
                                 kernel_initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=0.005),
                                 ones_size = b_size)(inputs)
tower2 = cLayers.staticBiasLayer(K,
                                 activation = None,
                                 use_bias=True,
                                 kernel_initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=0.005),
                                 ones_size = b_size)(tower2)

merged = tf.keras.layers.concatenate([tower1, tower2], axis=0)
# outputs = cLayers.staticBiasLayer(2,
#                                   activation = None,
#                                   use_bias=True,
#                                   kernel_initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=0.005),
#                                   ones_size = b_size)(merged)
outputs = tf.keras.layers.Dense(1, activation=None)(tf.transpose(merged))


model = tf.keras.Model(inputs, tf.transpose(outputs))
model.summary()

model.compile(optimizer= 'SGD', #tf.keras.optimizers.SGD(learning_rate=0.0001)
              loss=tf.keras.losses.MeanSquaredError(), 
              metrics=tf.keras.metrics.RootMeanSquaredError())

modelTest = model.fit(mlpNorm[:,4:7], mlpNorm[:,0:4], batch_size=b_size, epochs=5)

YNewTest = model.predict(mlpNorm[:,4:7], batch_size=b_size)

YReNorm = (YNewTest*std) + mean

trnErrTest = trn.MSError(YReNorm, L1.tempTrain[:,0:4])

models["Parallel"] = model

# %%
trn.cvTraining(L1, R1, 4, nFoldIter, Kfold, lag, b_size, epochs, models, "JDST")
trn.cvTraining(L1, R1, 4, nFoldIter, Kfold, lag, b_size, epochs, models, "Parallel")
# %%
# objects = []
# with (open("KerasVal.pickle", "rb")) as openfile:
#     while True:
#         try:
#             objects.append(pickle.load(openfile))
#         except EOFError:
#             break