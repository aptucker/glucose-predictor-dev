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
    lPat, rPat = pickle.load(f)

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
skip = 0 

b_size = 1
epochs = 5

# lPat.partitionData(partNum, partSize)
# rPat.partitionData(partNum, partSize)

# pat.createLagData(lPat.trainData, lag, skip = None, dropNaN=True)
# pat.createLagData(lPat.testData, lag, skip = None, dropNaN=True)
# pat.createLagData(rPat.trainData, lag, skip = None, dropNaN=True)
# pat.createLagData(rPat.testData, lag, skip = None, dropNaN=True)

# %% JDST Model 
partNum = 1
partSize = [0.1]
lag = 6

Kfold = 2
nFoldIter = 5

H = 3
K = 4
D = lag+1
skip = 0 

b_size = 1
epochs = 10

# lPat.randomizeTrainingData(Kfold, seed=1)
lPat.resetData()
rPat.resetData()

lPat.partitionData(partNum, partSize)
rPat.partitionData(partNum, partSize)

pat.createLagData(lPat.trainData, lag, skip = None, dropNaN=True)
pat.createLagData(lPat.testData, lag, skip = None, dropNaN=True)
pat.createLagData(rPat.trainData, lag, skip = None, dropNaN=True)
pat.createLagData(rPat.testData, lag, skip = None, dropNaN=True)

np.random.seed(1)
initializer1 = tf.keras.initializers.Constant(np.random.normal(0, 0.005, (H, 4)))
np.random.seed(1)
aInit = np.random.normal(0, 0.005, (H, D))

np.random.seed(4)
initializer2 = tf.keras.initializers.Constant(np.random.normal(0, 0.005, (H+1, K)))
np.random.seed(4)
bInit = np.random.normal(0, 0.005, (H+1, K))

[mlpNorm, mean, std] = trn.zscoreData(lPat.tempTrain)

b_size = 1

callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'loss',
                                             min_delta = 0.05,
                                             patience = 2,
                                             mode = "min",
                                             restore_best_weights = True)]

# initializers = [initializer1, initializer2]
initializers = [tf.keras.initializers.RandomNormal(mean=0, stddev=0.005),
                tf.keras.initializers.RandomNormal(mean=0, stddev=0.005)]

model = cModels.sbSeqModel(H,
                           K,
                           use_bias = True,
                           initializers = initializers,
                           bias_size = b_size,
                           activators = ["sigmoid", None])
model(tf.keras.Input(shape=H))

model.compile(optimizer= 'SGD', #tf.keras.optimizers.SGD(learning_rate=0.0001)
              loss=tf.keras.losses.MeanSquaredError(), 
              metrics=tf.keras.metrics.RootMeanSquaredError())

models["JDST"] = model

trn.cvTraining(lPat,
               rPat,
               4,
               nFoldIter,
               Kfold,
               lag,
               None,
               b_size,
               epochs,
               models,
               "JDST",
               callbacks)

# modelTest = model.fit(mlpNorm[:,4:7], mlpNorm[:,0:4], batch_size=b_size, epochs=5)

# YNewTest = model.predict(mlpNorm[:,4:7], batch_size=b_size)

# YReNorm = (YNewTest*std) + mean

# # Output Mapping -> Column 3 = 15min, Column 0 = 60min

# trnErrTest = trn.MSError(YReNorm, lPat.tempTrain[:,0:4])
# # trnErrTest = trn.MSError(np.reshape(YReNorm[:,3], [4671, 1]), tf.reshape(lPat.tempTrain[:,3], [4671,1]))


# %% Parallel Network
partNum = 1
partSize = [0.1]

lag = 6
skip = None 
# lag = 98
# skip = list(range(6, 95))

Kfold = 2
nFoldIter = 5

H = 3
K = 4
D = lag+1


b_size = 1
epochs = 5

lPat.resetData()
rPat.resetData()

lPat.partitionData(partNum, partSize)
rPat.partitionData(partNum, partSize)

pat.createLagData(lPat.trainData, lag, skip, dropNaN=True)
pat.createLagData(lPat.testData, lag, skip, dropNaN=True)
pat.createLagData(rPat.trainData, lag, skip, dropNaN=True)
pat.createLagData(rPat.testData, lag, skip, dropNaN=True)

lPat.randomizeTrainingData(Kfold, seed=1)
rPat.randomizeTrainingData(Kfold, seed=1)
[lNorm, lMean, lStd] = trn.zscoreData(lPat.tempTrain)
[rNorm, rMean, rStd] = trn.zscoreData(rPat.tempTrain)
[lValNorm, lValMean, lValStd] = trn.zscoreData(lPat.tempVal)
[rValNorm, rValMean, rValStd] = trn.zscoreData(rPat.tempVal)

inputs = tf.keras.Input(shape=(H*2,))

tower1 = cLayers.staticBiasLayer(H,
                                 activation = 'sigmoid',
                                 use_bias = True,
                                 kernel_initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=0.005),
                                 ones_size = b_size)(inputs[:, 0:H])
tower1 = cLayers.staticBiasLayer(K,
                                 activation = None,
                                 use_bias=True,
                                 kernel_initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=0.005),
                                 ones_size = b_size)(tower1)

tower2 = cLayers.staticBiasLayer(H,
                                 activation = 'sigmoid',
                                 use_bias = True,
                                 kernel_initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=0.005),
                                 ones_size = b_size)(inputs[:, H:])
tower2 = cLayers.staticBiasLayer(K,
                                 activation = None,
                                 use_bias=True,
                                 kernel_initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=0.005),
                                 ones_size = b_size)(tower2)

merged = tf.keras.layers.concatenate([tower1, tower2], axis=0)

outputs = tf.keras.layers.Dense(1, activation=None)(tf.transpose(merged))

callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'loss',
                                             min_delta = 0.05,
                                             patience = 4,
                                             mode = "min",
                                             restore_best_weights = True)]

parallelModel = cModels.parallelModel(H, K, use_bias=True, bias_size = b_size)
parallelModel.compile(optimizer= 'SGD', #tf.keras.optimizers.SGD(learning_rate=0.0001)
              loss=tf.keras.losses.MeanSquaredError(), 
              metrics=tf.keras.metrics.RootMeanSquaredError())
models["Parallel"] = parallelModel
# parallelModel(tf.keras.Input(shape=6))

trn.cvTrainingParallel(lPat,
                       rPat,
                       4,
                       nFoldIter,
                       Kfold,
                       lag,
                       None,
                       b_size,
                       epochs,
                       models,
                       "Parallel",
                       callbacks)


# normInputs = np.append(rNorm[:, 4:7], lNorm[0:len(rNorm), 4:7], axis = 1)
# normValInputs = np.append(rValNorm[:, 4:7], lValNorm[0:len(rValNorm), 4:7], axis = 1)

# modelLTest = parallelModel.fit(normInputs, lNorm[0:len(rNorm),0:4], batch_size=b_size, epochs=5)

# YNewTest = parallelModel.predict(normValInputs, batch_size=b_size)

# YReNorm = (YNewTest*lStd) + lMean

# trnErrTest = trn.MSError(YReNorm, lPat.tempTrain[0:len(rNorm),0:4])

# %% Parallel with 2 Hidden Layers
partNum = 1
partSize = [0.1]

lag = 6

Kfold = 2
nFoldIter = 5

H = 3
K = 4
skip = 0 

b_size = 1
epochs = 5

tower1Shapes = tower2Shapes = [H, H, K]
tower1Activators = tower2Activators = ['sigmoid', 'sigmoid', None]



lPat.resetData()
rPat.resetData()

lPat.partitionData(partNum, partSize)
rPat.partitionData(partNum, partSize)

pat.createLagData(lPat.trainData, lag, skip = None, dropNaN=True)
pat.createLagData(lPat.testData, lag, skip = None, dropNaN=True)
pat.createLagData(rPat.trainData, lag, skip = None, dropNaN=True)
pat.createLagData(rPat.testData, lag, skip = None, dropNaN=True)

lPat.randomizeTrainingData(Kfold, seed=1)
rPat.randomizeTrainingData(Kfold, seed=1)
[lNorm, lMean, lStd] = trn.zscoreData(lPat.tempTrain)
[rNorm, rMean, rStd] = trn.zscoreData(rPat.tempTrain)
[lValNorm, lValMean, lValStd] = trn.zscoreData(lPat.tempVal)
[rValNorm, rValMean, rValStd] = trn.zscoreData(rPat.tempVal)

parallelModelH2 = cModels.parallelModelH2(tower1Shapes,
                                          tower2Shapes,
                                          tower1Activators,
                                          tower2Activators,
                                          b_size)
parallelModelH2.compile(optimizer= 'SGD', #tf.keras.optimizers.SGD(learning_rate=0.0001)
              loss=tf.keras.losses.MeanSquaredError(), 
              metrics=tf.keras.metrics.RootMeanSquaredError())
# parallelModelH2(tf.keras.Input(shape=6))
models["Parallel H2"] = parallelModelH2

callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'loss',
                                             min_delta = 0.05,
                                             patience = 5,
                                             mode = "min",
                                             restore_best_weights = True)]

trn.cvTrainingParallel(lPat,
                       rPat,
                       4,
                       nFoldIter,
                       Kfold,
                       lag,
                       None,
                       b_size,
                       epochs,
                       models,
                       "Parallel",
                       callbacks)

# normInputs = np.append(rNorm[:, 4:7], lNorm[0:len(rNorm), 4:7], axis = 1)
# normValInputs = np.append(rValNorm[:, 4:7], lValNorm[0:len(rValNorm), 4:7], axis = 1)

# modelLTest = parallelModelH2.fit(normInputs, lNorm[0:len(rNorm),0:4], batch_size=b_size, epochs=30)

# YNewTest = parallelModelH2.predict(normValInputs, batch_size=b_size)

# YReNorm = (YNewTest*lStd) + lMean

# trnErrTest = trn.MSError(YReNorm, lPat.tempTrain[0:len(rNorm),0:4])



# %% Circadian Model 1
lPat.resetData()
rPat.resetData()

circLag = 98
circSkip = list(range(6, 95))

lPat.partitionData(partNum, partSize)
rPat.partitionData(partNum, partSize)

pat.createLagData(lPat.trainData, circLag, circSkip, dropNaN=True)
pat.createLagData(rPat.trainData, circLag, circSkip, dropNaN=True)

lPat.randomizeTrainingData(Kfold, seed=1)
rPat.randomizeTrainingData(Kfold, seed=1)
[lNorm, lMean, lStd] = trn.zscoreData(lPat.tempTrain)
[rNorm, rMean, rStd] = trn.zscoreData(rPat.tempTrain)
[lValNorm, lValMean, lValStd] = trn.zscoreData(lPat.tempVal)
[rValNorm, rValMean, rValStd] = trn.zscoreData(rPat.tempVal)

initializers = [tf.keras.initializers.RandomNormal(mean=0, stddev=0.005), tf.keras.initializers.RandomNormal(mean=0, stddev=0.005)]

model = cModels.sbSeqModel(6, K, use_bias = True, initializers = initializers, bias_size = b_size, activators = ["sigmoid", None])
model(tf.keras.Input(shape=6))

model.compile(optimizer= 'SGD', #tf.keras.optimizers.SGD(learning_rate=0.0001)
              loss=tf.keras.losses.MeanSquaredError(), 
              metrics=tf.keras.metrics.RootMeanSquaredError())

models["Circadian 1"] = model
callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'loss',
                                             min_delta = 0.05,
                                             patience = 5,
                                             mode = "min",
                                             restore_best_weights = True)]

trn.cvTraining(lPat,
               rPat,
               4,
               nFoldIter,
               Kfold,
               circLag,
               circSkip,
               b_size,
               epochs,
               models,
               "Circadian 1",
               callbacks)

# circTest = model.fit(lNorm[:, K:], lNorm[:, 0:K], batch_size = b_size, epochs = epochs)
# cirPredTest = model.predict(lValNorm[:, K:], batc_size = b_size)

# %% Sequential w/2 Hidden Layers
partNum = 1
partSize = [0.1]

lag = 6

Kfold = 2
nFoldIter = 5

H = 3
K = 4
D = lag+1
skip = 0 

b_size = 1
epochs = 5

initializers = [tf.keras.initializers.RandomNormal(mean=0, stddev=0.005),
                tf.keras.initializers.RandomNormal(mean=0, stddev=0.005),
                tf.keras.initializers.RandomNormal(mean=0, stddev=0.005)]

activators = ['sigmoid', 'sigmoid', None]

shapes = [H, H, K]

# lPat.randomizeTrainingData(Kfold, seed=1)
lPat.resetData()
rPat.resetData()

lPat.partitionData(partNum, partSize)
rPat.partitionData(partNum, partSize)

pat.createLagData(lPat.trainData, lag, skip = None, dropNaN=True)
pat.createLagData(lPat.testData, lag, skip = None, dropNaN=True)
pat.createLagData(rPat.trainData, lag, skip = None, dropNaN=True)
pat.createLagData(rPat.testData, lag, skip = None, dropNaN=True)

model = cModels.sbSeqModelH2(shapes, True, initializers, b_size, activators)
model(tf.keras.Input(shape=H))
model.compile(optimizer= 'SGD', #tf.keras.optimizers.SGD(learning_rate=0.0001)
              loss=tf.keras.losses.MeanSquaredError(), 
              metrics=tf.keras.metrics.RootMeanSquaredError())

models["Sequential H=2"] = model

callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'loss',
                                             min_delta = 0.05,
                                             patience = 10,
                                             mode = "min",
                                             restore_best_weights = True)]

# lPat.randomizeTrainingData(Kfold, seed=0)
# rPat.randomizeTrainingData(Kfold, seed=0)
# [lNorm, lMean, lStd] = trn.zscoreData(lPat.tempTrain)
# [rNorm, rMean, rStd] = trn.zscoreData(rPat.tempTrain)
# [lValNorm, lValMean, lValStd] = trn.zscoreData(lPat.tempVal)
# [rValNorm, rValMean, rValStd] = trn.zscoreData(rPat.tempVal)

# modelTest = model.fit(lNorm[:,4:7], lNorm[:,0:4], batch_size=b_size, epochs=20)
trn.cvTraining(lPat,
               rPat,
               4,
               nFoldIter,
               Kfold,
               lag,
               skip,
               b_size,
               20,
               models,
               "Sequential H=2",
               callbacks)


# %%

# %%
# trn.cvTraining(L1, R1, 4, nFoldIter, Kfold, lag, skip, b_size, epochs, models, "JDST")
# trn.cvTrainingParallel(lPat, rPat, 4, nFoldIter, Kfold, lag, skip, b_size, epochs, models, "Parallel")
# trn.cvTraining(lPat, rPat, 4, nFoldIter, Kfold, circLag, circSkip, b_size, epochs, models, "Circadian 1")
# %%
# objects = []
# with (open("KerasVal.pickle", "rb")) as openfile:
#     while True:
#         try:
#             objects.append(pickle.load(openfile))
#         except EOFError:
#             break