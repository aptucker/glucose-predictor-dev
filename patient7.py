# Written and designed by Aaron Tucker 
#
#
#
# ----------------------------------------------------------------------------
"""
Patient 7 analysis file
"""
# %% Imports and Data Loading
import tensorflow as tf
import numpy as np
import pickle

import patient as pat
import customLayers as cLayers
import customModels as cModels
import training as trn

with open("processed_data\\patient7.pickle", "rb") as f:
    lPat, rPat = pickle.load(f)

models = {}    

# %% JDST Model Definition
partNum = 1
partSize = [0.1]

lag = 6
skip = None
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

pat.createLagData(lPat.trainData, lag, dropNaN=True)
pat.createLagData(lPat.testData, lag, dropNaN=True)
pat.createLagData(rPat.trainData, lag, dropNaN=True)
pat.createLagData(rPat.testData, lag, dropNaN=True)

np.random.seed(1)
initializer1 = tf.keras.initializers.Constant(np.random.normal(0, 0.005, (H, 4)))
np.random.seed(1)
aInit = np.random.normal(0, 0.005, (H, D))

np.random.seed(4)
initializer2 = tf.keras.initializers.Constant(np.random.normal(0, 0.005, (H+1, K)))
np.random.seed(4)
bInit = np.random.normal(0, 0.005, (H+1, K))

callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'loss',
                                             min_delta = 0.05,
                                             patience = 2,
                                             mode = "min",
                                             restore_best_weights = True)]

initializers = [initializer1, initializer2]
# initializers = [tf.keras.initializers.RandomNormal(mean=0, stddev=0.005),
#                 tf.keras.initializers.RandomNormal(mean=0, stddev=0.005)]

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
               K,
               nFoldIter,
               Kfold,
               lag,
               skip,
               b_size,
               epochs,
               models,
               "JDST",
               callbacks)

print("JDST Done")
# %% Sequential w/2 Hidden Layers
partNum = 1
partSize = [0.1]

lag = 6

Kfold = 2
nFoldIter = 5

H = 3
K = 4
skip = None 

b_size = 1
epochs = 20

initializers = [tf.keras.initializers.RandomNormal(mean=0, stddev=0.005),
                tf.keras.initializers.RandomNormal(mean=0, stddev=0.005),
                tf.keras.initializers.RandomNormal(mean=0, stddev=0.005)]

activators = ['sigmoid', 'sigmoid', None]

shapes = [H, H, K]

callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'loss',
                                             min_delta = 0.05,
                                             patience = 10,
                                             mode = "min",
                                             restore_best_weights = True)]

lPat.resetData()
rPat.resetData()

lPat.partitionData(partNum, partSize)
rPat.partitionData(partNum, partSize)

pat.createLagData(lPat.trainData, lag, skip = None, dropNaN=True)
pat.createLagData(lPat.testData, lag, skip = None, dropNaN=True)
pat.createLagData(rPat.trainData, lag, skip = None, dropNaN=True)
pat.createLagData(rPat.testData, lag, skip = None, dropNaN=True)

model = cModels.sbSeqModelH2(shapes,
                             True,
                             initializers,
                             b_size,
                             activators)
model.compile(optimizer= 'SGD', #tf.keras.optimizers.SGD(learning_rate=0.0001)
              loss=tf.keras.losses.MeanSquaredError(), 
              metrics=tf.keras.metrics.RootMeanSquaredError())
model(tf.keras.Input(shape=H))

models["Sequential H=2"] = model

trn.cvTraining(lPat,
               rPat,
               K,
               nFoldIter,
               Kfold,
               lag,
               skip,
               b_size,
               epochs,
               models,
               "Sequential H=2",
               callbacks)

print("Sequential H=2 Done")

# %% Circadian Model 1
partNum = 1
partSize = [0.1]

lag = 98
skip = list(range(6, 95))

H = 6
K = 4

b_size = 1
epochs = 5

callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'loss',
                                             min_delta = 0.05,
                                             patience = 5,
                                             mode = "min",
                                             restore_best_weights = True)]

lPat.resetData()
rPat.resetData()

lPat.partitionData(partNum, partSize)
rPat.partitionData(partNum, partSize)

pat.createLagData(lPat.trainData, lag, skip, dropNaN=True)
pat.createLagData(rPat.trainData, lag, skip, dropNaN=True)


initializers = [tf.keras.initializers.RandomNormal(mean=0, stddev=0.005),
                tf.keras.initializers.RandomNormal(mean=0, stddev=0.005)]

model = cModels.sbSeqModel(H,
                           K,
                           use_bias = True,
                           initializers = initializers,
                           bias_size = b_size,
                           activators = ["sigmoid", None])
model.compile(optimizer= 'SGD', #tf.keras.optimizers.SGD(learning_rate=0.0001)
              loss=tf.keras.losses.MeanSquaredError(), 
              metrics=tf.keras.metrics.RootMeanSquaredError())
model(tf.keras.Input(shape=H))

models["Circadian 1"] = model

trn.cvTraining(lPat,
               rPat,
               K,
               nFoldIter,
               Kfold,
               lag,
               skip,
               b_size,
               epochs,
               models,
               "Circadian 1",
               callbacks)

print("Circadian 1 Done")

# %% Parallel Network
partNum = 1
partSize = [0.1]

lag = 6
skip = None 

Kfold = 2
nFoldIter = 5

H = 3
K = 4
D = lag+1


b_size = 1
epochs = 5

callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'loss',
                                             min_delta = 0.05,
                                             patience = 4,
                                             mode = "min",
                                             restore_best_weights = True)]

lPat.resetData()
rPat.resetData()

lPat.partitionData(partNum, partSize)
rPat.partitionData(partNum, partSize)

pat.createLagData(lPat.trainData, lag, skip, dropNaN=True)
pat.createLagData(lPat.testData, lag, skip, dropNaN=True)
pat.createLagData(rPat.trainData, lag, skip, dropNaN=True)
pat.createLagData(rPat.testData, lag, skip, dropNaN=True)

parallelModel = cModels.parallelModel(H,
                                      K,
                                      use_bias=True,
                                      bias_size = b_size)
parallelModel.compile(optimizer= 'SGD', #tf.keras.optimizers.SGD(learning_rate=0.0001)
              loss=tf.keras.losses.MeanSquaredError(), 
              metrics=tf.keras.metrics.RootMeanSquaredError())
models["Parallel"] = parallelModel
# parallelModel(tf.keras.Input(shape=6))

trn.cvTrainingParallel(lPat,
                       rPat,
                       K,
                       nFoldIter,
                       Kfold,
                       lag,
                       None,
                       b_size,
                       epochs,
                       models,
                       "Parallel",
                       callbacks)

print("Parallel Done")

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
epochs = 20

tower1Shapes = tower2Shapes = [H, H, K]
tower1Activators = tower2Activators = ['sigmoid', 'sigmoid', None]

callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'loss',
                                             min_delta = 0.05,
                                             patience = 10,
                                             mode = "min",
                                             restore_best_weights = True)]

lPat.resetData()
rPat.resetData()

lPat.partitionData(partNum, partSize)
rPat.partitionData(partNum, partSize)

pat.createLagData(lPat.trainData, lag, skip = None, dropNaN=True)
pat.createLagData(lPat.testData, lag, skip = None, dropNaN=True)
pat.createLagData(rPat.trainData, lag, skip = None, dropNaN=True)
pat.createLagData(rPat.testData, lag, skip = None, dropNaN=True)

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

trn.cvTrainingParallel(lPat,
                       rPat,
                       K,
                       nFoldIter,
                       Kfold,
                       lag,
                       None,
                       b_size,
                       epochs,
                       models,
                       "Parallel H2",
                       callbacks)

print("Parallel H2 Done")

# %% Parallel with Circadian Inputs
partNum = 1
partSize = [0.1]

lag = 98
skip = list(range(6, 95))

Kfold = 2
nFoldIter = 5

H = 3
K = 4
D = lag+1


b_size = 1
epochs = 20

callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'loss',
                                             min_delta = 0.05,
                                             patience = 10,
                                             mode = "min",
                                             restore_best_weights = True)]

lPat.resetData()
rPat.resetData()

lPat.partitionData(partNum, partSize)
rPat.partitionData(partNum, partSize)

pat.createLagData(lPat.trainData, lag, skip, dropNaN=True)
pat.createLagData(lPat.testData, lag, skip, dropNaN=True)
pat.createLagData(rPat.trainData, lag, skip, dropNaN=True)
pat.createLagData(rPat.testData, lag, skip, dropNaN=True)

parallelModel = cModels.parallelModel(H,
                                      K,
                                      use_bias=True,
                                      bias_size = b_size)
parallelModel.compile(optimizer= 'SGD', #tf.keras.optimizers.SGD(learning_rate=0.0001)
              loss=tf.keras.losses.MeanSquaredError(), 
              metrics=tf.keras.metrics.RootMeanSquaredError())
models["Parallel Circadian"] = parallelModel
# parallelModel(tf.keras.Input(shape=6))

trn.cvTrainingParallel(lPat,
                       rPat,
                       K,
                       nFoldIter,
                       Kfold,
                       lag,
                       None,
                       b_size,
                       epochs,
                       models,
                       "Parallel Circadian",
                       callbacks)

print("Parallel Circadian Done")

# %% Save Results

with open("results\\patient7_analysis.pickle", "wb") as f:
    pickle.dump([lPat, rPat], f)

