# Written and designed by Aaron Tucker 
#
#
#
# ----------------------------------------------------------------------------
"""
Patient 2 analysis file
"""
# %% Imports and Data Loading
import tensorflow as tf
import numpy as np
import pickle
import time

import patient as pat
import customLayers as cLayers
import customModels as cModels
import training as trn

models = {}
# %% Load w/Previous Results
with open("results\\patient2_analysis.pickle", "rb") as f:
    lPat, rPat = pickle.load(f)

# %% Load w/o Previous Results
with open("processed_data\\patient2.pickle", "rb") as f:
    lPat, rPat = pickle.load(f)  

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

# callbacks = []
callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'loss',
                                              min_delta = 0.05,
                                              patience = 2,
                                              mode = "min",
                                              restore_best_weights = False)]

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

ticJDST = time.perf_counter()

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

tocJDST = time.perf_counter()
timePat2JDST = tocJDST - ticJDST
print(tocJDST - ticJDST)

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
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# callbacks = [tf.keras.callbacks.LearningRateScheduler(scheduler),
#              tf.keras.callbacks.EarlyStopping(monitor = 'loss',
#                                               min_delta = 0.05,
#                                               patience = 20,
#                                               mode = "min",
#                                               restore_best_weights = True)]

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
parallelModelH2.compile(optimizer= tf.keras.optimizers.SGD(learning_rate=0.001),
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

# %% GRU H=1 Model

partNum = 1
partSize = [0.1]
lag = 6

Kfold = 2
nFoldIter = 5

H = 3
K = 4
D = lag+1
skip = 0 

shapes = [H, H, K]
activators = ['tanh', 'sigmoid', None]

b_size = 1
epochs = 100

# lPat.randomizeTrainingData(Kfold, seed=1)
lPat.resetData()
rPat.resetData()

lPat.partitionData(partNum, partSize)
rPat.partitionData(partNum, partSize)

pat.createLagData(lPat.trainData, lag, skip = None, dropNaN=True)
pat.createLagData(lPat.testData, lag, skip = None, dropNaN=True)
pat.createLagData(rPat.trainData, lag, skip = None, dropNaN=True)
pat.createLagData(rPat.testData, lag, skip = None, dropNaN=True)


[mlpNorm, mean, std] = trn.zscoreData(lPat.trainData.to_numpy())

callbacks = []
# callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'loss',
#                                              min_delta = 0.001,
#                                              patience = 2,
#                                              mode = "min",
#                                              restore_best_weights = True)]

inputs = tf.keras.Input(shape=(H,1))
gruLayer = tf.keras.layers.GRU(H, activation='tanh', recurrent_activation='sigmoid', use_bias=True, bias_initializer='ones')
x = gruLayer(inputs)
output = tf.keras.layers.Dense(K, activation=None, use_bias=True, bias_initializer='ones')(x)
model = tf.keras.Model(inputs=inputs, outputs=output)
model.compile(optimizer= 'SGD', #tf.keras.optimizers.SGD(learning_rate=0.0001)
              loss=tf.keras.losses.MeanSquaredError(), 
              metrics=tf.keras.metrics.RootMeanSquaredError())
models["GRU H=1"] = model

ticGRU = time.perf_counter()

trn.cvTraining(lPat,
                rPat,
                4,
                nFoldIter,
                Kfold,
                lag,
                skip,
                b_size,
                epochs,
                models,
                "GRU H=1",
                callbacks)

tocGRU = time.perf_counter()

timePat2GRU = tocGRU - ticGRU
print(tocGRU - ticGRU)

print("GRU H=1 Done")

# %% Save Results

with open("results\\patient2_analysis.pickle", "wb") as f:
    pickle.dump([lPat, rPat], f)

