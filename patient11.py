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

import patient as pat
import customLayers as cLayers
import customModels as cModels
import training as trn

with open("processed_data\\patient11.pickle", "rb") as f:
    lPat, rPat = pickle.load(f)

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

lPat.partitionData(partNum, partSize)
rPat.partitionData(partNum, partSize)

# pat.createLagData(lPat.trainData, lag, dropNaN=True)
# pat.createLagData(lPat.testData, lag, dropNaN=True)
# pat.createLagData(rPat.trainData, lag, dropNaN=True)
# pat.createLagData(rPat.testData, lag, dropNaN=True)

# %% JDST Model Definition

np.random.seed(1)
initializer1 = tf.keras.initializers.Constant(np.random.normal(0, 0.005, (H, 4)))
np.random.seed(1)
aInit = np.random.normal(0, 0.005, (H, D))

np.random.seed(4)
initializer2 = tf.keras.initializers.Constant(np.random.normal(0, 0.005, (H+1, K)))
np.random.seed(4)
bInit = np.random.normal(0, 0.005, (H+1, K))

# initializers = [initializer1, initializer2]
initializers = [tf.keras.initializers.RandomNormal(mean=0, stddev=0.005), tf.keras.initializers.RandomNormal(mean=0, stddev=0.005)]

model = cModels.sbSeqModel(H, K, use_bias = True, initializers = initializers, bias_size = b_size, activators = ["sigmoid", None])
model(tf.keras.Input(shape=H))

model.compile(optimizer= 'SGD', #tf.keras.optimizers.SGD(learning_rate=0.0001)
              loss=tf.keras.losses.MeanSquaredError(), 
              metrics=tf.keras.metrics.RootMeanSquaredError())

models["JDST"] = model

# lPat.models["JDST"] = model
# rPat.models["JDST"] = model


# %% 5x2 Cross Validation
trn.cvTraining(lPat, rPat, K, nFoldIter, Kfold, lag, b_size, epochs, models, "JDST")

# %% Save Results

with open("results\\patient11_analysis.pickle", "wb") as f:
    pickle.dump([lPat, rPat], f)

