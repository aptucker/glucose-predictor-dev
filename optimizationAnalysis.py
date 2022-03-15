# Written and designed by Aaron Tucker 
#
#
#
# ----------------------------------------------------------------------------
"""
Optimization analysis 
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from scipy import stats
import statsmodels as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import matplotlib.font_manager as font_manager
import time
import math
import control as ctrl
import control.matlab as ctrlmat

import patient as pat
import customLayers as cLayers
import customModels as cModels
import training as trn
import customPlots as cPlots
import customStats as cStats
import customCallbacks as cBacks
import optimizationFunctions as optFun

with open('results\\patient1_analysis.pickle', 'rb') as f:
    l1, r1 = pickle.load(f)

with open('results\\patient2_analysis.pickle', 'rb') as f:
    l2, r2 = pickle.load(f)
    
with open('results\\patient3_analysis.pickle', 'rb') as f:
    l3, r3 = pickle.load(f)
    
with open('results\\patient4_analysis.pickle', 'rb') as f:
    l4, r4 = pickle.load(f)

with open('results\\patient5_analysis.pickle', 'rb') as f:
    l5, r5 = pickle.load(f)

with open('results\\patient6_analysis.pickle', 'rb') as f:
    l6, r6 = pickle.load(f)

with open('results\\patient7_analysis.pickle', 'rb') as f:
    l7, r7 = pickle.load(f)
    
with open('results\\patient8_analysis.pickle', 'rb') as f:
    l8, r8 = pickle.load(f)
    
with open('results\\patient9_analysis.pickle', 'rb') as f:
    l9, r9 = pickle.load(f)
    
with open('results\\patient10_analysis.pickle', 'rb') as f:
    l10, r10 = pickle.load(f)
    
with open('results\\patient11_analysis.pickle', 'rb') as f:
    l11, r11 = pickle.load(f)
    
with open('results\\patient12_analysis.pickle', 'rb') as f:
    l12, r12 = pickle.load(f)

with open('results\\patient13_analysis.pickle', 'rb') as f:
    l13, r13 = pickle.load(f)
    
lPats = [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13]
rPats = [r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13]
# patNames = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
patNames = ['Patient' f' {i}' for i in range(1, 14)]

# %% Timing Testing

lPat = l2
rPat = r2

partNum = 1
partSize = [0.1]
lag = 6

Kfold = 2
nFoldIter = 5

H = 3
K = 4
D = lag+1
skip = 0 
outSize = 4

shapes = [H, H, K]
activators = ['tanh', 'sigmoid', None]

[lPatsTrain,
 rPatsTrain,
 lPatsTest,
 rPatsTest] = optFun.dataCombiner(lPats, rPats, partNum, partSize, lag)
# lPat.randomizeTrainingData(Kfold, seed=1)
# lPat.resetData()
# rPat.resetData()

# lPat.partitionData(partNum, partSize)
# rPat.partitionData(partNum, partSize)

# pat.createLagData(lPat.trainData, lag, skip = None, dropNaN=True)
# pat.createLagData(lPat.testData, lag, skip = None, dropNaN=True)
# pat.createLagData(rPat.trainData, lag, skip = None, dropNaN=True)
# pat.createLagData(rPat.testData, lag, skip = None, dropNaN=True)


[mlpNorm, mean, std] = trn.zscoreData(lPatsTrain.to_numpy())
[mlpNormTest, testMean, testStd] = trn.zscoreData(lPatsTest.to_numpy())


batch_end_loss = list()

# callbacks = [cBacks.EarlyStoppingAtMinLoss(patience=20, baseLoss=0.25),
#              cBacks.GetErrorOnBatch(batch_end_loss),
             # cBacks.lrScheduler(refLoss=0.2, lossHistory=batch_end_loss, gain=0.1)]

callbacks = [cBacks.EarlyStoppingAtMinLoss(patience=20, baseLoss=0.25),
             cBacks.batchErrorModel()]

learningRates = optFun.lrDict()
learningRates.storeLR(0.001, 'standard')

optimizers = optFun.optimizerDict()
optimizers.storeOptimizer(tf.keras.optimizers.Adam(), 'adam')

trialsToRun = 2

b_size = 100
epochs = 10

models = {}

inputs = tf.keras.Input(shape=(H,1))
gruLayer = tf.keras.layers.GRU(H,
                               activation='tanh',
                               recurrent_activation='sigmoid',
                               use_bias=True,
                               bias_initializer='ones')
x = gruLayer(inputs)
output = tf.keras.layers.Dense(K,
                               activation=None,
                               use_bias=True,
                               bias_initializer='ones')(x)
model = tf.keras.Model(inputs=inputs,
                       outputs=output)
# model.compile(optimizer= tf.keras.optimizers.SGD(learning_rate=0.01),
#               loss=tf.keras.losses.MeanSquaredError(), 
#               metrics=tf.keras.metrics.RootMeanSquaredError(),
#               loss_weights = [1.0, 1.0, 1.0, 1.0])
model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.MeanSquaredError(), 
              metrics=tf.keras.metrics.RootMeanSquaredError())#,
              # loss_weights = [1.0, 1.0, 1.0, 1.0])
models['gru'] = model
modelNames = list(['gru'])

optFun.timeTester(lPats,
                  rPats,
                  partNum,
                  partSize,
                  lag,
                  models,
                  modelNames,
                  learningRates,
                  'standard',
                  optimizers,
                  'adam',
                  b_size,
                  epochs,
                  trialsToRun,
                  callbacks)

# %%

ticGRU = time.perf_counter()

history = model.fit(mlpNorm[:, outSize:],
                    mlpNorm[:, 0:outSize],
                    batch_size=b_size,
                    epochs=epochs,
                    validation_data = (mlpNormTest[:, outSize:],
                                       mlpNormTest[:, 0:outSize]),
                    callbacks=callbacks)

tocGRU = time.perf_counter()

timePatGRU = tocGRU - ticGRU
# range(1, len(batch_end_loss) + 1
batchLossDF = pd.DataFrame(model.lossDict['newLoss'], index=np.linspace(0, timePatGRU, len(model.lossDict['newLoss'])), columns=['Batch Loss'])
# batchLossDF = pd.DataFrame(batch_end_loss, index=np.linspace(0, timePatGRU, len(batch_end_loss)), columns=['Batch Loss'])
batchLossDF['Batch Loss'].plot()

print(timePatGRU)








