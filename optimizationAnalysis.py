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

np.random.seed(1)
initializer1 = tf.keras.initializers.Constant(np.random.normal(0, 0.005, (H, 4)))
np.random.seed(1)
aInit = np.random.normal(0, 0.005, (H, D))

np.random.seed(4)
initializer2 = tf.keras.initializers.Constant(np.random.normal(0, 0.005, (H+1, K)))
np.random.seed(4)
bInit = np.random.normal(0, 0.005, (H+1, K))

initializers = [initializer1, initializer2]

batch_end_loss = list()

trialsToRun = 10

b_size = 1
epochs = 20

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

standardModel = tf.keras.Model(inputs=inputs,
                               outputs=output)
adamStandardModel = tf.keras.Model(inputs=inputs,
                                   outputs=output)
lrScheduledStandardModel = tf.keras.Model(inputs=inputs,
                                          outputs=output)
lrScheduledAdamModel = tf.keras.Model(inputs=inputs,
                                      outputs=output)
standardScheduleStandardModel = tf.keras.Model(inputs=inputs,
                                               outputs=output)
standardScheduleAdamModel = tf.keras.Model(inputs=inputs,
                                           outputs=output)

jdstModel = cModels.sbSeqModel(H, 
                            K,
                            use_bias = True,
                            initializers = initializers,
                            bias_size = b_size,
                            activators = ["sigmoid", None])
jdstModel(tf.keras.Input(shape=H))

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)

standardModel.compile(optimizer= tf.keras.optimizers.SGD(),
                      loss=tf.keras.losses.MeanSquaredError(), 
                      metrics=tf.keras.metrics.RootMeanSquaredError())

adamStandardModel.compile(optimizer= tf.keras.optimizers.Adam(),
                          loss=tf.keras.losses.MeanSquaredError(), 
                          metrics=tf.keras.metrics.RootMeanSquaredError())

lrScheduledStandardModel.compile(optimizer= tf.keras.optimizers.SGD(),
                                 loss=tf.keras.losses.MeanSquaredError(), 
                                 metrics=tf.keras.metrics.RootMeanSquaredError())

lrScheduledAdamModel.compile(optimizer= tf.keras.optimizers.Adam(),
                          loss=tf.keras.losses.MeanSquaredError(), 
                          metrics=tf.keras.metrics.RootMeanSquaredError())

standardScheduleStandardModel.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
                                      loss=tf.keras.losses.MeanSquaredError(),
                                      metrics=tf.keras.metrics.RootMeanSquaredError())

standardScheduleAdamModel.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
                                      loss=tf.keras.losses.MeanSquaredError(),
                                      metrics=tf.keras.metrics.RootMeanSquaredError())
jdstModel.compile(optimizer= 'SGD', 
              loss=tf.keras.losses.MeanSquaredError(), 
              metrics=tf.keras.metrics.RootMeanSquaredError())

callbacks = [cBacks.EarlyStoppingAtMinLoss(patience=20, baseLoss=0.25),
             cBacks.lrScheduler(refLoss=0.2, gain=0.1)]

standardModel.callbacks = [cBacks.EarlyStoppingAtMinLoss(patience=20, baseLoss=0.25),
                           cBacks.batchErrorModel()]
adamStandardModel.callbacks = [cBacks.EarlyStoppingAtMinLoss(patience=20, baseLoss=0.25),
                               cBacks.batchErrorModel()]
lrScheduledStandardModel.callbacks = [cBacks.EarlyStoppingAtMinLoss(patience=20, baseLoss=0.25),
                                      cBacks.lrScheduler(refLoss=0.23, gain=0.1)]
lrScheduledAdamModel.callbacks = [cBacks.EarlyStoppingAtMinLoss(patience=20, baseLoss=0.25),
                                  cBacks.lrScheduler(refLoss=0.23, gain=0.1)]
standardScheduleStandardModel.callbacks = [cBacks.EarlyStoppingAtMinLoss(patience=20, baseLoss=0.25),
                                           cBacks.batchErrorModel()]
standardScheduleAdamModel.callbacks = [cBacks.EarlyStoppingAtMinLoss(patience=20, baseLoss=0.25),
                                       cBacks.batchErrorModel()]
jdstModel.callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'loss',
                                              min_delta = 0.1,
                                              patience = 4,
                                              mode = "min",
                                              restore_best_weights = False),
                       cBacks.batchErrorModel()]

models['standard'] = standardModel
standardModel.save_weights('standardModel.start')

models['adam'] = adamStandardModel
adamStandardModel.save_weights('adamModel.start')

models['lrStandard'] = lrScheduledStandardModel
lrScheduledStandardModel.save_weights('lrStandardModel.start')

models['lrAdam'] = lrScheduledAdamModel
lrScheduledAdamModel.save_weights('lrAdamModel.start')

models['standardSchedule'] = standardScheduleStandardModel
standardScheduleStandardModel.save_weights('standardScheduleModel.start')

models['standardAdam'] = standardScheduleAdamModel
standardScheduleAdamModel.save_weights('standardAdamModel.start')

models['jdst'] = jdstModel
# jdstModel.save_weights('jdstModel.start')

modelNames = list(['standard', 'adam', 'lrStandard', 'lrAdam', 'standardSchedule', 'standardAdam', 'jdst'])
# modelNames = list(['jdst'])



outDict = optFun.timeTester(lPats,
                            rPats,
                            partNum,
                            partSize,
                            lag,
                            models,
                            modelNames,
                            b_size,
                            epochs,
                            trialsToRun,
                            maxDataSize=1000)

# %%

outDict = optFun.timeTester(lPats,
                            rPats,
                            partNum,
                            partSize,
                            lag,
                            models, 
                            modelNames,
                            b_size,
                            epochs,
                            trialsToRun,
                            maxDataSize=1000)

outDict2 = optFun.timeTester(lPats,
                            rPats,
                            partNum,
                            partSize,
                            lag,
                            models,
                            modelNames,
                            10,
                            epochs,
                            trialsToRun,
                            maxDataSize=10000)

# %% Time Results Analysis

timeDF1 = optFun.compileTimeTrialResults(outDict,
                                         modelNames,
                                         averageWindow=30,
                                         threshold=0.35)

timeDF2 = optFun.compileTimeTrialResults(outDict2,
                                         modelNames,
                                         averageWindow=30,
                                         threshold=0.35)

tempList = []
timeDF = pd.DataFrame()

for i in range(len(modelNames)):
    for e in range(len(outDict[modelNames[i]])):
        convTime = optFun.findConvergenceTime(outDict[modelNames[i]]['Loss It. ' f'{e+1}'], 
                                              averageWindow=30,
                                              threshold=0.25)
        
        tempList.append(convTime)
    
    timeDF[modelNames[i]] = tempList
    
    tempList = []

tempList = []
tcDF = pd.DataFrame()

for i in range(len(modelNames)):
    for e in range(len(outDict[modelNames[i]])):
        tc = optFun.findTimeConstant(outDict2[modelNames[i]]['Loss It. ' f'{e+1}'])
        
        tempList.append(tc)
    
    tcDF[modelNames[i]] = tempList
    
    tempList = []
    



# %% Plots
fig, ax = plt.subplots(1,1)

legendNames = []
# modelstoplot = ['standardSchedule', 'lrStandard', 'lrAdam', 'jdst']
# modelstoplot = ['standard', 'jdst', 'lrAdam']
modelstoplot = ['lrAdam', 'standard', 'adam', 'jdst', 'lrStandard']

rangeToPlot = range(1,2)

for i in range(len(modelstoplot)):
    for e in rangeToPlot:
        outDict2[modelstoplot[i]]['Loss It. ' f'{e+1}'].plot(ax=ax)
        
        if len(rangeToPlot) > 1:
            legendNames.append(f'{modelstoplot[i]}' ' It. ' f'{e+1}')
        
        else:
            legendNames.append(f'{modelstoplot[i]}')
        
ax.legend(legendNames)
ax.set_xlabel('TIME [s]')
ax.set_ylabel('LOSS')
ax.set_title('NETWORK LOSS DURING TRAINING')
ax.set_xlim([-0.5, 5])


plt.savefig('C:\Code\glucose-predictor-dev\speedtest.pdf', bbox_inches='tight')


# %% Control Law Testing

[lPatsTrain,
 rPatsTrain,
 lPatsTest,
 rPatsTest] = optFun.dataCombiner(lPats, rPats, partNum, partSize, lag)

[mlpNorm, mean, std] = trn.zscoreData(lPatsTrain.to_numpy())
[mlpNormTest, testMean, testStd] = trn.zscoreData(lPatsTest.to_numpy())

H = 3
K = 4
outSize = K

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

standardTestModel = tf.keras.Model(inputs=inputs,
                               outputs=output)

standardTestModel.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01),
                          loss=tf.keras.losses.MeanSquaredError(),
                          metrics=tf.keras.metrics.RootMeanSquaredError())

stModelCallbacks = [cBacks.batchErrorModel(),
                    cBacks.sinLRScheduler(freq=2, startLR=0.001)]

stModelCallbacks = [cBacks.batchErrorModel(),
                    cBacks.lrScheduler(refLoss=0.23, gain=.85)]



history = standardTestModel.fit(mlpNorm[:, outSize:],
                                mlpNorm[:, 0:outSize],
                                batch_size=10,
                                epochs=3,
                                validation_data = (mlpNormTest[:, outSize:],
                                                   mlpNormTest[:, 0:outSize]),
                                callbacks=stModelCallbacks)


plt.plot(np.linspace(0, standardTestModel.trainTime, len(standardTestModel.lossDict['newLoss'])), standardTestModel.lossDict['newLoss'])

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








