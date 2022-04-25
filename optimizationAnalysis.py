# Written and designed by Aaron Tucker 
#
#
#
# ----------------------------------------------------------------------------
"""
This is the file which optimizes for training time. There are numerous model 
declarations, each has it's own learning rate for testing. Models are saved 
in a dictionary with the results of their loss over time. 

See the customCallbacks file for custom learning rate and error storage 
functions. See the optimizationFunctions file for other functions and classes
used in this analysis file.

This file also includes analysis from the control package, although it is 
unclear if it always produces the same results as MATLAB. 

Results are stored in the time_results.pickle file. It is currently a large 
file; a fix is coming.
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
from sys import platform

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

# %% Dictionary Load

if platform == 'win32':
    with open('time_results.pickle', 'rb') as f:
        timeResults = pickle.load(f)
    
if platform == 'darwin':
    with open('time_results.pickle', 'rb') as f:
        timeResults = pickle.load(f)

# %% Model Definitions
"""
All model definitions are located in this section. The basic GRU structure is
defined once, all other models are defined using that same structure. The 
models are stored in a dictionary for later access.

Initial model weights are stored in the weights_storage file.
"""

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

# Combine the data to form larger sets; condition it for use with the models.
[lPatsTrain,
 rPatsTrain,
 lPatsTest,
 rPatsTest] = optFun.dataCombiner(lPats, rPats, partNum, partSize, lag)

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

b_size = 10
epochs = 20

models = {}


# General definition of the GRU structure, this is reused for each new model.
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

# R=1, Q=1 OLD
lr84 = tf.keras.Model(inputs=inputs,
                      outputs=output)

# R=5, Q=1 OLD
lr31 = tf.keras.Model(inputs=inputs,
                      outputs=output)

# R=10, Q=1 OLD
lr19 = tf.keras.Model(inputs=inputs,
                      outputs=output)

# R=1, Q=1
lr91 = tf.keras.Model(inputs=inputs,
                      outputs=output)

# R=5, Q=1
lr36 = tf.keras.Model(inputs=inputs,
                      outputs=output)

# R=10, Q=1
lr23 = tf.keras.Model(inputs=inputs,
                      outputs=output)


jdstModel = cModels.sbSeqModel(H, 
                            K,
                            use_bias = True,
                            initializers = initializers,
                            bias_size = b_size,
                            activators = ["sigmoid", None])
jdstModel(tf.keras.Input(shape=H))

lr_schedule_standard = tf.keras.optimizers.schedules.ExponentialDecay(
                                            initial_learning_rate=0.01,
                                            decay_steps=10000,
                                            decay_rate=0.9)

lr_schedule_adam = tf.keras.optimizers.schedules.ExponentialDecay(
                                            initial_learning_rate=0.01,
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

standardScheduleStandardModel.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule_standard),
                                      loss=tf.keras.losses.MeanSquaredError(),
                                      metrics=tf.keras.metrics.RootMeanSquaredError())

standardScheduleAdamModel.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule_adam),
                                      loss=tf.keras.losses.MeanSquaredError(),
                                      metrics=tf.keras.metrics.RootMeanSquaredError())
jdstModel.compile(optimizer= tf.keras.optimizers.SGD(learning_rate=0.01), 
              loss=tf.keras.losses.MeanSquaredError(), 
              metrics=tf.keras.metrics.RootMeanSquaredError())

# OLD
lr84.compile(optimizer=tf.keras.optimizers.SGD(),
             loss=tf.keras.losses.MeanSquaredError(),
             metrics=tf.keras.metrics.RootMeanSquaredError())
lr31.compile(optimizer=tf.keras.optimizers.SGD(),
             loss=tf.keras.losses.MeanSquaredError(),
             metrics=tf.keras.metrics.RootMeanSquaredError())
lr19.compile(optimizer=tf.keras.optimizers.SGD(),
             loss=tf.keras.losses.MeanSquaredError(),
             metrics=tf.keras.metrics.RootMeanSquaredError())

lr91.compile(optimizer=tf.keras.optimizers.SGD(),
             loss=tf.keras.losses.MeanSquaredError(),
             metrics=tf.keras.metrics.RootMeanSquaredError())
lr36.compile(optimizer=tf.keras.optimizers.SGD(),
             loss=tf.keras.losses.MeanSquaredError(),
             metrics=tf.keras.metrics.RootMeanSquaredError())
lr23.compile(optimizer=tf.keras.optimizers.SGD(),
             loss=tf.keras.losses.MeanSquaredError(),
             metrics=tf.keras.metrics.RootMeanSquaredError())

callbacks = [cBacks.EarlyStoppingAtMinLoss(patience=20, baseLoss=0.25),
             cBacks.lrScheduler(refLoss=0.2, gain=0.1)]

standardModel.callbacks = [cBacks.EarlyStoppingAtMinLoss(patience=20, baseLoss=0.25),
                           cBacks.batchErrorModel()]
adamStandardModel.callbacks = [cBacks.EarlyStoppingAtMinLoss(patience=20, baseLoss=0.25),
                               cBacks.batchErrorModel()]
lrScheduledStandardModel.callbacks = [cBacks.EarlyStoppingAtMinLoss(patience=20, baseLoss=0.25),
                                      cBacks.lrScheduler(refLoss=0.1, gain=0.85)]
lrScheduledAdamModel.callbacks = [cBacks.EarlyStoppingAtMinLoss(patience=20, baseLoss=0.25),
                                  cBacks.lrScheduler(refLoss=0.1, gain=0.1)]
standardScheduleStandardModel.callbacks = [cBacks.EarlyStoppingAtMinLoss(patience=20, baseLoss=0.25),
                                           cBacks.batchErrorModel()]
standardScheduleAdamModel.callbacks = [cBacks.EarlyStoppingAtMinLoss(patience=20, baseLoss=0.25),
                                       cBacks.batchErrorModel()]
jdstModel.callbacks = [cBacks.EarlyStoppingAtMinLoss(patience=20, baseLoss=0.25),
                       cBacks.batchErrorModel()]

# OLD
lr84.callbacks = [cBacks.EarlyStoppingAtMinLoss(patience=20, baseLoss=0.25),
                  cBacks.lrScheduler(refLoss=0.1, gain=0.84)]
lr31.callbacks = [cBacks.EarlyStoppingAtMinLoss(patience=20, baseLoss=0.25),
                  cBacks.lrScheduler(refLoss=0.1, gain=0.31)]
lr19.callbacks = [cBacks.EarlyStoppingAtMinLoss(patience=20, baseLoss=0.25),
                  cBacks.lrScheduler(refLoss=0.1, gain=0.19)]

lr91.callbacks = [cBacks.EarlyStoppingAtMinLoss(patience=20, baseLoss=0.25),
                  cBacks.lrScheduler(refLoss=0.1, gain=0.91)]
lr36.callbacks = [cBacks.EarlyStoppingAtMinLoss(patience=20, baseLoss=0.25),
                  cBacks.lrScheduler(refLoss=0.1, gain=0.36)]
lr23.callbacks = [cBacks.EarlyStoppingAtMinLoss(patience=20, baseLoss=0.25),
                  cBacks.lrScheduler(refLoss=0.1, gain=0.23)]

models['standard'] = standardModel
standardModel.save_weights('weights_storage\\standardModel.start')

models['adam'] = adamStandardModel
adamStandardModel.save_weights('weights_storage\\adamModel.start')

models['lrStandard'] = lrScheduledStandardModel
lrScheduledStandardModel.save_weights('weights_storage\\lrStandardModel.start')

models['lrAdam'] = lrScheduledAdamModel
lrScheduledAdamModel.save_weights('weights_storage\\lrAdamModel.start')

models['standardSchedule'] = standardScheduleStandardModel
standardScheduleStandardModel.save_weights('weights_storage\\standardScheduleModel.start')

models['standardAdam'] = standardScheduleAdamModel
standardScheduleAdamModel.save_weights('weights_storage\\standardAdamModel.start')

models['jdst'] = jdstModel
# jdstModel.save_weights('jdstModel.start')

# OLD
models['lr84'] = lr84
lr84.save_weights('weights_storage\\lr84Model.start')

models['lr31'] = lr31
lr31.save_weights('weights_storage\\lr31Model.start')

models['lr19'] = lr19
lr19.save_weights('weights_storage\\lr19Model.start')

models['lr91'] = lr91
lr91.save_weights('weights_storage\\lr91Model.start')

models['lr36'] = lr36
lr36.save_weights('weights_storage\\lr36Model.start')

models['lr23'] = lr23
lr23.save_weights('weights_storage\\lr23Model.start')

# modelNames = list(['standard', 'adam', 'lrStandard', 'lrAdam', 'standardSchedule', 'standardAdam', 'jdst'])
# modelNames = list(['standardSchedule', 'standardAdam'])

# %% Time trials for N=1000
"""
Time tester function run for only N=1000 observations. See optimizationFunctions
for function definition. 
"""

modelNames = list(['standard', 'adam', 'lrStandard', 'lrAdam', 'standardSchedule', 'standardAdam', 'jdst'])
# modelNames = list(['standardSchedule', 'standardAdam'])
# modelNames = list(['jdst'])

outDict = optFun.timeTester(lPats,
                            rPats,
                            partNum,
                            partSize,
                            lag,
                            models, 
                            modelNames,
                            b_size,
                            40,
                            trialsToRun,
                            maxDataSize=1000)


# %% Time trials for N=10000
"""
Time tester function for N=10000 observations. See optimizationFunctions for
function definition.
"""

modelNames = list(['standard', 'adam', 'lrStandard', 'lrAdam', 'standardSchedule', 'standardAdam', 'jdst'])
# modelNames = list(['standardSchedule', 'standardAdam'])

outDict2 = optFun.timeTester(lPats,
                            rPats,
                            partNum,
                            partSize,
                            lag,
                            models,
                            modelNames,
                            b_size,
                            epochs,
                            trialsToRun,
                            maxDataSize=10000)

# %% Time trials for N=71400
"""
Time tester function for N=71400 observations. The number of observations 
must be divisible by 10 for this method, therefore N=71400 is the largest amount
of data divisible by 10.
"""

# modelNames = list(['standard', 'adam', 'lrStandard', 'lrAdam', 'standardSchedule', 'standardAdam', 'jdst'])
# modelNames = list(['lr84', 'lr31', 'lr19'])
modelNames = list(['lr91', 'lr36', 'lr23'])

outDictAllData = optFun.timeTester(lPats,
                            rPats,
                            partNum,
                            partSize,
                            lag,
                            models,
                            modelNames,
                            b_size,
                            epochs,
                            trialsToRun,
                            maxDataSize=71400)

# %% Simulate System Response
"""
Use the control package to simulate the system response to the designed
controller. The values of the A,B,C,D matrices are taken from MATLAB because
it is unclear if the control package will generate the same values as MATLAB.
"""

tempList = []
tcDF = pd.DataFrame()
modelsForTC = ['standard']

for i in range(len(modelsForTC)):
    for e in range(len(timeResults['n=71400'][modelsForTC[i]])):
        tc = optFun.findTimeConstant(timeResults['n=71400'][modelsForTC[i]]['Loss It. ' f'{e+1}'])
        
        tempList.append(tc)
    
    tcDF[modelsForTC[i]] = tempList
    
    tempList = []

yss = 0.21
tc = float(tcDF.mean())
refIn = 0.01
kG = yss/refIn

x01 = 2.5

#OLD VALUES
# a1 = -12.81
# a5 = -5.11
# a10 = -3.394
# b1 = 4
# b5 = 2
# b10 = 2
# c1 = 3.032
# c5 = 2.212
# c10 = 1.335

a0 = -0.38
a1 = -7.75
a5 = -3.31
a10 = -2.28
b0 = 4
b1 = 2
b5 = 2
b10 = 1
c0 = 2.03
c1 = 3.68
c5 = 1.46
c10 = 1.89
d = 0

g = ctrlmat.ss(a0, b0, c0, d)
h1 = ctrlmat.ss(a1, b1, c1, d)
h5 = ctrlmat.ss(a5, b5, c5, d)
h10 = ctrlmat.ss(a10, b10, c10, d)

t = np.linspace(0, 20, 10000)
u = 0.02*np.ones([len(t), 1])

y0 = ctrlmat.lsim(g, 0.02*np.ones([len(t), 1]), t, 0.9)
y1 = ctrlmat.lsim(h1, u, t, x01)
y5 = ctrlmat.lsim(h5, u, t, x01)
y10 = ctrlmat.lsim(h10, u, t, x01)

simDF = pd.DataFrame(index=t)
simDF['gSim'] = y0[0]
simDF['lr91Sim'] = y1[0]
simDF['lr36Sim'] = y5[0]
simDF['lr23Sim'] = y10[0]


# plt.plot(t, y1[0])

# %% Dictionary Save
"""
Load old time trial results dictionary and save any new results that have been
generated.
"""

dictToSave = {
    'n=1000': timeResults['n=1000'],
    'n=10000': timeResults['n=10000'],
    'n=71400': timeResults['n=71400']}

# dictToSave['n=71400']['lr84'] = outDictAllData['lr84']
# dictToSave['n=71400']['lr31'] = outDictAllData['lr31']
# dictToSave['n=71400']['lr19'] = outDictAllData['lr19']

dictToSave['n=71400']['lr91'] = outDictAllData['lr91']
dictToSave['n=71400']['lr36'] = outDictAllData['lr36']
dictToSave['n=71400']['lr23'] = outDictAllData['lr23']

# %% Pickle Dump
if platform == 'win32':
    with open("time_results.pickle", "wb") as f:
        pickle.dump(dictToSave, f)

if platform == 'darwin':
    with open("time_results.pickle", "wb") as f:
        pickle.dump(dictToSave, f)


# %% Time Results Analysis
"""
Find the time constant of the GRU training response.
"""

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
    for e in range(len(timeResults[0]['n=71400'][modelNames[i]])):
        tc = optFun.findTimeConstant(timeResults[0]['n=71400'][modelNames[i]]['Loss It. ' f'{e+1}'])
        
        tempList.append(tc)
    
    tcDF[modelNames[i]] = tempList
    
    tempList = []

# %% Convergence Time Results
"""
Find the convergence times and time constants of training results. Save the
results to an excel file.
"""

modelNames = list(['standard', 'adam', 'lrStandard', 'jdst'])#, 'lr91', 'lr36', 'lr23'])

tempListTC = []
tempListConv = []
tempListMin = []
tempListConv10 = []
convDF = pd.DataFrame()
conv10DF = pd.DataFrame()
tcDF = pd.DataFrame()
minDF = pd.DataFrame()

for i in range(len(modelNames)):
    for e in range(len(timeResults['n=71400'][modelNames[i]])):
        convTime = optFun.findConvergenceTime(timeResults['n=71400'][modelNames[i]]['Loss It. ' f'{e+1}'],
                                              averageWindow=30,
                                              threshold=0.3)
        
        convTime10 = optFun.findConvergenceTime(timeResults['n=10000'][modelNames[i]]['Loss It. ' f'{e+1}'],
                                              averageWindow=30,
                                              threshold=0.3)
        
        tc = optFun.findTimeConstant(timeResults['n=71400'][modelNames[i]]['Loss It. ' f'{e+1}'])
        
        minVal = timeResults['n=71400'][modelNames[i]]['Loss It. ' f'{e+1}'].rolling(30).mean().min().values[0]
                
        tempListConv.append(convTime)
        tempListConv10.append(convTime10)
        tempListTC.append(tc)
        tempListMin.append(minVal)
    
    convDF[modelNames[i]] = tempListConv
    conv10DF[modelNames[i]] = tempListConv10
    tcDF[modelNames[i]] = tempListTC
    minDF[modelNames[i]] = tempListMin
    
    tempListConv = []
    tempListConv10 = []
    tempListTC = []
    tempListMin = []

# convDF.to_excel('G:\\My Drive\\Minnesota Files\\Erdman Research\\Final Paper\\convTimesRaw.xlsx', sheet_name='Raw_Python_Data', index=False)
# tcDF.to_excel('G:\\My Drive\\Minnesota Files\\Erdman Research\\Final Paper\\tcRaw.xlsx', sheet_name='Raw_Python_Data', index=False)


# %% JDST vs GRU Plot n=71400
"""
JDST vs GRU plot for dissertation. Saved in time_trial_plots file.
"""

colors = ['#3F7D6E',
          '#593560',
          '#A25756',
          '#E7B56D']

lossLabel = r'LOSS [$(\frac{\mathrm{mg}}{\mathrm{dL}})^2$]'

# modelsToPlot = ['standard', 'adam', 'jdst', 'lrStandard']
modelsToPlot = ['standard', 'jdst']
rangeToPlot = range(7,8)

fig3, ax3 = plt.subplots(1,1)
ax3.set_prop_cycle(color=colors)

legendNames3 = cPlots.timeTrialPlot(timeResults['n=71400'], modelsToPlot, rangeToPlot, ax3)
simDF['gSim'].plot(ax=ax3)
ax3.legend(['GRUNN', 'TDNN', 'Simulated GRUNN'])

# ax3.legend().set_visible(False)
ax3.set_xlabel('TIME [s]')
ax3.set_ylabel(r'LOSS [$(\frac{\mathrm{mg}}{\mathrm{dL}})^2$]')
# ax3.set_ylabel('ERROR')
ax3.set_title('NETWORK LOSS DURING TRAINING n=71400')
ax3.set_xlim([-0.1, 20])
# simDF['gSim'].plot(ax=ax3)

# ax3.get_lines()[0].set_color('black')

# plt.savefig('C:\\Code\\glucose-predictor-dev\\time_trial_plots\\timeTrialN71400.pdf', bbox_inches='tight')
plt.savefig('C:\\Code\\glucose-predictor-dev\\time_trial_plots\\timeTrialN71400JandSandSim.pdf', bbox_inches='tight')
# plt.savefig('C:\\Code\\glucose-predictor-dev\\time_trial_plots\\timeTrialN71400Example.pdf', bbox_inches='tight')

# %% GRU vs Simulated GRU
"""
GRU vs Sim GRU plot for dissertation. Saved in time_trial_plots file.
"""

colors = ['#3F7D6E',
          '#593560',
          '#A25756',
          '#E7B56D']

lossLabel = r'LOSS [$(\frac{\mathrm{mg}}{\mathrm{dL}})^2$]'

# modelsToPlot = ['standard', 'adam', 'jdst', 'lrStandard']
modelsToPlot = ['standard']
rangeToPlot = range(7,8)

fig3, ax3 = plt.subplots(1,1)
ax3.set_prop_cycle(color=colors)

legendNames3 = cPlots.timeTrialPlot(timeResults['n=71400'], modelsToPlot, rangeToPlot, ax3)
simDF['gSim'].plot(ax=ax3)

ax3.legend(['GRUNN', 'Simulated Response'])
ax3.set_xlabel('TIME [s]')
ax3.set_ylabel(lossLabel)
ax3.set_title('NETWORK LOSS DURING TRAINING n=71400')
ax3.set_xlim([-0.1, 20])

plt.savefig('C:\\Code\\glucose-predictor-dev\\time_trial_plots\\timeTrialN71400SimAndMeasured.pdf', bbox_inches='tight')

# %% GRU vs JDST vs LR vs Adam
"""
JDST vs GRU vs GRULR vs GRU Adam plot for dissertation. Saved in
time_trial_plots file.
"""

colors = ['#3F7D6E',
          '#593560',
          '#A25756',
          '#E7B56D']

lossLabel = r'LOSS [$(\frac{\mathrm{mg}}{\mathrm{dL}})^2$]'

modelsToPlot = ['standard', 'adam', 'jdst', 'lrStandard']
# modelsToPlot = ['standard']
rangeToPlot = range(7,8)

fig3, ax3 = plt.subplots(1,1)
ax3.set_prop_cycle(color=colors)

legendNames3 = cPlots.timeTrialPlot(timeResults['n=71400'], modelsToPlot, rangeToPlot, ax3)
# simDF['gSim'].plot(ax=ax3)

ax3.legend(['GRUNN', 'GRUNN w/Adam', 'TDNN', 'GRUNN w/LQR LR'])
ax3.set_xlabel('TIME [s]')
ax3.set_ylabel(lossLabel)
ax3.set_title('NETWORK LOSS DURING TRAINING n=71400')
ax3.set_xlim([-0.1, 20])

plt.savefig('C:\\Code\\glucose-predictor-dev\\time_trial_plots\\timeTrialN71400AllMethods.pdf', bbox_inches='tight')

# %% Different Learning Rates
"""
GRU with different learning rates plot for dissertation for N=71400. Saved in 
time_trial_plots file.
"""
colors = ['#3F7D6E',
          '#593560',
          '#E7B56D',
          '#A25756']

modelsToPlot = ['lr91', 'lr36', 'lr23']
rangeToPlot = range(7,8)

fig5, ax5 = plt.subplots(1, 1)
ax5.set_prop_cycle(color=colors)

legendNames5 = cPlots.timeTrialPlot(timeResults['n=71400'], modelsToPlot, rangeToPlot, ax5)
ax5.legend([r'LQR K=0.91 $\rightarrow$ Q=1, R=1',
            r'LQR K=0.36 $\rightarrow$ Q=1, R=5',
            r'LQR K=0.23 $\rightarrow$ Q=1, R=10'])
ax5.set_xlabel('TIME [s]')
ax5.set_ylabel(r'LOSS [$(\frac{\mathrm{mg}}{\mathrm{dL}})^2$]')
ax5.set_title('NETWORK LOSS DURING TRAINING n=71400')
ax5.set_xlim([-0.01, 2])
ax5.set_ylim([0, 3])
# props = dict(boxstyle='round', facecolor='white', alpha=0.25)
# textstr = r'K=0.84 $\rightarrow$ Q=1, R=1' '\n' r'K=0.31 $\rightarrow$ Q=1, R=5' '\n' r'K=0.19 $\rightarrow$ Q=1, R=10'
# ax5.text(0.45, 0.975, textstr, transform=ax4.transAxes, fontsize=10, verticalalignment='top', bbox=props)

plt.savefig('C:\\Code\\glucose-predictor-dev\\time_trial_plots\\timeTrialLR.pdf', bbox_inches='tight')

# %% Learning Rate Results n=10000
"""
JDST vs GRU vs GRULR vs GRU Adam plot for dissertation for N=10000. Saved in 
time_trial_plots file.
"""

colors = ['#3F7D6E',
          '#593560',
          '#A25756',
          '#E7B56D']

lossLabel = r'LOSS [$(\frac{\mathrm{mg}}{\mathrm{dL}})^2$]'

modelsToPlot = ['standard', 'adam', 'jdst', 'lrStandard']
rangeToPlot = range(7,8)

fig2, ax2 = plt.subplots(1,1)
ax2.set_prop_cycle(color=colors)

legendNames2 = cPlots.timeTrialPlot(timeResults['n=10000'], modelsToPlot, rangeToPlot, ax2)

ax2.legend(['GRUNN', 'GRUNN w/Adam', 'TDNN', 'GRUNN w/LQR LR'])
ax2.set_xlabel('TIME [s]')
ax2.set_ylabel(lossLabel)
ax2.set_title('NETWORK LOSS DURING TRAINING n=10000')
ax2.set_xlim([-0.1, 8])

plt.savefig('C:\\Code\\glucose-predictor-dev\\time_trial_plots\\timeTrialN10000.pdf', bbox_inches='tight')

# %% Different Learning Rates w/SIM
"""
GRU with different learning rates plot with simulated responses 
for dissertation for N=71400. Saved in time_trial_plots file.
"""

fig4, ax4 = plt.subplots(1,1)

modelsToPlot = ['lr91', 'lr36', 'lr23']

legendNames4 = cPlots.timeTrialPlot(timeResults['n=71400'], modelsToPlot, rangeToPlot, ax4)
simDF.plot(ax=ax4)
ax4.legend(legendNames4 + simDF.columns.values.tolist())
ax4.set_xlabel('TIME [s]')
ax4.set_ylabel(r'LOSS [$(\frac{\mathrm{mg}}{\mathrm{dL}})^2$]')
ax4.set_title('NETWORK LOSS vs SIMULATED RESPONSES n=71400')
ax4.set_xlim([-0.01, 2])
ax4.set_ylim([0, 3])
props = dict(boxstyle='round', facecolor='white', alpha=0.25)
textstr = r'K=0.84 $\rightarrow$ Q=1, R=1' '\n' r'K=0.31 $\rightarrow$ Q=1, R=5' '\n' r'K=0.19 $\rightarrow$ Q=1, R=10'
ax4.text(0.4, 0.965, textstr, transform=ax4.transAxes, fontsize=10, verticalalignment='top', bbox=props)

# plt.savefig('C:\\Code\\glucose-predictor-dev\\time_trial_plots\\timeTrialLRwSIM.pdf', bbox_inches='tight')


