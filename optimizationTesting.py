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

# %% Callbacks Testing

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
[mlpNormTest, testMean, testStd] = trn.zscoreData(lPat.testData.to_numpy())


# callbacks = [cBacks.EarlyStoppingAtMinLoss(patience = 20, baseLoss = 0.15),
#              cBacks.EarlyStoppingAtMinLoss(patience = 20, baseLoss = 0.15),
#              cBacks.EarlyStoppingAtMinLoss(patience = 20, baseLoss = 0.15),
#              cBacks.EarlyStoppingAtMinLoss(patience = 20, baseLoss = 0.15)]

class lrScheduler(tf.keras.callbacks.Callback):
    
    def __init__(self, refLoss, lossHistory, gain):
        super(lrScheduler, self).__init__()
        self.refLoss = refLoss
        self.lossHistory = lossHistory
        self.gain = gain
        
    def on_train_batch_begin(self, batch, logs=None):
        self.tic = time.perf_counter()
        
    def on_train_batch_end(self, batch, logs=None):
        # lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        
        self.toc = time.perf_counter()
        
        dt = self.toc - self.tic
        
        new_lr = float(self.gain * (logs['loss'] - self.refLoss))
        # new_lr = float(self.gain * ( (logs['loss'] - self.refLoss)  +  
        #                             1.0*((logs['loss'] - self.lossHistory[-1])/dt) ))
        # new_lr = float(self.gain * ( 0.38*(logs['loss'] - self.refLoss)  +  
        #                             0.0*((logs['loss'] - self.lossHistory[-1])/dt) + 
        #                             (dt/1.0)*(self.refLoss - (logs['loss'] + self.lossHistory[-1])/2 ) ))
        # if batch%100:
            # print(new_lr)
        # print(self.lossHistory[-1])
        
        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)


batch_end_loss = list()

callbacks = [cBacks.EarlyStoppingAtMinLoss(patience=20, baseLoss=0.25),
             cBacks.GetErrorOnBatch(batch_end_loss),
             lrScheduler(refLoss=0.2, lossHistory=batch_end_loss, gain=0.1)]

# callbacks = [cBacks.earlyStoppingBatchLoss(patience=2, baseLoss=0.40),
#              cBacks.GetErrorOnBatch(batch_end_loss),
#              lrScheduler(refLoss=0.2, lossHistory=batch_end_loss, gain=1.1)]

lossWeights = [[1.0, 1.0, 1.0, 1.0],
               [1.0, 1.0, 1.0, 1.0],
               [1.0, 1.0, 1.0, 1.0],
               [1.0, 1.0, 1.0, 1.0]]

b_size = 10
epochs = 10

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
              metrics=tf.keras.metrics.RootMeanSquaredError(),
              loss_weights = [1.0, 1.0, 1.0, 1.0])
# models["GRU H=1"] = model

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
batchLossDF = pd.DataFrame(batch_end_loss, index=np.linspace(0, timePatGRU, len(batch_end_loss)), columns=['Batch Loss'])
batchLossDF['Batch Loss'].plot()

print(timePatGRU)

# lPat.timeStorage["GRU H=1"] = timePatGRU
# rPat.timeStorage["GRU H=1"] = timePatGRU

 # %% System ID
dt = batchLossDF.index.values[1] - batchLossDF.index.values[0]

batchLossDF = pd.DataFrame(batch_end_loss, index=np.linspace(0, timePatGRU, len(batch_end_loss)), columns=['Batch Loss'])
batchLossDF.insert(0, 'Batch Loss db/dt', batchLossDF['Batch Loss'].diff()/dt)
batchLossDF.insert(0, 'Batch Loss db2/dt2', batchLossDF['Batch Loss db/dt'].diff()/dt)


batchLossDF.dropna(axis=0, inplace=True)

uBatch = 0.2*np.ones(len(batchLossDF))

batchLossDF['Input'] = uBatch

constants = np.matmul(np.linalg.pinv(batchLossDF[['Batch Loss db2/dt2', 'Batch Loss db/dt', 'Batch Loss']].values), batchLossDF['Input'])
constants1 = np.matmul(np.linalg.pinv(batchLossDF[['Batch Loss db/dt', 'Batch Loss']].values), batchLossDF['Input'])

a1 = [-constants1[1]/constants1[0]]
b1 = [1/constants1[0]]
c1 = [1]
d1 = [0]

a = [[-constants[1]/constants[0], -constants[2]/constants[0]],
     [1, 0]]

b = [[1/constants[0]],
     [0]]

c = [0, 1]

d = [0]

sysTest = ctrlmat.ss(a, b, c, d)

tTest = batchLossDF.index.values

u = 0.2*np.ones([len(tTest), 1])

x0 = [0, 0]
x0t = [0, batchLossDF['Batch Loss'].iloc[0]]

yout, tout, xout = ctrlmat.lsim(sysTest, u, tTest, x0t)

outDF = pd.DataFrame(yout, index=tout)
outDF.plot()

# %% Control Systems Testing
wn = 2
zeta = .7
k = 1

num= [k*wn**2]
deno= [1, 2*zeta*wn, wn**2]
g = ctrlmat.tf(num, deno)
t = ctrlmat.feedback(g,1)
# [A, B, C, D] = ctrlmat.tf2ss(num, deno)
sysmid = ctrlmat.tf2ss(num, deno)
sys = ctrlmat.ss(sysmid.A, -sysmid.B, -sysmid.C, sysmid.D)

t = np.linspace(0, 5, 100)
x0 = [0, 1]
u = 0.2*np.ones([len(t), 1])

yout, tout, xout = ctrlmat.lsim(sys, u, t, x0)

outDF = pd.DataFrame(yout, index=tout)
outDF.plot()







