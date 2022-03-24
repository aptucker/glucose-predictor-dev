# Written and designed by Aaron Tucker 
#
#
#
# ----------------------------------------------------------------------------
"""
Custom callbacks and related classes
"""

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
import time
import math

class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
    
    """
    Custom callback for stopping at a set error to prevent overfitting
    
    Inputs: 
        Patience - The number of epochs to wait to stop if metric has not been 
        achieved
        baseLoss - The floor for loss to stop the model training
        
    Outputs:
        EarlyStoppingAtMinLoss - Stops training early if metrics are met
        
    """
    
    def __init__(self, patience=0, baseLoss=0):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        self.baseLoss = baseLoss
        self.best_weights = None
        
    
    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf
        
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if np.less(current, self.best):
            if np.less(current, self.baseLoss):
                self.best = current
                self.wait = 0
                self.stopped_epoch = epoch    
                self.best_weights = self.model.get_weights()
                self.model.stop_training = True
            
            self.best = current
            self.wait = 0
            self.best_weights = self.model.get_weights()
            
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch")
                self.model.set_weights(self.best_weights)
                
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

class earlyStoppingBatchLoss(tf.keras.callbacks.Callback):
    
    def __init__(self, patience=0, baseLoss=0):
        super(earlyStoppingBatchLoss, self).__init__()
        self.patience = patience
        self.baseLoss = baseLoss
        self.best_weights=None
        
    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_batch = 0
        self.best = np.Inf
        
    def on_batch_end(self, batch, logs=None):
        current = logs.get('loss')
        if np.less(current, self.best):
            if np.less(current, self.baseLoss):
                self.best = current
                self.wait = 0
                self.stopped_batch = batch
                self.best_weights = self.model.get_weights()
                self.model.stop_training = True
            
            self.best = current
            self.wait = 0
            self.best_weights = self.model.get_weights()
        
        else:
            self.wait += 1
            if self. wait >= self.patience:
                self.stopped_batch = batch
                self.model.stop_training = True
                print("Restoring model weights from end of best batch")
                self.model.set_weights(self.best_weights)
                
    def on_epoch_end(self, epoch, logs=None):
        if self.model.stop_training:
            self.model.stop_training = True
                
    
    def on_train_end(self, logs=None):
        if self.stopped_batch > 0:
            print("Batch %05d: early stopping" % (self.stopped_batch + 1))
        
    
    
        

class GetErrorOnBatch(tf.keras.callbacks.Callback):
    
    def __init__(self, lossList):
        super(GetErrorOnBatch, self).__init__()
        self.lossList = lossList
        
        
    def on_train_batch_end(self, batch, logs=None):
        self.lossList.append(logs['loss'])
        
class batchErrorModel(tf.keras.callbacks.Callback):
    
    def __init__(self):
        super(batchErrorModel, self).__init__()
        self.lossList = []
        # self.epochTimes = []
        
    def on_train_begin(self, logs=None):
        self.lossList = []
        self.model.lossDict = {}
        self.trainStart = time.perf_counter()
    
    def on_train_batch_end(self, batch, logs=None):
        self.lossList.append(logs['loss'])
    
    def on_epoch_end(self, batch, logs=None):
        print('\nEpoch Done:' f'{time.perf_counter()-self.trainStart}')
    
    def on_train_end(self, logs=None):
        self.model.lossDict['newLoss'] = self.lossList
        self.trainStop = time.perf_counter()
        self.model.trainTime = self.trainStop - self.trainStart

class PlotLearning(tf.keras.callbacks.Callback):
    """
    Callback to plot the learning curves of the model during training.
    """
    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []
            

    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]
        
        # Plotting
        metrics = [x for x in logs if 'val' not in x]
        
        f, axs = plt.subplots(1, len(metrics), figsize=(15,5))
        clear_output(wait=True)

        for i, metric in enumerate(metrics):
            axs[i].plot(range(1, epoch + 2), 
                        self.metrics[metric], 
                        label=metric)
            if logs['val_' + metric]:
                axs[i].plot(range(1, epoch + 2), 
                            self.metrics['val_' + metric], 
                            label='val_' + metric)
                
            axs[i].legend()
            axs[i].grid()

        plt.tight_layout()
        plt.show()
        

class lrScheduler(tf.keras.callbacks.Callback):
    
    def __init__(self, refLoss, gain):
        super(lrScheduler, self).__init__()
        self.refLoss = refLoss
        self.gain = gain
        self.lossList = []
    
    def on_train_begin(self, logs=None):
        self.lossList = []
        self.model.lossDict = {}
        self.trainStart = time.perf_counter()
        
    def on_train_batch_begin(self, batch, logs=None):
        self.tic = time.perf_counter()
        
    def on_train_batch_end(self, batch, logs=None):
        
        self.lossList.append(logs['loss'])
        
        self.toc = time.perf_counter()
        
        dt = self.toc - self.tic
        
        new_lr = float(self.gain * (logs['loss'] - self.refLoss))
        
        # Derivative Equation
        # new_lr = float(self.gain * ( (logs['loss'] - self.refLoss)  +  
        #                             1.0*((logs['loss'] - self.lossHistory[-1])/dt) ))
        
        # Proportional Integral Derivative Equation
        # new_lr = float(self.gain * ( 0.38*(logs['loss'] - self.refLoss)  +  
        #                             0.0*((logs['loss'] - self.lossHistory[-1])/dt) + 
        #                             (dt/1.0)*(self.refLoss - (logs['loss'] + self.lossHistory[-1])/2 ) ))
        
        
        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
    
    def on_train_end(self, logs=None):
        self.model.lossDict['newLoss'] = self.lossList
        self.trainStop = time.perf_counter()
        
        self.model.trainTime = self.trainStop - self.trainStart
    
    def resetVars(self, refLoss, gain):
        self.refLoss = refLoss
        self.gain = gain
        



class sinLRScheduler(tf.keras.callbacks.Callback):
    
    def __init__(self, freq, startLR):
        super(sinLRScheduler, self).__init__()
        self.freq = freq
        self.startLR = startLR
        
    def on_train_begin(self, logs=None):
        self.trainStart = time.perf_counter()
    
    def on_train_batch_begin(self, batch, logs=None):
        self.tic = time.perf_counter()
        
    def on_train_batch_end(self, batch, logs=None):
        self.toc = time.perf_counter()
        
        tNow = self.toc - self.trainStart
        
        new_lr = float(self.startLR * math.sin(self.freq * tNow))
        
        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
    
    def on_train_end(self, logs=None):
        self.trainStop = time.perf_counter()
        self.model.trainTime = self.trainStop - self.trainStart
        
        
