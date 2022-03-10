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


class GetErrorOnBatch(tf.keras.callbacks.Callback):
    
    def __init__(self, lossList):
        super(GetErrorOnBatch, self).__init__()
        self.lossList = lossList
        
        
    def on_train_batch_end(self, batch, logs=None):
        self.lossList.append(logs['loss'])
        


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
        
    
    