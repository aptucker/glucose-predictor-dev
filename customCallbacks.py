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

class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
    
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
