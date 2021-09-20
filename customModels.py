# Written and designed by Aaron Tucker 
#
#
#
# ----------------------------------------------------------------------------
"""
Custom models and related classes
"""

import tensorflow as tf
import customLayers as cLayers

class sbSeqModel(tf.keras.Model):
    """Simple sequential model using the custom staticBiasLayer
    
    Arguments:
        
    Returns:
    """
    
    def __init__(self,
                 inShape,
                 outShape,
                 use_bias,
                 initializers,
                 bias_size,
                 activators):
        super(sbSeqModel, self).__init__()
        self.inShape = inShape
        self.outShape = outShape
        self.use_bias = use_bias
        self.initializers = initializers
        self.bias_size = bias_size
        self.activators = activators
        
        self.inLayer = tf.keras.Input(shape=(self.inShape, ))
        self.hLayer = cLayers.staticBiasLayer(units = inShape,
                                     activation = self.activators[0],
                                     use_bias = self.use_bias,
                                     kernel_initializer = self.initializers[0],
                                     ones_size = self.bias_size)
        self.outLayer = cLayers.staticBiasLayer(units = outShape,
                                                activation = self.activators[1],
                                                use_bias = self.use_bias,
                                                kernel_initializer = self.initializers[1],
                                                ones_size = self.bias_size)
        
        def call(self, inputs):
            x = self.hLayer(inputs)
            return self.outLayer(x)
        
