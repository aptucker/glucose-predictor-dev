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
        inShape = shape of the input (columns)
        outShape = shape of the output
        use_bias = whether to use bias (currently unused)
        initializers = pass initializers to the model if necessary
        bias_size = size of the bias to add (also same as batch size)
        activators = activation functions to pass to layers
        
    Returns:
        outLayer = custom layer with linear activation
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
        

class parallelModel(tf.keras.Model):
    
    def __init__(self,
                 inShape,
                 outShape,
                 use_bias,
                 bias_size):
        
        super(parallelModel, self).__init__()
        self.inShape = inShape
        self.outShape = outShape
        self.use_bias = use_bias
        self.bias_size = bias_size
                
        self.inLayer = tf.keras.Input(shape=(self.inShape*2, ))
        self.hLayer1 = cLayers.staticBiasLayer(units = inShape,
                                     activation = 'sigmoid',
                                     use_bias = self.use_bias,
                                     kernel_initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=0.005),
                                     ones_size = self.bias_size)
        self.hLayer2 = cLayers.staticBiasLayer(units = inShape,
                                     activation = 'sigmoid',
                                     use_bias = self.use_bias,
                                     kernel_initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=0.005),
                                     ones_size = self.bias_size)
        self.outLayer1 = cLayers.staticBiasLayer(units = outShape,
                                                activation = None,
                                                use_bias = self.use_bias,
                                                kernel_initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=0.005),
                                                ones_size = self.bias_size)
        self.outLayer2 = cLayers.staticBiasLayer(units = outShape,
                                                activation = None,
                                                use_bias = self.use_bias,
                                                kernel_initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=0.005),
                                                ones_size = self.bias_size)
        
        self.denseLayer = tf.keras.layers.Dense(1, activation=None)
        
        
    def call(self, inputs):
        tower1 = self.hLayer1(inputs[:, 0:self.inShape])
        tower2 = self.hLayer2(inputs[:, self.inShape:])
        
        tower1a = self.outLayer1(tower1)
        tower2a = self.outLayer2(tower2)
        
        merged = tf.keras.layers.concatenate([tower1a, tower2a], axis=0)
        outputs = self.denseLayer(tf.transpose(merged))
        return tf.transpose(outputs)
                 
                 
                 
                 
                 
                 
                 
                 
                 