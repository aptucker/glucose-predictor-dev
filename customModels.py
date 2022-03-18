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
        self.hLayer = cLayers.staticBiasLayer(units = self.inShape,
                                     activation = self.activators[0],
                                     use_bias = self.use_bias,
                                     kernel_initializer = self.initializers[0],
                                     ones_size = self.bias_size)
        self.outLayer = cLayers.staticBiasLayer(units = self.outShape,
                                                activation = self.activators[1],
                                                use_bias = self.use_bias,
                                                kernel_initializer = self.initializers[1],
                                                ones_size = self.bias_size)
        
    def call(self, inputs):
        x = self.hLayer(inputs)
        return self.outLayer(x)

class sbSeqModelH2(tf.keras.Model):
    
    def __init__(self,
                 shapes,
                 use_bias,
                 initializers,
                 bias_size,
                 activators):
        super(sbSeqModelH2, self).__init__()
        self.inShape = shapes[0]
        self.midShape = shapes[1]
        self.outShape = shapes[2]
        self.use_bias = use_bias
        self.initializers = initializers
        self.activators = activators
        self.bias_size = bias_size
        
        self.inLayer = tf.keras.Input(shape=(self.inShape, ))
        self.hLayer = cLayers.staticBiasLayer(units = self.inShape,
                                     activation = self.activators[0],
                                     use_bias = self.use_bias,
                                     kernel_initializer = self.initializers[0],
                                     ones_size = self.bias_size)
        self.hLayer2 = cLayers.staticBiasLayer(units = self.midShape,
                                               activation = self.activators[1],
                                               use_bias = self.use_bias,
                                               kernel_initializer = self.initializers[1],
                                               ones_size = self.bias_size)
        self.outLayer = cLayers.staticBiasLayer(units = self.outShape,
                                                activation = self.activators[2],
                                                use_bias = self.use_bias,
                                                kernel_initializer = self.initializers[2],
                                                ones_size = self.bias_size)
    
    def call(self, inputs):
        x = self.hLayer(inputs)
        x = self.hLayer2(x)
        return self.outLayer(x)
        

class parallelModel(tf.keras.Model):
    """Parallel NN which takes training from right and left arms
    
    Arguments:
        inShape = input shape
        outShape = output shape
        use_bias = toggle bias layer (currently unused)
        bias_size = size of bias matrix to add in layer (same as batch size)
        
    Returns:
        Transpose of output layer which concatenates the two 'towers'
    """
    
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
                 
                 
class parallelModelH2(tf.keras.Model):
    
    def __init__(self,
                 tower1Shapes,
                 tower2Shapes,
                 tower1Activators,
                 tower2Activators,
                 bias_size):
        super(parallelModelH2, self).__init__()
        self.tower1Shapes = tower1Shapes
        self.tower2Shapes = tower2Shapes
        self.tower1Activators = tower1Activators
        self.tower2Activators = tower2Activators
        self.bias_size = bias_size
                
        self.tower1 = cLayers.staticBiasTowerH2(self.tower1Shapes,
                                                self.tower1Activators,
                                                self.bias_size)
        self.tower2 = cLayers.staticBiasTowerH2(self.tower2Shapes,
                                                self.tower2Activators,
                                                self.bias_size)
        self.denseLayer = tf.keras.layers.Dense(1, activation = None)
        
    def call(self, inputs):
        x1 = self.tower1(inputs[:, 0:self.tower1Shapes[0]])
        x2 = self.tower2(inputs[:, self.tower2Shapes[0]:])
        
        merged = tf.keras.layers.concatenate([x1, x2], axis=0)
        outputs = self.denseLayer(tf.transpose(merged))
        return tf.transpose(outputs)
                 
                 
                 
class gruH1(tf.keras.Model):
    
    def __init__(self,
                 shapes,
                 use_bias,
                 bias_size,
                 activators):
        super(gruH1, self).__init__()
        self.inShape = shapes[0]
        self.gruShape = shapes[1]
        self.outShape = shapes[2]
        self.use_bias = use_bias
        self.bias_size = bias_size
        self.gruActivator = activators[0]
        self.recActivator = activators[1]
        self.outActivator = activators[2]
        
        # self.inLayer = tf.keras.Input(shape=[None , None, self.inShape])
        
        self.gruLayer = tf.keras.layers.GRU(units=self.gruShape,
                                            activation = self.gruActivator,
                                            recurrent_activation = self.recActivator,
                                            use_bias = self.use_bias,
                                            bias_initializer = 'ones')
        
        self.outLayer = tf.keras.layers.Dense(units=self.outShape,
                                              activation = self.outActivator,
                                              use_bias = self.use_bias,
                                              bias_initializer = 'ones')
        
    def call(self, inputs):
        x = self.gruLayer(inputs)
        outputs = self.outLayer(x)
        return outputs
    
    