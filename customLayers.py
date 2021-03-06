# Written and designed by Aaron Tucker 
#
#
#
# ----------------------------------------------------------------------------
"""
Custom layers built from Keras API
"""

import tensorflow as tf

class staticBiasLayer(tf.keras.layers.Layer):
    """ This is the custom layer which uses the equation presented in Alpaydin. 
    
    Arguments: 
        units - number of output units
        activation - pulls from list of keras built-in activations
        use_bias - set to True for now
        kernel_initializer - pulls from keras built-in initializers
        ones_size - size of bias matrix to add (should match training batch size)
        
    Returns:
        Dot product of W and inputs with bias (b) always set to a matrix of 1s
        and concatenated to the inputs - the keras activation wraps the entire
        equation in this layer
    """

    def __init__(self, 
                 units,      
                 activation,
                 use_bias,
                 kernel_initializer,
                 ones_size):        
        super(staticBiasLayer, self).__init__()
        self.units = int(units)
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.ones_size = ones_size
        
        
    def build(self, input_shape):
        self.w = self.add_weight(shape=(self.units, input_shape[-1]+1),      
                                 initializer = self.kernel_initializer,           
                                 trainable=True,
                                 ) 

        self.b = tf.ones([self.ones_size, 1], tf.float32) 
        
    def call(self, inputs):
        return self.activation(tf.transpose(tf.matmul(self.w, tf.transpose(tf.concat([self.b, inputs], 1)))))
    

class staticBiasTowerH2(tf.keras.layers.Layer):
    
    def __init__(self,
                 shapes,
                 activators,
                 ones_size):
        super(staticBiasTowerH2, self).__init__()
        self.hLayer1Units = shapes[0]
        self.hLayer2Units = shapes[1]
        self.outLayerUnits = shapes[2]
        self.hLayer1Activator = activators[0]
        self.hLayer2Activator = activators[1]
        self.outLayerActivator = activators[2]
        self.ones_size = ones_size
        
        initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=0.005)
        
        self.hLayer1 = staticBiasLayer(self.hLayer1Units,
                                       self.hLayer1Activator,
                                       True,
                                       initializer,
                                       ones_size)
        self.hLayer2 = staticBiasLayer(self.hLayer2Units,
                                       self.hLayer2Activator,
                                       True,
                                       initializer,
                                       ones_size)
        self.outLayer = staticBiasLayer(self.outLayerUnits,
                                        self.outLayerActivator,
                                        True,
                                        initializer,
                                        ones_size)
        
    def call(self, inputs):
        x = self.hLayer1(inputs)
        x = self.hLayer2(x)
        x = self.outLayer(x)
        return x
        
        
        
        