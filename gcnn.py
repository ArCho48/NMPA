import pdb
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple

class GCL(layers.Layer):
    """Graph convolutional layer"""
    def __init__(self, num_filters=3, input_dim=1, output_dim=1, use_bias=True, non_lin=tf.nn.leaky_relu, name="gcl", **kwargs):
        super(GCL, self).__init__(name=name,dtype=tf.float64,**kwargs)
        self.num_filters = num_filters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.non_lin = non_lin
        
        # Initialize trinable variables
        with tf.compat.v1.variable_scope(name):
            self.w = tf.compat.v1.get_variable( name='w', shape=(1, self.num_filters, self.input_dim, self.output_dim), initializer=tf.initializers.glorot_uniform(), dtype=tf.float64, trainable=True)
            if self.use_bias:
                self.b = tf.compat.v1.get_variable( name='b', shape=(1, self.num_filters, 1, self.output_dim), initializer=tf.initializers.glorot_uniform(), dtype=tf.float64, trainable=True)
                
    def call(self, S, x):        
        # Container to store inputs for K filter taps
        z = tf.expand_dims( x, axis=1 ) # Bx1xNxG
        
        # Graph polynomial
        for k in range(1,self.num_filters):
            x = tf.matmul(S, x) #BxNxG
            z = tf.concat([z,tf.expand_dims( x, axis=1 )], axis=1) #BxKxNxG
        
        # LSIGF
        x = tf.matmul(z,self.w) #BxKxNxF_l
        if self.use_bias: 
            x = x + self.b #BxKxNxF_l
        x = tf.reduce_sum( x, axis=1 ) #BxNxF_l

        # Non-linearity
        return( self.non_lin(x) ) 

class GCNN(tf.keras.Model):
  """Graph Convolutional Neural network."""
  def __init__(self, num_layers=4, num_filters=3, input_dim=1, intermediate_dim=4, output_dim=1, non_lin=tf.nn.leaky_relu, name='gcnn', **kwargs):
    super(GCNN, self).__init__(name=name, dtype=tf.float64, **kwargs)
    self.num_layers = num_layers
    self.input_dim = input_dim
    self.gc = []

    # Initialize GC layers
    self.gc.append( GCL(num_filters=num_filters, input_dim=input_dim, output_dim=intermediate_dim, use_bias=True, non_lin=tf.nn.leaky_relu, name=name+"gc1", **kwargs) )
    for l in range(num_layers-2):
        self.gc.append( GCL(num_filters=num_filters, input_dim=intermediate_dim, output_dim=intermediate_dim, use_bias=True, non_lin=tf.nn.leaky_relu, name=name+"gc"+str(l+1), **kwargs) )
    self.gc.append( GCL(num_filters=num_filters, input_dim=intermediate_dim, output_dim=output_dim, use_bias=True, non_lin=non_lin, name=name+"gc"+str(num_layers), **kwargs) )

  def call(self, S, x):
    if self.input_dim == 1:
        x = tf.expand_dims( x, axis=-1 )
    # pdb.set_trace()
    for l in range( self.num_layers ):
        x = self.gc[l](S,x)
    
    return (x) 
