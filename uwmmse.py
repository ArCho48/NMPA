import pdb
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple

from gcnn_uwmmse import *

def U_block(H, V, var, nNodes, eps):
    # H_ii^2
    Hsq = tf.math.square(H)
    
    # H_ii * v_i
    num = tf.math.multiply( tf.compat.v1.matrix_diag_part(H), V )

    # sigma^2 + sum_j( (H_ji)^2 * (v_j)^2 )
    den = tf.reshape( tf.matmul( tf.transpose( Hsq, perm=[0,2,1] ), tf.reshape( tf.math.square( V ), [-1, nNodes, 1] ) ), [-1, nNodes] ) + var
    
    # U = num/den
    return( tf.math.divide( num, den+eps ) )

def W_block(H, U, V, eps):
    # 1 - u_i * H_ii * v_i
    den = 1. - tf.math.multiply( tf.compat.v1.matrix_diag_part(H), tf.math.multiply( U, V ) )
    
    # W = 1/den
    return( tf.math.reciprocal( den+eps ) )

def V_block(H, U, W, mu, nNodes, eps):
    # H_ii^2
    Hsq = tf.math.square(H)
    
    # H_ii * u_i * w_i
    num = tf.math.multiply( tf.compat.v1.matrix_diag_part(H), tf.math.multiply( U, W ) )
    
    # mu + sum_j( (H_ij)^2 * (u_j)^2 *w_j )
    den = tf.math.add( tf.reshape( tf.matmul( Hsq, tf.reshape( tf.math.multiply( tf.math.square( U ), W ), [-1, nNodes, 1] ) ), [-1, nNodes] ), tf.nn.relu(mu) )
    
    # V = num/den
    return( tf.math.divide( num, den+eps ) )

class UWL(layers.Layer):
    """Unfolded WMMMSE layer"""
    def __init__(self, Vmax=1.0, Vmin=0.0, nNodes=6, gc_layers=2, gc_filters=3, var=7e-10, name="uwl", eps=1e-9, **kwargs):
        super(UWL, self).__init__(name=name,dtype=tf.float64,**kwargs)
        self.var = var
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.nNodes = nNodes
        self.eps = eps
        
        # Initialize trinable variables
        self.gcnn_a = GCNN(num_layers=gc_layers, num_filters=gc_filters, input_dim=2, output_dim=1, non_lin=tf.nn.sigmoid, name=name+'_gcnn_a')
        self.gcnn_b = GCNN(num_layers=gc_layers, num_filters=gc_filters, input_dim=2, output_dim=1, non_lin=tf.nn.sigmoid, name=name+'_gcnn_b')
        self.mu = tf.compat.v1.get_variable( name=name+'mu', shape=[nNodes], initializer=tf.random_uniform_initializer(minval=0.0, maxval=1.0),dtype=tf.float64 )
        # self.mu = 0.01*tf.ones(shape=[nNodes],dtype=tf.float64)

    def call(self, H, V):        
        # Compute U^l
        U = U_block( H, V, self.var, self.nNodes, self.eps )
        
        # Compute W^l
        W_wmmse = W_block( H, U, V, self.eps )

        # X
        x = tf.concat( tf.concat([tf.expand_dims(U, -1), tf.expand_dims(V, -1)], -1), -1 )

        # Learn a^l
        a = tf.reshape( self.gcnn_a(H,x), [-1, self.nNodes] ) 

        # Learn b^l
        b = tf.reshape( self.gcnn_b(H,x), [-1, self.nNodes] ) 
    
        # Compute Wcap^l = a^l * W^l + b^l
        W = tf.math.add( tf.math.multiply( a, W_wmmse ), b )

        # Compute V^l
        V = V_block( H, U, W, self.mu, self.nNodes, self.eps )
        
        ## Saturation non-linearity  ->  0 <= V <= Vmax
        V = tf.math.minimum(V, self.Vmax) + tf.math.maximum(V, self.Vmin) - V
        
        return( V ) 

class UWMMSE(tf.keras.Model):
  """Unfolded WMMSE network."""
  def __init__(self, Vmax=1.0, Vmin=0.0, nNodes=10, uw_layers=4, gc_layers=2, gc_filters=3, var=7e-10, batch_size=1, name='uwmmse', eps=1e-9, **kwargs):
    super(UWMMSE, self).__init__(name=name, dtype=tf.float64, **kwargs)
    self.Vmax = tf.cast(Vmax, tf.float64)
    self.Vmin = tf.cast(Vmin, tf.float64)
    self.nNodes = nNodes
    self.uw_layers = uw_layers
    self.batch_size = batch_size
    
    # Initialize UW layers
    self.uw = UWL(Vmax=self.Vmax, Vmin=self.Vmin, nNodes=nNodes, gc_layers=gc_layers, gc_filters=gc_filters, var=var, name=name+'_uwlayer', eps=eps, **kwargs) 

  def call(self, S):
    x = self.Vmax * tf.ones([self.batch_size,self.nNodes], dtype=tf.float64)

    for l in range( self.uw_layers ):
        x = self.uw(S,x)

    return (x) 
