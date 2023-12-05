import pdb
import numpy as np
import tensorflow as tf

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float64).eps.item()

# WMMSE
class WMMSE(object):
    # Initialize
    def __init__( self, Pmax=1., var=7e-10, layers=100, eps=eps ):
        self.Pmax              = tf.cast( Pmax, tf.float64 )
        self.var               = var
        self.layers            = layers
        self.eps               = eps

    # Building network
    def call(self,H):
        # Squared H 
        self.Hsq = tf.math.square(H)
        
        # Diag H
        dH =  tf.linalg.diag_part( H ) 
        self.dH = tf.compat.v1.matrix_diag( dH )
        
        # Retrieve number of nodes for initializing V
        self.nNodes = tf.shape( H )[-1]
        self.batch_size = tf.shape( H )[0]

        # Maximum V = sqrt(Pmax)
        Vmax = tf.math.sqrt(self.Pmax)
        
        # Initial V
        V = Vmax * tf.ones([self.batch_size, self.nNodes], dtype=tf.float64)

        # Iterate over layers l
        for l in range(self.layers):
            # Compute U^l
            U = self.U_block( V, H )
            
            # Compute W^l
            W = self.W_block( U, V, H )
            
            # Compute V^l
            V = self.V_block( U, W, H, 0. )                
            
            # Saturation non-linearity  ->  0 <= V <= Vmax
            V = tf.math.minimum(V, Vmax) + tf.math.maximum(V, 0) - V
        
        # Rate
        return( V )

    def U_block(self, V, H):
        # H_ii * v_i
        num = tf.math.multiply( tf.compat.v1.matrix_diag_part(H), V )
        
        # sigma^2 + sum_j( (H_ji)^2 * (v_j)^2 )
        den = tf.reshape( tf.matmul( tf.transpose( self.Hsq, perm=[0,2,1] ), tf.reshape( tf.math.square( V ), [-1, self.nNodes, 1] ) ), [-1, self.nNodes] ) + self.var 
        
        # U = num/den
        return( tf.math.divide( num, den+self.eps  ) )

    def W_block(self, U, V, H):
        # 1 - u_i * H_ii * v_i
        den = 1. - tf.math.multiply( tf.compat.v1.matrix_diag_part(H), tf.math.multiply( U, V ) ) 
        
        # W = 1/den
        return( tf.math.reciprocal( den+self.eps ) )
    
    def V_block(self, U, W, H, mu):
        # H_ii * u_i * w_i
        num = tf.math.multiply( tf.compat.v1.matrix_diag_part(H), tf.math.multiply( U, W ) )
        
        # mu + sum_j( (H_ij)^2 * (u_j)^2 *w_j )
        den = tf.math.add( tf.reshape( tf.matmul( self.Hsq, tf.reshape( tf.math.multiply( tf.math.square( U ), W ), [-1, self.nNodes, 1] ) ), [-1, self.nNodes] ), mu) 
        
        # V = num/den
        return( tf.math.divide( num, den+self.eps ) )    
    
    def sum_rate(self, V):
        # (H_ii)^2 * (v_i)^2
        num = tf.math.multiply( tf.compat.v1.matrix_diag_part(self.Hsq), tf.math.square(V) )
        
        # sigma^2 + sum_j j ~= i ( (H_ji)^2 * (v_j)^2 ) 
        den = tf.reshape( tf.matmul( tf.transpose( self.Hsq, perm=[0,2,1] ), tf.reshape( tf.math.square(V), [-1, self.nNodes, 1] ) ), [-1, self.nNodes] ) - num + self.var 
        
        # rate
        rate = tf.math.log( 1. + tf.math.divide( num, den+self.eps) ) / tf.cast( tf.math.log( 2.0 ), tf.float64 )

        return( tf.reduce_sum( rate, axis=1 ) )
