import tensorflow as tf
from tensorflow.keras import layers

from gcnn import *

def Actor( Vmax, Vmin, nNodes, nlayers, nfilters, name ):
    last_init = tf.random_normal_initializer(stddev=0.00005)

    Ht = tf.keras.layers.Input(shape=[nNodes,nNodes], dtype=tf.float64)
    bt = tf.keras.layers.Input(shape=[nNodes], dtype=tf.float64)
    flag = tf.keras.layers.Input(shape=[], dtype=tf.bool)

    action = GCNN(num_layers=nlayers, num_filters=nfilters, input_dim=1, intermediate_dim=32, output_dim=8, non_lin=tf.nn.leaky_relu, name=name+'actor_gcnn')(Ht,bt)
    # action = tf.keras.layers.BatchNormalization(dtype=tf.float64)(action)
    action = tf.keras.layers.Dense(1, dtype=tf.float64, kernel_initializer=last_init, name='actor_linear')(action) #

    # action = Vmin + (Vmax - Vmin) * tf.nn.sigmoid(action)
    # action = tf.math.multiply( tf.expand_dims(uvt,axis=-1), tf.nn.sigmoid(action) )
    action = tf.reshape(tf.nn.sigmoid(action),[-1,nNodes])

    return( tf.keras.Model([Ht, bt, flag], action) )

def Critic( nlayers, nfilters, nNodes, name ):
    last_init = tf.random_normal_initializer(stddev=0.00005)

    Ht = tf.keras.layers.Input(shape=[nNodes,nNodes], dtype=tf.float64)
    bt = tf.keras.layers.Input(shape=[nNodes], dtype=tf.float64)
    vt = tf.keras.layers.Input(shape=[nNodes], dtype=tf.float64)
    
    state_out = GCNN(num_layers=nlayers, num_filters=nfilters, input_dim=1, intermediate_dim=32, output_dim=8, name=name+'state_gcnn')(Ht, bt)
    # state_out = tf.keras.layers.Dense(1, activation=tf.nn.leaky_relu, dtype=tf.float64, name='state_linear')(state_out)
    
    action_out = GCNN(num_layers=nlayers, num_filters=nfilters, intermediate_dim=32, output_dim=2, name=name+'action_gcnn')(Ht, vt)
    # action_out = tf.keras.layers.Dense(1, activation=tf.nn.leaky_relu, dtype=tf.float64, name='action_linear')(action_out)
    
    value = tf.keras.layers.Concatenate()([state_out, action_out])
    
    # value = tf.keras.layers.BatchNormalization(dtype=tf.float64)(value)
    value = tf.keras.layers.Dense(32, activation=tf.nn.leaky_relu, dtype=tf.float64, name='fc1')(value)
    # value = tf.keras.layers.BatchNormalization(dtype=tf.float64)(value)
    value = tf.keras.layers.Dense(1, kernel_initializer=last_init, dtype=tf.float64, name='fc2')(value) 

    value = tf.reduce_sum( tf.squeeze(value), axis=-1 )

    return( tf.keras.Model([Ht, bt, vt], value) )

# def Actor( Vmax, Vmin, nNodes, nlayers, nfilters, name ):
#     last_init = tf.random_normal_initializer(stddev=0.00005)

#     Ht = tf.keras.layers.Input(shape=[nNodes,nNodes], dtype=tf.float64)
#     bt = tf.keras.layers.Input(shape=[nNodes], dtype=tf.float64)
#     flag = tf.keras.layers.Input(shape=[], dtype=tf.bool)
    
#     state1 = tf.keras.layers.Flatten()(Ht)
#     state1 = tf.keras.layers.Dense(512, dtype=tf.float64, name='state_linear11')(state1)
#     # state1 = tf.keras.layers.BatchNormalization(dtype=tf.float64)(state1,training=flag)
#     state1 = tf.keras.layers.Dense(128, dtype=tf.float64, name='state_linear12')(state1)
#     # state1 = tf.keras.layers.BatchNormalization(dtype=tf.float64)(state1,training=flag)
#     state1 = tf.keras.layers.Dense(32, dtype=tf.float64, name='state_linear13')(state1)
#     # state1 = tf.keras.layers.BatchNormalization(dtype=tf.float64)(state1,training=flag)
#     state1 = tf.keras.layers.Dense(8, dtype=tf.float64, name='state_linear14')(state1) 

#     state2 = tf.keras.layers.Dense(64, dtype=tf.float64, name='state_linear21')(bt)
#     # state2 = tf.keras.layers.BatchNormalization(dtype=tf.float64)(state2,training=flag)
#     state2 = tf.keras.layers.Dense(32, dtype=tf.float64, name='state_linear22')(state2)
#     # state2 = tf.keras.layers.BatchNormalization(dtype=tf.float64)(state2,training=flag)
#     state2 = tf.keras.layers.Dense(2, dtype=tf.float64, name='state_linear23')(state2)

#     action = tf.keras.layers.Concatenate()([state1, state2])

#     action = tf.keras.layers.Dense(32, dtype=tf.float64, name='actor_linear4')(action)  
#     # action = tf.keras.layers.BatchNormalization(dtype=tf.float64)(action,training=flag)
#     action = tf.keras.layers.Dense(10, dtype=tf.float64, kernel_initializer=last_init, name='actor_linear5')(action)

#     return( tf.keras.Model([Ht,bt,flag], tf.nn.sigmoid(action)) ) #, uvt  tf.nn.tanh(action)

# def Critic( nlayers, nfilters, nNodes, name ):
#     last_init = tf.random_normal_initializer(stddev=0.00005)

#     Ht = tf.keras.layers.Input(shape=[nNodes,nNodes], dtype=tf.float64)
#     bt = tf.keras.layers.Input(shape=[nNodes], dtype=tf.float64)
#     vt = tf.keras.layers.Input(shape=[nNodes], dtype=tf.float64)
    
#     state1 = tf.keras.layers.Flatten()(Ht)
#     state1 = tf.keras.layers.Dense(512, dtype=tf.float64, name='state_linear11')(state1)
#     # state1 = tf.keras.layers.BatchNormalization(dtype=tf.float64)(state1,training=False)
#     state1 = tf.keras.layers.Dense(128, dtype=tf.float64, name='state_linear12')(state1)
#     # state1 = tf.keras.layers.BatchNormalization(dtype=tf.float64)(state1,training=False)
#     state1 = tf.keras.layers.Dense(32, dtype=tf.float64, name='state_linear13')(state1)
#     # state1 = tf.keras.layers.BatchNormalization(dtype=tf.float64)(state1,training=False)
#     state1 = tf.keras.layers.Dense(8, dtype=tf.float64, name='state_linear14')(state1)
    
#     state2 = tf.keras.layers.Dense(64, dtype=tf.float64, name='state_linear21')(bt)
#     # state2 = tf.keras.layers.BatchNormalization(dtype=tf.float64)(state2,training=False)
#     state2 = tf.keras.layers.Dense(32, dtype=tf.float64, name='state_linear22')(state2)
#     # state2 = tf.keras.layers.BatchNormalization(dtype=tf.float64)(state2,training=False)
#     state2 = tf.keras.layers.Dense(2, dtype=tf.float64, name='state_linear23')(state2)

#     action = tf.keras.layers.Dense(64, dtype=tf.float64, name='action_linear11')(vt)
#     # action = tf.keras.layers.BatchNormalization(dtype=tf.float64)(action,training=False)
#     action = tf.keras.layers.Dense(32, dtype=tf.float64, name='action_linear12')(action)
#     # action = tf.keras.layers.BatchNormalization(dtype=tf.float64)(action,training=False)
#     action = tf.keras.layers.Dense(2, dtype=tf.float64, name='action_linear13')(action)
    
#     value = tf.keras.layers.Concatenate()([state1, state2, action])
    
#     value = tf.keras.layers.Dense(32, activation=tf.nn.leaky_relu, dtype=tf.float64, name='fc1')(value)
#     # value = tf.keras.layers.BatchNormalization(dtype=tf.float64)(value,training=False)
#     value = tf.keras.layers.Dense(16, activation=tf.nn.leaky_relu, dtype=tf.float64, name='fc2')(value)
#     # value = tf.keras.layers.BatchNormalization(dtype=tf.float64)(value,training=False)
#     value = tf.keras.layers.Dense(1, dtype=tf.float64, kernel_initializer=last_init, name='fc3')(value) #

#     value = tf.squeeze(value)#tf.reduce_sum( tf.squeeze(value), axis=-1 )

#     return( tf.keras.Model([Ht, bt, vt], value) )