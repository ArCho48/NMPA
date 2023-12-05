import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
import tqdm
import pickle
import datetime
import statistics
import collections
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cdist
tf.get_logger().setLevel('ERROR')
from tensorflow.keras import layers, optimizers
import tensorflow_addons as tfa
from matplotlib import pyplot as plt
from typing import Any, List, Sequence, Tuple

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#   except RuntimeError as e:
#     print(e)
from utils import *
from uwmmse import *
from wmmse import *
from datagen import fading

Pmax = 1.0
Vmax = tf.cast(tf.math.sqrt(Pmax),tf.float64) 
Vmin = tf.cast(0.0,tf.float64)
nNodes = 10 
var_db = -25
var = 10**(var_db/10)
nlayers=2
nfilters=3
uw_layers = 4
batch_size = 32
eps = np.finfo(np.float64).eps.item()
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

with tf.device('/device:cpu:0'):
    utilities = Utilities( 15.0, nNodes, var, eps )         
    model = UWMMSE( Vmax=Vmax, Vmin=Vmin, nNodes=nNodes, uw_layers=uw_layers, gc_layers=nlayers, gc_filters=nfilters, var=var, batch_size=batch_size, eps=eps, name='lower_level' )   
    #change uw_layers in uwmmse def # 
    channel_model = fading(num_tx=nNodes, num_rx=nNodes, seed=seed) 
    wmmse = WMMSE(Pmax=Pmax, var=var, eps=eps)
    if sys.argv[1] == 'train':
        opt = tfa.optimizers.NovoGrad(learning_rate=1e-3)

        best_rate = 0.0
        best_epoch = 0
        for i in range(10000):
            Hlist = []
            channel = channel_model.generate_H_gb() 
            for j in range(batch_size):
                Hlist.append(next(channel))
            Hlist = tf.concat(Hlist,axis=0)
            with tf.GradientTape(persistent=True) as tape:
                V = model(Hlist)
                _, sum_rates = utilities.compute_reward(Hlist,V)
                loss = -tf.reduce_mean(sum_rates)

            grads = tape.gradient(loss,[model.trainable_variables])
            clip_grads, mnorm = tf.clip_by_global_norm(grads[0], 5.0)
            opt.apply_gradients(zip(clip_grads, model.trainable_variables))

            if (i+1)%500 == 0:
                print('Epoch: ',i+1)

                wp = wmmse.call(tf.squeeze(Hlist))
                wsr = wmmse.sum_rate(wp)

                print('\nTrain:')
                print('WMMSE:', np.mean(wsr.numpy()))
                print('UWMMSE:', -loss.cpu().numpy())

                channel_te = channel_model.generate_H_gb() 
                Hlist_te = []
                for k in range(batch_size):
                    Hlist_te.append(next(channel_te))
                Hlist_te = tf.concat(Hlist_te,axis=0)
                V_te = model(Hlist_te)
                _, sum_rate_te = utilities.compute_reward(Hlist_te,V_te)
                sum_rate_te = np.mean(sum_rate_te)
                wp_te = wmmse.call(tf.squeeze(Hlist_te))
                wsr_te = wmmse.sum_rate(wp_te)
                
                print('\nTest:')
                print('WMMSE:',np.mean(wsr_te))
                print('UWMMSE:',sum_rate_te)
                
                if sum_rate_te > best_rate:
                    best_rate = sum_rate_te
                    best_epoch = i
                    model.save_weights('models/pretrained/uwmmse/gb/uwmmse-model-'+str(i+1))
                    print('\nBest model saved at epoch ',i+1)
    
    model.load_weights('models/pretrained/uwmmse/gb/uwmmse-model-1000').expect_partial()
    print('\nModel Loaded')
    channel_te = channel_model.generate_H_gb() 
    Hlist_te = []
    for k in range(batch_size):
        Hlist_te.append(next(channel_te))
    Hlist_te = tf.concat(Hlist_te,axis=0)
    V_te = model(Hlist_te)
    _, sum_rate_te = utilities.compute_reward(Hlist_te,V_te)
    sum_rate_te = np.mean(sum_rate_te)
    wp_te = wmmse.call(tf.squeeze(Hlist_te))
    wsr_te = wmmse.sum_rate(wp_te)
    
    print('\nTest:')
    print('WMMSE:',np.mean(wsr_te))
    print('UWMMSE:',sum_rate_te)