"""Main file"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import time
import sys
import pdb
import pickle
import random
import logging
import datetime
import argparse
import statistics
import collections
import numpy as np
from tqdm import trange
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from typing import Any, List, Sequence, Tuple

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#   except RuntimeError as e:
#     print(e)

from wmmse import *
from uwmmse import *
from model import Agent   
from datagen import fading
from utils import Utilities
from common_definitions import TOTAL_EPISODES, NOISE_VAR, GAMMA, GCLr, GCF, UNBALANCE_P

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

# Set seed for experiment reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser(
prog="CRL-MANETs",
description="DDPG for contrained power allocation in MANETs"
)
parser.add_argument('--size', type=int, nargs='?', default=10,
                help='Number of transceivers in the network')
parser.add_argument('--pmax', type=float, nargs='?', default=1.0,
                help='Maximum instantaneous power at each node')
parser.add_argument('--bmax', type=float, nargs='?', default=20.0,
                help='Maximum available battery at each node')
parser.add_argument('--train', type=bool, nargs='?', default=True,
                help='Train the network')
parser.add_argument('--use_noise', type=bool, nargs='?', default=True,
                help='OU Noise will be applied to the policy action')
parser.add_argument('--eps_greedy', type=float, nargs='?', default=0.95,
                help="The epsilon for Epsilon-greedy in the policy's action")
parser.add_argument('--save_weights', type=bool, nargs='?', default=True,
                help='Save the weight of the network in the defined checkpoint file '
                        'directory.')
parser.add_argument('--set', type=str, nargs='?', default='set1',
                help='Exp ID')

args = parser.parse_args()
SIZE = args.size
PMAX = args.pmax
BMAX = args.bmax
LEARN = args.train
USE_NOISE = args.use_noise
SAVE_WEIGHTS = args.save_weights
EPS_GREEDY = args.eps_greedy
SET = args.set

# general parameters
CHECKPOINTS_PATH = 'checkpoints/DDPG/'+SET+'/'

# Channel model
channel_model = fading(num_tx=SIZE, num_rx=SIZE, seed=seed) 

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float64).eps.item()   

# Max instant power
action_space_high = tf.math.sqrt( PMAX )

# DDPG agent
agent = Agent(action_high=action_space_high, nNodes=SIZE)

# WMMSE
wmmse = WMMSE(Pmax=PMAX, var=NOISE_VAR, eps=eps)

# UWMMSE
uwmmse = UWMMSE( nNodes=SIZE, gc_layers=GCLr, gc_filters=GCF, var=NOISE_VAR, batch_size=1, eps=eps, name='base_network' )
uwmmse.load_weights('models/pretrained/uwmmse/gb/uwmmse-model-1000').expect_partial()

# Utilties
utilities = Utilities( BMAX, SIZE, NOISE_VAR, eps )

# load weights if available
agent.load_weights(CHECKPOINTS_PATH)

# To store reward history of each episode
ep_srs = []
violations = []
lamdas = []
    
## Debugging
tf.config.run_functions_eagerly(True)

# Pickle dump
def pdump( dump, path ):
    f = open(path,'wb')
    pickle.dump(dump, f)
    f.close()

# Reward function
def compute_reward(r,b,s):
    r = r.numpy()
    b = b.numpy()
    s = s.numpy()

    txs = np.where(s>=0.5)
    b_ = b[txs]
    violation = np.sum( 1.0 * (b_ > 0.0) - 1.0 )
    reward = np.sum(r[np.intersect1d(txs,np.where(b>0.0))])

    return( reward, violation )

# Run episode
def evaluate(eval_episodes=10):
    step_list = list(np.linspace(30,150,num=13,dtype=np.int32))
    nmpa_mean = []
    nmpa_std = []
    mpa_mean = []
    mpa_std = []

    for steps in step_list:
        uw_store = []
        mpa_store = []
        nmpa_store = []
        scl_store = []
        vio_store = []
        bat_store = []

        for ep in range(eval_episodes):
            # Storage
            uw_epr = tf.TensorArray(dtype=tf.float64, size=0, dynamic_size=True)
            nmpa_epsr = tf.TensorArray(dtype=tf.float64, size=0, dynamic_size=True)
            mpa_epsr = tf.TensorArray(dtype=tf.float64, size=0, dynamic_size=True)
            scl_ep = tf.TensorArray(dtype=tf.float64, size=0, dynamic_size=True)
            vio_ep = tf.TensorArray(dtype=tf.float64, size=0, dynamic_size=True)
            bat_ep = tf.TensorArray(dtype=tf.float64, size=0, dynamic_size=True)

            # Initialize channel model and action noise
            channel = channel_model.generate_H_gb()

            # Initial state
            prev_H = next(channel)
            prev_uv = uwmmse(prev_H)
            prev_b = tf.math.rint(BMAX*tf.random.uniform(shape=[1,SIZE],minval=0.5,maxval=1.0,dtype=tf.dtypes.float64,seed=67,))
            prev_b_mpa = prev_b

            # Iterations
            for i in range(steps):
                # Unconstrained
                uw_rate, _  = utilities.compute_reward( prev_H, prev_uv ) 
                
                # MPA
                vc = tf.math.sqrt( tf.math.minimum( tf.math.square( prev_uv ), prev_b_mpa ) )
                _, uwc_sumrate = utilities.compute_reward( prev_H, vc )
                curr_vc = prev_uv+((0.0+tf.math.ceil(prev_uv - 0.0009))*tf.cast(0.5,tf.float64))
                b_mpa = utilities.evolve_battery(prev_b_mpa, curr_vc)
                prev_b_mpa = b_mpa

                # NMPA
                scale = tf.reshape( agent.act(prev_H, prev_b, flag=False, noise=False), [1,-1]) 
                curr_v = tf.math.sqrt(tf.math.minimum( tf.math.square( tf.math.rint(scale) * prev_uv), prev_b ))
                
                # Recieve reward, violation and next state from environment
                nmpa_r, _ = utilities.compute_reward( prev_H, curr_v )
                curr_v = curr_v+((0.0+tf.math.ceil(curr_v - 0.0009))*tf.cast(0.5,tf.float64))
                reward, violation = tf.cast(compute_reward(tf.squeeze(nmpa_r), tf.squeeze(prev_b), tf.squeeze(scale)),tf.float64)#
                H = next(channel)
                uv = uwmmse(H)
                b = utilities.evolve_battery(prev_b, curr_v)

                # Store
                uw_epr = uw_epr.write(i,uw_rate)
                nmpa_epsr = nmpa_epsr.write(i,reward)
                mpa_epsr = mpa_epsr.write(i,uwc_sumrate)
                scl_ep = scl_ep.write(i,scale)
                vio_ep = vio_ep.write(i,violation)
                bat_ep = bat_ep.write(i,prev_b)

                # post update for next step
                prev_H = H
                prev_uv = uv
                prev_b = b
   
            uw_store.append( tf.squeeze( uw_epr.stack() ).numpy() )
            mpa_store.append( tf.squeeze( mpa_epsr.stack() ).numpy() )
            nmpa_store.append( tf.squeeze( nmpa_epsr.stack() ).numpy() )
            scl_store.append( tf.squeeze( scl_ep.stack() ).numpy() )
            vio_store.append( tf.squeeze( vio_ep.stack() ).numpy() )
            bat_store.append( tf.squeeze( bat_ep.stack() ).numpy())

        if steps == 100:
            pickle.dump(mpa_store,open('results/gb/wc.pkl','wb'))
            pickle.dump(nmpa_store,open('results/gb/uw.pkl','wb'))
            pickle.dump(uw_store,open('results/gb/rate_achievable.pkl','wb'))
            pickle.dump(scl_store,open('results/gb/scale.pkl','wb'))
            pickle.dump(bat_store,open('results/gb/battery_available.pkl','wb'))

            print('Avg improvement in episodic sum-rate: ',np.mean((np.sum(nmpa_store,axis=1)-np.sum(mpa_store,axis=1))/(np.sum(mpa_store,axis=1))))
            print('Avg violations per node: ',-np.mean(vio_store)/SIZE)
            
        nmpa_mean.append( np.mean(np.sum(nmpa_store,axis=1)) )
        nmpa_std.append( np.std(np.sum(nmpa_store,axis=1)) )
        mpa_mean.append( np.mean(np.sum(mpa_store,axis=1)) )
        mpa_std.append( np.std(np.sum(mpa_store,axis=1)) )

    pickle.dump(step_list,open('results/gb/step_list.pkl','wb'))
    pickle.dump((mpa_mean,mpa_std),open('results/gb/wmmse_con.pkl','wb'))
    pickle.dump((nmpa_mean,nmpa_std),open('results/gb/crl.pkl','wb'))
      
if __name__ == '__main__':
    evaluate()
  