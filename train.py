"""Main file"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
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
    prog="CRL-WANETs",
    description="DDPG for contrained power allocation in WANETs"
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
uwmmse = UWMMSE( nNodes=SIZE, gc_layers=GCLr, gc_filters=GCF, var=NOISE_VAR, batch_size=1, eps=eps, name='lower_level' )
uwmmse.load_weights('models/pretrained/uwmmse/gb/uwmmse-model-1000').expect_partial()

# Utilties
utilities = Utilities( BMAX, SIZE, NOISE_VAR, eps )

# load weights if available
# logging.info("Loading weights from %s*, make sure the folder exists", CHECKPOINTS_PATH)
# agent.load_weights(CHECKPOINTS_PATH)

# To store reward history of each episode
ep_srs = []
violations = []
lamdas = []
        
# Debugging
tf.config.run_functions_eagerly(True)

# Pickle dump
def pdump( dump, path ):
    f = open(path,'wb')
    pickle.dump(dump, f)
    f.close()

# Reward
def compute_reward(r,b,s):
    r = r.numpy()
    b = b.numpy()
    s = s.numpy()

    txs = np.where(s>=0.5)[0]
    b_ = b[txs]
    violation = np.sum( 1.0 * (b_ > 0.0) - 1.0 ) 
    reward = np.sum(r[np.intersect1d(txs,np.where(b>0.0)[0])])

    return( reward + violation )

# Evaluation 
def evaluate(tr_ep, eval_episodes=5):
    con_er = []
    timesteps = 100
    
    for ep in range(eval_episodes):
        # Storage
        con_r = tf.TensorArray(dtype=tf.float64, size=0, dynamic_size=True)

        # Initialize channel model and action noise
        channel = channel_model.generate_H_gb()

        # Initial state
        prev_H = next(channel)
        prev_uv = uwmmse(prev_H)
        prev_b = tf.math.rint(BMAX*tf.random.uniform(shape=[1,SIZE],minval=0.5,maxval=1.0,dtype=tf.dtypes.float64,seed=67,))

        # Iterations
        for i in range(timesteps):
            scale = tf.reshape( agent.act(prev_H, prev_b, flag=False, noise=False), [1,-1]) 
            curr_v = tf.math.sqrt(tf.math.minimum( tf.math.square( tf.math.rint(scale) * prev_uv), prev_b ))
            
            # Recieve reward, violation and next state from environment
            uwmmse_r, uwmmse_sr = utilities.compute_reward( prev_H, curr_v )
            reward = tf.cast(compute_reward(tf.squeeze(uwmmse_r), tf.squeeze(prev_b), tf.squeeze(scale)),tf.float64)#
            H = next(channel)
            uv = uwmmse(H)
            curr_v = curr_v+((0.0+tf.math.ceil(curr_v - 0.0009))*tf.cast(0.5,tf.float64))
            b = utilities.evolve_battery(prev_b, curr_v)

            # post update for next step
            prev_H = H
            prev_uv = uv
            prev_b = b
            
            # Store
            con_r = con_r.write(i,reward)

        con_er.append( tf.squeeze( con_r.stack() ).numpy() )

    con_er = np.mean([np.sum(arr) for arr in con_er])

    log = "\nEpisode {}/{}, \nTest performance = {:.3f}"
    print(log.format( tr_ep, TOTAL_EPISODES, con_er))

    return(con_er)

if __name__ == "__main__":
    eval_window = 50 
    timesteps = 100
    best_er = 0.0
    tr_ep_r = []

    # Episodes
    for ep in range(TOTAL_EPISODES):
        # all the metrics
        ep_r = tf.TensorArray(dtype=tf.float64, size=0, dynamic_size=True)
        
        # Initialize channel model and action noise
        channel = channel_model.generate_H_gb()
        agent.noise.reset()

        # Initial state
        prev_H = next(channel)
        prev_uv = uwmmse(prev_H)
        prev_b = tf.math.rint(BMAX*tf.random.uniform(shape=[1,SIZE],minval=0.5,maxval=1.0,dtype=tf.dtypes.float64,seed=42,))

        # Iterations
        for i in range(timesteps):
            # Act
            scale = tf.reshape( agent.act(prev_H, prev_b, flag=False, noise=True), [1,-1]) 
            curr_v = tf.math.sqrt(tf.math.minimum( tf.math.square( tf.math.rint(scale) * prev_uv), prev_b ))
          
            # Recieve reward, violation and next state from environment
            uwmmse_r, uwmmse_sr = utilities.compute_reward( prev_H, curr_v )
            reward = tf.cast(compute_reward(tf.squeeze(uwmmse_r), tf.squeeze(prev_b), tf.squeeze(scale)),tf.float64)

            H = next(channel)
            uv = uwmmse(H)

            curr_v = curr_v+((0.0+tf.math.ceil(curr_v - 0.0009))*tf.cast(0.5,tf.float64))
            b = utilities.evolve_battery(prev_b, curr_v)

            agent.remember(prev_H, prev_b, reward, H, b)

            # update weights
            if LEARN:
                agent.learn()

            # post update for next step
            prev_H = H
            prev_uv = uv
            prev_b = b
            
            # Store
            ep_r = ep_r.write(i, tf.squeeze(reward))

        # Store episodes
        tr_ep_r.append( tf.reduce_sum( ep_r.stack() ) )
      
        if (ep+1) % eval_window == 0: 
            er = evaluate(ep+1)
            print('Training performance:', tf.reduce_mean(tr_ep_r[-eval_window:-1]).numpy())
            # save weights
            er = tf.reduce_mean(tr_ep_r[-eval_window:-1]).numpy()
            if er >= best_er:
                agent.save_weights(CHECKPOINTS_PATH)
                best_er = er
    
    logging.info("Training done...")

    ############################################# END ##########################
