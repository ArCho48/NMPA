"""
Common definitions of variables that can be used across files
"""
import tensorflow as tf

# agent parameters
ACT_HIGH = tf.math.sqrt(1.0) 
ACT_LOW = tf.math.sqrt(0.0) 
NODES = 10 #change here
GAMMA = 0.99  # for the temporal difference
RHO = 0.001  # to update the target networks
GCLr = 2
GCF = 3
NOISE_VAR = 10**(-25/10) #change here
MAX_RECHARGE = 0.5
FW_LR = 0.05

# buffer params
UNBALANCE_P = 0.8  # newer entries are prioritized
BUFFER_UNBALANCE_GAP = 0.5

# training parameters
STD_DEV = 0.2
BATCH_SIZE = 32
BUFFER_SIZE = 1e5
TOTAL_EPISODES = 10000
CRITIC_LR = 1e-3
ACTOR_LR = 5e-4
LAMDA_LR = 5e-5
WARM_UP = 1000  # num of warm up epochs
MAX_GRAD_NORM = 5.0 # gradient clipping
UPDATE_ACTOR_INTERVAL = 2
UPDATE_LAMDA_INTERVAL = 4