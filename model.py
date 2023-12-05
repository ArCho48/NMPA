"""
The main model declaration
"""
import os
import sys
import pdb
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from common_definitions import (ACT_HIGH, ACT_LOW, NODES, GAMMA, RHO, BATCH_SIZE, GCLr, GCF, NOISE_VAR, STD_DEV, BUFFER_SIZE, CRITIC_LR, ACTOR_LR, LAMDA_LR, MAX_GRAD_NORM, WARM_UP, UPDATE_ACTOR_INTERVAL, UPDATE_LAMDA_INTERVAL, UNBALANCE_P)
from buffer import ReplayBuffer
from utils import OUActionNoise

from gcnn import *
from uwmmse import *
from networks import *

def update_target(model_target, model_ref, rho=0):
    """
    Update target's weights with the given model reference

    Args:
        model_target: the target model to be changed
        model_ref: the reference model
        rho: the ratio of the new and old weights
    """
    model_target.set_weights([rho * ref_weight + (1 - rho) * target_weight
                              for (target_weight, ref_weight) in
                              list(zip(model_target.get_weights(), model_ref.get_weights()))])


class Agent:
    """
    The Agent that contains all the models
    """
    def __init__(self, action_high=ACT_HIGH, action_low=ACT_LOW, nNodes=NODES, gamma=GAMMA, rho=RHO, batch_size=BATCH_SIZE, gc_layers=GCLr, gc_filters=GCF, var=NOISE_VAR, std_dev=STD_DEV, buffer_size=BUFFER_SIZE, max_gradient_norm=MAX_GRAD_NORM, warmup=WARM_UP, update_actor_interval=UPDATE_ACTOR_INTERVAL):
        # initialize everything
        self.action_high = tf.cast(action_high, tf.float64)
        self.action_low = tf.cast(action_low, tf.float64)
        self.rep_buf = ReplayBuffer(buffer_size, batch_size)
        self.gamma = tf.cast(tf.constant(gamma), tf.float64)
        self.rho = tf.cast(rho, tf.float64)
        self.batch_size = batch_size
        self.max_gradient_norm = tf.cast(max_gradient_norm, tf.float64)
        self.nNodes = nNodes
        self.noise = OUActionNoise(mean=np.zeros((1,nNodes,1)), std_deviation=float(std_dev) * np.ones(1))
        self.warmup = warmup
        self.update_actor_interval = update_actor_interval
        self.learn_step_cntr = 0
        self.time_step = 0
        
        self.actor_network = Actor(self.action_high, self.action_low, nNodes, gc_layers, gc_filters, 'actor_network')
        self.critic_1_network = Critic( gc_layers, gc_filters, nNodes, 'critic_1_network' )
        self.critic_2_network = Critic( gc_layers, gc_filters, nNodes, 'critic_2_network' )
        self.actor_target = Actor(self.action_high, self.action_low, nNodes, gc_layers, gc_filters, 'actor_target')
        self.critic_1_target = Critic( gc_layers, gc_filters, nNodes, 'critic_1_target' )
        self.critic_2_target = Critic( gc_layers, gc_filters, nNodes, 'critic_2_target' )

        # Making the weights equal initially
        self.actor_target.set_weights(self.actor_network.get_weights())
        self.critic_1_target.set_weights(self.critic_1_network.get_weights())
        self.critic_2_target.set_weights(self.critic_2_network.get_weights())

        # optimizers
        self.critic_1_optimizer = tf.keras.optimizers.Adam(CRITIC_LR)#,amsgrad=True)
        self.critic_2_optimizer = tf.keras.optimizers.Adam(CRITIC_LR)#,amsgrad=True)
        self.actor_optimizer = tf.keras.optimizers.Adam(ACTOR_LR)#,amsgrad=True)

        # temporary variable for side effects
        self.cur_action = None

        # define update weights with tf.function for improved performance
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(None, nNodes, nNodes), dtype=tf.float64),
                tf.TensorSpec(shape=(None, nNodes), dtype=tf.float64),
                tf.TensorSpec(shape=(None, nNodes), dtype=tf.float64),
                tf.TensorSpec(shape=(None, nNodes), dtype=tf.float64),
                tf.TensorSpec(shape=(None,), dtype=tf.float64),
                tf.TensorSpec(shape=(None, nNodes, nNodes), dtype=tf.float64),
                tf.TensorSpec(shape=(None, nNodes), dtype=tf.float64),
            ])
        def update_weights(H, b, v, r, Hn, bn, noise=True): #uv,
            """
            Function to update weights with optimizer
            """
            # pdb.set_trace()
            with tf.GradientTape() as tape:
                # Target Action
                vn = self.actor_target([Hn, bn, False]) + (tf.random.normal(shape=[1,self.nNodes],stddev=0.001, dtype=tf.float64) if noise else 0)
                vn = tf.clip_by_value( vn, self.action_low, self.action_high )
                
                # Target Critic 1 & 2
                tq1 = self.critic_1_target([Hn, bn, vn])
                tq2 = self.critic_2_target([Hn, bn, vn])

                # Target Q
                tq = tf.math.minimum( tq1, tq2 )

                # Critic 1 & 2
                q1 = self.critic_1_network( [H, b, v] )
                q2 = self.critic_2_network( [H, b, v] )

                # Q
                y = tf.squeeze(r) + self.gamma * tq 
                
                # Loss
                critic_loss = tf.math.reduce_mean( tf.math.square(tf.stop_gradient(y) - q1) + tf.math.square(tf.stop_gradient(y) - q2) )
            
            # Gradient Descent
            critic_1_grad, critic_2_grad = tape.gradient(critic_loss, [self.critic_1_network.trainable_variables,self.critic_2_network.trainable_variables])
            critic_1_grad, c1g_norm = tf.clip_by_global_norm(critic_1_grad, self.max_gradient_norm)
            critic_2_grad, c2g_norm = tf.clip_by_global_norm(critic_2_grad, self.max_gradient_norm)
            self.critic_1_optimizer.apply_gradients(zip(critic_1_grad, self.critic_1_network.trainable_variables))
            self.critic_2_optimizer.apply_gradients(zip(critic_2_grad, self.critic_2_network.trainable_variables))

            self.learn_step_cntr += 1
            if self.learn_step_cntr % self.update_actor_interval != 0:
                return

            with tf.GradientTape() as tape:
                # Delta mu
                actor_loss = -tf.math.reduce_mean( self.critic_1_network( [H, b, self.actor_network( [H, b, False] )] ) )

            actor_grad = tape.gradient(actor_loss, self.actor_network.trainable_variables)
            actor_grad, ag_norm = tf.clip_by_global_norm(actor_grad, self.max_gradient_norm)
            self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_network.trainable_variables))
                      
        self.update_weights = update_weights
    
    def compute_violation(self, b, v):
        # Detect (b-p)<0
        vio = tf.math.sign(b-tf.math.square(v))
        
        # Assign -1 for each violation 
        vio = -0.5 * tf.math.multiply(vio,vio-1) + 1e-2 * tf.nn.relu(vio) 
        
        return( vio )

    def act(self, Ht, bt, flag, noise=True):
        """
        Run action by the actor network

        Args:
            state: the current state
            noise: whether noise is to be added to the result action (this improves exploration)

        Returns:
            the resulting action
        """
        self.cur_action = self.actor_network([Ht, bt, flag]) + (tf.random.normal(shape=[1,self.nNodes],stddev=0.001, dtype=tf.float64) if noise else 0)
        self.cur_action = tf.clip_by_value( self.cur_action, self.action_low, self.action_high )

        return self.cur_action

    def remember(self, H, b, r, Hn, bn):
        """
        Store states, reward value to the buffer
        """
        # record it in the buffer based on its reward
        self.rep_buf.append(H, b, self.cur_action, r, Hn, bn)

    def learn(self):
        """
        Run update for all networks (for training)
        """
        if len( self.rep_buf.buffer ) < self.batch_size:
            return 

        batch = self.rep_buf.get_batch(unbalance_p=UNBALANCE_P)
        H, b, v, r, Hn, bn = zip(*batch)

        self.update_weights(tf.concat(H, axis=0), tf.concat(b,axis=0), tf.concat(v,axis=0), tf.stack(r), tf.concat(Hn, axis=0), tf.concat(bn,axis=0))

        update_target(self.actor_target, self.actor_network, self.rho)
        update_target(self.critic_1_target, self.critic_1_network, self.rho)
        update_target(self.critic_2_target, self.critic_2_network, self.rho)

    def save_weights(self, path):
        """
        Save weights to `path`
        """
        parent_dir = os.path.dirname(path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

        # Save the weights
        self.actor_network.save_weights(path + "an.h5")
        self.critic_1_network.save_weights(path + "c1n.h5")
        self.critic_1_target.save_weights(path + "c1t.h5")
        self.critic_2_network.save_weights(path + "c2n.h5")
        self.critic_2_target.save_weights(path + "c2t.h5")
        self.actor_target.save_weights(path + "at.h5")

    def load_weights(self, path):
        """
        Load weights from path
        """
        try:
            self.actor_network.load_weights(path + "an.h5")
            self.critic_1_network.load_weights(path + "c1n.h5")
            self.critic_1_target.load_weights(path + "c1t.h5")
            self.critic_2_network.load_weights(path + "c2n.h5")
            self.critic_2_target.load_weights(path + "c2t.h5")
            self.actor_target.load_weights(path + "at.h5")
        except OSError as err:
            logging.warning("Weights files cannot be found, %s", err)
