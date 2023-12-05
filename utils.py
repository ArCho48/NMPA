"""
The extra classes or function that will be used in the main ones
"""

import pdb
import datetime
import numpy as np
import tensorflow as tf
from common_definitions import MAX_RECHARGE

class Utilities:
    """
    Utilites to compute rewards, violations, state evolution
    """
    def __init__(self, Bmax, size, var, eps):
        self.Bmax = Bmax
        self.size = size
        self.var = var
        self.eps = eps
    
    # Reward computation
    def compute_reward(self,H,v):
        # H^2
        Hsq = tf.math.square(H)
        
        # (H_ii)^2 * p_i
        num = tf.math.multiply( tf.compat.v1.matrix_diag_part(Hsq), tf.math.square(v) )
        # pdb.set_trace()
        # sigma^2 + sum_j j ~= i ( (H_ji)^2 * p_j^2 )
        den = tf.reshape( tf.matmul( tf.transpose( Hsq, perm=[0,2,1] ), tf.reshape(tf.math.square(v), [-1, self.size, 1] ) ), [-1, self.size] ) - num + self.var + self.eps #small eps for stability
        
        # Rate
        rate = tf.math.log( 1. + tf.math.divide( num, den ) ) / tf.cast( tf.math.log( 2.0 ), tf.float64 )
        
        # Sum-rate
        return( rate, tf.reduce_sum( rate, axis=1, keepdims=True ) )

    # Battery evolution
    def evolve_battery(self,b,v,Bmax=None):
        if not Bmax:
            Bmax = self.Bmax

        # Random charging
        w = tf.random.uniform([1,self.size], maxval=MAX_RECHARGE, dtype=tf.dtypes.float64)

        # Evolve
        b = b - tf.math.square(v)#+ w
        
        # Maximum battery cannot be exceeded
        return( tf.math.minimum(tf.nn.relu(b), Bmax) )  

    # Constraint violation
    def compute_violations(self,b,v):
        # Detect (b-p)<0
        vio = tf.math.sign(b-tf.math.square(v))
        
        # Assign -1 for each violation and +1e-2 for each non-violation
        vio = -0.5 * tf.math.multiply(vio,vio-1) #+ 1e-1 * tf.nn.relu(vio) #1e-2
        
        return( vio )

class OUActionNoise:
    """
    Noise as defined in the DDPG algorithm
    """

    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
                self.x_prev
                + self.theta * (self.mean - self.x_prev) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)
            

class Tensorboard:
    """
    Custom tensorboard for the training loop
    """

    def __init__(self, log_dir):
        """
        Args:
            log_dir: directory of the logging
        """
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = log_dir + current_time + '/train'
        test_log_dir = log_dir + current_time + '/test'
        self.test_counter = 0
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    def write_train(self, epoch, reward, vio, lam):#(self, epoch, reward, val, Q_loss, A_loss, vio):
        """
        Storing all relevant variables
        """

        with self.train_summary_writer.as_default():
            with tf.name_scope("Training"):
                tf.summary.scalar('Sum-rate', reward, step=epoch)
                #tf.summary.scalar('Values', val, step=epoch)
                # tf.summary.scalar('Critic loss', Q_loss.result(), step=epoch)
                # tf.summary.scalar('Actor loss', A_loss.result(), step=epoch)
                tf.summary.scalar('Violations', vio, step=epoch)
                tf.summary.scalar('Lambdas', lam, step=epoch)

    def write_test(self, csr, usr, wsr_uncon, wsr_con, wsr_uni, vio):#(self, epoch, reward, val, Q_loss, A_loss, vio):
        """
        Storing all relevant variables
        """
        epoch = self.test_counter
        with self.test_summary_writer.as_default():
            with tf.name_scope("Test"):
                tf.summary.scalar('WMMSE_uncon', wsr_uncon, step=epoch)
                tf.summary.scalar('WMMSE_con', wsr_con, step=epoch)
                tf.summary.scalar('WMMSE_uni', wsr_uni, step=epoch)
                tf.summary.scalar('UWMMSE_uncon', usr, step=epoch)
                tf.summary.scalar('UWMMSE_con', csr, step=epoch)
                tf.summary.scalar('Violations', vio, step=epoch)
        self.test_counter = epoch + 1
