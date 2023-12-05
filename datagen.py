import os
import sys
#sys.path.append('code/')
import pdb
import math
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.special import j0
from scipy.spatial.distance import cdist

from contact import *

class fading(object):
    def __init__( self, num_tx=10, num_rx=10, transmitter_xlims=[0,60], transmitter_ylims=[0,60], transmitter_range=20, seed=None):
        self.K        = num_tx
        self.N        = num_rx
        self.tx_xlim  = transmitter_xlims
        self.tx_ylim  = transmitter_ylims
        self.tx_range = transmitter_range
        self.set_randomseed(seed)        
    
    # Set random seed
    def set_randomseed( self, seed=None ):
        if not seed:
            seed = np.random.randint(2**20)
        np.random.seed(seed)
    
    # Sample tx coordinates
    def sample_tx_coord( self, nNodes, xlim, ylim ):
        x = np.random.uniform(low=xlim[0], high=xlim[1], size=(nNodes,1))
        y = np.random.uniform(low=ylim[0], high=ylim[1], size=(nNodes,1))
        
        return( np.concatenate((x,y),1) )

    # Sample rx coordinates
    def sample_rx_coord( self, nNodes, tx_range, ctx ):
        x = np.random.uniform(low=-int(0.5*tx_range), high=int(0.5*tx_range), size=(nNodes,1))
        y = np.random.uniform(low=-int(0.5*tx_range), high=int(0.5*tx_range), size=(nNodes,1))
            
        return( ctx + np.concatenate((x,y),1) )

    def sample_good(self,nNodes, xlim, ylim):
        tx_x = [xlim[0],int(0.5*(xlim[1]-xlim[0])),xlim[1]] #
        tx_y = [ylim[0],int(0.5*(ylim[1]-ylim[0])),ylim[1]] #
        tx_x, tx_y = np.meshgrid(tx_x,tx_y)

        rx_x = np.concatenate([np.random.uniform(0,10,(3,1)),np.random.uniform(-5,5,(3,1)), np.random.uniform(-10,0,(3,1))],axis=1) 
        rx_x += tx_x
        rx_y = np.concatenate([np.random.uniform(0,10,(1,3)),np.random.uniform(-5,5,(1,3)),np.random.uniform(-10,0,(1,3))],axis=0) 
        rx_y += tx_y

        tx_x = np.reshape(tx_x,(nNodes-1,1))
        tx_y = np.reshape(tx_y,(nNodes-1,1))
        rx_x = np.reshape(rx_x,(nNodes-1,1))
        rx_y = np.reshape(rx_y,(nNodes-1,1))

        tx, rx = np.concatenate((tx_x,tx_y),1), np.concatenate((rx_x,rx_y),1)
        
        # Add 10th
        tx = np.concatenate((tx,[[15,15]]),axis=0)
        rx = np.concatenate((rx,[[15+random.uniform(-5,5),15+random.uniform(-5,5)]]),axis=0)
        
        return( tx,rx )

    def sample_bad(self,nNodes, xlim, ylim):
        ctr = [int(0.5*(xlim[1]-xlim[0])),int(0.5*(ylim[1]-ylim[0]))]
        tx_rad = np.random.uniform(5,10,(nNodes,1))
        rx_rad = np.random.uniform(6,9,(nNodes,1))

        theta = [0,20,40,80,120,160,200,240,280,320] 
        tx_x, tx_y = tx_rad*np.expand_dims(np.cos(theta),axis=-1), tx_rad*np.expand_dims(np.sin(theta),axis=-1)
        tx_x += ctr[0]
        tx_y += ctr[1]
        random.shuffle(theta)
        rx_x, rx_y = rx_rad*np.expand_dims(np.cos(theta),axis=-1), rx_rad*np.expand_dims(np.sin(theta),axis=-1)
        rx_x += ctr[0]
        rx_y += ctr[1]

        return( np.concatenate((tx_x,tx_y),1), np.concatenate((rx_x,rx_y),1) )

    # Compute pairwise distance
    def pdist( self, A, B):
        d = cdist( A, B )
        return(d)

    # Contact
    def get_contacts( self, tx, rx ):
        d = cdist(tx,rx)
        contacts = [tuple(coord) for coord in np.argwhere(d<(self.tx_range*np.sqrt(2))).tolist()]

        return( contacts )
    
    def visualize( self, tx, rx, contacts, step, name ):
        fig, ax = plt.subplots(1,2)
        ax[0].set_xlim(self.tx_xlim[0]-self.tx_range,self.tx_xlim[1]+self.tx_range)
        ax[0].set_ylim(self.tx_ylim[0]-self.tx_range,self.tx_ylim[1]+self.tx_range)
        ax[1].set_xlim(self.tx_xlim[0]-self.tx_range,self.tx_xlim[1]+self.tx_range)
        ax[1].set_ylim(self.tx_ylim[0]-self.tx_range,self.tx_ylim[1]+self.tx_range)

        ax[0].scatter(tx[:,0],tx[:,1],color='r',label='transmitter')
        ax[0].scatter(rx[:,0],rx[:,1],marker='D',color='g',label='receiver')

        ax[1].scatter(tx[:,0],tx[:,1],color='r',label='transmitter')
        ax[1].scatter(rx[:,0],rx[:,1],marker='D',color='g',label='receiver')
        circle1 = plt.Circle((tx[0,0],tx[0,1]), (self.tx_range*np.sqrt(2)), color='r', ls='--',fill=False)
        
        for (i,j) in contacts:
            if i == j:
                ax[0].plot([tx[i,0],rx[j,0]], [tx[i,1],rx[j,1]],'-b')
            #elif i == 0:
            else:
                ax[0].plot([tx[i,0],rx[j,0]], [tx[i,1],rx[j,1]],':m')

        for (i,j) in contacts:
            if i == j:
                ax[1].plot([tx[i,0],rx[j,0]], [tx[i,1],rx[j,1]],'-b')
            elif i == 0:
            #else:
                ax[1].plot([tx[i,0],rx[j,0]], [tx[i,1],rx[j,1]],':m')
        
        ax[1].add_patch(circle1)
        plt.legend()
        plt.savefig('./visuals/'+name+'/fig'+str(step)+'.png')
    
    # Generate good-bad channel
    def generate_H_gb( self):
        if os.path.exists('data/topologies/gb.pkl'):
            gb = pickle.load(open('data/topologies/gb.pkl','rb'))

            coord_tx_g,coord_rx_g = gb['good']
            coord_tx_b,coord_rx_b = gb['bad']
        else:
            coord_tx_g,coord_rx_g = self.sample_good( self.K, self.tx_xlim, self.tx_ylim )
            coord_tx_b,coord_rx_b = self.sample_bad( self.K, self.tx_xlim, self.tx_ylim ) 
            
            pickle.dump({'good':(coord_tx_g,coord_rx_g),'bad':(coord_tx_b,coord_rx_b)}, open('data/topologies/gb.pkl','wb'))

        contacts_g = self.get_contacts(coord_tx_g, coord_rx_g) 
        # self.visualize(  coord_tx_g, coord_rx_g, contacts_g, 1, 'gb' )
        contacts_g = list(zip(*contacts_g))
        mask_g = sparse.coo_matrix((np.ones(len(contacts_g[0])), contacts_g ), shape=(self.K,self.N), dtype=np.float64).todense()
        d_g = self.pdist( coord_tx_g, coord_rx_g )
        pl_g = np.divide(1, 1 + d_g)

        contacts_b = self.get_contacts(coord_tx_b, coord_rx_b) 
        # self.visualize(  coord_tx_b, coord_rx_b, contacts_b, 2, 'gb' )
        contacts_b = list(zip(*contacts_b))
        mask_b = sparse.coo_matrix((np.ones(len(contacts_b[0])), contacts_b ), shape=(self.K,self.N), dtype=np.float64).todense()
        d_b = self.pdist( coord_tx_b, coord_rx_b )
        pl_b = np.divide(1, 1 + d_b)

        while 1:       
            if random.random() > 0.5:
                pl = pl_g
                mask = mask_g
            else:
                pl = pl_b
                mask = mask_b
            
            h_t = np.random.randn(self.K,self.N) + 1j * np.random.randn(self.K,self.N)
            h_modsq_t = np.absolute(h_t) ** 2
            tmp_h = np.sqrt(h_modsq_t * pl) 
    
            yield np.expand_dims( np.multiply( mask, tmp_h ), axis=0 )

    # Generate fixed-topology channel
    def generate_H_ft( self):
        if os.path.exists('data/topologies/rand.pkl'):
            coord_tx,coord_rx = pickle.load(open('data/topologies/rand.pkl','rb'))
        else:
            coord_tx = self.sample_tx_coord( self.K, self.tx_xlim, self.tx_ylim )#xx#
            coord_rx = self.sample_rx_coord( self.N, self.tx_range, coord_tx  )#yy#

            pickle.dump((coord_tx,coord_rx), open('data/topologies/rand.pkl','wb'))

        contacts = self.get_contacts(coord_tx, coord_rx) 
        # self.visualize(  coord_tx, coord_rx, contacts, 1, 'fixed'  )

        contacts = list(zip(*contacts))
        mask = sparse.coo_matrix((np.ones(len(contacts[0])), contacts ), shape=(self.K,self.N), dtype=np.float64).todense()

        d = self.pdist( coord_tx, coord_rx )
        pl = np.divide(1, 1 + d)

        while 1:         
            h_t = np.random.randn(self.K,self.N) + 1j * np.random.randn(self.K,self.N)
            h_modsq_t = np.absolute(h_t) ** 2            
            tmp_h = np.sqrt(h_modsq_t * pl) 
    
            yield np.expand_dims( np.multiply( mask, tmp_h ), axis=0 )    
    
    # Generate varying topology channel
    def generate_H_vt( self):
        while 1:
            coord_tx = self.sample_tx_coord( self.K, self.tx_xlim, self.tx_ylim )#xx#
            coord_rx = self.sample_rx_coord( self.N, self.tx_range, coord_tx  )#yy#        
            
            contacts = self.get_contacts(coord_tx, coord_rx) 
            # self.visualize(  coord_tx, coord_rx, contacts, counter, 'varying'  )
            
            contacts = list(zip(*contacts))
            mask = sparse.coo_matrix((np.ones(len(contacts[0])), contacts ), shape=(self.K,self.N), dtype=np.float64).todense()
            
            h_t = np.random.randn(self.K,self.N) + 1j * np.random.randn(self.K,self.N)
            h_modsq_t = np.absolute(h_t) ** 2
            d = self.pdist( coord_tx, coord_rx )
            pl = np.divide(1, 1 + d)
            
            tmp_h = np.sqrt(h_modsq_t * pl) 
    
            yield np.expand_dims( np.multiply( mask, tmp_h ), axis=0 )#, np.hstack([coord_tx, vel_tx, coord_rx, vel_rx])     

if __name__ == '__main__':
    channel_model = fading(seed=42)
    channel = channel_model.generate_H_vt()
    for i in range(64):
        _ = next(channel) 