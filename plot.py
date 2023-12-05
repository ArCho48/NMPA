import os
import pdb
import pickle
import matplotlib.pyplot as plt 
import numpy as np
from  matplotlib.colors import LinearSegmentedColormap

######################################### Figure 2a ##########################################

uw = pickle.load(open('results/gb/uw.pkl','rb'))
wc = pickle.load(open('results/gb/wc.pkl','rb'))
if not os.path.exists('results/gb/Figures/Fig_2a/'):
    os.mkdir('results/gb/Figures/Fig_2a/')
for i in range(10):
    fig,ax = plt.subplots()
    ax.plot(range(len(wc[i])),np.cumsum(wc[i]),label='MPA')
    ax.plot(range(len(uw[i])),np.cumsum(uw[i]),label='NMPA')
    ax.grid()
    plt.legend(fontsize=14)
    ax.set_ylabel('Cumulative sum-rate',fontsize=16)
    ax.set_xlabel('Time',fontsize=16)
    plt.xticks([0,20,40,60,80,100],fontsize=14)
    plt.yticks(fontsize=14) 
    plt.savefig('results/gb/Figures/Fig_2a/Fig_2a_episode_'+str(i+1)+'.png')

######################################### Figure 2b ##########################################

c = ["darkred","red","lightcoral","white", "palegreen","green","darkgreen"]
v = [0,.15,.4,.5,0.6,.9,1.]
l = list(zip(v,c))
cmap=LinearSegmentedColormap.from_list('rg',l, N=256)

s = pickle.load(open('results/gb/scale.pkl','rb'))
b = pickle.load(open('results/gb/battery_available.pkl','rb'))
r = pickle.load(open('results/gb/rate_achievable.pkl','rb'))
s = np.reshape(s,(-1,))
b = np.reshape(b,(-1,))
r = np.reshape(r,(-1,))
fig, ax = plt.subplots()
X = [0,2,4,6,8,10,12,14,16,18,20.1]
Y = [0,2,4,6,8,11]
Z = np.zeros((10,5))
for i in range(10):
    tmpx = np.intersect1d(np.where(b>=X[i])[0],np.where(b<X[i+1])[0])
    for j in range(5):
        tmpy = np.intersect1d(np.where(r>=Y[j])[0],np.where(r<Y[j+1])[0])
        tmp = np.intersect1d(tmpx,tmpy)
        Z[i][j] = np.mean(s[tmp])
        
fig, ax = plt.subplots()
plt.pcolormesh(X, Y, np.transpose(Z), cmap=cmap)
cbar = plt.colorbar()
cbar.set_label('Scale',size=16)
cbar.ax.tick_params(labelsize=14)
ax.set_xlabel('Available Battery',fontsize=16)
ax.set_ylabel('Achievable Rate',fontsize=16)
X[-1] = 20
plt.xticks(X,fontsize=14)
plt.yticks(Y,fontsize=14)
plt.savefig('results/gb/Figures/Fig_2b.png')

######################################### Figure 2c ##########################################

uw = pickle.load(open('results/gb/crl.pkl','rb'))
wc = pickle.load(open('results/gb/wmmse_con.pkl','rb'))
x = pickle.load(open('results/gb/step_list.pkl','rb'))
fig, ax = plt.subplots()
ax.errorbar([x_-0.75 for x_ in x],wc[0],yerr=wc[1],marker ='.',color='b', ms='20',label='myopic (MPA)',linestyle='None')
ax.errorbar([x_+0.75 for x_ in x],uw[0],yerr=uw[1],marker ='D',color='g', ms='10', label='non-myopic (NMPA)',linestyle='None')
ax.set_ylabel('Average episodic sum-rate', fontsize=16)
ax.set_xlabel('Episode length',fontsize=16)
ax.yaxis.grid()
plt.xticks(x,fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=14,loc='lower right')
plt.savefig('results/gb/Figures/Fig_2c.png')

######################################### End ##########################################