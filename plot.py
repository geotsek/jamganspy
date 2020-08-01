# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 2019

@author: georgios
"""

import os
import sys

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib import rc
from matplotlib import rcParams


#import matplotlib.style
#matplotlib.style.use('classic')

topDir = '/Users/georgio/MyStuff/MyData/jamGANsOut'
    
#print(baseDir)

outFig1pdf = topDir + os.sep + 'jamGANsOut' + '_.pdf'
outFig1png = topDir + os.sep + 'jamGANsOut' + '_.png'

lbls = np.array(['step','d_error(d_loss)','g_error(g_loss)','d_pred_real','d_pred_fake'])

#cmap = cm.get_cmap("rainbow")
#cmap = cm.get_cmap("prism")
clr=cm.rainbow(np.linspace(0,1,len(lbls)))
clr1=cm.rainbow(np.linspace(0,1,len(lbls)))

markerLineStyle=('x-','s-','o-','^-','+-','v-')
markerStyle=('x','s','o','^','+','v')

font = {'size' :15}
matplotlib.rc('font', **font)
plt.figure()
rcParams.update({'figure.autolayout': True})

fig, axs = plt.subplots( 2, 2, sharex=True, gridspec_kw={'hspace': 0.5,'wspace': 0.5}, figsize=(6,5))

fileFs = topDir + os.sep + 'jamGANsOut_.dat'

S=[]
S = np.loadtxt(fileFs,dtype=np.float64)

axs[0,0].plot(S[:,0],S[:,1], markerLineStyle[0], color=clr[0], markeredgewidth=0.0, markersize=2, fillstyle='none', label=('s='+lbls[1]))
axs[0,1].plot(S[:,0],S[:,2], markerLineStyle[0], color=clr[1], markeredgewidth=0.0, markersize=2, fillstyle='none', label=('s='+lbls[2]))
axs[1,0].plot(S[:,0],S[:,2], markerLineStyle[0], color=clr[2], markeredgewidth=0.0, markersize=2, fillstyle='none', label=('s='+lbls[3]))
axs[1,1].plot(S[:,0],S[:,3], markerLineStyle[0], color=clr[3], markeredgewidth=0.0, markersize=2, fillstyle='none', label=('s='+lbls[4]))



axs[0,0].set_xlabel('step',size=12)
axs[0,0].set_ylabel(lbls[1],size=12)
axs[0,1].set_xlabel('step',size=12)
axs[0,1].set_ylabel(lbls[2],size=12)
axs[1,0].set_xlabel('step',size=12)
axs[1,0].set_ylabel(lbls[3],size=12)
axs[1,1].set_xlabel('step',size=12)
axs[1,1].set_ylabel(lbls[4],size=12)


#fig.text(0.5, 0.015, r'$h$', ha='center', rotation='horizontal')
#fig.text(0.0, 0.52, r'$\mathrm{D}(h)$', va='center', rotation='vertical')
#fig.text(0.82, 0.89, 'HEX', va='center', rotation='horizontal',fontsize=16)
#fig.text(0.82, 0.48, 'FCC', va='center', rotation='horizontal',fontsize=16)

#plt.xlim([1e-9,1e0])
#plt.xticks([1e-8,1e-6,1e-4,1e-2,1e0])
#plt.ylim([2e-5,6e+4])
#plt.yticks([1e-4,1e-2,1e+0,1e+2,1e+4])

#plt.legend(loc = 'lower left',prop={'size': 11})
#axs[0].legend(loc = 'lower left',prop={'size': 8})
#axs[1].legend(loc = 'lower left',prop={'size': 8})
    
#title = crystalFlag + ',Gaps3,N=' + str(nParticles) + ',Ns=' + str(numFiles)
#fig.suptitle(title,fontsize=20)
fig.savefig(outFig1pdf)
fig.savefig(outFig1png,dpi=400)

print(" - - - - - The End! - - - - - ")





