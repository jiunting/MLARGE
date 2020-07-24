import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
A=np.load('Test81.npy',allow_pickle=True)
B=np.load('Test81_sampl5.npy')

A=A.item()

plt.plot(np.arange(50000)+1,np.log10(A['loss']),color=[0.8,0.8,0.8],linewidth=0.8)
plt.plot(np.arange(50000)+1,np.log10(A['val_loss']),color=[0.3,0.3,0.3],linewidth=0.8)
plt.plot(B[:,0],np.log10(B[:,1]),'r.',markersize=5)
plt.plot(B[-1,0],np.log10(B[-1,1]),'r*',markersize=12)
plt.xlabel('Training epochs',fontsize=16)
plt.ylabel('log(MSE)',fontsize=16)
#plt.yscale('log')
plt.xlim([-20,50020])
plt.ylim([-3.92,-3.0])
plt.legend(['training','validation','checkpoint'],fontsize=14,frameon=True)
plt.savefig('Model_losses.png',r=300)
plt.show()
