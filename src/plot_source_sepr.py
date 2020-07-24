#plot source separation

import matplotlib.pyplot as plt
import numpy as np
import obspy
import seaborn as sns

home='/Users/timlin/Documents/Project/MLARGE'
ALLEQ=np.genfromtxt(home+'/data/'+'Chile_full_new_source_slip.txt')

hypo_lon=ALLEQ[:,2]
hypo_lat=ALLEQ[:,3]
cen_lon=ALLEQ[:,5]
cen_lat=ALLEQ[:,6]

sav_sepr=[]
for i in range(len(hypo_lon)):
    d=obspy.geodetics.locations2degrees(lat1=hypo_lat[i],long1=hypo_lon[i],lat2=cen_lat[i],long2=cen_lon[i])
    sav_sepr.append(d)


sns.set()
plt.plot(ALLEQ[:,1],sav_sepr,'k.')
plt.xlabel('Mw',fontsize=16)
plt.ylabel(r'epi-centroid separation($^\circ$)',fontsize=16)
plt.grid(True)
plt.savefig('Source_separ.png',r=300)
plt.show()
