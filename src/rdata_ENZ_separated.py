#read data and output E,N,Z.npy time serirs for the desired sampling rate
import glob
import obspy
import numpy as np
import os


#------setting in the .fq.py------#
home='/projects/tlalollin/jiunting/Fakequakes/'  #Path where all the fakequakes located
project_name='Chile_full_new'
run_name='subduction'
#---------------------------------#

def readMw(logf):
    IN1=open(logf,'r')
    for line in IN1.readlines():
        if 'Actual magnitude' in line:
            Mw=float(line.split()[-1])
            break
    IN1.close()
    return Mw


sep_window=np.arange(5,515,5) #window for time series

ruptures=glob.glob(home+project_name+'/'+'output/waveforms/'+run_name+'*')
ruptures.sort()

A_Mw=[] #final Mw for all fakequakes

sav_path='Chile_full_new_ENZ' #Where you save all the output data

'''
sav_path='Chile_27200_ENZ'


if not(os.path.exists(sav_path)):
    os.makedirs(sav_path)
'''

#sav_E=[] #this is the E array for all ruptures
#sav_N=[] #this is the N array for all ruptures
#sav_Z=[] #this is the Z array for all ruptures
for nrupt,rupt in enumerate(ruptures):
    eqid=rupt.split('/')[-1].split('.')[-1]
    logf=home+project_name+'/'+'output/ruptures/'+run_name+'.'+eqid+'.log'
    Mw=readMw(logf)
    #print(rupt,eqid,logf,Mw)
    print(rupt,eqid)
    A_Mw.append(Mw) #save Mw in a list
    ############start load all of the sac files##################
    #use the first all_stname as the reference, others should follow this order
    '''
    if nrupt==0:
        all_stname=[]
        all_sacs=glob.glob(rupt+'/*.sac')
        for tmpsac in all_sacs:
            all_stname.append(tmpsac.split('/')[-1].split('.')[0])
        all_stname=list(set(all_stname))
        all_stname.sort()
        #np.savetxt('ALL_staname_order.txt',all_stname,fmt='%s') #next time you can directly load this file, so that the "order of features" is the same
    '''
    
    all_stname=np.genfromtxt('ALL_staname_order.txt','S') #careful! python3 might have issue. use .decode() of all stations
    sav_E_sta=[] #this is the E array for all stations (i.e. 121)
    sav_N_sta=[] #this is the N array for all stations
    sav_Z_sta=[] #this is the Z array for all stations
    for stname in all_stname:
        if os.path.exists(rupt+'/'+stname+'.LYE.sac') and os.path.exists(rupt+'/'+stname+'.LYN.sac') and os.path.exists(rupt+'/'+stname+'.LYZ.sac'):
            O_E=obspy.read(rupt+'/'+stname+'*LYE.sac') #find E component sac file
            O_N=obspy.read(rupt+'/'+stname+'*LYN.sac') #find N component sac file
            O_Z=obspy.read(rupt+'/'+stname+'*LYZ.sac') #find Z component sac file
            E=O_E[0].data
            N=O_N[0].data
            Z=O_Z[0].data
            time=O_E[0].times()
            O_E.clear();O_N.clear();O_Z.clear()
            E_interp=np.interp(sep_window,time,E)
            N_interp=np.interp(sep_window,time,N)
            Z_interp=np.interp(sep_window,time,Z)
        else:
            #no data because the station is too far (still output zeros for training)
            E_interp=np.zeros(len(sep_window))
            N_interp=np.zeros(len(sep_window))
            Z_interp=np.zeros(len(sep_window))

        sav_E_sta.append(E_interp) #n points(102) for each station by m stations(121)
        sav_N_sta.append(N_interp)
        sav_Z_sta.append(Z_interp)
    sav_E_sta=np.array(sav_E_sta)
    sav_N_sta=np.array(sav_N_sta)
    sav_Z_sta=np.array(sav_Z_sta)
    #save data individually
    np.save(sav_path+'/'+project_name+'.'+eqid+'.E.npy',sav_E_sta)
    np.save(sav_path+'/'+project_name+'.'+eqid+'.N.npy',sav_N_sta)
    np.save(sav_path+'/'+project_name+'.'+eqid+'.Z.npy',sav_Z_sta)

    #sav_E.append(sav_E_sta.transpose())
    #sav_N.append(sav_N_sta.transpose())
    #sav_Z.append(sav_Z_sta.transpose())

    #new_sav_PGD_sta=np.array(sav_PGD_sta)
    #new_sav_PGD_sta=new_sav_PGD_sta.transpose()
    #A.append(new_sav_PGD_sta)

#sav_E=np.array(sav_E)
#sav_N=np.array(sav_N)
#sav_Z=np.array(sav_Z)
A_Mw=np.array(A_Mw)
#np.save('E_Chile_3400.npy',sav_E)
#np.save('N_Chile_3400.npy',sav_N)
#np.save('Z_Chile_3400.npy',sav_Z)
#np.save('Mw_Chile_3400.npy',A_Mw)
#np.save('Chile_27200_Mw.npy',A_Mw)

#np.save('Chile_small_Mw.npy',A_Mw) #M7.2~7.8

#np.save('Chile_small_2_Mw.npy',A_Mw) #more 7.2~7.8 events
np.save('Chile_full_new_Mw.npy',A_Mw) #re-generated fakequakes


