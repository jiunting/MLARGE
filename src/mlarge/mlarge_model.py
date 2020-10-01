#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 22:13:15 2020

@author: timlin
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
import os,glob,sys,datetime
from tensorflow.keras import models
from tensorflow.keras import layers
from mudpy.hfsims import windowed_gaussian,apply_spectrum
from mudpy.forward import gnss_psd
import obspy

###########dealing with stations ordering and find every station locations#############
#sta_loc_file='../data/Chile_GNSS.gflist'   #Full GFlist file
#station_order_file='../data/ALL_staname_order.txt' #this is the order in training (PGD_Chile_3400.npy)

#STAINFO=np.genfromtxt(sta_loc_file,'S12',skip_header=1)
#STAINFO={ista[0].decode():[float(ista[1]),float(ista[2])]  for ista in STAINFO}
#STA=np.genfromtxt(station_order_file,'S6')
#STA={ista:STAINFO[sta.decode()] for ista,sta in enumerate(STA)}



class feature_gen(keras.utils.Sequence):
    #######Generator should inherit the "Sequence" class in order to run multi-processing of fit_generator###########
    def __init__(self,Dpath,E_path,N_path,Z_path,y_path,EQinfo,STAinfo,Nstan=121,add_code=True,add_noise=True,noise_p=0.5,  
                 rmN=(10,110),Noise_level=[1,10,20,30,40,50,60,70,80,90],Min_stan_dist=[4,3],scale=(0,1), 
                 BatchSize=128,Mwfilter=8.0,save_ID='sav_pickedID_1_valid.npy',shuffle=True):
        self.Dpath=Dpath #The path of the individual [E/N/U].npy data (should be a list or a numpy array)
        self.E_path=E_path #The path of the individual [E/N/U].npy data (should be a list or a numpy array)
        self.N_path=N_path
        self.Z_path=Z_path
        self.y_path=y_path
        self.EQinfo=EQinfo   #EQinfo file with the same number of lines of X,y list
        self.STAinfo=STAinfo #Stainfo files
        self.Nstan=Nstan
        self.add_code=add_code
        self.add_noise=add_noise
        self.noise_p=noise_p
        self.rmN=rmN
        self.Noise_level=Noise_level  
        self.Min_stan_dist=Min_stan_dist #e.g. [4,3]=minimum 4 stations within 3 degree
        self.scale=scale
        self.BatchSize=BatchSize
        self.Mwfilter=Mwfilter #Threshold magnitude for generated events, or False means everything
        self.save_ID=save_ID #save the original eqid with this name (e.g. sav_pickedID_73_2_valid.npy)
        self.shuffle=shuffle  #shuffle always True (shuffle station and eqs?)
        #self.__check_shape__()
    def __len__(self):
        #length
        return len(self.E_path)
    def __check_shape__(self):
        #size of data
        #if self.Edata.ndim==3 and self.Ndata.ndim==3 and self.Zdata.ndim==3 and self.y.ndim==2:
        if self.E_path.ndim==3 and self.N_path.ndim==3 and self.Z_path.ndim==3:
            print('Inputs are matrix')
            #return True
        else:
            print('Inputs are eqids')
            #return False #the inputs are eqid
    def __getitem__(self,index):
        def get_hypo(logfile):
            #Input log file path from the rupture directory
            #output hypo lon,lat
            IN1=open(logfile,'r')
            for line in IN1.readlines():
                if 'Hypocenter (lon,lat,z[km])' in line:
                    Hypo_xyz=line.split(':')[-1].replace('(','').replace(')','').strip().split(',')
                    break
            IN1.close()
            return float(Hypo_xyz[0]),float(Hypo_xyz[1])
        
        def check_PGDs_hypoInfo(Data,STA,hypo,dist_thres,min_Nsta):
            #Given the hypocenter, check if after the removal, the PGDs are still meaningful
            '''
                Data: the PGD data [steps,features]
                STA: dictionary of station sorted by the 'ALL_staname_order.txt' file
                e.g. STA[37]=[-70.131718, -20.273540] because the STA[37]='IQQE' (check the location of IQQE)
                hypo: hypocenter of the rupture
                dist_thres:a distance from hypo
                min_Nsta: at least min_Nsta stations closer than the above dist_thres
            '''
            rec_idx=np.where(Data[-1,:121]!=0.0)[0] #the last PGD is not zero, means the station is not removed
            #print('number of non-removed=',len(rec_idx))
            if len(rec_idx)<min_Nsta:
                #oops, you removed all the near and far field stations
                return False
            Nclose=0
            for n_idx in rec_idx:
                stlon,stlat=STA[n_idx]
                dist=obspy.geodetics.locations2degrees(lat1=hypo[1],long1=hypo[0],lat2=stlat,long2=stlon)
                if dist<=dist_thres:
                    Nclose=Nclose+1
                if Nclose>=min_Nsta: #actually it stops at Nclose==min_Nsta 
                    return True
            return False #run through all the stations, Nclose still smaller than min_Nsta

        def D2PGD(data):
            #displacement to peak ground displacement 
            PGD=[]
            if np.ndim(data)==2:
                for i in range(data.shape[0]):
                    PGD.append(np.max(data[:i+1,:],axis=0))
            elif np.ndim(data)==1:
                for i in range(len(data)):
                    PGD.append(np.max(data[:i+1]))
            PGD=np.array(PGD)
            return(PGD)

        def make_noise(n_steps,f,Epsd,Npsd,Zpsd,PGD=False):
            #define sample rate
            dt=1.0 #make this 1 so that the length of PGD can be controlled by duration
            #noise model
            #level='median'  #this can be low, median, or high
            #duration of time series
            duration=n_steps
            # get white noise
            E_noise=windowed_gaussian(duration,dt,window_type=None)
            N_noise=windowed_gaussian(duration,dt,window_type=None)
            Z_noise=windowed_gaussian(duration,dt,window_type=None)
            noise=windowed_gaussian(duration,dt,window_type=None)
            #get PSDs
            #f,Epsd,Npsd,Zpsd=gnss_psd(level=level,return_as_frequencies=True,return_as_db=False)
            #control the noise level
            #scale=np.abs(np.random.randn()) #normal distribution
            #Epsd=Epsd*scale
            #Npsd=Npsd*scale
            #Npsd=Npsd*scale
            #Covnert PSDs to amplitude spectrum
            Epsd = Epsd**0.5
            Npsd = Npsd**0.5
            Zpsd = Zpsd**0.5
            #apply the spectrum
            E_noise=apply_spectrum(E_noise,Epsd,f,dt,is_gnss=True)[:n_steps]
            N_noise=apply_spectrum(N_noise,Npsd,f,dt,is_gnss=True)[:n_steps]
            Z_noise=apply_spectrum(Z_noise,Zpsd,f,dt,is_gnss=True)[:n_steps]
            if PGD:
                GD=(np.real(E_noise)**2.0+np.real(N_noise)**2.0+np.real(Z_noise)**2.0)**0.5
                PGD=D2PGD(GD)
                return PGD
            else:
                return np.real(E_noise),np.real(N_noise),np.real(Z_noise)        

        def get_mw(logfile):
            #Input log file path from the rupture directory
            # ***the fnction is no longer used, load directly from EQinfo file***
            IN1=open(logfile,'r')
            for line in IN1.readlines():
                if 'Actual magnitude' in line:
                    mw=float(line.split()[3])
                    break    
            IN1.close()
            return mw

        np.random.seed()
        #RNDID=np.random.rand()
        '''
        Nstan:number of stations
        add_code: do you want to add station existence code?
        add_nose:do you want to add color noise on the X?
        noise_p: possibility of the output is noise data (i.e. generate data with Mw=0)
        rmN: remove stations from these two number (10,110) or rmN=(0,0) for not remove
        scale: scale the added noise (if any) to the same scale as Normalization (i.e. PGD_mean,PGD_var), if not scale, simply set scale=(0,1)
        index is useless here since I want every batches to be different
        '''
        Dpath,E,N,Z,y,EQinfo,STAinfo,Nstan,add_code,add_noise,noise_p,rmN,level,Min_stan_dist,scale,BatchSize,Mwfilter,save_ID,shuffle= \
        (self.Dpath,self.E_path,self.N_path,self.Z_path,self.y_path,self.EQinfo,self.STAinfo,self.Nstan,self.add_code,self.add_noise,self.noise_p,self.rmN,self.Noise_level,self.Min_stan_dist,self.scale,self.BatchSize,self.Mwfilter,self.save_ID,self.shuffle)
        #Get station information and ordering in X
        sta_loc_file=STAinfo['sta_loc_file']
        station_order_file=STAinfo['station_order_file']
        STAINFO=np.genfromtxt(sta_loc_file,'S12',skip_header=1)
        STAINFO={ista[0].decode():[float(ista[1]),float(ista[2])]  for ista in STAINFO}
        STA=np.genfromtxt(station_order_file,'S6')
        STA={ista:STAINFO[sta.decode()] for ista,sta in enumerate(STA)}
        
        np.random.shuffle(level)
        #f,Epsd,Npsd,Zpsd=gnss_psd(level=level[0],return_as_frequencies=True,return_as_db=False) #these are always the same so only needs to be generated once
        #ngen=1
        #while True:
        #    print('Num of generator=%d'%(ngen))
        #    print('bs=    %d'%(BatchSize))
        rndEQidx=np.arange(len(E)) #To randomly pick an earthquake
        rndSTAidx=np.arange(Nstan) #Total of 121 stations to be removed
        X_batch=[]
        y_batch=[]
        #save picked EQ name
        sav_picked_EQ=[]
        nb=0 #number of batch generated
        #for nb in range(BatchSize):
        EQ_flag=0 #force it to be an earthquake if EQ_flag=1
        while nb<BatchSize:
            #print('Num of batch=',nb)
            if Dpath==None:
                #E,N,Z are already a muge matrix (not recommended!)
                #Station existence code, generally doesn't matter but you want the value close to features
                #Data=np.ones([E[0].shape[0],int(Nstan*2)]) #filling the data matrix, set the status code as zero when remove station
                Data=0.5*np.ones([E[0].shape[0],int(Nstan*2)]) #filling the data matrix, set the status code as zero when remove station
            else:
                #read the data from Directory, now the E/N/Z should be EQids (e.g. '002356')
                #Dpath=/projects/tlalollin/jiunting/Fakequakes/run/Chile_27200_ENZ   Chile_full.002709.Z.npy
                #test_read=glob.glob(Dpath+'/'+'Chile_full.'+E[0]+'.Z.npy')
                #test_read=np.load(test_read[0])
                test_read=np.load(E[0])
                Data=0.5*np.ones([test_read.shape[1],int(Nstan*2)]) #filling the data matrix, set the status code as zero when remove station
            if EQ_flag==0:
                EQ_or_noise=np.random.rand() #EQ or noise?
            else:
                EQ_or_noise=1 #EQ_flag==1, last round was EQ but remove too many stations, so try again with EQ
            if EQ_or_noise>=noise_p:
                #this is data (+noise? if add_noise=True)
                while True:
                    #randomly pull out an event (its Mw should larger than Mwfilter, otherwise, repeat)
                    np.random.shuffle(rndEQidx) #shuffle the index
                    #####Get the path of log file#####
                    #from: /projects/tlalollin/jiunting/Fakequakes/run/Chile_27200_ENZ/Chile_full.002709.Z.npy to 
                    real_EQid=E[int(rndEQidx[0])].split('/')[-1].split('.')[1] #this will be, for example '002709'
                    pre_pend=E[int(rndEQidx[0])].split('/')[-1].split('.')[0] #This will be, for example 'Chile_full' or Chile_small
                    #logfile='/projects/tlalollin/jiunting/Fakequakes/'+pre_pend+'/output/ruptures/subduction.'+real_EQid+'.log'
                    if not Mwfilter:
                        break
                    #checkMw=get_mw(logfile)
                    checkMw=EQinfo[int(rndEQidx[0])][1]     #y[int(rndEQidx[0])]
                    #print('checkMw,Mwfiter',checkMw,Mwfilter)
                    if checkMw>=Mwfilter:
                        break
                if Dpath==None:
                    tmp_E=E[int(rndEQidx[0])].copy() #Ok, use this Eq
                    tmp_N=N[int(rndEQidx[0])].copy()
                    tmp_Z=Z[int(rndEQidx[0])].copy()
                    #y_batch.append(y[int(rndEQidx[0])].reshape(-1,1)) #and its label
                else:
                    #print('# %d EQ selected:'%(nb),E[int(rndEQidx[0])])
                    #tmp_E=glob.glob(Dpath+'/'+'Chile_full.'+E[int(rndEQidx[0])]+'.E.npy') #Note E[int(rndEQidx[0])]==N[int(rndEQidx[0])]==Z[int(rndEQidx[0])]
                    tmp_E=np.load(E[int(rndEQidx[0])])
                    tmp_N=np.load(N[int(rndEQidx[0])])
                    tmp_Z=np.load(Z[int(rndEQidx[0])])
                    tmp_E=tmp_E.T
                    tmp_N=tmp_N.T
                    tmp_Z=tmp_Z.T
                    #y_batch.append(y[int(rndEQidx[0])].reshape(-1,1)) #and its label

                #beore started, should I add noise?
                if add_noise:
                    #add noise in every stations
                    for n in range(Nstan):
                        np.random.shuffle(level) #randomly pick from the level list 
                        f,Epsd,Npsd,Zpsd=gnss_psd(level=level[0],return_as_frequencies=True,return_as_db=False) 
                        Noise_add_E,Noise_add_N,Noise_add_Z=make_noise(tmp_E.shape[0],f,Epsd,Npsd,Zpsd,PGD=False)
                        tmp_E[:,n] = tmp_E[:,n] + Noise_add_E
                        tmp_N[:,n] = tmp_N[:,n] + Noise_add_N
                        tmp_Z[:,n] = tmp_Z[:,n] + Noise_add_Z
                rm_Nstan=np.random.randint(rmN[0],rmN[1]+1) #remove a random number of stations. Can remove zero station (not remove)
                np.random.shuffle(rndSTAidx) #shuffle the index of stations, pick the first ":rm_Nsta" to be removed
                #dealing with missing stations
                for rmidx in rndSTAidx[:rm_Nstan]:
                    #remove the data part
                    tmp_E[:,rmidx]=np.zeros(tmp_E.shape[0])
                    tmp_N[:,rmidx]=np.zeros(tmp_N.shape[0])
                    tmp_Z[:,rmidx]=np.zeros(tmp_Z.shape[0])
                    #Also set the "status code" to 0,Nstan means skip ["station"] and go to status code
                    Data[:,Nstan+rmidx]=np.zeros(tmp_E.shape[0])
                PGD=D2PGD((tmp_E**2+tmp_N**2+tmp_Z**2)**0.5)
                PGD=(PGD-scale[0])/scale[1]
                Data[:,:Nstan]=PGD.copy()
                #--------check if the removed Data is meaningful---------------
                ##############get hypocenter of the eq###################
                #logfile='/projects/tlalollin/jiunting/Fakequakes/Chile_full/output/ruptures/subduction.'+E[int(rndEQidx[0])]+'.log'
                eqinfo=EQinfo[int(rndEQidx[0])]
                eqlon=eqinfo[2]
                eqlat=eqinfo[3]
                #eqlon,eqlat=get_hypo(logfile)
                #print('Input E,y:',E,y)
                #print('rndEQIDX=%s,ID=%s, ID_from_EQinfo=%s eqlon,eqlat=%f %f'%(int(rndEQidx[0]),E[int(rndEQidx[0])],eqinfo[0],eqlon,eqlat))
                #########################################################
                #if check_PGDs_hypoInfo(Data,STA,hypo=[eqlon,eqlat],dist_thres=5.0,min_Nsta=8)==False:
                #if check_PGDs_hypoInfo(Data,STA,hypo=[eqlon,eqlat],dist_thres=5.0,min_Nsta=5)==False: #This is for Test#48, and Test#49
                #if check_PGDs_hypoInfo(Data,STA,hypo=[eqlon,eqlat],dist_thres=5.0,min_Nsta=3)==False: #This is for Test#48, and Test#49, #63
                #if check_PGDs_hypoInfo(Data,STA,hypo=[eqlon,eqlat],dist_thres=3.0,min_Nsta=4)==False: #This is for Test#72 (Even the Melinka has 5 stations within 3deg),#73,#75
                #if check_PGDs_hypoInfo(Data,STA,hypo=[eqlon,eqlat],dist_thres=5.0,min_Nsta=10)==False: #This is for Test#74 
                if check_PGDs_hypoInfo(Data,STA,hypo=[eqlon,eqlat],dist_thres=Min_stan_dist[1],min_Nsta=Min_stan_dist[0])==False: 
                    #print('Not enough near field stations......try again! at nb=%d'%(nb)) #but not just try again, make sure next one should be an earthquake, not noise.
                    EQ_flag=1 #remove too many near-field station, run again in "EQ" case (not noise) so the possibility of EQ/noise states the same
                    continue #skip this generation, try again......
                nb=nb+1 #the generated Data is okay, save it
                ##########save the picked EQ name#############
                sav_picked_EQ.append(real_EQid)
                #-----What labeling do you want to use??-----
                if y is 'flat':
                    #1.use the "flat" label (assuming strong determinism)
                    y_batch.append(y[int(rndEQidx[0])][-1] * np.ones(tmp_E.shape[0],1)) #the flat label
                else:
                    #2.none-determinism
                    #_t,sumMw=get_accM0(E[int(rndEQidx[0])]) #E[int(rndEQidx[0])] is the eqID (e.g. '002340')
                    #sumMw=np.load('/projects/tlalollin/jiunting/Fakequakes/run/Chile_27200_STF/Chile_full.'+E[int(rndEQidx[0])]+'.npy') #or directly loaded from .npy file
                    #sumMw=np.load('/projects/tlalollin/jiunting/Fakequakes/run/Chile_27200_STF/Chile_full.'+real_EQid+'.npy') #or directly loaded from .npy file
                    #sumMw=np.load('/projects/tlalollin/jiunting/Fakequakes/run/Chile_full_new_STF/Chile_full_new.'+real_EQid+'.npy') #or directly loaded from .npy file
                    sumMw=np.load(y[int(rndEQidx[0])]) #or directly load from the sorted data list
                    y_batch.append(sumMw.reshape(-1,1))
                #--------------------------------------------
                if add_code:
                    X_batch.append(Data)
                else:
                    X_batch.append(PGD)
                EQ_flag=0
            else:
                #this is just noise
                if Dpath==None:
                    1
                    #y_batch.append(np.zeros([E[0].shape[0],1])) #Mw=0 label from 0~n epochs
                else:
                    #test_read=glob.glob(Dpath+'/'+'Chile_full.'+E[0]+'.Z.npy')
                    #test_read=np.load(test_read[0]) #test_read=[Nsta,time_epochs]
                    test_read=np.load(E[0])
                    #y_batch.append(np.zeros([test_read.shape[1],1])) #Mw=0 label from 0~n epochs
                rm_Nstan=np.random.randint(rmN[0],rmN[1]+1) #remove a random number of stations, can be remove zero station (not remove)
                np.random.shuffle(rndSTAidx) #shuffle the index of stations, pick the first ":rm_Nsta" to be removed
                #dealing with missing stations
                for n in range(Nstan):
                    if n in rndSTAidx[:rm_Nstan]:
                        if Dpath==None:
                            #removed data
                            #Data[:,n]=np.zeros(E[0].shape[0])# but not just zero, this should be SCALED!!!
                            Data[:,n]=(np.zeros(E[0].shape[0])-scale[0])/scale[1] # but not just zero, this should be SCALED!!! in order to keep consistent to EQ part. (Xnew[:,:,:121]*10)+5=original
                            #remove code
                            Data[:,Nstan+n]=np.zeros(E[0].shape[0])
                        else:
                            #test_read=glob.glob(Dpath+'/'+'Chile_full.'+E[0]+'.Z.npy')
                            #test_read=np.load(test_read[0]) #test_read=[Nsta,time_epochs]
                            test_read=np.load(E[0])
                            #removed data
                            #Data[:,n]=np.zeros(test_read.shape[1])
                            Data[:,n]=(np.zeros(test_read.shape[1])-scale[0])/scale[1] #but not just zero! this should be SCALED!!!
                            #remove code
                            Data[:,Nstan+n]=np.zeros(test_read.shape[1])
                    else:
                        if Dpath==None:
                            np.random.shuffle(level)
                            f,Epsd,Npsd,Zpsd=gnss_psd(level=level[0],return_as_frequencies=True,return_as_db=False) #
                            Noise_add_E,Noise_add_N,Noise_add_Z=make_noise(E[0].shape[0],f,Epsd,Npsd,Zpsd,PGD=False)
                            PGD=D2PGD((Noise_add_E**2+Noise_add_N**2+Noise_add_Z**2)**0.5)
                            PGD=(PGD-scale[0])/scale[1]
                            Data[:,n]=PGD.copy()
                        else:
                            #test_read=glob.glob(Dpath+'/'+'Chile_full.'+E[0]+'.Z.npy')
                            #test_read=np.load(test_read[0]) #test_read=[Nsta,time_epochs]
                            test_read=np.load(E[0])
                            np.random.shuffle(level)
                            f,Epsd,Npsd,Zpsd=gnss_psd(level=level[0],return_as_frequencies=True,return_as_db=False) #randomly make noise psd
                            Noise_add_E,Noise_add_N,Noise_add_Z=make_noise(test_read.shape[1],f,Epsd,Npsd,Zpsd,PGD=False)
                            PGD=D2PGD((Noise_add_E**2+Noise_add_N**2+Noise_add_Z**2)**0.5)
                            PGD=(PGD-scale[0])/scale[1]
                            Data[:,n]=PGD.copy()
                #When it is noise, don't care the hypo
                #if check_PGDs_hypoInfo(Data,STA,hypo,dist_thres=5.0,min_Nsta=3):
                #    continue #skip this generation, try again......
                nb=nb+1
                #save the real ID, which is just noise (-1)
                sav_picked_EQ.append(-1)
                #there's no STF to worry about, just set a flat line of zero
                #y_batch.append(np.zeros([len(y[0,:]) ,1])) #Mw=0 label from 0~n epochs
                #####1.Assuming Mw_noise=0#####
                y_batch.append(np.zeros([test_read.shape[1],1])) #Mw=0 label from 0~n epochs
                #####2.Assuming Mw_noise=a constant here, 5.0#####
                #y_batch.append(5.0*np.ones([test_read.shape[1],1])) #Mw=0 label from 0~n epochs
                if add_code:
                    X_batch.append(Data)
                else:
                    X_batch.append(Data[:,:Nstan])
                EQ_flag=0
        X_batch=np.array(X_batch)
        y_batch=np.array(y_batch)
        y_batch=y_batch/10.0 #normalized the y so that they are closer 

        #######save the real_EQID########
        #Save the real EQid if you curious about the real EQID (e.g. to compare them with GFAST)
        if save_ID:
            sav_picked_EQ=np.array(sav_picked_EQ)
            np.save(save_ID,sav_picked_EQ)
        #######---END-------------#######
        #if len(X_batch)==32:
        #    print('Return a Test batch, value=%f RNDID=%7.5f'%(Data[23,55],RNDID))
        #else:
        #    print('Return a Training batch, value= %f RNDID=%7.5f'%(Data[23,55],RNDID))
        #ngen+=1
        #X_batch[:,:,:121]=X_batch[:,:,:121]**0.5 #take the sqrt #this is rerun #14
        #X_batch[:,:,:121]=(X_batch[:,:,:121]**0.5)/10.0 #take the sqrt and /10. This is rerun #19
        #If feature is smaller than 0.01, set to 0.01 (this is necessarily because log(0) is -inf will cause problem)
        X_batch[:,:,:121]=np.where(X_batch[:,:,:121]>=0.01,X_batch[:,:,:121],0.01) #this mean if X>=0.01, return X, otherwise(i.e. <0.01), return 0.01
        X_batch[:,:,:121]=np.log10(X_batch[:,:,:121]) #take the log10(x), starting from #67
        return X_batch,y_batch



class feature_gen_multi(keras.utils.Sequence):
    #######Generator should inherit the "Sequence" class in order to run multi-processing of fit_generator###########
    def __init__(self,Dpath,E_path,N_path,Z_path,y_path,EQinfo,STAinfo,Nstan=121,add_code=True,add_noise=True,noise_p=0.5,  
                 rmN=(10,110),Noise_level=[1,10,20,30,40,50,60,70,80,90],Min_stan_dist=[4,3],scale=(0,1), 
                 BatchSize=128,Mwfilter=8.0,save_ID='sav_pickedID_1_valid.npy',Xout='PGD',shuffle=True):
        self.Dpath=Dpath #The path of the individual [E/N/Z].npy data (should be a list or a numpy array)
        self.E_path=E_path #The path of the individual [E/N/Z].npy data (should be a list or a numpy array)
        self.N_path=N_path
        self.Z_path=Z_path
        self.y_path=y_path   # file name of y_path, string or list with multie string.
        self.EQinfo=EQinfo   #EQinfo file with the same number of lines of X,y list
        self.STAinfo=STAinfo #Stainfo files
        self.Nstan=Nstan
        self.add_code=add_code
        self.add_noise=add_noise
        self.noise_p=noise_p
        self.rmN=rmN
        self.Noise_level=Noise_level  
        self.Min_stan_dist=Min_stan_dist #e.g. [4,3]=minimum 4 stations within 3 degree
        self.scale=scale
        self.BatchSize=BatchSize
        self.Mwfilter=Mwfilter #Threshold magnitude for generated events, or False means everything
        self.save_ID=save_ID #save the original eqid with this name (e.g. sav_pickedID_73_2_valid.npy)
        self.Xout=Xout  #define the output feature X: a string of 'PGD' or 'ENZ'.
        self.shuffle=shuffle  #shuffle always True (shuffle station and eqs?)
        #self.__check_shape__()
    def __len__(self):
        #length
        return len(self.E_path)
    def __check_shape__(self):
        #size of data
        #if self.Edata.ndim==3 and self.Ndata.ndim==3 and self.Zdata.ndim==3 and self.y.ndim==2:
        if self.E_path.ndim==3 and self.N_path.ndim==3 and self.Z_path.ndim==3:
            print('Inputs are matrix')
            #return True
        else:
            print('Inputs are eqids')
            #return False #the inputs are eqid
    def __getitem__(self,index):
        def get_hypo(logfile):
            #Input log file path from the rupture directory
            #output hypo lon,lat
            IN1=open(logfile,'r')
            for line in IN1.readlines():
                if 'Hypocenter (lon,lat,z[km])' in line:
                    Hypo_xyz=line.split(':')[-1].replace('(','').replace(')','').strip().split(',')
                    break
            IN1.close()
            return float(Hypo_xyz[0]),float(Hypo_xyz[1])
        
        def check_PGDs_hypoInfo(Data,STA,nsta,hypo,dist_thres,min_Nsta):
            #Given the hypocenter, check if after the removal, the PGDs are still meaningful
            '''
                Data: the PGD data [steps,features]
                STA: dictionary of station sorted by the 'ALL_staname_order.txt' file
                e.g. STA[37]=[-70.131718, -20.273540] because the STA[37]='IQQE' (check the location of IQQE)
                hypo: hypocenter of the rupture
                dist_thres:a distance from hypo
                min_Nsta: at least min_Nsta stations closer than the above dist_thres
            '''
            rec_idx=np.where(Data[-1,:nsta]!=0.0)[0] #the last PGD is not zero, means the station is not removed
            #print('number of non-removed=',len(rec_idx))
            if len(rec_idx)<min_Nsta:
                #oops, you removed all the near and far field stations
                return False
            Nclose=0
            for n_idx in rec_idx:
                stlon,stlat=STA[n_idx]
                dist=obspy.geodetics.locations2degrees(lat1=hypo[1],long1=hypo[0],lat2=stlat,long2=stlon)
                if dist<=dist_thres:
                    Nclose=Nclose+1
                if Nclose>=min_Nsta: #actually it stops at Nclose==min_Nsta 
                    return True
            return False #run through all the stations, Nclose still smaller than min_Nsta

        def D2PGD(data):
            #displacement to peak ground displacement 
            PGD=[]
            if np.ndim(data)==2:
                for i in range(data.shape[0]):
                    PGD.append(np.max(data[:i+1,:],axis=0))
            elif np.ndim(data)==1:
                for i in range(len(data)):
                    PGD.append(np.max(data[:i+1]))
            PGD=np.array(PGD)
            return(PGD)

        def make_noise(n_steps,f,Epsd,Npsd,Zpsd,PGD=False):
            #define sample rate
            dt=1.0 #make this 1 so that the length of PGD can be controlled by duration
            #noise model
            #level='median'  #this can be low, median, or high
            #duration of time series
            duration=n_steps
            # get white noise
            E_noise=windowed_gaussian(duration,dt,window_type=None)
            N_noise=windowed_gaussian(duration,dt,window_type=None)
            Z_noise=windowed_gaussian(duration,dt,window_type=None)
            noise=windowed_gaussian(duration,dt,window_type=None)
            #get PSDs
            #f,Epsd,Npsd,Zpsd=gnss_psd(level=level,return_as_frequencies=True,return_as_db=False)
            #control the noise level
            #scale=np.abs(np.random.randn()) #normal distribution
            #Epsd=Epsd*scale
            #Npsd=Npsd*scale
            #Npsd=Npsd*scale
            #Covnert PSDs to amplitude spectrum
            Epsd = Epsd**0.5
            Npsd = Npsd**0.5
            Zpsd = Zpsd**0.5
            #apply the spectrum
            E_noise=apply_spectrum(E_noise,Epsd,f,dt,is_gnss=True)[:n_steps]
            N_noise=apply_spectrum(N_noise,Npsd,f,dt,is_gnss=True)[:n_steps]
            Z_noise=apply_spectrum(Z_noise,Zpsd,f,dt,is_gnss=True)[:n_steps]
            if PGD:
                GD=(np.real(E_noise)**2.0+np.real(N_noise)**2.0+np.real(Z_noise)**2.0)**0.5
                PGD=D2PGD(GD)
                return PGD
            else:
                return np.real(E_noise),np.real(N_noise),np.real(Z_noise)        

        def get_mw(logfile):
            #Input log file path from the rupture directory
            # ***the fnction is no longer used, load directly from EQinfo file***
            IN1=open(logfile,'r')
            for line in IN1.readlines():
                if 'Actual magnitude' in line:
                    mw=float(line.split()[3])
                    break    
            IN1.close()
            return mw

        np.random.seed()
        #RNDID=np.random.rand()
        '''
        Nstan:number of stations
        add_code: do you want to add station existence code?
        add_nose:do you want to add color noise on the X?
        noise_p: possibility of the output is noise data (i.e. generate data with Mw=0)
        rmN: remove stations from these two number (10,110) or rmN=(0,0) for not remove
        scale: scale the added noise (if any) to the same scale as Normalization (i.e. PGD_mean,PGD_var), if not scale, simply set scale=(0,1)
        index is useless here since I want every batches to be different
        '''
        Dpath,E,N,Z,y,EQinfo,STAinfo,Nstan,add_code,add_noise,noise_p,rmN,level,Min_stan_dist,scale,BatchSize,Mwfilter,save_ID,Xout,shuffle= \
        (self.Dpath,self.E_path,self.N_path,self.Z_path,self.y_path,self.EQinfo,self.STAinfo,self.Nstan,self.add_code,self.add_noise,self.noise_p,self.rmN,self.Noise_level,self.Min_stan_dist,self.scale,self.BatchSize,self.Mwfilter,self.save_ID,self.Xout,self.shuffle)
        #Get station information and ordering in X
        sta_loc_file=STAinfo['sta_loc_file']
        station_order_file=STAinfo['station_order_file']
        STAINFO=np.genfromtxt(sta_loc_file,'S12',skip_header=1)
        STAINFO={ista[0].decode():[float(ista[1]),float(ista[2])]  for ista in STAINFO}
        STA=np.genfromtxt(station_order_file,'S6')
        STA={ista:STAINFO[sta.decode()] for ista,sta in enumerate(STA)}
        
        np.random.shuffle(level)
        #f,Epsd,Npsd,Zpsd=gnss_psd(level=level[0],return_as_frequencies=True,return_as_db=False) #these are always the same so only needs to be generated once
        #ngen=1
        #while True:
        #    print('Num of generator=%d'%(ngen))
        #    print('bs=    %d'%(BatchSize))
        rndEQidx=np.arange(len(E)) #To randomly pick an earthquake
        rndSTAidx=np.arange(Nstan) #Total of 121 stations to be removed
        X_batch=[]
        y_batch=[]
        #save picked EQ name
        sav_picked_EQ=[]
        nb=0 #number of batch generated
        #for nb in range(BatchSize):
        EQ_flag=0 #force it to be an earthquake if EQ_flag=1
        while nb<BatchSize:
            #print('Num of batch=',nb)
            if Dpath==None:
                #E,N,Z are already a muge matrix (not recommended!)
                #Station existence code, generally doesn't matter but you want the value close to features
                #Data=np.ones([E[0].shape[0],int(Nstan*2)]) #filling the data matrix, set the status code as zero when remove station
                if Xout=='PGD':
                    Data=0.5*np.ones([E[0].shape[0],int(Nstan*(1+1))]) #filling the data matrix, set the status code as zero when remove station
                elif Xout=='ENZ':
                    Data=0.5*np.ones([E[0].shape[0],int(Nstan*(3+1))]) #filling the data matrix, set the status code as zero when remove station
            else:
                #read the data from Directory, now the E/N/Z should be EQids (e.g. '002356')
                #Dpath=/projects/tlalollin/jiunting/Fakequakes/run/Chile_27200_ENZ   Chile_full.002709.Z.npy
                #test_read=glob.glob(Dpath+'/'+'Chile_full.'+E[0]+'.Z.npy')
                #test_read=np.load(test_read[0])
                test_read=np.load(E[0])  #shape=[Nsta,Timesteps]
                if Xout=='PGD':
                    Data=0.5*np.ones([test_read.shape[1],int(Nstan*(1+1))]) #filling the data matrix, set the status code as zero when remove station
                elif Xout=='ENZ':
                    Data=0.5*np.ones([test_read.shape[1],int(Nstan*(3+1))])
            if EQ_flag==0:
                EQ_or_noise=np.random.rand() #EQ or noise?
            else:
                EQ_or_noise=1 #EQ_flag==1, last round was EQ but remove too many stations, so try again with EQ
            if EQ_or_noise>=noise_p:
                #this is data (+noise? if add_noise=True)
                while True:
                    #randomly pull out an event (its Mw should larger than Mwfilter, otherwise, repeat)
                    np.random.shuffle(rndEQidx) #shuffle the index
                    #####Get the path of log file#####
                    #from: /projects/tlalollin/jiunting/Fakequakes/run/Chile_27200_ENZ/Chile_full.002709.Z.npy to 
                    real_EQid=E[int(rndEQidx[0])].split('/')[-1].split('.')[1] #this will be, for example '002709'
                    pre_pend=E[int(rndEQidx[0])].split('/')[-1].split('.')[0] #This will be, for example 'Chile_full' or Chile_small
                    #logfile='/projects/tlalollin/jiunting/Fakequakes/'+pre_pend+'/output/ruptures/subduction.'+real_EQid+'.log'
                    if not Mwfilter:
                        break
                    #checkMw=get_mw(logfile) #check Mw from .log file
                    checkMw=EQinfo[int(rndEQidx[0])][1]   #or simply check Mw from EQinf file
                    #print('checkMw,Mwfiter',checkMw,Mwfilter)
                    if checkMw>=Mwfilter:
                        break
                if Dpath==None:
                    tmp_E=E[int(rndEQidx[0])].copy() #Ok, use this Eq
                    tmp_N=N[int(rndEQidx[0])].copy()
                    tmp_Z=Z[int(rndEQidx[0])].copy()
                    #y_batch.append(y[int(rndEQidx[0])].reshape(-1,1)) #and its label
                else:
                    #print('# %d EQ selected:'%(nb),E[int(rndEQidx[0])])
                    #tmp_E=glob.glob(Dpath+'/'+'Chile_full.'+E[int(rndEQidx[0])]+'.E.npy') #Note E[int(rndEQidx[0])]==N[int(rndEQidx[0])]==Z[int(rndEQidx[0])]
                    tmp_E=np.load(E[int(rndEQidx[0])]) #shape=[Nsta,Timesteps]
                    tmp_N=np.load(N[int(rndEQidx[0])])
                    tmp_Z=np.load(Z[int(rndEQidx[0])])
                    tmp_E=tmp_E.T  #shape=[Timesteps,Nsta]
                    tmp_N=tmp_N.T
                    tmp_Z=tmp_Z.T
                    #y_batch.append(y[int(rndEQidx[0])].reshape(-1,1)) #and its label

                #beore started, should I add noise?
                if add_noise:
                    #add noise in every stations
                    for n in range(Nstan):
                        np.random.shuffle(level) #randomly pick from the level list 
                        f,Epsd,Npsd,Zpsd=gnss_psd(level=level[0],return_as_frequencies=True,return_as_db=False) 
                        Noise_add_E,Noise_add_N,Noise_add_Z=make_noise(tmp_E.shape[0],f,Epsd,Npsd,Zpsd,PGD=False)
                        tmp_E[:,n] = tmp_E[:,n] + Noise_add_E
                        tmp_N[:,n] = tmp_N[:,n] + Noise_add_N
                        tmp_Z[:,n] = tmp_Z[:,n] + Noise_add_Z
                rm_Nstan=np.random.randint(rmN[0],rmN[1]+1) #remove a random number of stations. Can remove zero station (not remove)
                np.random.shuffle(rndSTAidx) #shuffle the index of stations, pick the first ":rm_Nsta" to be removed
                #dealing with missing stations
                for rmidx in rndSTAidx[:rm_Nstan]:
                    #remove the data part
                    tmp_E[:,rmidx]=np.zeros(tmp_E.shape[0])
                    tmp_N[:,rmidx]=np.zeros(tmp_N.shape[0])
                    tmp_Z[:,rmidx]=np.zeros(tmp_Z.shape[0])
                    #Also set the "status code" to 0,Nstan means skip station columns and go to status code
                    if Xout=='PGD':
                        Data[:,Nstan+rmidx]=np.zeros(tmp_E.shape[0])
                    elif Xout=='ENZ':
                        Data[:,Nstan*3+rmidx]=np.zeros(tmp_E.shape[0])
                if Xout=='PGD':
                    PGD=D2PGD((tmp_E**2+tmp_N**2+tmp_Z**2)**0.5)
                    PGD=(PGD-scale[0])/scale[1]
                    Data[:,:Nstan]=PGD.copy()
                elif Xout=='ENZ':
                    ENZ_Data=np.hstack([tmp_E,tmp_N,tmp_Z])
                    ENZ_Data=(ENZ_Data-scale[0])/scale[1]
                    Data[:,:Nstan*3]=ENZ_Data.copy()

                #--------check if the removed Data is meaningful---------------
                ##############get hypocenter of the eq###################
                #logfile='/projects/tlalollin/jiunting/Fakequakes/Chile_full/output/ruptures/subduction.'+E[int(rndEQidx[0])]+'.log'
                eqinfo=EQinfo[int(rndEQidx[0])]
                eqlon=eqinfo[2]
                eqlat=eqinfo[3]
                #eqlon,eqlat=get_hypo(logfile)
                #print('Input E,y:',E,y)
                #print('rndEQIDX=%s,ID=%s, ID_from_EQinfo=%s eqlon,eqlat=%f %f'%(int(rndEQidx[0]),E[int(rndEQidx[0])],eqinfo[0],eqlon,eqlat))
                #########################################################
                #if check_PGDs_hypoInfo(Data,STA,hypo=[eqlon,eqlat],dist_thres=3.0,min_Nsta=4)==False: #This is for Test#72 (Even the Melinka has 5 stations within 3deg),#73,#75
                #if check_PGDs_hypoInfo(Data,STA,hypo=[eqlon,eqlat],dist_thres=5.0,min_Nsta=10)==False: #This is for Test#74 
                if check_PGDs_hypoInfo(Data,STA,Nstan,hypo=[eqlon,eqlat],dist_thres=Min_stan_dist[1],min_Nsta=Min_stan_dist[0])==False:
                    #print('Not enough near field stations......try again! at nb=%d'%(nb)) #but not just try again, make sure next one should be an earthquake, not noise.
                    EQ_flag=1 #remove too many near-field station, run again in "EQ" case (not noise) so the possibility of EQ/noise states the same
                    continue #skip this generation, try again......
                nb=nb+1 #the generated Data is okay, save it
                ##########save the picked EQ name#############
                sav_picked_EQ.append(real_EQid)
                #-----What labeling do you want to use??-----
                if y is 'flat':
                    #1.use the "flat" label (assuming strong determinism)
                    y_batch.append(y[int(rndEQidx[0])][-1] * np.ones(tmp_E.shape[0],1)) #the flat label
                else:
                    #2.none-determinism or multiple output
                    if y.ndim==2:
                        merg_y=[]
                        for iy in range(len(y)):
                            tmp_y=np.load(y[iy][int(rndEQidx[0])])
                            if len(merg_y)==0:
                                merg_y=tmp_y.reshape(-1,1)
                            else:
                                merg_y=np.hstack([merg_y,tmp_y.reshape(-1,1)])
                            #merg_y.append(tmp_y.reshape(-1,1))
                        merg_y=np.array(merg_y)
                        y_batch.append(merg_y)
                    elif y.ndim==1:
                        #the original label, only Mw
                        #_t,sumMw=get_accM0(E[int(rndEQidx[0])]) #E[int(rndEQidx[0])] is the eqID (e.g. '002340')
                        #sumMw=np.load('/projects/tlalollin/jiunting/Fakequakes/run/Chile_27200_STF/Chile_full.'+E[int(rndEQidx[0])]+'.npy') #or directly loaded from .npy file
                        #sumMw=np.load('/projects/tlalollin/jiunting/Fakequakes/run/Chile_27200_STF/Chile_full.'+real_EQid+'.npy') #or directly loaded from .npy file
                        #sumMw=np.load('/projects/tlalollin/jiunting/Fakequakes/run/Chile_full_new_STF/Chile_full_new.'+real_EQid+'.npy') #or directly loaded from .npy file
                        sumMw=np.load(y[int(rndEQidx[0])]) #or directly load from the sorted data list
                        y_batch.append(sumMw.reshape(-1,1))
                    else:
                        print('Unknown y format, please check input y!')
                #--------------------------------------------
                if add_code:
                    X_batch.append(Data)
                else:
                    X_batch.append(PGD)
                EQ_flag=0
            else:
                #this is just noise
                if Dpath==None:
                    1
                    #y_batch.append(np.zeros([E[0].shape[0],1])) #Mw=0 label from 0~n epochs
                else:
                    #test_read=glob.glob(Dpath+'/'+'Chile_full.'+E[0]+'.Z.npy')
                    #test_read=np.load(test_read[0]) #test_read=[Nsta,time_epochs]
                    test_read=np.load(E[0])
                    #y_batch.append(np.zeros([test_read.shape[1],1])) #Mw=0 label from 0~n epochs
                rm_Nstan=np.random.randint(rmN[0],rmN[1]+1) #remove a random number of stations, can be remove zero station (not remove)
                np.random.shuffle(rndSTAidx) #shuffle the index of stations, pick the first ":rm_Nsta" to be removed
                #dealing with missing stations
                for n in range(Nstan):
                    if n in rndSTAidx[:rm_Nstan]:
                        if Dpath==None:
                            #removed data
                            #Data[:,n]=np.zeros(E[0].shape[0])# but not just zero, this should be SCALED!!!
                            Data[:,n]=(np.zeros(E[0].shape[0])-scale[0])/scale[1] # but not just zero, this should be SCALED!!! in order to keep consistent to EQ part. (Xnew[:,:,:121]*10)+5=original
                            #remove code
                            Data[:,Nstan+n]=np.zeros(E[0].shape[0])
                        else:
                            #test_read=glob.glob(Dpath+'/'+'Chile_full.'+E[0]+'.Z.npy')
                            #test_read=np.load(test_read[0]) #test_read=[Nsta,time_epochs]
                            test_read=np.load(E[0])
                            #removed data
                            #Data[:,n]=np.zeros(test_read.shape[1])
                            Data[:,n]=(np.zeros(test_read.shape[1])-scale[0])/scale[1] #but not just zero! this should be SCALED!!!
                            #remove code
                            Data[:,Nstan+n]=np.zeros(test_read.shape[1])
                    else:
                        if Dpath==None:
                            np.random.shuffle(level)
                            f,Epsd,Npsd,Zpsd=gnss_psd(level=level[0],return_as_frequencies=True,return_as_db=False) #
                            Noise_add_E,Noise_add_N,Noise_add_Z=make_noise(E[0].shape[0],f,Epsd,Npsd,Zpsd,PGD=False)
                            PGD=D2PGD((Noise_add_E**2+Noise_add_N**2+Noise_add_Z**2)**0.5)
                            PGD=(PGD-scale[0])/scale[1]
                            Data[:,n]=PGD.copy()
                        else:
                            #test_read=glob.glob(Dpath+'/'+'Chile_full.'+E[0]+'.Z.npy')
                            #test_read=np.load(test_read[0]) #test_read=[Nsta,time_epochs]
                            test_read=np.load(E[0])
                            np.random.shuffle(level)
                            f,Epsd,Npsd,Zpsd=gnss_psd(level=level[0],return_as_frequencies=True,return_as_db=False) #randomly make noise psd
                            Noise_add_E,Noise_add_N,Noise_add_Z=make_noise(test_read.shape[1],f,Epsd,Npsd,Zpsd,PGD=False)
                            PGD=D2PGD((Noise_add_E**2+Noise_add_N**2+Noise_add_Z**2)**0.5)
                            PGD=(PGD-scale[0])/scale[1]
                            Data[:,n]=PGD.copy()
                #When it is noise, don't care the hypo
                #if check_PGDs_hypoInfo(Data,STA,hypo,dist_thres=5.0,min_Nsta=3):
                #    continue #skip this generation, try again......
                nb=nb+1
                #save the real ID, which is just noise (-1)
                sav_picked_EQ.append(-1)
                #there's no STF to worry about, just set a flat line of zero
                #y_batch.append(np.zeros([len(y[0,:]) ,1])) #Mw=0 label from 0~n epochs
                #####1.Assuming Mw_noise=0#####
                y_batch.append(np.zeros([test_read.shape[1],1])) #Mw=0 label from 0~n epochs
                #####2.Assuming Mw_noise=a constant here, 5.0#####
                #y_batch.append(5.0*np.ones([test_read.shape[1],1])) #Mw=0 label from 0~n epochs
                if add_code:
                    X_batch.append(Data)
                else:
                    X_batch.append(Data[:,:Nstan])
                EQ_flag=0
        X_batch=np.array(X_batch)
        y_batch=np.array(y_batch)
        y_batch=y_batch/10.0 #normalized the y so that they are closer

        #######save the real_EQID########
        #Save the real EQid if you curious about the real EQID (e.g. to compare them with GFAST)
        if save_ID:
            sav_picked_EQ=np.array(sav_picked_EQ)
            np.save(save_ID,sav_picked_EQ)
        #######---END-------------#######
        #if len(X_batch)==32:
        #    print('Return a Test batch, value=%f RNDID=%7.5f'%(Data[23,55],RNDID))
        #else:
        #    print('Return a Training batch, value= %f RNDID=%7.5f'%(Data[23,55],RNDID))
        #ngen+=1
        #X_batch[:,:,:121]=X_batch[:,:,:121]**0.5 #take the sqrt #this is rerun #14
        #X_batch[:,:,:121]=(X_batch[:,:,:121]**0.5)/10.0 #take the sqrt and /10. This is rerun #19
        #If feature is smaller than 0.01, set to 0.01 (this is necessarily because log(0) is -inf will cause problem)
        if Xout=='PGD':
            X_batch[:,:,:Nstan]=np.where(X_batch[:,:,:Nstan]>=0.01,X_batch[:,:,:Nstan],0.01) #this mean if X>=0.01, return X, otherwise(i.e. <0.01), return 0.01
            X_batch[:,:,:Nstan]=np.log10(X_batch[:,:,:Nstan]) #take the log10(x), starting from #67
        elif Xout=='ENZ':
            X_batch[:,:,:Nstan*3]=X_batch[:,:,:Nstan*3] * 0.1 #simple scaling
            #X_batch[:,:,:Nstan*3]=np.where(X_batch[:,:,:Nstan*3]>=0.01,X_batch[:,:,:Nstan*3],0.01) #this mean if X>=0.01, return X, otherwise(i.e. <0.01), return 0.01
            #X_batch[:,:,:Nstan*3]=np.log10(X_batch[:,:,:Nstan*3]) #take the log10(x), starting from #67
        
        return X_batch,y_batch
    

    
       
def train(files,train_params):
    STAinfo={}
    STAinfo['sta_loc_file']=files['GFlist']
    STAinfo['station_order_file']=files['Sta_ordering']
    EQinfo_file=files['EQinfo']
    E_file=files['E']
    N_file=files['N']
    Z_file=files['Z']
    y_file=files['y']
    E=np.genfromtxt(E_file,'S')
    N=np.genfromtxt(N_file,'S')
    Z=np.genfromtxt(Z_file,'S')
    #For python3, reading from the genfromtxt will be \b prepended
    E=np.array([i.decode() for i in E])
    N=np.array([i.decode() for i in N])
    Z=np.array([i.decode() for i in Z])
    #dealing with label
    if y_file is 'flat':
        y='flat'
    elif type(y_file) is list:
        y=np.array([ [tmpy.decode() for tmpy in np.genfromtxt(iy,'S')] for iy in y_file  ])
    else:
        y=np.genfromtxt(y_file,'S')
        y=np.array([i.decode() for i in y])
    
    ###If more than 1 output, y.shape is [Nparams X Samples] (i.e. Mw, Lon, Lat, Depth, Length, Width. -> Nparams=6)
    ###
    ###
    
    #load EQinfo file into array
    EQinfo=np.genfromtxt(EQinfo_file)
    
    
    #train-test split
    train_idx,valid_and_test_idx=train_test_split(np.arange(0,len(E)),test_size=0.3, random_state=16)
    X_train_E=E[train_idx]
    X_train_N=N[train_idx]
    X_train_Z=Z[train_idx]
    if y.ndim==2:
        y_train=y[:,train_idx]
    else:
        y_train=y[train_idx]
    EQinfo_train=EQinfo[train_idx]

    X_valid_test_E=E[valid_and_test_idx]
    X_valid_test_N=N[valid_and_test_idx]
    X_valid_test_Z=Z[valid_and_test_idx]
    if y.ndim==2:
        y_valid_test=y[:,valid_and_test_idx]
    else:
        y_valid_test=y[valid_and_test_idx]
    EQinfo_valid_test=EQinfo[valid_and_test_idx]
    #Split the valid+test again to 0.2 and 0.1
    valid_idx,test_idx=train_test_split(np.arange(0,len(valid_and_test_idx)),test_size=0.1/0.3, random_state=16) #0.1 out of 0.3 is the testing dataset; 0.2 out of 0.3 is the validation

    X_valid_E=X_valid_test_E[valid_idx]
    X_valid_N=X_valid_test_N[valid_idx]
    X_valid_Z=X_valid_test_Z[valid_idx]
    if y.ndim==2:
        y_valid=y_valid_test[:,valid_idx]
    else:
        y_valid=y_valid_test[valid_idx]
    EQinfo_valid=EQinfo_valid_test[valid_idx]

    X_test_E=X_valid_test_E[test_idx]
    X_test_N=X_valid_test_N[test_idx]
    X_test_Z=X_valid_test_Z[test_idx]
    if y.ndim==2:
        y_test=y_valid_test[:,test_idx]
    else:
        y_test=y_valid_test[test_idx]
    EQinfo_test=EQinfo_valid_test[test_idx]
    #print('Total training data, labels=',len(X_train_E),len(y_train))
    #print(X_train_E,y_train)
    #print('Total validation data, labels=',len(X_valid_E),len(y_valid))
    #print(X_valid_E,y_valid)
    #print('Total test data, labels=',len(X_test_E),len(y_test))
    #print(X_test_E,y_test)
    
    
    #Build structure
    HP=train_params['Neurons']
    epochs=train_params['epochs']
    Drops=train_params['Drops']
    BS=train_params['BS'] #Batch Size for training
    BS_valid=train_params['BS_valid'] #####################################CHANGE it later!!!!! 1024
    BS_test=train_params['BS_test'] #batch size for testing
    scales=train_params['scales'] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by log10(X)
    Testnum=train_params['Testnum']
    #FlatY=False #just change y='flat' in the y input
    NoiseP=train_params['NoiseP'] #possibility of noise event
    Noise_level=train_params['Noise_level']
    rm_stans=train_params['rm_stans'] #remove number of stations from 0~115
    Min_stan_dist=train_params['Min_stan_dist'] #minumum 4 stations within 3 degree
    Loss_func=train_params['Loss_func'] #can be default loss function string or self defined loss
    
    #add a tensorflow callback
    logdir = "logs/scalars/Test"+Testnum+'_'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    
    #MLARGE structure
    network = models.Sequential()
    network.add(tf.keras.layers.TimeDistributed(layers.Dense(HP[0]),input_shape=(102,242,)))
    network.add(layers.LeakyReLU(alpha=0.1))
    network.add(tf.keras.layers.TimeDistributed(layers.Dense(HP[1])))
    network.add(layers.LeakyReLU(alpha=0.1))
    network.add(layers.Dropout(Drops[0]))
    network.add(layers.LSTM(HP[2],return_sequences=True,input_shape=(102,242,),dropout=0.2, recurrent_dropout=0.2))
    network.add(tf.keras.layers.TimeDistributed(layers.Dense(HP[3])))
    network.add(layers.LeakyReLU(alpha=0.1))
    network.add(tf.keras.layers.TimeDistributed(layers.Dense(HP[4])))
    network.add(layers.LeakyReLU(alpha=0.1))
    network.add(tf.keras.layers.TimeDistributed(layers.Dense(HP[5])))
    network.add(layers.LeakyReLU(alpha=0.1))
    network.add(tf.keras.layers.TimeDistributed(layers.Dense(HP[6])))
    network.add(layers.LeakyReLU(alpha=0.1))
    network.add(layers.Dropout(Drops[1]))
    network.add(tf.keras.layers.TimeDistributed(layers.Dense(1,activation='relu')))
    network.summary()
    network.compile(loss=Loss_func,optimizer='adam')

    #build generator
    Dpath='Path_defined_in_file'
    #print('Training_X inp:',X_train_E)
    #print('Training_y inp:',y_train)
    gtrain=feature_gen(Dpath,X_train_E,X_train_N,X_train_Z,y_train,EQinfo_train,STAinfo,Nstan=121,add_code=True,add_noise=True,noise_p=NoiseP,rmN=(rm_stans[0],rm_stans[1]),Noise_level=Noise_level,Min_stan_dist=Min_stan_dist,scale=(scales[0],scales[1]),BatchSize=BS,Mwfilter=7.0,save_ID=False,shuffle=True) #Use the "flat y"
    gvalid=feature_gen(Dpath,X_valid_E,X_valid_N,X_valid_Z,y_valid,EQinfo_valid,STAinfo,Nstan=121,add_code=True,add_noise=True,noise_p=NoiseP,rmN=(rm_stans[0],rm_stans[1]),Noise_level=Noise_level,Min_stan_dist=Min_stan_dist,scale=(scales[0],scales[1]),BatchSize=BS_valid,Mwfilter=7.0,save_ID='Run%s_valid_EQID.npy'%(Testnum),shuffle=True) #Use the "flat y"
    gtest=feature_gen(Dpath,X_test_E,X_test_N,X_test_Z,y_test,EQinfo_test,STAinfo,Nstan=121,add_code=True,add_noise=True,noise_p=NoiseP,rmN=(rm_stans[0],rm_stans[1]),Noise_level=Noise_level,Min_stan_dist=Min_stan_dist,scale=(scales[0],scales[1]),BatchSize=BS_test,Mwfilter=7.0,save_ID='Run%s_test_EQID.npy'%(Testnum),shuffle=True) #Use the "flat y"

    #check file/dir exist,otherwise mkdir
    if not(os.path.exists('./Test'+Testnum)):
        os.makedirs('./Test'+Testnum)
    #Add callback
    CB=keras.callbacks.ModelCheckpoint(filepath='./Test'+Testnum+'/weights.{epoch:04d}-{val_loss:.6f}.hdf5',monitor='val_loss',save_best_only=True,mode='min',period=5)
    
    #print('Start generating validation data')
    X_valid_out,y_valid_out=gvalid.__getitem__(1)
    np.save('Xvalid'+Testnum+'.npy',X_valid_out)
    np.save('yvalid'+Testnum+'.npy',y_valid_out)

    #Also save the testing data 
    #print('Start generating testing data')
    X_test_out,y_test_out=gtest.__getitem__(1)
    np.save('Xtest'+Testnum+'.npy',X_test_out)
    np.save('ytest'+Testnum+'.npy',y_test_out)
    
    
    #start training
    model_hist=network.fit_generator(gtrain,validation_data=(X_valid_out,y_valid_out),use_multiprocessing=True,workers=40,validation_steps=1,steps_per_epoch=1,epochs=epochs,callbacks=[CB,tensorboard_callback]) #so that total steps=1+7=8

    #save training result and training curve
    tf.keras.models.save_model(network,'./Test'+Testnum+'.h5py')
    np.save('./Test'+Testnum+'.npy',model_hist.history)




class Model():
    def __init__(self,Model_path,X,y,scale_X,back_scale_X,scale_y,back_scale_y):
        self.Model_path = Model_path
        self.X = X
        self.y = y
        self.scale_X = scale_X
        self.back_scale_X = back_scale_X
        self.scale_y = scale_y
        self.back_scale_y = back_scale_y
        self.model=None
        self.predictions=None
        self.real=None
        self.sav_acc=None
    
    def check(self):
        import numpy as np
        assert len(self.X)==len(self.y)
        print('Nsamples={}'.format(len(self.X)))
        assert np.max(self.y)<10.0, 'error:Mw>10'
        print('Input data are okay')

    def predict(self):
        import tensorflow as tf
        import tensorflow.keras as keras
        import matplotlib.pyplot as plt
        Model_path,X,y,scale_X,back_scale_X,scale_y,back_scale_y=(self.Model_path,self.X,self.y,self.scale_X,self.back_scale_X,
        self.scale_y,self.back_scale_y)
        if Model_path=='Lin2020':
            import mlarge
            model_loaded=tf.keras.models.load_model(mlarge.__path__[0].replace('src/mlarge','models/Test81_weights.49475-0.000131.hdf5'),compile=False)
        else:
            model_loaded=tf.keras.models.load_model(Model_path,compile=False)

        self.model=model_loaded
        X1=scale_X(X)
        predictions=model_loaded.predict(X1)
        #scale the labels back to the real sense
        predictions=back_scale_y(predictions)
        self.predictions=predictions
        y1=back_scale_y(y)
        self.real=y1

    def accuracy(self,tolerance=0.3,current=True):
        import numpy as np
        def get_accuracy(pred_Mw,real_Mw,tolorance=0.3,NoiseMw=False,tolorance_noise=False):
            #Make accuracy calculation
            #tolorance: constant +- tolorance consider as corrected prediction
            #NoiseMw: what are the noise event magnitude? or False if no noise events
            #tolorance_noise can be +- tolorance for noise magnitude, or False (do not count noise event)
            if NoiseMw==False and tolorance_noise==False:
                #only EQ events
                #accuracy for EQ
                n_success_EQ=np.where( np.abs(pred_Mw-real_Mw)<=tolorance )[0] #Mw is the Mw prediction at certain time t
                return (len(n_success_EQ)/len(pred_Mw))*100, None
            else:
                #noise and EQ events
                EQidx=np.where(real_Mw!=NoiseMw)[0]
                Noiseidx=np.where(real_Mw==NoiseMw)[0]
                #accuracy for EQ
                n_success_EQ=np.where(np.abs(pred_Mw[EQidx]-real_Mw[EQidx])<=tolorance)[0] #Mw is the Mw prediction at certain time t
                #accuracy for Noise
                n_success_Noise=np.where(np.abs(pred_Mw[Noiseidx]-real_Mw[Noiseidx])<=tolorance_noise )[0] #Mw is the Mw prediction at certain time t
                return (len(n_success_EQ)/len(pred_Mw))*100, (len(n_success_Noise)/len(Noiseidx))*100
        predictions=self.predictions
        y1=self.real
        if predictions is None:
            print('Please make prediction first by .prediction() method')
            return
        else: 
            sav_acc=[]
            if current:
                for i_epoch in range(predictions.shape[1]):
                    acc_EQ,acc_noise=get_accuracy(predictions[:,i_epoch],y1[:,i_epoch],tolerance)
                    sav_acc.append(acc_EQ)
            else:
                for i_epoch in range(predictions.shape[1]):
                    acc_EQ,acc_noise=get_accuracy(predictions[:,i_epoch],y1[:,-1],tolerance)
                    sav_acc.append(acc_EQ)
        sav_acc=np.array(sav_acc)
        self.sav_acc=sav_acc

    def plot_acc(self,T,show=True):
        if self.sav_acc is None:
            print('Please make prediction first by .accuracy() method')
            return
        else:
            import matplotlib.pyplot as plt
            plt.plot(T,self.sav_acc)
            plt.xlabel('Time(s)',fontsize=16)
            plt.ylabel('Accuracy(%)',fontsize=16)
            plt.xlim([10,510])
            plt.grid(True)
            if show:
                plt.show()

    def plot_snap(self,T=[60,120,240],err_range=0.3):
        import matplotlib.pyplot as plt
        import numpy as np
        #plot prediction snap shot
        tcs_time=np.arange(102)*5+5
        if self.predictions is None:
            print('Please make prediction first by .predict() method')
            return
        else:
            y=self.real
            y_pred=self.predictions
            props = dict(boxstyle='round', facecolor='white', alpha=0.7)
            if not(T is 'All'):
                subplot_w=int(len(T)**0.5)
                subplot_l=int(np.ceil(len(T)/subplot_w))
                plt.figure()
                for i,t in enumerate(T):
                    idx = np.where(tcs_time==t)[0][0]
                    plt.subplot(subplot_w,subplot_l,i+1)
                    plt.plot(y[:,-1,0],y_pred[:,idx,0],'o',markerfacecolor=[0.65,0.65,0.65],markeredgecolor='k',mew=0.25,ms=3.5)
                    plt.grid(True)
                    plt.plot([y.min(),y.max()],[y.min(),y.max()],'k--',linewidth=2)
                    plt.plot([y.min(),y.max()],[y.min()-err_range,y.max()-err_range],'k--',linewidth=0.5)
                    plt.plot([y.min(),y.max()],[y.min()+err_range,y.max()+err_range],'k--',linewidth=0.5)
                    plt.fill_between([y.min(),y.max()],[y.min()-err_range,y.max()-err_range],[y.min()+err_range,y.max()+err_range],facecolor='k',alpha=0.25)
                    plt.xlim([7.4,9.52])
                    plt.ylim([7.4,9.52])
                    Xlim=plt.xlim()
                    Ylim=plt.ylim()
                    Xpos=(Xlim[1]-Xlim[0])*0.06+Xlim[0]
                    Ypos=(Ylim[1]-Ylim[0])*0.9+Ylim[0]
                    plt.text(Xpos,Ypos,'%03d sec'%(idx*5+5),bbox=props,fontsize=12)
                    plt.ylabel('Predicted Mw',fontsize=14)
                    plt.xlabel('Final Mw',fontsize=14)
                    ax1=plt.gca()
                    ax1.tick_params(pad=0.3)
                plt.subplots_adjust(left=0.08,top=0.88,right=0.97,bottom=0.1,wspace=0.07)
                plt.show()
            else:
                #plot and save all figures need to be done
                pass
        

    def show_data(self,show=True):
        import matplotlib.pyplot as plt
        if self.real is None:
            print('Please make prediction first by .predict() method')
            return
        plt.hist(self.real[:,-1,0])
        plt.xlabel('Mw',fontsize=16)
        plt.ylabel('frequency',fontsize=16)
        if show:
            plt.show()
    


def train_cont(files,train_params,Model_path,):
    STAinfo={}
    STAinfo['sta_loc_file']=files['GFlist']
    STAinfo['station_order_file']=files['Sta_ordering']
    EQinfo_file=files['EQinfo']
    E_file=files['E']
    N_file=files['N']
    Z_file=files['Z']
    y_file=files['y']
    E=np.genfromtxt(E_file,'S')
    N=np.genfromtxt(N_file,'S')
    Z=np.genfromtxt(Z_file,'S')
    y=np.genfromtxt(y_file,'S')
    #For python3, reading from the genfromtxt will be \b prepended
    E=np.array([i.decode() for i in E])
    N=np.array([i.decode() for i in N])
    Z=np.array([i.decode() for i in Z])
    y=np.array([i.decode() for i in y])
    
    #load EQinfo file into array
    EQinfo=np.genfromtxt(EQinfo_file)
    
    
    #train-test split
    train_idx,valid_and_test_idx=train_test_split(np.arange(0,len(E)),test_size=0.3, random_state=16)
    X_train_E=E[train_idx]
    X_train_N=N[train_idx]
    X_train_Z=Z[train_idx]
    y_train=y[train_idx]
    EQinfo_train=EQinfo[train_idx]
    
    X_valid_test_E=E[valid_and_test_idx]
    X_valid_test_N=N[valid_and_test_idx]
    X_valid_test_Z=Z[valid_and_test_idx]
    y_valid_test=y[valid_and_test_idx]
    EQinfo_valid_test=EQinfo[valid_and_test_idx]
    #Split the valid+test again to 0.2 and 0.1
    valid_idx,test_idx=train_test_split(np.arange(0,len(valid_and_test_idx)),test_size=0.1/0.3, random_state=16) #0.1 out of 0.3 is the testing dataset; 0.2 out of 0.3 is the validation
    
    X_valid_E=X_valid_test_E[valid_idx]
    X_valid_N=X_valid_test_N[valid_idx]
    X_valid_Z=X_valid_test_Z[valid_idx]
    y_valid=y_valid_test[valid_idx]
    EQinfo_valid=EQinfo_valid_test[valid_idx]
    
    X_test_E=X_valid_test_E[test_idx]
    X_test_N=X_valid_test_N[test_idx]
    X_test_Z=X_valid_test_Z[test_idx]
    y_test=y_valid_test[test_idx]
    EQinfo_test=EQinfo_valid_test[test_idx]
    #print('Total training data, labels=',len(X_train_E),len(y_train))
    #print(X_train_E,y_train)
    #print('Total validation data, labels=',len(X_valid_E),len(y_valid))
    #print(X_valid_E,y_valid)
    #print('Total test data, labels=',len(X_test_E),len(y_test))
    #print(X_test_E,y_test)
    
    #Build structure
    HP=train_params['Neurons']
    epochs=train_params['epochs']
    Drops=train_params['Drops']
    BS=train_params['BS'] #Batch Size for training
    BS_valid=train_params['BS_valid'] #batch size for validation
    BS_test=train_params['BS_test'] #batch size for testing
    scales=train_params['scales'] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by log10(X)
    Testnum=train_params['Testnum']
    #FlatY=False #just change y='flat' in the y input
    NoiseP=train_params['NoiseP'] #possibility of noise event
    Noise_level=train_params['Noise_level']
    rm_stans=train_params['rm_stans'] #remove number of stations from 0~115
    Min_stan_dist=train_params['Min_stan_dist'] #minumum 4 stations within 3 degree
    Loss_func=train_params['Loss_func'] #can be default loss function string or self defined loss
    
    #add a tensorflow callback
    logdir = "logs/scalars/Test"+Testnum+'_'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    
    if Model_path=='Lin2020':
        network=tf.keras.models.load_model(mlarge.__path__[0].replace('src/mlarge','models/Test81_weights.49475-0.000131.hdf5'),compile=False)
    else:
        network=tf.keras.models.load_model(Model_path,compile=False)

    #build generator
    Dpath='Path_defined_in_file'
    #print('Training_X inp:',X_train_E)
    #print('Training_y inp:',y_train)
    gtrain=feature_gen(Dpath,X_train_E,X_train_N,X_train_Z,y_train,EQinfo_train,STAinfo,Nstan=121,add_code=True,add_noise=True,noise_p=NoiseP,rmN=(rm_stans[0],rm_stans[1]),Noise_level=Noise_level,Min_stan_dist=Min_stan_dist,scale=(scales[0],scales[1]),BatchSize=BS,Mwfilter=7.0,save_ID=False,shuffle=True) #Use the "flat y"
    gvalid=feature_gen(Dpath,X_valid_E,X_valid_N,X_valid_Z,y_valid,EQinfo_valid,STAinfo,Nstan=121,add_code=True,add_noise=True,noise_p=NoiseP,rmN=(rm_stans[0],rm_stans[1]),Noise_level=Noise_level,Min_stan_dist=Min_stan_dist,scale=(scales[0],scales[1]),BatchSize=BS_valid,Mwfilter=7.0,save_ID='Run%s_valid_EQID.npy'%(Testnum),shuffle=True) #Use the "flat y"
    gtest=feature_gen(Dpath,X_test_E,X_test_N,X_test_Z,y_test,EQinfo_test,STAinfo,Nstan=121,add_code=True,add_noise=True,noise_p=NoiseP,rmN=(rm_stans[0],rm_stans[1]),Noise_level=Noise_level,Min_stan_dist=Min_stan_dist,scale=(scales[0],scales[1]),BatchSize=BS_test,Mwfilter=7.0,save_ID='Run%s_test_EQID.npy'%(Testnum),shuffle=True) #Use the "flat y"
    
    #check file/dir exist,otherwise mkdir
    if not(os.path.exists('./Test'+Testnum)):
        os.makedirs('./Test'+Testnum)
    #Add callback
    CB=keras.callbacks.ModelCheckpoint(filepath='./Test'+Testnum+'/weights.{epoch:04d}-{val_loss:.6f}.hdf5',monitor='val_loss',save_best_only=True,mode='min',period=5)

    #print('Start generating validation data')
    X_valid_out,y_valid_out=gvalid.__getitem__(1)
    np.save('Xvalid'+Testnum+'.npy',X_valid_out)
    np.save('yvalid'+Testnum+'.npy',y_valid_out)

    #Also save the testing data
    #print('Start generating testing data')
    X_test_out,y_test_out=gtest.__getitem__(1)
    np.save('Xtest'+Testnum+'.npy',X_test_out)
    np.save('ytest'+Testnum+'.npy',y_test_out)

    #start training
    model_hist=network.fit_generator(gtrain,validation_data=(X_valid_out,y_valid_out),use_multiprocessing=True,workers=40,validation_steps=1,steps_per_epoch=1,epochs=epochs,callbacks=[CB,tensorboard_callback]) #so that total steps=1+7=8
    
    #save training result and training curve
    tf.keras.models.save_model(network,'./Test'+Testnum+'.h5py')
    np.save('./Test'+Testnum+'.npy',model_hist.history)












