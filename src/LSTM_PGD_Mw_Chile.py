#This model is to predict the final Mw by time evolving PGD
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
import os
import glob
# from sklearn.preprocessing import StandardScaler
from tensorflow.keras import models
from tensorflow.keras import layers
#from MudpyRand.hfsims import windowed_gaussian,apply_spectrum
#from MudpyRand.forward import gnss_psd
from mudpy.hfsims import windowed_gaussian,apply_spectrum
from mudpy.forward import gnss_psd
import obspy
#from mudpy import viewFQ
#from scipy.integrate import cumtrapz
import sys
import datetime


def D2PGD(data):
    PGD=[]
    if np.ndim(data)==2:
        for i in range(data.shape[0]):
            PGD.append(np.max(data[:i+1,:],axis=0))
    elif np.ndim(data)==1:
        for i in range(len(data)):
            PGD.append(np.max(data[:i+1]))
    
    PGD=np.array(PGD)
    return(PGD)


def M02Mw(M0):
    Mw=(2.0/3)*np.log10(M0*1e7)-10.7 #M0 input is dyne-cm, convert to N-M by 1e7
    return(Mw)

def mapping_Mw(mag,num_step,fill_type=1):
    #fill_type=1: set label as a flat plateau to final Mw from 0 sec to final sec
    #fill_type=2: Mw smoothly increase from 0-Mw_final, all magnitude have the same growing time
    #fill_type=3: Mw growing time is based on a reasonable STF from Duputel et al., 2013
    large_mag=[] #A large matrix of mag
    for neq,tmpmag in enumerate(mag):
        if fill_type==1: #let all the mag to be the same (at PGD(5sec)=Mw9.0, PGD(10sec)=Mw9.0...)
            tmp=np.ones([num_step,1])*mag[neq]
        elif fill_type==2: #let all the mag smoothly increase from 0~Mw (at PGD(5sec)=Mw0, PGD(500sec)=Mw9.0 )
            X=np.array([0,num_step]);Y=np.array([0,mag[neq]]) #linear interpolate from X,Y
            tmp=np.interp(range(num_step),X,Y)
        elif fill_type==3: #interpl the label by source time function
            thalf=Tr(neq) #half duration
            t=np.arange(1,num_step+1)*5
            #find first stage (where Mw grow from 0~neq within thalf)
            idx=np.where(t<=thalf)[0]
            if len(idx)!=0:
                X1=np.array([0,idx[-1]]);Y1=np.array([0,neq])
                tmp=np.interp(range(num_step),X1,Y1)
            else:
                #idx is empty
                X1=np.array([0,num_step]);Y1=np.array([0,neq])
                tmp=np.interp(range(num_step),X1,Y1)
        elif fill_type==4: #interpl by moment
            t=np.arange(1,num_step+1)*5
            tdur=Tr(neq)*2 #total duration
            M0=Mw2M0(neq)
            H=2*M0/tdur #Height of the triangular function
            h=t*H*2/tdur #Height of the triangular function at time t
            area=2*(t*h/2) #know t*h/2, and because symmetric*2
            tmp=M02Mw(area)
            idx=np.where(t>=tdur/2)[0]
            tmp[idx]=neq
        
        large_mag.append(tmp)
    large_mag=np.array(large_mag)
    large_mag=large_mag.reshape(len(mag),num_step,1)
    return(large_mag)


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


def get_mw(logfile):
    #Input log file path from the rupture directory
    IN1=open(logfile,'r')
    for line in IN1.readlines():
        if 'Actual magnitude' in line:
            mw=float(line.split()[3])
            break    
    IN1.close()
    return mw


###########dealing with stations ordering and find every station locations#############
sta_loc_file='../data/Chile_GNSS.gflist'   #Full GFlist file
station_order_file='../data/ALL_staname_order.txt' #this is the order in training (PGD_Chile_3400.npy)

STAINFO=np.genfromtxt(sta_loc_file,'S12',skip_header=1)
STAINFO={ista[0].decode():[float(ista[1]),float(ista[2])]  for ista in STAINFO}
STA=np.genfromtxt(station_order_file,'S6')
STA={ista:STAINFO[sta.decode()] for ista,sta in enumerate(STA)}

#simple test:
#STA[17] check the 18th station's(according to ALL_staname_order.txt) stlon,stlat from the .gflist
#####################################################################################



#######Generator should inherit the "Sequence" class in order to run multi-processing of fit_generator###########
class feature_gen(keras.utils.Sequence):
    def __init__(self,Dpath,E_path,N_path,Z_path,y,Nstan=121,add_code=True,add_noise=True,noise_p=0.5,rmN=(10,110),Noise_level=[1,10,20,30,40,50,60,70,80,90],Min_stan_dist=[4,3],scale=(0,1),BatchSize=128,Mwfilter=8.0,save_ID='sav_pickedID_1_valid.npy',shuffle=True):
        self.Dpath=Dpath #The path of the individual [E/N/U].npy data (should be a list or a numpy array)
        self.E_path=E_path #The path of the individual [E/N/U].npy data (should be a list or a numpy array)
        self.N_path=N_path
        self.Z_path=Z_path
        self.y=y
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
        np.random.seed()
        RNDID=np.random.rand()
        #if self.BatchSize==32:
        #    print('Making Test data start:%7.5f'%(RNDID))
        #else:
        #    print('Making Training start:%7.5f'%(RNDID))
        '''
        X:data (feature matrix)
        y:label
        Nstan:number of stations
        add_code: do you want to add station existence code?
        add_nose:do you want to add color noise on the X?
        noise_p: possibility of the output is noise data (i.e. generate data with Mw=0)
        rmN: remove stations from these two number (10,110) or rmN=(0,0) for not remove
        scale: scale the added noise (if any) to the same scale as Normalization (i.e. PGD_mean,PGD_var), if not scale, simply set scale=(0,1)
        index is useless here since I want every batches to be different
        '''
        Dpath,E,N,Z,y,Nstan,add_code,add_noise,noise_p,rmN,level,Min_stan_dist,scale,BatchSize,Mwfilter,save_ID,shuffle=(self.Dpath,self.E_path,self.N_path,self.Z_path,self.y,self.Nstan,self.add_code,self.add_noise,self.noise_p,self.rmN,self.Noise_level,self.Min_stan_dist,self.scale,self.BatchSize,self.Mwfilter,self.save_ID,self.shuffle)
        #level=[1,10,20,30,40,50,60,70,80,90] #shuffle and take only the first one
        #level=[1,10,20,30,40,50] #shuffle and take only the first one
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
                #E,N,Z are already a muge matrix
                #Station existence code, generally doesn't matter but you want the value it to be smaller
                Data=np.ones([E[0].shape[0],int(Nstan*2)]) #filling the data matrix, set the status code as zero when remove station
                #Data=0.5*np.ones([E[0].shape[0],int(Nstan*2)]) #filling the data matrix, set the status code as zero when remove station
            else:
                #read the data from Directory, now the E/N/Z should be EQids (e.g. '002356')
                #Dpath=/projects/tlalollin/jiunting/Fakequakes/run/Chile_27200_ENZ   Chile_full.002709.Z.npy
                #test_read=glob.glob(Dpath+'/'+'Chile_full.'+E[0]+'.Z.npy')
                #test_read=np.load(test_read[0])
                test_read=np.load(E[0])
                Data=np.ones([test_read.shape[1],int(Nstan*2)]) #filling the data matrix, set the status code as zero when remove station
                #Data=0.5*np.ones([test_read.shape[1],int(Nstan*2)]) #filling the data matrix, set the status code as zero when remove station
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
                    logfile='/projects/tlalollin/jiunting/Fakequakes/'+pre_pend+'/output/ruptures/subduction.'+real_EQid+'.log'
                    #logfile='/projects/tlalollin/jiunting/Fakequakes/Chile_full/output/ruptures/subduction.'+E[int(rndEQidx[0])]+'.log'
                    if not Mwfilter:
                        break
                    checkMw=get_mw(logfile)
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
                    #tmp_N=glob.glob(Dpath+'/'+'Chile_full.'+N[int(rndEQidx[0])]+'.N.npy')
                    #tmp_Z=glob.glob(Dpath+'/'+'Chile_full.'+Z[int(rndEQidx[0])]+'.Z.npy')
                    #tmp_E=np.load(tmp_E[0])
                    #tmp_N=np.load(tmp_N[0])
                    #tmp_Z=np.load(tmp_Z[0])
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
                eqlon,eqlat=get_hypo(logfile)
                #print('ID=%s eqlon,eqlat=%f %f'%(E[int(rndEQidx[0])],eqlon,eqlat))
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
                if not (y is None):
                    #1.use the "flat" label (assuming strong determinism)
                    y_batch.append(y[int(rndEQidx[0])].reshape(-1,1)) #the flat label
                else:
                    #2.none-determinism
                    #_t,sumMw=get_accM0(E[int(rndEQidx[0])]) #E[int(rndEQidx[0])] is the eqID (e.g. '002340')
                    #sumMw=np.load('/projects/tlalollin/jiunting/Fakequakes/run/Chile_27200_STF/Chile_full.'+E[int(rndEQidx[0])]+'.npy') #or directly loaded from .npy file
                    #sumMw=np.load('/projects/tlalollin/jiunting/Fakequakes/run/Chile_27200_STF/Chile_full.'+real_EQid+'.npy') #or directly loaded from .npy file
                    sumMw=np.load('/projects/tlalollin/jiunting/Fakequakes/run/Chile_full_new_STF/Chile_full_new.'+real_EQid+'.npy') #or directly loaded from .npy file
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



#gtest=feature_gen(E[:2000],N[:2000],Z[:2000],y[:2000],Nstan=121,add_code=True,add_noise=True,noise_p=0.0,rmN=(0,0),scale=(0,1),BatchSize=32,shuffle=True)
#gtrain=feature_gen(E[2000:],N[2000:],Z[2000:],y[2000:],Nstan=121,add_code=True,add_noise=True,noise_p=0.0,rmN=(0,0),scale=(0,1),BatchSize=128,shuffle=True)
#gg=g.__getitem__(1)




'''
test the generator

E=np.load('./Data/E_Chile_3400.npy')
N=np.load('./Data/N_Chile_3400.npy')
Z=np.load('./Data/Z_Chile_3400.npy')
y=np.load('./Data/Mw_Chile_3400.npy')
y=mapping_Mw(y,102,fill_type=1)[:,:,0]

g=feature_gen(E,N,Z,y,Nstan=121,add_code=True,add_noise=True,noise_p=0.5,rmN=(0,110),scale=(0,1),BatchSize=256)
X_test,y_test=next(g)

plt.plot(X_test[0],'k')
D=(E[0]**2+N[0]**2+Z[0]**2)**0.5
PGD=D2PGD(D)
plt.plot(PGD,'r')
plt.show()
'''


#########Load features and labels##########
#sav_all=np.load('./PGD_Chile_3400.npy') #feature matrix [Num_Eqs,Num_time_steps,PGD at diff stations]
#sav_mag=np.load('./Mw_Chile_3400.npy') #label array [final Mws]

'''
E=np.load('./Data/E_Chile_3400.npy')
N=np.load('./Data/N_Chile_3400.npy')
Z=np.load('./Data/Z_Chile_3400.npy')
'''

'''
#Old loading method
n_senarios=27200
EQids=np.array(['%06d'%(i) for i in range(n_senarios)])
E=EQids.copy()
N=EQids.copy()
Z=EQids.copy()
All_Mw=np.load('/projects/tlalollin/jiunting/Fakequakes/run/Chile_27200_Mw.npy')
n_steps=102 #number of time steps
n_stan=121 #number of stations
'''


#The new method allows input data from different directory
'''
#-------------------------------------------------
#Merging Large+small Mw events
Mw_large=np.load('/projects/tlalollin/jiunting/Fakequakes/run/Chile_27200_Mw.npy') #Mw for 27200 dataset (M7.5~9.6)
Mw_small=np.load('/projects/tlalollin/jiunting/Fakequakes/run/Chile_small_Mw.npy') #Mw for 9600 dataset (M7.2~7.7)
Mw_small_2=np.load('/projects/tlalollin/jiunting/Fakequakes/run/Chile_small_2_Mw.npy') #More Mw for 96000 dataset (M7.2~7.7)
#All_Mw=np.hstack([Mw_large,Mw_small])
All_Mw=np.hstack([Mw_large,Mw_small,Mw_small_2])
E=np.genfromtxt('/projects/tlalollin/jiunting/Fakequakes/run/E_allEQlist.txt','S')
N=np.genfromtxt('/projects/tlalollin/jiunting/Fakequakes/run/N_allEQlist.txt','S')
Z=np.genfromtxt('/projects/tlalollin/jiunting/Fakequakes/run/Z_allEQlist.txt','S')
#For python3, reading from the genfromtxt will be \b prepended
E=np.array([i.decode() for i in E])
N=np.array([i.decode() for i in N])
Z=np.array([i.decode() for i in Z])
#-------------------------------------------------
'''


#-------------------------------------------------
#Only large Mw events
#Mw_large=np.load('/projects/tlalollin/jiunting/Fakequakes/run/Chile_27200_Mw.npy') #Mw for 27200 dataset (M7.5~9.6)
Mw_large=np.load('/projects/tlalollin/jiunting/Fakequakes/run/Chile_full_new_Mw.npy') #Mw for 27200 dataset (M7.5~9.6)
#Mw_small=np.load('/projects/tlalollin/jiunting/Fakequakes/run/Chile_small_Mw.npy') #Mw for 9600 dataset (M7.2~7.7)
#Mw_small_2=np.load('/projects/tlalollin/jiunting/Fakequakes/run/Chile_small_2_Mw.npy') #More Mw for 96000 dataset (M7.2~7.7)
#All_Mw=np.hstack([Mw_large,Mw_small])
All_Mw=Mw_large
#E=np.genfromtxt('/projects/tlalollin/jiunting/Fakequakes/run/E_LargeEQlist.txt','S')
#N=np.genfromtxt('/projects/tlalollin/jiunting/Fakequakes/run/N_LargeEQlist.txt','S')
#Z=np.genfromtxt('/projects/tlalollin/jiunting/Fakequakes/run/Z_LargeEQlist.txt','S')
E=np.genfromtxt('/projects/tlalollin/jiunting/Fakequakes/run/E_full_newEQlist.txt','S')
N=np.genfromtxt('/projects/tlalollin/jiunting/Fakequakes/run/N_full_newEQlist.txt','S')
Z=np.genfromtxt('/projects/tlalollin/jiunting/Fakequakes/run/Z_full_newEQlist.txt','S')
#For python3, reading from the genfromtxt will be \b prepended
E=np.array([i.decode() for i in E])
N=np.array([i.decode() for i in N])
Z=np.array([i.decode() for i in Z])
#-------------------------------------------------


'''
All_Mw=np.load('/projects/tlalollin/jiunting/Fakequakes/run/Chile_27200_Mw.npy') #Mw for 27200 dataset (M7.5~9.6)
#Mw_small=np.load('/projects/tlalollin/jiunting/Fakequakes/run/Chile_small_Mw.npy') #Mw for 9600 dataset (M7.2~7.7)
#All_Mw=np.hstack([Mw_large,Mw_small])
E=np.genfromtxt('/projects/tlalollin/jiunting/Fakequakes/run/E_LargeEQlist.txt','S')
N=np.genfromtxt('/projects/tlalollin/jiunting/Fakequakes/run/N_LargeEQlist.txt','S')
Z=np.genfromtxt('/projects/tlalollin/jiunting/Fakequakes/run/Z_LargeEQlist.txt','S')
#For python3, reading from the genfromtxt will be \b prepended
E=np.array([i.decode() for i in E])
N=np.array([i.decode() for i in N])
Z=np.array([i.decode() for i in Z])
'''

n_steps=102 #number of time steps
n_stan=121 #number of stations

#Expand the label to time evolving Mw by different assumptions.......
 #1. Here, based on strong determinism (i.e. 5 sec is enough to tell M7 and M9)
#print('Use flat Mw')
y=mapping_Mw(All_Mw,n_steps,fill_type=1)[:,:,0]
#y=y/10.0 # scale the label so that LSTM can better predict the Step function!!!!!!!!!!!!!!!!!!!!!!!!!!!
 #2. label the y based on the real STF, it will read the .rupt so don't worry about the y

'''
##########Split train and test dataset as id this is the old one###############
train_idx,test_idx=train_test_split(np.arange(0,len(All_Mw)),test_size=0.2, random_state=16)
X_train_E=E[train_idx]
X_train_N=N[train_idx]
X_train_Z=Z[train_idx]
y_train=y[train_idx]

X_test_E=E[test_idx]
X_test_N=N[test_idx]
X_test_Z=Z[test_idx]
y_test=y[test_idx]
#########Load features and labels END##########
'''


##########Split train and test dataset as id this is the old one###############
train_idx,valid_and_test_idx=train_test_split(np.arange(0,len(All_Mw)),test_size=0.3, random_state=16)
X_train_E=E[train_idx]
X_train_N=N[train_idx]
X_train_Z=Z[train_idx]
y_train=y[train_idx]

X_valid_test_E=E[valid_and_test_idx]
X_valid_test_N=N[valid_and_test_idx]
X_valid_test_Z=Z[valid_and_test_idx]
y_valid_test=y[valid_and_test_idx]
#Split the valid+test again to 0.2 and 0.1
valid_idx,test_idx=train_test_split(np.arange(0,len(valid_and_test_idx)),test_size=0.1/0.3, random_state=16) #0.1 out of 0.3 is the testing dataset; 0.2 out of 0.3 is the validation


X_valid_E=X_valid_test_E[valid_idx]
X_valid_N=X_valid_test_N[valid_idx]
X_valid_Z=X_valid_test_Z[valid_idx]
y_valid=y_valid_test[valid_idx]

X_test_E=X_valid_test_E[test_idx]
X_test_N=X_valid_test_N[test_idx]
X_test_Z=X_valid_test_Z[test_idx]
y_test=y_valid_test[test_idx]


#########Load features and labels END##########
print('Total training data=',len(X_train_E))
print('Total validation data=',len(X_valid_E))
print('Total test data=',len(X_test_E))


#######Test if the random events are always the same first (reproducable)######
#np.save('EQID_train_73.txt',train_idx)
#np.save('EQID_valid_73.txt',valid_and_test_idx[valid_idx])
#np.save('EQID_test_73.txt',valid_and_test_idx[test_idx])

#import sys
#sys.exit(2)


#Norm the features by "training data"
#PGD_mean=X_train.mean()
#PGD_var=X_train.std()
#X_train_norm=(X_train-PGD_mean)/PGD_var
#X_test_norm=(X_test-PGD_mean)/PGD_var

#Do NOT norm the features since you'll add more information later
#X_train_norm=X_train.copy()
#X_test_norm=X_test.copy()


'''
Now, the X,y dataset are
  Training:
    features:X_train_[E,N,Z]
    labels:y_train
  Testing:
    features:X_test_[E,N,Z]
    labels:y_test
'''
#This is not for training yet!, use feature_gen to make PGD+noise.
#Test the generator and show example of what it looks like
'''
gen=feature_gen(X_train_E,X_train_N,X_train_Z,y_train,Nstan=121,add_code=True,add_noise=True,noise_p=0.5,rmN=(10,110),scale=(0,1),BatchSize=128)
testX,testy=next(gen)
testid=123
#plt.pcolor(testX[testid].transpose(),vmin=0,vmax=2,cmap=plt.cm.seismic)
plt.pcolor(testX[testid].transpose(),cmap=plt.cm.seismic)
plt.xlabel('Time steps',fontsize=15)
plt.ylabel('PGD,code',fontsize=15)
plt.title('Generator example Mw=%3.1f'%(testy[testid][0]),fontsize=15)
plt.colorbar()
plt.show()
'''


import tensorflow.keras.backend as K


def weight_mse(y_true,y_pred):
    #ones = K.ones_like(y_true[0,:]) #array vector with ones shaped
    #idx = K.cumsum(ones) #weights from 1~len(y_true)
    #idx=np.arange(1,len(y_true)+1)
    #idx = K.reverse(K.cumsum(ones),axes=0) #reverse the order so that "1/idx" become -> small~large, the later data has higher weight
    idx=K.arange(1,K.shape(y_true)[1]+1,dtype='float32') #the 0-th sample, ? time steps
    idx = K.reverse(1/idx,axes=0) #reverse the order so that "1/idx" become -> small~large, the later data has higher weighting
    dif=K.square(y_pred - y_true) #differences in every samples, accross time steps it should be [batch_size,102,1]
    difT=K.transpose(dif[:,:,0]) #transpose so that can be dotted  [102,bs]
    idxT=K.reshape(idx,[1,-1]) #matrix of the idx (weighting)  [1,102] mul [102,bs] = [1,bs]
    return K.mean(K.dot(idxT,difT))
    #return K.mean(K.square(1.0/idx)*K.square(y_pred[:,0] - y_true[:,0]) )
    #return idx[0,:]+y_true[:,1]*0.0+y_pred[:,0]*0.0
    #return K.mean(idx) + K.square(y_pred[:,0] - y_true[:,0])*0.0

def weight_mae(y_true,y_pred):
    #ones = K.ones_like(y_true[0,:]) #array vector with ones shaped
    #idx = K.cumsum(ones) #weights from 1~len(y_true)
    #idx=np.arange(1,len(y_true)+1)
    #idx = K.reverse(K.cumsum(ones),axes=0) #reverse the order so that "1/idx" become -> small~large, the later data has higher weight
    idx=K.arange(1,K.shape(y_true)[1]+1,dtype='float32') #the 0-th sample, ? time steps [1,102]
    idx = K.reverse(1/idx,axes=0) #reverse the order so that "1/idx" become -> small~large, the later data has higher weighting
    dif=K.abs(y_pred - y_true) #differences in every samples, accross time steps it should be [batch_size,102,1]
    difT=K.transpose(dif[:,:,0]) #transpose so that can be dotted  [102,bs]
    idxT=K.reshape(idx,[1,-1]) #matrix of the idx (weighting)  [1,102] mul [102,bs] = [1,bs]
    return K.mean(K.dot(idxT,difT))


def weightMw_mse(y_true, y_pred):
    #if not K.is_tensor(y_pred):
    #    y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    #return K.mean((K.square(y_pred - y_true) * (y_true + 0.1))/K.mean(y_true + 0.1,axis=-1), axis=-1) #dim need to figure out
    #return K.mean((K.square(y_pred - y_true) * (y_true + 0.1)), axis=-1) #Test41
    #return K.mean((K.square(y_pred - y_true) * (y_true + 1.0)), axis=-1) #Test42
    #return K.mean((K.square(y_pred - y_true) * (y_true + 2.0)), axis=-1) #Test43
    #return K.mean((K.square(y_pred - y_true) * (y_true + 5.0)), axis=-1) #Test44
    #return K.mean((K.square(y_pred - y_true) * (y_true + 2.0)), axis=-1) #Test45
    #return K.mean((K.square(y_pred - y_true) * (y_true + 2.0)), axis=-1) #Test46
    return K.mean((K.square(y_pred - y_true) * (y_true + 2.0)), axis=-1) #Test47, and #48


def weightMw_time_mse(y_true, y_pred):
    #if not K.is_tensor(y_pred):
    #    y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    idx=K.arange(1,K.shape(y_true)[1]+1,dtype='float32') #the 0-th sample, ? time steps [1,102]
    #max_idx=102 thus sqrt(102)~10 and the last data is 10 times important than the first point
    #idx = K.reverse(1/(idx)**0.5,axes=0) #reverse the order so that "1/idx" become -> small~large, the later data has higher weighting ###For test #54
    #idx = K.reverse(10/(idx),axes=0) #reverse the order so that "1/idx" become -> small~large, the later data has higher weighting  ###For test #55
    #idx = K.reverse(100/(idx),axes=0) #reverse the order so that "1/idx" become -> small~large, the later data has higher weighting  ###For test #56,#57
    idx =  1 - (K.exp(-1*idx/2)) # become -> small~large, the later data has higher weighting  ###For test #58
    idx = K.reshape(idx,(102,1))
    #return K.mean((K.square(y_pred - y_true) * (y_true + 0.1))/K.mean(y_true + 0.1,axis=-1), axis=-1) #dim need to figure out
    #return K.mean((K.square(y_pred - y_true) * (y_true + 0.1)), axis=-1) #Test41
    #return K.mean((K.square(y_pred - y_true) * (y_true + 1.0)), axis=-1) #Test42
    #return K.mean((K.square(y_pred - y_true) * (y_true + 2.0)), axis=-1) #Test43
    #return K.mean((K.square(y_pred - y_true) * (y_true + 5.0)), axis=-1) #Test44
    #return K.mean((K.square(y_pred - y_true) * (y_true + 2.0)), axis=-1) #Test45
    #return K.mean((K.square(y_pred - y_true) * (y_true + 2.0)), axis=-1) #Test46
    #return K.mean( K.dot( (K.square(y_pred[:,:,0] - y_true[:,:,0]) * (y_true[:,:,0] + 2.0)),idx ), axis=-1) #Test47, and #48
    return K.mean( K.dot( ( K.square(y_pred[:,:,0] - y_true[:,:,0]) * (y_true[:,:,0] + 2.0)),idx ), axis=-1) #Test58, Test61


def weight_time_mse(y_true, y_pred):
    #if not K.is_tensor(y_pred):
    #    y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    idx=K.arange(1,K.shape(y_true)[1]+1,dtype='float32') #the 0-th sample, ? time steps [1,102]
    #max_idx=102 thus sqrt(102)~10 and the last data is 10 times important than the first point
    #idx = K.reverse(1/(idx)**0.5,axes=0) #reverse the order so that "1/idx" become -> small~large, the later data has higher weighting ###For test #54
    #idx = K.reverse(10/(idx),axes=0) #reverse the order so that "1/idx" become -> small~large, the later data has higher weighting  ###For test #55
    #idx = K.reverse(100/(idx),axes=0) #reverse the order so that "1/idx" become -> small~large, the later data has higher weighting  ###For test #56,#57
    idx =  1 - (K.exp(-1*idx/2)) # become -> small~large, the later data has higher weighting  ###For test #58
    idx = K.reshape(idx,(102,1))
    #return K.mean((K.square(y_pred - y_true) * (y_true + 0.1))/K.mean(y_true + 0.1,axis=-1), axis=-1) #dim need to figure out
    return K.mean( K.dot( (K.square(y_pred[:,:,0] - y_true[:,:,0]) ),idx ), axis=-1) #Test70



def weightMw_time_mae(y_true, y_pred):
    #if not K.is_tensor(y_pred):
    #    y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    idx=K.arange(1,K.shape(y_true)[1]+1,dtype='float32') #the 0-th sample, ? time steps [1,102]
    #max_idx=102 thus sqrt(102)~10 and the last data is 10 times important than the first point
    #idx = K.reverse(1/(idx)**0.5,axes=0) #reverse the order so that "1/idx" become -> small~large, the later data has higher weighting ###For test #54
    #idx = K.reverse(10/(idx),axes=0) #reverse the order so that "1/idx" become -> small~large, the later data has higher weighting  ###For test #55
    #idx = K.reverse(100/(idx),axes=0) #reverse the order so that "1/idx" become -> small~large, the later data has higher weighting  ###For test #56,#57
    idx =  1 - (K.exp(-1*idx/2)) # become -> small~large, the later data has higher weighting  ###For test #58
    idx = K.reshape(idx,(102,1))
    #return K.mean((K.square(y_pred - y_true) * (y_true + 0.1))/K.mean(y_true + 0.1,axis=-1), axis=-1) #dim need to figure out
    #return K.mean((K.square(y_pred - y_true) * (y_true + 0.1)), axis=-1) #Test41
    #return K.mean((K.square(y_pred - y_true) * (y_true + 1.0)), axis=-1) #Test42
    #return K.mean((K.square(y_pred - y_true) * (y_true + 2.0)), axis=-1) #Test43
    #return K.mean((K.square(y_pred - y_true) * (y_true + 5.0)), axis=-1) #Test44
    #return K.mean((K.square(y_pred - y_true) * (y_true + 2.0)), axis=-1) #Test45
    #return K.mean((K.square(y_pred - y_true) * (y_true + 2.0)), axis=-1) #Test46
    #return K.mean( K.dot( (K.square(y_pred[:,:,0] - y_true[:,:,0]) * (y_true[:,:,0] + 2.0)),idx ), axis=-1) #Test47, and #48
    return K.mean( K.dot( (K.abs(y_pred[:,:,0] - y_true[:,:,0]) * (y_true[:,:,0])),idx ), axis=-1) #Test58, Test61




#axis=-1
    #return K.sum(1.0/idx)+K.square(y_pred - y_true)*0.0


#from keras import backend as K
#tf_session = K.get_session()
#val = np.array([[1, 2], [3, 4]])
#kvar = K.variable(value=val)
#input = keras.backend.placeholder(shape=(2, 4, 5))
#idx2=K.arange(2,K.shape(kvar)[0]+60,dtype='float64')
#idx=K.arange(1,K.shape(kvar)[0]+5)
#idx=1/idx
#idx = K.reverse(idx,axes=0)
#
#idx2.eval(session=tf_session)
#idx.eval(session=tf_session)
#idx3=idx*idx2
#idx3.eval(session=tf_session)
#K.shape(input)
#K.shape(kvar).eval(session=tf_session)
#K.shape(input).eval(session=tf_session)

#HypoParameters
#First test :20191023:14:41
HP=[512,512,512,512,256,0.5] #Dense+Dense+LSTM+Dense+Dense+Dropout
BS=64 #Batch Size
#2 test :20191023:15:38
HP=[1024,1024,512,512,512,0.5] #Dense+Dense+LSTM+Dense+Dense+Dropout
BS=64 #Batch Size
#3 test :20191023:15:45
HP=[1024,1024,1024,512,512,0.5] #Dense+Dense+LSTM+Dense+Dense+Dropout
BS=64 #Batch Size
#4 test :20191023:15:46
HP=[1024,1024,1024,512,512,0.5] #Dense+Dense+LSTM+Dense+Dense+Dropout
BS=128 #Batch Size
#5 test :20191028:14:01
HP=[1024,1024,1024,512,512] #Dense+Dense+Dropout+LSTM+Dense+Dense+Dropout
Drops=[0.2,0.2]
BS=128 #Batch Size
#6 test :20191028:15:58
HP=[2048,2048,2048,1024,1024] # very large! Dense+Dense+Dropout+LSTM+Dense+Dense+Dropout but only trained by 1000 epochs
Drops=[0.2,0.2]
BS=128 #Batch Size
#7 test :20191029:15:36
HP=[1024,1024,1024,512,512] # make sure keep at least 3 near-field (<5.0 deg) for EQ.  Dense+Dense+Dropout+LSTM+Dense+Dense+Dropout
Drops=[0.2,0.2]
BS=128 #Batch Size
#8 test :20191031:23:08
HP=[1024,1024,512,256,256] # make sure keep at least 5 near-field (<5.0 deg) for EQ.  Dense+Dense+Dropout+LSTM+Dense+Dense+Dropout
Drops=[0.3,0.3]
BS=128 #Batch Size
#9 test :20191111:11:08
HP=[1024,1024,512,256,256] # make sure keep at least 5 near-field (<5.0 deg) for EQ.  Dense+Dense+Dropout+LSTM+Dense+Dense+Dropout
Drops=[0.3]
BS=128 #Batch Size

#10 test: 20191111:1130
HP=[512,512,512] #LSTM+LSTM+Dense
Drops=[0.3]
BS=128 #Batch Size

#11 test: 20191111:1200
HP=[512,512,256] #LSTM+LSTM+LSTM
Drops=[0.3]
BS=128 #Batch Size

#12 test: 20191112:1231
#Use the real STF for Mw labeling
#scale y by y=y/10.0
#scale X by (X-5)/10.0
HP=[512,512,512] #LSTM+LSTM+Dense(relu)+output Dense
Drops=[0.5]
BS=128 #Batch Size


#13 test: 20191112:1930
#Use the real STF for Mw labeling
#scale y by y=y/10.0
#scale X by (X-5)/10.0
#The station existence code is 0 and 0.5 (instead of 1)
HP=[512,512,512] #LSTM+LSTM+Dense(relu)+output Dense
Drops=[0.5]
BS=128 #Batch Size


#14 test: 20191112:1935
#Use the real STF for Mw labeling
#scale y by y=y/10.0
#scale X by (X-5)/10.0
#Activation of the Dense to be tanh, output to be relu
#The station existence code is 0 and 0.5 (instead of 1)
HP=[512,512,512] #LSTM+LSTM+Dense(tanh)+output Dense(relu)
Drops=[0.5]
BS=128 #Batch Size

#15 test: 20191112:2000
#Use the real STF for Mw labeling
#scale y by y=y/10.0
#scale X by (X-5)/10.0
#Activation of the Dense to be tanh, output to be relu
#The station existence code is 0 and 0.5 (instead of 1)
#use BS=???,epoch=5000
HP=[128,256,512,512] #LSTM+LSTM+Dense(tanh)+Dense(tanh)+Droup(0.5)+output Dense(relu)
Drops=[0.5]
BS=64 #Batch Size


#16 test: 20191113:1118
#By comparision of the #13,#14 the relu in #13 is probably better than tanh in #14
#This experiment is to compare the #13. but with a different X-scaling. Now, the scaling of X=(X-30)/30
#Use the real STF for Mw labeling
#scale y by y=y/10.0
#scale X by (X-30)/30.0 #make a different scaling
#Activation of the Dense is relu, no activation for output dense
#The station existence code is 0 and 0.5 (instead of 1)
#use BS=128,epoch=2000
HP=[512,512,512] #LSTM+LSTM+Dense(relu)+Droup(0.5)+output Dense
Drops=[0.5]
BS=128 #Batch Size

#17 test: 20191113:1118
#Similar to #16 but add a Callback
#By comparision of the #13,#14 the relu in #13 is probably better than tanh in #14
#This experiment is to compare the #13. but with a different X-scaling. Now, the scaling of X=(X-30)/30
#Use the real STF for Mw labeling
#scale y by y=y/10.0
#scale X by (X-20)/20.0 #make a different scaling
#Activation of the Dense is relu, no activation for output dense
#The station existence code is 0 and 0.5 (instead of 1)
#use BS=128 for training,epoch=5000
HP=[512,512,512] #LSTM+LSTM+Dense(relu)+Droup(0.5)+output Dense
Drops=[0.5]
BS=128 #Batch Size, 
###validation BS set to 1024, so there's more data and unlikely to lucky!

#18 test: 20191114:1150
#Things become weird. So back to the original large sandwitch model
#same as #17, at least 5 stations close to hypocenter <=5.0deg 
#Use the real STF for Mw labeling
#scale y by y=y/10.0
#scale X by (X-20)/20.0 #make a different scaling
#Activation of the Dense is relu, no activation for output dense
#The station existence code is 0 and 0.5 (instead of 1)
#use BS=128 for training,epoch=5000
HP=[1024,1024,1024,512,512] #Dense(relu)+Dense(relu)+LSTM(1024)+Dense(relu)+Dense(relu)+Droup(0.5)+output Dense
Drops=[0.5]
BS=128 #Batch Size, 
###validation BS set to 1024, so there's more data and unlikely to lucky!

#Found a bug when scaling the X... so before the #19 (#11~18 needs to be rerun)
#19 test: 20191114:1203
#Things become weird. So back to the original large sandwitch model, and use 2-LSTM
#same as #18, at least 5 stations close to hypocenter <=5.0deg 
#Use the real STF for Mw labeling
#scale y by y=y/10.0
#scale X by (X-20)/20.0 #make a different scaling
#Activation of the Dense is relu, no activation for output dense
#The station existence code is 0 and 0.5 (instead of 1)
#use BS=128 for training,epoch=5000
HP=[1024,1024,1024,512,512] #Dense(relu)+Dense(relu)+LSTM(1024)+LSTM(512)+Dense(relu)+Droup(0.5)+output Dense
Drops=[0.5]
BS=128 #Batch Size, 
###validation BS set to 1024, so there's more data and unlikely to lucky!


#Re-run 13 test: 20191114:1930 since I found a bug in when run#18, after #19 is corrected!
#But modify the scaling 
#Use the real STF for Mw labeling
#scale y by y=y/10.0
#scale X by (X-20)/20.0
#The station existence code is 0 and 0.5 (instead of 1)
HP=[512,512,512] #LSTM+LSTM+Dense(relu)+output Dense
Drops=[0.5]
BS=128 #Batch Size
BS_test=1024 #batch size for validation
scales=[20,20.0] #(x-scaels[0])/scales[1]
#if this doesn't work then change back to the (X-5)/10


#Re-Run 14 test: 20191114:1330
#But structure the same as in #13 (use relu)
#Use the real STF for Mw labeling
#scale y by y=y/10.0
#scale X by (X-5)/10.0 #note the scaling!
#The station existence code is 0 and 0.5 (instead of 1)
HP=[512,512,512] #LSTM+LSTM+Dense(relu)+output Dense
Drops=[0.5]
BS=128 #Batch Size
BS_test=1024 #batch size for validation
scales=[5,10.0] #(x-scaels[0])/scales[1]


#Re-Run 15 test: 20191114:1420
#Modify the scaling of sandwith model #19
#Why the mse for re-run 13,14 is very large?
#Use the real STF for Mw labeling
#scale y by y=y/10.0
#scale X by (X)/20.0 #note the scaling!
#The station existence code is 0 and 0.5 (instead of 1)
HP=[512,512,512] #LSTM+LSTM+Dense(relu)+output Dense
Drops=[0.5]
BS=128 #Batch Size for training
BS_test=1024 #batch size for validation
scales=[0,20.0] #(x-scaels[0])/scales[1]???????????????????have to check this


#Re-Run 16 test: 20191114:1440
#Modify the scaling of sandwith model two-LSTM  #19
#Why the mse for re-run 13,14 is very large?
#Use the real STF for Mw labeling
#scale y by y=y/10.0
#scale X by (X)/10.0 #note the scaling!
#The station existence code is 0 and 0.5 (instead of 1)
HP=[1024,1024,1024,512,512] #Dense(relu)+Dense(relu)+LSTM(1024)+LSTM(512)+Dense(relu)+Droup(0.5)+output Dense
Drops=[0.5]
BS=128 #Batch Size for training
BS_test=128 #batch size for validation
scales=[0,10.0] #(x-scaels[0])/scales[1]


#Re-Run 17 test: 20191114:1530
#Modify the scaling of sandwith model of rerun #16 above. one-LSTM
#Use the real STF for Mw labeling
#scale y by y=y/10.0
#scale X by (X)/10.0 #note the scaling!
#The station existence code is 0 and 0.5 (instead of 1)
HP=[1024,1024,1024,512,512] #Dense(relu)+Dense(relu)+LSTM(1024)+"Dense(relu)"+Dense(relu)+Droup(0.5)+output Dense
Drops=[0.5]
BS=128 #Batch Size for training
BS_test=128 #batch size for validation
scales=[0,10.0] #(x-scaels[0])/scales[1]


#Re-Run 13 test: 20191115:1000
#The val_loss for #13 was very high! the reason is probably because there are many negative X due to scaling
#Compare to #13 v.s. #16  I found that without a negative sign, the loss is weight lower.
#Modify the scaling of sandwith model two-LSTM  #16
#use a smaller batch
#Use the real STF for Mw labeling
#scale y by y=y/10.0
#scale X by (X)/20.0 #note the scaling from 10 to 20!
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu replace the original Relu
#use a smaller BS=64
HP=[1024,1024,1024,512,512] #Dense(Leakyrelu)+Dense(Leakyrelu)+LSTM(1024)+LSTM(512)+Dense(Leakyrelu)+Droup(0.5)+output Dense
Drops=[0.5]
BS=64 #Batch Size for training
BS_test=128 #batch size for validation
scales=[0,20.0] #(x-scaels[0])/scales[1]
Testnum='13'


#Re-Run 14 test: 20191115:1120
#Modify the scaling by applying a sqrt(data)
#The val_loss for #13 was very high! the reason is probably because there are many negative X due to scaling
#Compare to #13 v.s. #16  I found that without a negative sign, the loss is weight lower.
#Modify the scaling of sandwith model two-LSTM  #16
#use a smaller batch
#Use the real STF for Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X) 
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu replace the original Relu
#use a smaller BS=64
HP=[1024,1024,1024,512,512] #Dense(Leakyrelu)+Dense(Leakyrelu)+LSTM(1024)+LSTM(512)+Dense(Leakyrelu)+Droup(0.5)+output Dense
Drops=[0.5]
BS=64 #Batch Size for training
BS_test=128 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT
Testnum='14'

#----------------#
'''
compare to #Re-run13,14, the Sqrt(X) feature scaling is better than just X/20
'''
#----------------#


#Re-Run 19 test: 20191115:1120, since the val_loss for previous #19 is really large!
#i.e. it stop improving: epochs 0225, val_loss=0.171731
#This run is only different from the rerun #14(above) by an additional /10 of the sqrt(X)
#Modify the scaling by applying a sqrt(data) then /10
#The val_loss for #13 was very high! the reason is probably because there are many negative X due to scaling
#Compare to #13 v.s. #16  I found that without a negative sign, the loss is weight lower.
#Modify the scaling of sandwith model two-LSTM  #16
#use a smaller batch
#Use the real STF for Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is all in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu replace the original Relu
#use a smaller BS=64
HP=[1024,1024,1024,512,512] #Dense(Leakyrelu)+Dense(Leakyrelu)+LSTM(1024)+LSTM(512)+Dense(Leakyrelu)+Droup(0.5)+output Dense
#I really want to try this HP=[1024,512,1024,1024,512,512] #LSTM(1024)+LSTM(512)+Dense(Leakyrelu)+Dense(Leakyrelu)+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.5)+output Dense(Relu)
Drops=[0.5]
BS=64 #Batch Size for training
BS_test=128 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='19'

#Run 20 test: 20191117:2340, since the val_loss for all training so far have a limit of ~0.01
#Similar to rerun #19, but use only single LSTM and 2-dropout
#Modify the scaling by applying a sqrt(data) then /10
#Use the real STF for Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu replace the original Relu
#GNSS noise level[1~90], remove #stations(0~115), close stations(5 stations within 5deg)
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(1024)+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense
HP=[1024,1024,1024,512,512]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_test=128 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='20'
#GNSS noise level[1~90], remove #stations(0~115), close stations(5 stations within 5deg)
#If this still doesn't work, modify the noise level[~50], remove #stations(0~110), close stations(8 stations within 5deg) to make it more easier!!



#Run 21 test:20191117:2350
#Similar to #20, but change the training data criteria
#Modify the scaling by applying a sqrt(data) then /10
#Use the real STF for Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu replace the original Relu
#!!!GNSS noise level[1~50], remove #stations(0~110), close stations(8 stations within 5deg)
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(1024)+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense
HP=[1024,1024,1024,512,512]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_test=128 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='21'


#Run 22 test:20191117:2355
#Similar to #21, but an easier model (smaller neurons)
#Modify the scaling by applying a sqrt(data) then /10
#Use the real STF for Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu0.2 replace the original Relu
#!!!GNSS noise level[1~50], remove #stations(0~110), close stations(8 stations within 5deg)
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(1024)+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense
HP=[512,512,256,256,128]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_test=128 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='22'


#Run 23 test:20191118:1026
#Similar to #21, but an more easier model (smaller neurons)
#Modify the scaling by applying a sqrt(data) then /10
#Use the real STF for Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~50], remove #stations(0~110), close stations(8 stations within 5deg)
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128)+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense
HP=[256,256,128,128,64]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_test=128 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='23'


#Run 24 test:20191118:1120
#Similar to #22, but an easist model with only 1-LSTM
#Modify the scaling by applying a sqrt(data) then /10
#Use the real STF for Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#!!!GNSS noise level[1~50], remove #stations(0~110), close stations(8 stations within 5deg)
#use a larger BS=128, since #16 is better than #13
#LSTM(256)+Droup(0.2)+output Dense
HP=[256]
Drops=[0.2]
BS=128 #Batch Size for training
BS_test=128 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='24'


#Run 25 test:20191118:1500
#re_calculate the #16 since this is the best model, add a TimeDistributed
#Modify the scaling of sandwith model two-LSTM  #19
#Why the mse for re-run 13,14 is very large?
#Use the real STF for Mw labeling
#scale y by y=y/10.0
#scale X by (X)/10.0 #note the scaling!
#The station existence code is 0 and 0.5 (instead of 1)
#the original setting: GNSS noise level[1~90], remove #stations(0~115), close stations(5 stations within 5deg)
HP=[1024,1024,1024,512,512] #Dense(relu)+Dense(relu)+LSTM(1024)+LSTM(512)+Dense(relu)+Droup(0.5)+output Dense
Drops=[0.5]
BS=128 #Batch Size for training
BS_test=128 #batch size for validation
scales=[0,10.0] #(x-scaels[0])/scales[1]
Testnum='25'
#Time distributed did not improve the model


#Run 26 test:20191120:1010
#Everythings are the same as the #23(the best model), but with harder X features
#Modify the scaling by applying a sqrt(data) then /10
#Use the real STF for Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10.0 #note the scaling!
#The station existence code is 0 and 0.5 (instead of 1)
#the feature setting: GNSS noise level[1~50], remove #stations(0~115), close stations(5 stations within 5deg) (In Test23 was:remove #stations(0~110), close stations(8 stations within 5deg))
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128)+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense
HP=[256,256,128,128,64] 
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_test=128 #batch size for validation
scales=[0,1.0] #(x-scaels[0])/scales[1]
Testnum='26'


#Run 27 test:20191120:1010
#Everythings are the same as the #23(the best model), but with much more easier than features in Test23
#Modify the scaling by applying a sqrt(data) then /10
#Use the real STF for Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10.0 #note the scaling!
#The station existence code is 0 and 0.5 (instead of 1)
#the feature setting: GNSS noise level[1~50], remove #stations(0~100), close stations(10 stations within 5deg) (In Test23 was:remove #stations(0~110), close stations(8 stations within 5deg))
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128)+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense
HP=[256,256,128,128,64] 
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_test=128 #batch size for validation
scales=[0,1.0] #(x-scaels[0])/scales[1]
Testnum='27'


#Run 28 test:20191120:2228
#Similar to #23, but use the "flat" y
#Modify the scaling by applying a sqrt(data) then /10
#Use the flat Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~50], remove #stations(0~110), close stations(8 stations within 5deg)
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128)+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense
HP=[256,256,128,128,64]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_test=128 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='28'



#Run 29 test:20191121:1112
#Similar to #28, but use a smaller neurons
#Modify the scaling by applying a sqrt(data) then /10
#Use the flat Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~50], remove #stations(0~110), close stations(8 stations within 5deg)
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128)+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense
HP=[128,128,64,64,32]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_test=128 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='29'


#Run 30 test:20191121:1112
#Everything is the same as #29, but apply MAE (Used to mse)
#Modify the scaling by applying a sqrt(data) then /10
#Use the flat Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~50], remove #stations(0~110), close stations(8 stations within 5deg)
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128)+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense
HP=[128,128,64,64,32]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_test=128 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='30'



#Run 31 test:20191126:1850
#Everything is the same as #30, but larger neurons
#Modify the scaling by applying a sqrt(data) then /10
#Use the flat Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~50], remove #stations(0~110), close stations(8 stations within 5deg)
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128)+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense
HP=[256,256,128,128,64]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_test=128 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='31'


#Run 32 test:20191126:1900
#Everything is the same as #31, but larger neurons
#Modify the scaling by applying a sqrt(data) then /10
#Use the flat Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~50], remove #stations(0~110), close stations(8 stations within 5deg)
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128)+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense
HP=[512,512,256,128,64]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_test=128 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='32'


#Run 33 test:20191126:1915
#Everything is the same as #31, but use the "REAL" Mw
#Modify the scaling by applying a sqrt(data) then /10
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~50], remove #stations(0~110), close stations(8 stations within 5deg)
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128)+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense
HP=[256,256,128,128,64]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_test=128 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='33'
FlatY=False #not flat Mw means use the real Mw


#Run 34 test:20191126:2015
#Everything is the same as #33, but use larger neurons as #32
#Modify the scaling by applying a sqrt(data) then /10
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~50], remove #stations(0~110), close stations(8 stations within 5deg)
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128)+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense
HP=[512,512,256,128,64]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_test=128 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='34'
FlatY=False #not flat Mw means use the real Mw


#Run 35 test:20191126:1915
#Everything is the same as #33 but add a tanh in final Dense output
#Modify the scaling by applying a sqrt(data) then /10
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~50], remove #stations(0~110), close stations(8 stations within 5deg)
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128)+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense(tanh)
HP=[256,256,128,128,64]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_test=128 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='35'
FlatY=False #not flat Mw means use the real Mw



#Run 36 test:20191128:2327
#Everything is the same as #33 but without Noisedata (noise still added in the PGD)
#Modify the scaling by applying a sqrt(data) then /10
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~50], remove #stations(0~110), close stations(8 stations within 5deg)
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128)+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense(tanh)
HP=[256,256,128,128,64]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_test=128 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='36'
FlatY=False #not flat Mw means use the real Mw



#Run 37 test:20191129:0045
#Everything is the same as #35 but use weighted_mae
#Modify the scaling by applying a sqrt(data) then /10
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~50], remove #stations(0~110), close stations(8 stations within 5deg)
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128)+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense(tanh)
HP=[256,256,128,128,64]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_test=128 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='37'
FlatY=False #not flat Mw means use the real Mw


#Run 38 test:20191129:0128
#Start using stacked LSTM, weighted mae?(YES)
#Modify the scaling by applying a sqrt(data) then /10
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
######use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~50], remove #stations(0~110), close stations(8 stations within 5deg)
#LSTM+LSTM+Dense(tanh)
HP=[256,128,64]
Drops=[0.2]
BS=128 #Batch Size for training
BS_test=128 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='38'
FlatY=False #not flat Mw means use the real Mw


#Run 39 test:20191129:1252
#Start using stacked LSTM, same as #38 but smaller neuron
#Regular MAE
#Modify the scaling by applying a sqrt(data) then /10
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
######use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~50], remove #stations(0~110), close stations(8 stations within 5deg)
#LSTM+LSTM+Dense out(tanh)
HP=[128,64,32]
Drops=[0.2]
BS=128 #Batch Size for training
BS_test=128 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='39'
FlatY=False #not flat Mw means use the real Mw



#Run 40 test:20191129:1338
#using stacked LSTM, same as #39 but more dense layer after 2-LSTM
#Regular MAE
#Modify the scaling by applying a sqrt(data) then /10
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
######use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~50], remove #stations(0~110), close stations(8 stations within 5deg)
#LSTM+LSTM+Drop(0.2)+Dense(LekyRely)+Dense(LekyRely)+Dense(LekyRely)+Dense(LekyRely) +Dense out(tanh)
HP=[128,64,32,32,16,8]
Drops=[0.2]
BS=128 #Batch Size for training
BS_test=128 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='40'
FlatY=False #not flat Mw means use the real Mw



#Run 41 test:20191130:2350
#Everything is the same as #31,but use weightedMw_mse and use tanh for final Dense
#use weightMw_mse
#Modify the scaling by applying a sqrt(data) then /10
#Use the flat Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~50], remove #stations(0~110), close stations(8 stations within 5deg)
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128)+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense(tanh)
HP=[256,256,128,128,64]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_test=128 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='41'
FlatY=True



#Run 42 test:20191201:1203
#Everything is the same as #41,but use weightedMw_mse +1 and use tanh for final Dense
#use weightMw_mse, use a larger validation BS(1024)
#Modify the scaling by applying a sqrt(data) then /10
#Use the flat Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~50], remove #stations(0~110), close stations(8 stations within 5deg)
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128)+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense(tanh)
HP=[256,256,128,128,64]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_test=1024 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='42'
FlatY=True


#Run 43 test:20191201:1220
#Everything is the same as #42,but use weightedMw_mse +2 and use tanh for final Dense
#use weightMw_mse, use a larger validation BS(1024)
#Modify the scaling by applying a sqrt(data) then /10
#Use the flat Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~50], remove #stations(0~110), close stations(8 stations within 5deg)
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128)+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense(tanh)
HP=[256,256,128,128,64]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_test=1024 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='43'
FlatY=True


#Run 44 test:20191201:1220
#Everything is the same as #43,but use weightedMw_mse +5 and use tanh for final Dense
#use weightMw_mse, use a larger validation BS(1024)
#Modify the scaling by applying a sqrt(data) then /10
#Use the flat Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~50], remove #stations(0~110), close stations(8 stations within 5deg)
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128)+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense(tanh)
HP=[256,256,128,128,64]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_test=1024 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='44'
FlatY=True



#Run 45 test:20191206:1320
#Everything is the same as #43,use weightedMw_mse +2, add 2 more dense layer and, recurrent dropout and use tanh for final Dense
#use weightMw_mse, use a larger validation BS(1024)
#Modify the scaling by applying a sqrt(data) then /10
#Use the flat Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~50], remove #stations(0~110), close stations(8 stations within 5deg)
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128)+Dense(Leakyrelu)+Dense(Leakyrelu)+Dense(LeakyRelu)+Dense(LeakyRelu)+Droup(0.2)+output Dense(tanh)
HP=[256,256,128,128,64,32,8]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_test=1024 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='45'
FlatY=True


#Run 46 test:20200123:1540
#Everything is the same as #45, but ONLY EQ, no noise data!
#use weightMw_mse, use a larger validation BS(1024)
#Modify the scaling by applying a sqrt(data) then /10
#Use the flat Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~50], remove #stations(0~110), close stations(8 stations within 5deg)
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128)+Dense(Leakyrelu)+Dense(Leakyrelu)+Dense(LeakyRelu)+Dense(LeakyRelu)+Droup(0.2)+output Dense(tanh)
HP=[256,256,128,128,64,32,8]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_test=1024 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='46'
FlatY=True



#Run 47 test:20200123:1610
#Everything is the same as #46 but use a different Leakyrelu alpha=0.05
#use weightMw_mse, use a larger validation BS(1024)
#Modify the scaling by applying a sqrt(data) then /10
#Use the flat Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.05) replace the original LeakyRelu(0.2)
#!!!GNSS noise level[1~50], remove #stations(0~110), close stations(8 stations within 5deg)
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128)+Dense(Leakyrelu)+Dense(Leakyrelu)+Dense(LeakyRelu)+Dense(LeakyRelu)+Droup(0.2)+output Dense(tanh)
HP=[256,256,128,128,64,32,8]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_test=1024 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='47'
FlatY=True


#Re-train the #43 (with noise, the performance on 5-real look the best)
#Run 48 test:20200302:1605
#Everything is the same as #43, but with Recurrent dropout, 5 stations within 5 deg, and changed step_per_epoch from 10 to 1 but total epochs from 5000 to 50000, should be the same!
#use weightMw_mse, use a larger validation BS(1024)
#Modify the scaling by applying a sqrt(data) then /10
#Use the flat Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~50], remove #stations(0~110), close stations(5 stations within 5deg)
#Dense(Leakyrelu 0.2)+Dense(Leakyrelu 0.2)+Dropout(0.2)+LSTM(128 recurrent D=0.2)+Dense(Leakyrelu 0.2)+Dense(Leakyrelu 0.2)+Dense(LeakyRelu 0.2)+Dense(LeakyRelu 0.2)+Droup(0.2)+output Dense(tanh)
HP=[256,256,128,128,64,32,8]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_test=1024 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='48'
FlatY=True



#Run 49 test:20200302:1625
#Re-Run 46 test
#Everything is the same as #46, but harder noise, station rm, less near-field station
#use weightMw_mse, use a larger validation BS(1024)
#Modify the scaling by applying a sqrt(data) then /10
#Use the flat Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~90], remove #stations(0~115), close stations(5 stations within 5deg)
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128)+Dense(Leakyrelu)+Dense(Leakyrelu)+Dense(LeakyRelu)+Dense(LeakyRelu)+Droup(0.2)+output Dense(tanh)
HP=[256,256,128,128,64,32,8]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_test=1024 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='49'
FlatY=True



#Re-train the #48 
#50 test:20200303:1250
#Everything is the same as #48, replace tanh with sigmoid in the final output, station rm=(0~115), larger neurons size, 30% of training is Noise(Mw=0) event.
#use weightMw_mse, use a larger validation BS(1024)
#Modify the scaling by applying a sqrt(data) then /10
#Use the flat Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~50], remove #stations(0~115), close stations(5 stations within 5deg)
#Dense(Leakyrelu 0.2)+Dense(Leakyrelu 0.2)+Dropout(0.2)+LSTM(128 recurrent D=0.2)+Dense(Leakyrelu 0.2)+Dense(Leakyrelu 0.2)+Dense(LeakyRelu 0.2)+Dense(LeakyRelu 0.2)+Droup(0.2)+output Dense(sigmoid)
#HP=[256,256,128,128,64,32,8]
HP=[512,512,256,128,128,64,32]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_test=1024 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='50'
FlatY=True



#51 test:20200303:1250
#More layers, replace tanh with sigmoid in the final output, station rm=(0~115), larger neurons size, 30% of training is Noise(Mw=0) event.
#use weightMw_mse, use a larger validation BS(1024)
#Modify the scaling by applying a sqrt(data) then /10
#Use the flat Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~50], remove #stations(0~115), close stations(5 stations within 5deg)
#Dense(Leakyrelu 0.2)+Dense(Leakyrelu 0.2)+Dropout(0.2)+LSTM(128 recurrent D=0.2)+Dense(Leakyrelu 0.2)+Dense(Leakyrelu 0.2)+Dense(LeakyRelu 0.2)+Dense(LeakyRelu 0.2)+Dense(LeakyRelu 0.2)+Dense(LeakyRelu 0.2)+Droup(0.2)+output Dense(sigmoid)
#HP=[256,256,128,128,64,32,8]
HP=[512,512,256,128,128,64,32,32,16] #Added 2more dense layers
Drops=[0.2,0.2]
BS=64 #Batch Size for training
BS_test=1024 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='51'
FlatY=True


#########Starting from #52, adding Mw filter NO!doesn't make sense for 5-real events(some are small)################
#Run 52 test:20200304:1120
#Everything is the same as #43,but noise lever [1~90], 50% noise events,rm stations 0~115, close stations(5 stations within 5deg),  training events only Mw>=8.0 and Mw=0
#use weightMw_mse, use a larger validation BS(1024)
#Modify the scaling by applying a sqrt(data) then /10
#Use the flat Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~90], remove #stations(0~115), close stations(5 stations within 5deg)
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128 (recurrent Drop0.2))+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense(tanh)
HP=[256,256,128,128,64]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_test=1024 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='52'
FlatY=True



#Run 53 test:20200304:1430
#Everything is the same as #52,but 70% noise events, change Mw_noise from 0 to 5.0, the average of label should be (7.5+9.5)/2*0.3+5*0.7=~6.0
#use weightMw_mse, use a larger validation BS(1024)
#Modify the scaling by applying a sqrt(data) then /10
#Use the flat Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~90], remove #stations(0~115), close stations(5 stations within 5deg)
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128 (recurrent Drop0.2))+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense(tanh)
HP=[256,256,128,128,64]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_test=1024 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='53'
FlatY=True


#Run 54 test:20200304:1915
#Everything is the same as #53,which is 70% noise events, change Mw_noise from 0 to 5.0, the average of label should be (7.5+9.5)/2*0.3+5*0.7=~6.0
#use weightMw_time_mse!!!!! instead of weightMw_mse
#Modify the scaling by applying a sqrt(data) then /10
#Use the flat Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~90], remove #stations(0~115), close stations(5 stations within 5deg)
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128 (recurrent Drop0.2))+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense(tanh)
HP=[256,256,128,128,64]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_test=1024 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='54'
FlatY=True


#Run 55 test:20200305:1635
#Everything is the same as #54,which is 70% noise events, change Mw_noise from 0 to 5.0, the average of label should be (7.5+9.5)/2*0.3+5*0.7=~6.0, use a larger neurons
#use weightMw_time_mse!!!!! instead of weightMw_mse, the idx= 10/max_idx~10/1
#Modify the scaling by applying a sqrt(data) then /10
#Use the flat Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~90], remove #stations(0~115), close stations(5 stations within 5deg)
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128 (recurrent Drop0.2))+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense(tanh)
HP=[512,512,256,256,256]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_test=1024 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='55'
FlatY=True


#Run 56 test:20200305:1645
#Everything is the same as #55, larger idx for weightMw_time_mse
#use weightMw_time_mse!!!!! instead of weightMw_mse, the idx= 100/max_idx~100/1 larger than the #55
#Modify the scaling by applying a sqrt(data) then /10
#Use the flat Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~90], remove #stations(0~115), close stations(5 stations within 5deg)
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128 (recurrent Drop0.2))+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense(tanh)
HP=[512,512,256,256,256]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_test=1024 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='56'
FlatY=True


#Run 57 test:20200306:1145
#Everything is the same as #56, use sample_weights in Keras
#also weight the Mw same as 56
#Modify the scaling by applying a sqrt(data) then /10
#Use the flat Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~90], remove #stations(0~115), close stations(5 stations within 5deg)
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128 (recurrent Drop0.2))+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense(tanh)
HP=[512,512,256,256,256]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
#BS_test=1024 #batch size for validation
BS_test=128 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='57'
FlatY=True


#Run 58 test:20200306:1532
#Everything is the same as #57, but a different recurrent weight
#also weight the Mw same as 56
#Modify the scaling by applying a sqrt(data) then /10
#Use the flat Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~90], remove #stations(0~115), close stations(5 stations within 5deg)
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128 (recurrent Drop0.2))+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense(tanh)
HP=[512,512,256,256,256]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
#BS_test=1024 #batch size for validation
BS_test=128 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='58'
FlatY=True


######Testing the new generator!!!
#Run 59 test:20200310:1150 Repeat the #58 with the new generator!
#Everything is the same as #58, but training only by EQs (withour noise), use weight_time_mw_mse
#Modify the scaling by applying a sqrt(data) then /10
#Use the flat Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~90], remove #stations(0~115), close stations(5 stations within 5deg) noise_P=0.7
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128 (recurrent Drop0.2))+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense(tanh)
HP=[512,512,256,256,256]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_test=128 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='59'
FlatY=True


#Run 60 test:20200310:1550 Repeat the #46 with the new generator!
#Everything is the same as #46, training only by EQs (withour noise), use weightMw_time_mse (adding smaller Mw EQs)
#Modify the scaling by applying a sqrt(data) then /10
#Use the flat Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~90], remove #stations(0~115), close stations(5 stations within 5deg) noise_P=0.7
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128 (recurrent Drop0.2))+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense(tanh)
HP=[256,256,128,128,64,32,8]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_test=1024 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='60'
FlatY=True



#Run 61 test:20200310:1640
#Everything is the same as #60, training only by EQs Large+small Mw (without noise), use "weight_time_mw_mse"
#Modify the scaling by applying a sqrt(data) then /10
#Use the flat Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~90], remove #stations(0~115), close stations(5 stations within 5deg) noise_P=0.7
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128 (recurrent Drop0.2))+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense(tanh)
HP=[256,256,128,128,64,32,8]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_test=1024 #batch size for validation
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='61'
FlatY=True


###Test61 is cheating, use a better Train(0.7)_Validation(0.2)_Test(0.1) split
#Run 62 test:20200310:1720
#Other than the dataset splitting, everything is the same as #61, training only by EQs Large+small Mw (without noise), use "weight_time_mw_mse"
#Modify the scaling by applying a sqrt(data) then /10
#Use the flat Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~90], remove #stations(0~115), close stations(5 stations within 5deg) noise_P=0.7
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128 (recurrent Drop0.2))+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense(tanh)
HP=[256,256,128,128,64,32,8]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_valid=1024 #######################################################################################CHANGE it later!!!!! 1024
BS_test=8192 #batch size for testing
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='62'
FlatY=True


#Run 63 test:20200407:1220
#everything is the same as #62, training with more smaller Mw events, use "weight_time_mw_mse"
#Modify the scaling by applying a sqrt(data) then /10
#Use the flat Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~90], remove #stations(0~115), close stations(5 stations within 5deg) noise_P=0.7
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128 (recurrent Drop0.2))+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense(relu)
HP=[256,256,128,128,64,32,8]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_valid=1024 #######################################################################################CHANGE it later!!!!! 1024
BS_test=8192 #batch size for testing
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='63'
FlatY=True
NoiseP=0.0 #probability of noise event



#Run 64 test:20200407:1750
#everything is the same as #62, training with more smaller Mw events, use "weight_time_mw_mse"
#Modify the scaling by applying a sqrt(data) then /10
#Use the flat Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~90], remove #stations(0~115), close stations(5 stations within 5deg) noise_P=0.7
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128 (recurrent Drop0.2))+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense(relu)
#HP=[256,256,128,128,64,32,8]
HP=[64,128,256,256,256,256,256] #Try a different structure
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_valid=1024 #######################################################################################CHANGE it later!!!!! 1024
BS_test=8192 #batch size for testing
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='64'
FlatY=True
NoiseP=0.0 #possibility of noise event


#Run 65 test:20200407:1800
#everything is the same as #63, use "weight_time_mw_mae"
#Modify the scaling by applying a sqrt(data) then /10
#Use the flat Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~90], remove #stations(0~115), close stations(5 stations within 5deg) noise_P=0.7
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128 (recurrent Drop0.2))+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense(relu)
HP=[256,256,128,128,64,32,8]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_valid=1024 #######################################################################################CHANGE it later!!!!! 1024
BS_test=8192 #batch size for testing
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='65'
FlatY=True
NoiseP=0.0 #possibility of noise event



#Run 66 test:20200407:1900
#everything is the same as #65, use "weight_time_mw_mae", Noise P=0.5
#Modify the scaling by applying a sqrt(data) then /10
#Use the flat Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~90], remove #stations(0~115), close stations(5 stations within 5deg) noise_P=0.7
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128 (recurrent Drop0.2))+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense(relu)
HP=[256,256,128,128,64,32,8]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_valid=1024 #######################################################################################CHANGE it later!!!!! 1024
BS_test=8192 #batch size for testing
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='66'
FlatY=True
NoiseP=0.5 #possibility of noise event


#Since #67, change the feature scaling to log(x)
#use "weight_time_mw_mse", Noise P=0.5
#Modify the scaling by applying a np.log(x)
#Use the flat Mw labeling
#scale y by y=y/10.0
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~90], remove #stations(0~115), close stations(3 stations within 5deg) noise_P=0.5
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128 (recurrent Drop0.2))+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense(relu)
HP=[256,256,128,128,64,32,8]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_valid=1024 #######################################################################################CHANGE it later!!!!! 1024
BS_test=8192 #batch size for testing
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='67'
FlatY=True
NoiseP=0.5 #possibility of noise event



#Since #68, same as #67, without noise(i.e. NoiseP=0)
#use "weightMw_time_mse" 
#Modify the scaling by applying a np.log(x)
#Use the flat Mw labeling
#scale y by y=y/10.0
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~90], remove #stations(0~115), close stations(3 stations within 5deg) noise_P=0.5
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128 (recurrent Drop0.2))+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense(relu)
HP=[256,256,128,128,64,32,8]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_valid=1024 #######################################################################################CHANGE it later!!!!! 1024
BS_test=8192 #batch size for testing
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by SQRT(X) and /10
Testnum='68'
FlatY=True
NoiseP=0.0 #possibility of noise event



#Back to the original sense, use only large Mw, and NoiseP=0.5, Mw_Noise=0. Introducing log(X)
##69
#use "weightMw_time_mse" Use weight(Mw+2.0) because Mw_noise=0 makes no constraint
#Modify the scaling by applying a np.log10(x)
#Use the flat Mw labeling
#scale y by y=y/10.0
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~90], remove #stations(0~115), close stations(3 stations within 5deg) noise_P=0.5
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128 (recurrent Drop0.2))+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense(relu)
HP=[256,256,128,128,64,32,8]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_valid=1024 #######################################################################################CHANGE it later!!!!! 1024
BS_test=8192 #batch size for testing
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by log10(X)
Testnum='69'
FlatY=True
NoiseP=0.5 #possibility of noise event



##70, Same as #69 (i.e. introducing log(X))
#use "weight_time_mse" (Without weighted Mw)
#Modify the scaling by applying a np.log10(x)
#Use the flat Mw labeling
#scale y by y=y/10.0
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~90], remove #stations(0~115), close stations(3 stations within 5deg) noise_P=0.5
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128 (recurrent Drop0.2))+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense(relu)
HP=[256,256,128,128,64,32,8]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_valid=1024 #######################################################################################CHANGE it later!!!!! 1024
BS_test=8192 #batch size for testing
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by log10(X)
Testnum='70'
FlatY=True
NoiseP=0.5 #possibility of noise event



##71, Same as #70 , but using real labeling
#use "weight_time_mse" (Without weighted Mw)
#Modify the scaling by applying a np.log10(x)
#scale y by y=y/10.0
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~90], remove #stations(0~115), close stations(3 stations within 5deg) noise_P=0.5
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128 (recurrent Drop0.2))+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense(relu)
HP=[256,256,128,128,64,32,8]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_valid=1024 #######################################################################################CHANGE it later!!!!! 1024
BS_test=8192 #batch size for testing
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by log10(X)
Testnum='71'
FlatY=False #The only difference here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
NoiseP=0.5 #possibility of noise event



##72, Same as #71, but regulat MSE and very strict station removel
#use "regular MSE" (Without weighted time)
#Modify the scaling by applying a np.log10(x)
#scale y by y=y/10.0
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~90], remove #stations(0~115), close stations(4 stations within 3deg)!!!very strict.  noise_P=0.5
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128 (recurrent Drop0.2))+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense(relu)
HP=[256,256,128,128,64,32,8]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_valid=1024 #######################################################################################CHANGE it later!!!!! 1024
BS_test=8192 #batch size for testing
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by log10(X)
Testnum='72'
FlatY=False #The only difference here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
NoiseP=0.5 #possibility of noise event


##73, Same as #72 but without Noise events
#use "regular MSE" (Without weighted time)
#Modify the scaling by applying a np.log10(x)
#scale y by y=y/10.0
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~90], remove #stations(0~115), close stations(4 stations within 3deg)!!!very strict.  
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128 (recurrent Drop0.2))+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense(relu)
HP=[256,256,128,128,64,32,8]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_valid=1024 #######################################################################################CHANGE it later!!!!! 1024
BS_test=8192 #batch size for testing
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by log10(X)
Testnum='73'
FlatY=False #The only difference here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
NoiseP=0.0 #possibility of noise event
Noise_level=[1,10,20,30,40,50,60,70,80,90]
rm_stans=[0,115] #remove number of stations from 0~115
Min_stan_dist=[4,3] #minumum 4 stations within 3 degree
Loss_func='mse' #can be default loss function string or self defined loss

#Get the testing data
#Write all parameters into function input
#Dpath='Path is not important'
#gtrain=feature_gen(Dpath,X_train_E,X_train_N,X_train_Z,None,Nstan=121,add_code=True,add_noise=True,noise_p=NoiseP,rmN=(rm_stans[0],rm_stans[1]),Noise_level=Noise_level,Min_stan_dist=Min_stan_dist,scale=(scales[0],scales[1]),BatchSize=BS,Mwfilter=7.0,save_ID=False,shuffle=True) #Use the "flat y"
#gvalid=feature_gen(Dpath,X_valid_E,X_valid_N,X_valid_Z,None,Nstan=121,add_code=True,add_noise=True,noise_p=NoiseP,rmN=(rm_stans[0],rm_stans[1]),Noise_level=Noise_level,Min_stan_dist=Min_stan_dist,scale=(scales[0],scales[1]),BatchSize=BS_valid,Mwfilter=7.0,save_ID='Run73_valid_EQID.npy',shuffle=True) 
#gtest=feature_gen(Dpath,X_test_E,X_test_N,X_test_Z,None,Nstan=121,add_code=True,add_noise=True,noise_p=NoiseP,rmN=(rm_stans[0],rm_stans[1]),Noise_level=Noise_level,Min_stan_dist=Min_stan_dist,scale=(scales[0],scales[1]),BatchSize=BS_test,Mwfilter=7.0,save_ID='Run73_test_EQID.npy',shuffle=True) #Use the "flat y"

#Also save the testing data 
#X_test_out,y_test_out=gtest.__getitem__(1)
#np.save('Xtest'+Testnum+'_2.npy',X_test_out)
#np.save('ytest'+Testnum+'_2.npy',y_test_out)

#sys.exit(2)

##74, Reproduce the Test#27
#use "regular MSE" change the Noise part!
#Modify the scaling by applying a np.log10(x)
#scale y by y=y/10.0
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~50], remove #stations(0~100), close stations(10 stations within 5deg)  noise_P=0.5
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128 (recurrent Drop0.2))+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense(relu)
HP=[256,256,128,128,64,32,8]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_valid=1024 #######################################################################################CHANGE it later!!!!! 1024
BS_test=8192 #batch size for testing
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by log10(X)
Testnum='74'
FlatY=False #The only difference here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
NoiseP=0.5 #possibility of noise event



##75, Same as #73 but flat labeling
#use "regular MSE" (Without weighted time)
#Modify the scaling by applying a np.log10(x)
#scale y by y=y/10.0
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~90], remove #stations(0~115), close stations(4 stations within 3deg)!!!very strict.  
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128 (recurrent Drop0.2))+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense(relu)
HP=[256,256,128,128,64,32,8]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_valid=1024 #######################################################################################CHANGE it later!!!!! 1024
BS_test=8192 #batch size for testing
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by log10(X)
Testnum='75'
FlatY=True #The only difference here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
NoiseP=0.0 #possibility of noise event



##76, Same as #72 but flat labeling
#use "regular MSE" (Without weighted time)
#Modify the scaling by applying a np.log10(x)
#scale y by y=y/10.0
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~90], remove #stations(0~115), close stations(4 stations within 3deg)!!!very strict.  
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128 (recurrent Drop0.2))+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense(relu)
HP=[256,256,128,128,64,32,8]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_valid=1024 #######################################################################################CHANGE it later!!!!! 1024
BS_test=8192 #batch size for testing
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by log10(X)
Testnum='76'
FlatY=True #The only difference here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
NoiseP=0.5 #possibility of noise event
Noise_level=[1,10,20,30,40,50,60,70,80,90]
rm_stans=[0,115] #remove number of stations from 0~115
Min_stan_dist=[4,3] #minumum 4 stations within 3 degree
Loss_func='mse' #can be default loss function string or self defined loss


##77, Same as #72(real STF), change noise level
#Modify the scaling by applying a np.log10(x)
#scale y by y=y/10.0
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128 (recurrent Drop0.2))+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense(relu)
HP=[256,256,128,128,64,32,8]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_valid=1024 #######################################################################################CHANGE it later!!!!! 1024
BS_test=8192 #batch size for testing
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by log10(X)
Testnum='77'
FlatY=False #
NoiseP=0.5 #possibility of noise event
Noise_level=[1] #Try the easiest noise level
rm_stans=[0,115] #remove number of stations from 0~115
Min_stan_dist=[4,3] #minumum 4 stations within 3 degree
Loss_func='mse' #can be default loss function string or self defined loss


##78, Same as #72(real STF), change noise level
#Modify the scaling by applying a np.log10(x)
#scale y by y=y/10.0
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128 (recurrent Drop0.2))+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense(relu)
HP=[256,256,128,128,64,32,8]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_valid=1024 #######################################################################################CHANGE it later!!!!! 1024
BS_test=8192 #batch size for testing
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by log10(X)
Testnum='78'
FlatY=False #
NoiseP=0.5 #possibility of noise event
Noise_level=[1,10,20,30] #Try the easiest noise level
rm_stans=[0,115] #remove number of stations from 0~115
Min_stan_dist=[4,3] #minumum 4 stations within 3 degree
Loss_func='mse' #can be default loss function string or self defined loss



##79, Same as #72(real STF), change noise level
#Modify the scaling by applying a np.log10(x)
#scale y by y=y/10.0
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128 (recurrent Drop0.2))+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense(relu)
HP=[256,256,128,128,64,32,8]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_valid=1024 #######################################################################################CHANGE it later!!!!! 1024
BS_test=8192 #batch size for testing
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by log10(X)
Testnum='79'
FlatY=False #
NoiseP=0.5 #possibility of noise event
Noise_level=[1,10,20,30,40,50,60] #Try the easiest noise level
rm_stans=[0,115] #remove number of stations from 0~115
Min_stan_dist=[4,3] #minumum 4 stations within 3 degree
Loss_func='mse' #can be default loss function string or self defined loss


####TRY again #69, or 70 but with regular MSE





##80, Same as #73 but use new fakequakes dataset without 25% issue
#use "regular MSE" (Without weighted time)
#Modify the scaling by applying a np.log10(x)
#scale y by y=y/10.0
#The station existence code is 0 and 0.5 (instead of 1)
#use LeakyRelu(0.2) replace the original Relu
#!!!GNSS noise level[1~90], remove #stations(0~115), close stations(4 stations within 3deg)!!!very strict.  
#use a larger BS=128, since #16 is better than #13
#Dense(Leakyrelu)+Dense(Leakyrelu)+Dropout(0.2)+LSTM(128 (recurrent Drop0.2))+Dense(Leakyrelu)+Dense(Leakyrelu)+Droup(0.2)+output Dense(relu)
HP=[256,256,128,128,64,32,8]
Drops=[0.2,0.2]
BS=128 #Batch Size for training
BS_valid=1024 #######################################################################################CHANGE it later!!!!! 1024
BS_test=8192 #batch size for testing
scales=[0,1] #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by log10(X)
Testnum='80'
FlatY=False #The only difference here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
NoiseP=0.0 #possibility of noise event
Noise_level=[1,10,20,30,40,50,60,70,80,90]
rm_stans=[0,115] #remove number of stations from 0~115
Min_stan_dist=[4,3] #minumum 4 stations within 3 degree
Loss_func='mse' #can be default loss function string or self defined loss

#Get the testing data
#Write all parameters into function input
#Dpath='Path is not important'
#gtrain=feature_gen(Dpath,X_train_E,X_train_N,X_train_Z,None,Nstan=121,add_code=True,add_noise=True,noise_p=NoiseP,rmN=(rm_stans[0],rm_stans[1]),Noise_level=Noise_level,Min_stan_dist=Min_stan_dist,scale=(scales[0],scales[1]),BatchSize=BS,Mwfilter=7.0,save_ID=False,shuffle=True) #Use the "flat y"
#gvalid=feature_gen(Dpath,X_valid_E,X_valid_N,X_valid_Z,None,Nstan=121,add_code=True,add_noise=True,noise_p=NoiseP,rmN=(rm_stans[0],rm_stans[1]),Noise_level=Noise_level,Min_stan_dist=Min_stan_dist,scale=(scales[0],scales[1]),BatchSize=BS_valid,Mwfilter=7.0,save_ID='Run80_valid_EQID.npy',shuffle=True) 
#gtest=feature_gen(Dpath,X_test_E,X_test_N,X_test_Z,None,Nstan=121,add_code=True,add_noise=True,noise_p=NoiseP,rmN=(rm_stans[0],rm_stans[1]),Noise_level=Noise_level,Min_stan_dist=Min_stan_dist,scale=(scales[0],scales[1]),BatchSize=BS_test,Mwfilter=7.0,save_ID='Run80_test_EQID.npy',shuffle=True) #Use the "flat y"

#Also save the testing data 
#X_test_out,y_test_out=gtest.__getitem__(1)
#np.save('Xtest'+Testnum+'.npy',X_test_out)
#np.save('ytest'+Testnum+'.npy',y_test_out)






#Use the real STF for Mw labeling
#scale y by y=y/10.0
#scale X by sqrt(X)/10  #note the scaling is written in function

#add a tensorboard callback to test #25, what the difference between TimeDistributed and None
logdir = "logs/scalars/Test"+Testnum+'_'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)



##############Start using stacked LSTM#####################
'''
network = models.Sequential()
network.add(layers.LSTM(HP[0],return_sequences=True,input_shape=(102,242,))) #the time steps:102 can be None
network.add(layers.LSTM(HP[1],return_sequences=True,input_shape=(102,242,))) #the time steps:102 can be None
network.add(layers.Dropout(Drops[0]))
network.add(tf.keras.layers.TimeDistributed(layers.Dense(HP[2])))
network.add(layers.LeakyReLU(alpha=0.2))
network.add(tf.keras.layers.TimeDistributed(layers.Dense(HP[3])))
network.add(layers.LeakyReLU(alpha=0.2))
network.add(tf.keras.layers.TimeDistributed(layers.Dense(HP[4])))
network.add(layers.LeakyReLU(alpha=0.2))
network.add(tf.keras.layers.TimeDistributed(layers.Dense(HP[5])))
network.add(layers.LeakyReLU(alpha=0.2))
#network.add(tf.keras.layers.TimeDistributed(layers.Dense(1))) #Note that "dense layer" also called "fully connected" layer
network.add(tf.keras.layers.TimeDistributed(layers.Dense(1,activation='tanh'))) #Note that "dense layer" also called "fully connected" layer
#network.add( tf.keras.layers.TimeDistributed(layers.Dense(1)) )  #Note that "dense layer" also called "fully connected" layer
network.summary() #see the structure of network
'''


##############Start Training for #20 with LeakyRelu 0.2#####################
network = models.Sequential()
network.add(tf.keras.layers.TimeDistributed(layers.Dense(HP[0]),input_shape=(102,242,)))
network.add(layers.LeakyReLU(alpha=0.1))
network.add(tf.keras.layers.TimeDistributed(layers.Dense(HP[1])))
network.add(layers.LeakyReLU(alpha=0.1))
network.add(layers.Dropout(Drops[0]))
network.add(layers.LSTM(HP[2],return_sequences=True,input_shape=(102,242,),dropout=0.2, recurrent_dropout=0.2)) #the time steps:102 can be None
#network.add(layers.LSTM(HP[3],return_sequences=True,input_shape=(102,242,))) #the time steps:102 can be None
#network.add(layers.Dense(HP[3],activation='relu',input_shape=(102,242,)))
#network.add(layers.LeakyReLU(alpha=0.2))
network.add(tf.keras.layers.TimeDistributed(layers.Dense(HP[3])))
network.add(layers.LeakyReLU(alpha=0.1))
#network.add(layers.Dropout(Drops[1]))
network.add(tf.keras.layers.TimeDistributed(layers.Dense(HP[4])))
network.add(layers.LeakyReLU(alpha=0.1))
network.add(tf.keras.layers.TimeDistributed(layers.Dense(HP[5])))
network.add(layers.LeakyReLU(alpha=0.1))
network.add(tf.keras.layers.TimeDistributed(layers.Dense(HP[6])))
network.add(layers.LeakyReLU(alpha=0.1))
#network.add(tf.keras.layers.TimeDistributed(layers.Dense(HP[7])))
#network.add(layers.LeakyReLU(alpha=0.2))
#network.add(tf.keras.layers.TimeDistributed(layers.Dense(HP[8])))
#network.add(layers.LeakyReLU(alpha=0.2))
network.add(layers.Dropout(Drops[1]))
#network.add(tf.keras.layers.TimeDistributed(layers.Dense(1))) #Note that "dense layer" also called "fully connected" layer
network.add(tf.keras.layers.TimeDistributed(layers.Dense(1,activation='relu'))) #Note that "dense layer" also called "fully connected" layer
#network.add( tf.keras.layers.TimeDistributed(layers.Dense(1)) )  #Note that "dense layer" also called "fully connected" layer
network.summary() #see the structure of network


##############Start Training#####################
'''
network = models.Sequential()
network.add(layers.LSTM(HP[0],return_sequences=True,input_shape=(102,242,))) #the time steps:102 can be None
network.add(layers.LSTM(HP[1],return_sequences=True,input_shape=(102,242,))) #the time steps:102 can be None
#network.add(layers.Dropout(Drops[0]))
#network.add(layers.LSTM(HP[2],return_sequences=True,input_shape=(102,242,))) #the time steps:102 can be None
#network.add(layers.Dense(HP[2],activation='relu',input_shape=(102,242,)))
network.add(layers.Dense(HP[2],activation='relu',input_shape=(102,242,)))
#network.add(layers.Dense(HP[3],activation='relu',input_shape=(102,242,)))
network.add(layers.Dropout(Drops[0]))
#network.add(layers.LSTM(HP[2],return_sequences=True,input_shape=(102,242))) #the time steps:102 can be None
#network.add(layers.Dense(HP[3],activation='relu'))
#network.add(layers.Dense(HP[4],activation='relu'))
#network.add(layers.Dropout(Drops[0]))
network.add(layers.Dense(1)) #Note that "dense layer" also called "fully connected" layer
#network.add(layers.Dense(1,activation='relu')) #Note that "dense layer" also called "fully connected" layer
network.summary() #see the structure of network
'''

##############Start Training for #13,#14 with LeakyRelu 0.2#####################
'''
network = models.Sequential()
network.add(layers.Dense(HP[0],input_shape=(102,242,)))
network.add(layers.LeakyReLU(alpha=0.2))
network.add(layers.Dense(HP[1],input_shape=(102,242,)))
network.add(layers.LeakyReLU(alpha=0.2))
network.add(layers.LSTM(HP[2],return_sequences=True,input_shape=(102,242,))) #the time steps:102 can be None
network.add(layers.LSTM(HP[3],return_sequences=True,input_shape=(102,242,))) #the time steps:102 can be None
#network.add(layers.Dense(HP[3],activation='relu',input_shape=(102,242,)))
#network.add(layers.Dense(HP[3],activation='relu',input_shape=(102,242,)))
network.add(layers.Dense(HP[4],input_shape=(102,242,)))
network.add(layers.LeakyReLU(alpha=0.2))
network.add(layers.Dropout(Drops[0]))
network.add(layers.Dense(1)) #Note that "dense layer" also called "fully connected" layer
network.summary() #see the structure of network
'''

'''
##############Start Training #19#####################
network = models.Sequential()
network.add(layers.Dense(HP[0],activation='relu',input_shape=(102,242,)))
network.add(layers.Dense(HP[1],activation='relu',input_shape=(102,242,)))
network.add(layers.LSTM(HP[2],return_sequences=True,input_shape=(102,242,))) #the time steps:102 can be None
network.add(layers.LSTM(HP[3],return_sequences=True,input_shape=(102,242,))) #the time steps:102 can be None
#network.add(layers.Dense(HP[3],activation='relu',input_shape=(102,242,)))
#network.add(layers.Dense(HP[3],activation='relu',input_shape=(102,242,)))
network.add(layers.Dense(HP[4],activation='relu',input_shape=(102,242,)))
network.add(layers.Dropout(Drops[0]))
network.add(layers.Dense(1)) #Note that "dense layer" also called "fully connected" layer
network.summary() #see the structure of network
'''


#network.compile(loss='mse',optimizer='adam') #Regular mse
#network.compile(loss='mae',optimizer='adam') #mean absolute error
#network.compile(loss=[weight_mse],optimizer='adam') #weighted mae
#network.compile(loss=[weight_mae],optimizer='adam') #weighted mae
#network.compile(loss=[weightMw_mse],optimizer='adam') #weighted Mw mse

#network.compile(loss=[weightMw_mse],optimizer='adam') #weighted Mw mse #Test60

#network.compile(loss=[weightMw_time_mse],optimizer='adam') #weighted time Mw mse #Test61~64(WMW) , #67(WMW), #68(WMW), #69(WMw+2.0)
#network.compile(loss=[weightMw_time_mae],optimizer='adam') #weighted time Mw mse #Test65
#network.compile(loss=[weight_time_mse],optimizer='adam') #weighted time mse # #70, #71
#network.compile(loss='mse',optimizer='adam') #weighted time mse # #72
network.compile(loss=Loss_func,optimizer='adam') #weighted time mse # #72


####Test57 not the fit_generator doesn't support sample_weight#####
#network.compile(loss=[weightMw_mse],optimizer='adam',sample_weight_mode="temporal") #weighted Mw & time mse


#model_hist=network.fit(X_train_norm_LL_rm1,Y_train_norm_LL_rm1,batch_size=500,epochs=50,validation_split=0.2)
#gen_train_data=feature_gen(X_train_norm_code[:2000],y_train[:2000],Nstan=121,add_noise=True,noise_p=0.5,BatchSize=256) #

#
#gen_train_data=feature_gen(X_train_E[:2000],X_train_N[:2000],X_train_Z[:2000],y_train[:2000],Nstan=121,add_code=True,add_noise=True,noise_p=0.5,rmN=(0,115),scale=(0,1),BatchSize=256)
#gen_valid_data=feature_gen(X_train_E[2000:],X_train_N[2000:],X_train_Z[2000:],y_train[2000:],Nstan=121,add_code=True,add_noise=True,noise_p=0.5,rmN=(0,115),scale=(0,1),BatchSize=256)
#X_valid,y_valid=next(gen_valid_data)
#
##X_train,y_train=next(gen_valid_data)
#
#g_train=feature_gen(E[:2000],N[:2000],Z[:2000],y[:2000],Nstan=121,add_code=True,add_noise=True,noise_p=0.5,rmN=(10,110),scale=(0,1),BatchSize=64,shuffle=True)
#g_test=feature_gen(E[2000:],N[2000:],Z[2000:],y[2000:],Nstan=121,add_code=True,add_noise=True,noise_p=0.5,rmN=(10,110),scale=(0,1),BatchSize=16,shuffle=True)
#
#
#g_train=make_feature(E[:2000],N[:2000],Z[:2000],y[:2000],Nstan=121,add_code=True,add_noise=True,noise_p=0,rmN=(0,0),scale=(0,1),BatchSize=128,shuffle=True)
#g_test=make_feature(E[2000:],N[2000:],Z[2000:],y[2000:],Nstan=121,add_code=True,add_noise=True,noise_p=0,rmN=(0,0),scale=(0,1),BatchSize=32,shuffle=True)
#X_test,ytest=next(g_test.it)
#

#model_hist=network.fit(x=X_train,y=y_train,validation_data=(X_valid, y_valid),batch_size=32,steps_per_epoch=1,epochs=500)

#gtrain=feature_gen(X_train_E[:2000],X_train_N[:2000],X_train_Z[:2000],y_train[:2000],Nstan=121,add_code=True,add_noise=True,noise_p=0.5,rmN=(0,115),scale=(0,1),BatchSize=256,shuffle=True)
#gtest=feature_gen(X_train_E[2000:],X_train_N[2000:],X_train_Z[2000:],y_train[2000:],Nstan=121,add_code=True,add_noise=True,noise_p=0.5,rmN=(0,115),scale=(0,1),BatchSize=256,shuffle=True)
#X_valid,y_valid=gtest.__getitem__(1)

#Dpath='/projects/tlalollin/jiunting/Fakequakes/run/Chile_27200_ENZ'


#1. y is a flat line
#gtrain=feature_gen(Dpath,X_train_E[:20000],X_train_N[:20000],X_train_Z[:20000],y_train[:20000],Nstan=121,add_code=True,add_noise=True,noise_p=0.5,rmN=(0,115),scale=(10,10.0),BatchSize=BS,shuffle=True)
#gtest=feature_gen(Dpath,X_train_E[20000:],X_train_N[20000:],X_train_Z[20000:],y_train[20000:],Nstan=121,add_code=True,add_noise=True,noise_p=0.5,rmN=(0,115),scale=(10,10.0),BatchSize=BS,shuffle=True)
#2. read y from the .rupt file, so just input None now

Dpath='Path is not important!'

#gtrain=feature_gen(Dpath,X_train_E[:20000],X_train_N[:20000],X_train_Z[:20000],None,Nstan=121,add_code=True,add_noise=True,noise_p=0.5,rmN=(0,115),scale=(scales[0],scales[1]),BatchSize=BS,shuffle=True)
#gtrain=feature_gen(Dpath,X_train_E[:20000],X_train_N[:20000],X_train_Z[:20000],None,Nstan=121,add_code=True,add_noise=True,noise_p=0.5,rmN=(0,100),scale=(scales[0],scales[1]),BatchSize=BS,shuffle=True) #remove lesser stations
if FlatY:
    #gtrain=feature_gen(Dpath,X_train_E[:20000],X_train_N[:20000],X_train_Z[:20000],y_train[:20000],Nstan=121,add_code=True,add_noise=True,noise_p=0.5,rmN=(0,110),scale=(scales[0],scales[1]),BatchSize=BS,shuffle=True) #Use the "flat y"
    #gtest=feature_gen(Dpath,X_train_E[20000:],X_train_N[20000:],X_train_Z[20000:],y_train[20000:],Nstan=121,add_code=True,add_noise=True,noise_p=0.5,rmN=(0,110),scale=(scales[0],scales[1]),BatchSize=BS_test,shuffle=True) #Use the "flat y"

    #This is for(without noise Mw=0 data) careful the rmN may be different!!!!!
    #gtrain=feature_gen(Dpath,X_train_E[:20000],X_train_N[:20000],X_train_Z[:20000],y_train[:20000],Nstan=121,add_code=True,add_noise=True,noise_p=0.7,rmN=(0,115),scale=(scales[0],scales[1]),BatchSize=BS,Mwfilter=7.0,shuffle=True) #Use the "flat y"
    #gtest=feature_gen(Dpath,X_train_E[20000:],X_train_N[20000:],X_train_Z[20000:],y_train[20000:],Nstan=121,add_code=True,add_noise=True,noise_p=0.7,rmN=(0,115),scale=(scales[0],scales[1]),BatchSize=BS_test,Mwfilter=7.0,shuffle=True) #Use the "flat y"

    #This is for(without noise Mw=0 data) careful the rmN may be different!!!!! #Test58
    #gtrain=feature_gen(Dpath,X_train_E,X_train_N,X_train_Z,y_train,Nstan=121,add_code=True,add_noise=True,noise_p=0.7,rmN=(0,115),scale=(scales[0],scales[1]),BatchSize=BS,Mwfilter=7.0,shuffle=True) #Use the "flat y"
    #gtest=feature_gen(Dpath,X_test_E,X_test_N,X_test_Z,y_test,Nstan=121,add_code=True,add_noise=True,noise_p=0.7,rmN=(0,115),scale=(scales[0],scales[1]),BatchSize=BS_test,Mwfilter=7.0,shuffle=True) #Use the "flat y"

    #This is for(without noise Mw=0 data) careful the rmN may be different!!!!! #Test60
    #gtrain=feature_gen(Dpath,X_train_E,X_train_N,X_train_Z,y_train,Nstan=121,add_code=True,add_noise=True,noise_p=NoiseP,rmN=(0,115),scale=(scales[0],scales[1]),BatchSize=BS,Mwfilter=7.0,shuffle=True) #Use the "flat y"
    #gvalid=feature_gen(Dpath,X_valid_E,X_valid_N,X_valid_Z,y_valid,Nstan=121,add_code=True,add_noise=True,noise_p=NoiseP,rmN=(0,115),scale=(scales[0],scales[1]),BatchSize=BS_valid,Mwfilter=7.0,shuffle=True) #Use the "flat y"
    #gtest=feature_gen(Dpath,X_test_E,X_test_N,X_test_Z,y_test,Nstan=121,add_code=True,add_noise=True,noise_p=NoiseP,rmN=(0,115),scale=(scales[0],scales[1]),BatchSize=BS_test,Mwfilter=7.0,shuffle=True) #Use the "flat y"

    #This is for(without noise Mw=0 data) careful the rmN may be different!!!!! #Test75
    #gtrain=feature_gen(Dpath,X_train_E,X_train_N,X_train_Z,y_train,Nstan=121,add_code=True,add_noise=True,noise_p=NoiseP,rmN=(0,115),scale=(scales[0],scales[1]),BatchSize=BS,Mwfilter=7.0,shuffle=True) #Use the "flat y"
    #gvalid=feature_gen(Dpath,X_valid_E,X_valid_N,X_valid_Z,y_valid,Nstan=121,add_code=True,add_noise=True,noise_p=NoiseP,rmN=(0,115),scale=(scales[0],scales[1]),BatchSize=BS_valid,Mwfilter=7.0,shuffle=True) #Use the "flat y"
    #gtest=feature_gen(Dpath,X_test_E,X_test_N,X_test_Z,y_test,Nstan=121,add_code=True,add_noise=True,noise_p=NoiseP,rmN=(0,115),scale=(scales[0],scales[1]),BatchSize=BS_test,Mwfilter=7.0,shuffle=True) #Use the "flat y"


    #Write all parameters into function
    gtrain=feature_gen(Dpath,X_train_E,X_train_N,X_train_Z,y_train,Nstan=121,add_code=True,add_noise=True,noise_p=NoiseP,rmN=(rm_stans[0],rm_stans[1]),Noise_level=Noise_level,Min_stan_dist=Min_stan_dist,scale=(scales[0],scales[1]),BatchSize=BS,Mwfilter=7.0,save_ID=False,shuffle=True) #Use the "flat y"
    gvalid=feature_gen(Dpath,X_valid_E,X_valid_N,X_valid_Z,y_valid,Nstan=121,add_code=True,add_noise=True,noise_p=NoiseP,rmN=(rm_stans[0],rm_stans[1]),Noise_level=Noise_level,Min_stan_dist=Min_stan_dist,scale=(scales[0],scales[1]),BatchSize=BS_valid,Mwfilter=7.0,save_ID='Run80_valid_EQID.npy',shuffle=True) #Use the "flat y"
    gtest=feature_gen(Dpath,X_test_E,X_test_N,X_test_Z,y_test,Nstan=121,add_code=True,add_noise=True,noise_p=NoiseP,rmN=(rm_stans[0],rm_stans[1]),Noise_level=Noise_level,Min_stan_dist=Min_stan_dist,scale=(scales[0],scales[1]),BatchSize=BS_test,Mwfilter=7.0,save_ID='Run80_test_EQID.npy',shuffle=True) #Use the "flat y"


    #gtrain=feature_gen(Dpath,X_train_E[:20000],X_train_N[:20000],X_train_Z[:20000],y_train[:20000],Nstan=121,add_code=True,add_noise=True,noise_p=0.0,rmN=(0,110),scale=(scales[0],scales[1]),BatchSize=BS,shuffle=True) #Use the "flat y"
    #gtest=feature_gen(Dpath,X_train_E[20000:],X_train_N[20000:],X_train_Z[20000:],y_train[20000:],Nstan=121,add_code=True,add_noise=True,noise_p=0.0,rmN=(0,110),scale=(scales[0],scales[1]),BatchSize=BS_test,shuffle=True) #Use the "flat y"
else:
    #This is for #71
    #gtrain=feature_gen(Dpath,X_train_E,X_train_N,X_train_Z,None,Nstan=121,add_code=True,add_noise=True,noise_p=NoiseP,rmN=(0,115),scale=(scales[0],scales[1]),BatchSize=BS,Mwfilter=7.0,shuffle=True) #Use the "flat y"
    #gvalid=feature_gen(Dpath,X_valid_E,X_valid_N,X_valid_Z,None,Nstan=121,add_code=True,add_noise=True,noise_p=NoiseP,rmN=(0,115),scale=(scales[0],scales[1]),BatchSize=BS_valid,Mwfilter=7.0,shuffle=True) #Use the "flat y"
    #gtest=feature_gen(Dpath,X_test_E,X_test_N,X_test_Z,None,Nstan=121,add_code=True,add_noise=True,noise_p=NoiseP,rmN=(0,115),scale=(scales[0],scales[1]),BatchSize=BS_test,Mwfilter=7.0,shuffle=True) #Use the "flat y"

    #This is for #74
    #gtrain=feature_gen(Dpath,X_train_E,X_train_N,X_train_Z,None,Nstan=121,add_code=True,add_noise=True,noise_p=NoiseP,rmN=(0,100),scale=(scales[0],scales[1]),BatchSize=BS,Mwfilter=7.0,shuffle=True) #Use the "flat y"
    #gvalid=feature_gen(Dpath,X_valid_E,X_valid_N,X_valid_Z,None,Nstan=121,add_code=True,add_noise=True,noise_p=NoiseP,rmN=(0,100),scale=(scales[0],scales[1]),BatchSize=BS_valid,Mwfilter=7.0,shuffle=True) #Use the "flat y"
    #gtest=feature_gen(Dpath,X_test_E,X_test_N,X_test_Z,None,Nstan=121,add_code=True,add_noise=True,noise_p=NoiseP,rmN=(0,100),scale=(scales[0],scales[1]),BatchSize=BS_test,Mwfilter=7.0,shuffle=True) #Use the "flat y"
 
    #Write all parameters into function input
    gtrain=feature_gen(Dpath,X_train_E,X_train_N,X_train_Z,None,Nstan=121,add_code=True,add_noise=True,noise_p=NoiseP,rmN=(rm_stans[0],rm_stans[1]),Noise_level=Noise_level,Min_stan_dist=Min_stan_dist,scale=(scales[0],scales[1]),BatchSize=BS,Mwfilter=7.0,save_ID=False,shuffle=True) #Use the "flat y"
    gvalid=feature_gen(Dpath,X_valid_E,X_valid_N,X_valid_Z,None,Nstan=121,add_code=True,add_noise=True,noise_p=NoiseP,rmN=(rm_stans[0],rm_stans[1]),Noise_level=Noise_level,Min_stan_dist=Min_stan_dist,scale=(scales[0],scales[1]),BatchSize=BS_valid,Mwfilter=7.0,save_ID='Run80_valid_EQID.npy',shuffle=True) #Use the "flat y"
    gtest=feature_gen(Dpath,X_test_E,X_test_N,X_test_Z,None,Nstan=121,add_code=True,add_noise=True,noise_p=NoiseP,rmN=(rm_stans[0],rm_stans[1]),Noise_level=Noise_level,Min_stan_dist=Min_stan_dist,scale=(scales[0],scales[1]),BatchSize=BS_test,Mwfilter=7.0,save_ID='Run80_test_EQID.npy',shuffle=True) #Use the "flat y"


    #gtrain=feature_gen(Dpath,X_train_E[:20000],X_train_N[:20000],X_train_Z[:20000],None,Nstan=121,add_code=True,add_noise=True,noise_p=0.5,rmN=(0,110),scale=(scales[0],scales[1]),BatchSize=BS,shuffle=True) #remove lesser stations
    #gtest=feature_gen(Dpath,X_train_E[20000:],X_train_N[20000:],X_train_Z[20000:],None,Nstan=121,add_code=True,add_noise=True,noise_p=0.5,rmN=(0,110),scale=(scales[0],scales[1]),BatchSize=BS_test,shuffle=True)

    #This is for #46,#47
    #gtrain=feature_gen(Dpath,X_train_E[:20000],X_train_N[:20000],X_train_Z[:20000],None,Nstan=121,add_code=True,add_noise=True,noise_p=0.0,rmN=(0,110),scale=(scales[0],scales[1]),BatchSize=BS,Mwfilter=8.0,shuffle=True) #remove lesser stations
    #gtest=feature_gen(Dpath,X_train_E[20000:],X_train_N[20000:],X_train_Z[20000:],None,Nstan=121,add_code=True,add_noise=True,noise_p=0.0,rmN=(0,110),scale=(scales[0],scales[1]),BatchSize=BS_test,Mwfilter=8.0,shuffle=True)

    #gtrain=feature_gen(Dpath,X_train_E[:20000],X_train_N[:20000],X_train_Z[:20000],None,Nstan=121,add_code=True,add_noise=True,noise_p=0.0,rmN=(0,110),scale=(scales[0],scales[1]),BatchSize=BS,shuffle=True) #remove lesser stations
    #gtest=feature_gen(Dpath,X_train_E[20000:],X_train_N[20000:],X_train_Z[20000:],None,Nstan=121,add_code=True,add_noise=True,noise_p=0.0,rmN=(0,110),scale=(scales[0],scales[1]),BatchSize=BS_test,shuffle=True)
    

'''
#gtest=feature_gen(Dpath,X_train_E[20000:],X_train_N[20000:],X_train_Z[20000:],None,Nstan=121,add_code=True,add_noise=True,noise_p=0.5,rmN=(0,115),scale=(scales[0],scales[1]),BatchSize=BS,shuffle=True)
#gtest=feature_gen(X_test_E,X_test_N,X_test_Z,y_test,Nstan=121,add_code=True,add_noise=True,noise_p=0.7,rmN=(0,115),scale=(scales[0],scales[1]),BatchSize=BS,Mwfilter,shuffle=True)
print('IDs for training:',X_train_E)
print('IDs for testing:',X_test_E)
print('Number of X_train:',len(X_train_E))
X_valid,y_valid=gtest.__getitem__(1)
#print('-------------------------')
print('-------------------------')

np.save('Xvalid_test59.npy',X_valid)
np.save('yvalid_test59.npy',y_valid)
sys.exit(2)
'''

#check file/dir exist,otherwise mkdir
if not(os.path.exists('./TrainingResults2/Test'+Testnum)):
    os.makedirs('./TrainingResults2/Test'+Testnum)

#Add callback
CB=keras.callbacks.ModelCheckpoint(filepath='./TrainingResults2/Test'+Testnum+'/weights.{epoch:04d}-{val_loss:.6f}.hdf5',monitor='val_loss',save_best_only=True,mode='min',period=5)

################Make another validation data#########################
#with the batch_size=1024, so it is very hard
#gtest=feature_gen(Dpath,X_train_E[20000:],X_train_N[20000:],X_train_Z[20000:],None,Nstan=121,add_code=True,add_noise=True,noise_p=0.5,rmN=(0,115),scale=(scales[0],scales[1]),BatchSize=BS_test,shuffle=True)
#gtest=feature_gen(Dpath,X_train_E[20000:],X_train_N[20000:],X_train_Z[20000:],None,Nstan=121,add_code=True,add_noise=True,noise_p=0.5,rmN=(0,100),scale=(scales[0],scales[1]),BatchSize=BS_test,shuffle=True) #remove lesser stations
#gtest=feature_gen(Dpath,X_train_E[20000:],X_train_N[20000:],X_train_Z[20000:],y_train[20000:],Nstan=121,add_code=True,add_noise=True,noise_p=0.5,rmN=(0,110),scale=(scales[0],scales[1]),BatchSize=BS_test,shuffle=True) #Use the "flat y"
X_valid_out,y_valid_out=gvalid.__getitem__(1)
np.save('Xvalid'+Testnum+'.npy',X_valid_out)
np.save('yvalid'+Testnum+'.npy',y_valid_out)

#Also save the testing data 
X_test_out,y_test_out=gtest.__getitem__(1)
np.save('Xtest'+Testnum+'.npy',X_test_out)
np.save('ytest'+Testnum+'.npy',y_test_out)

#np.save('Xvalid_test'+'30'+'.npy',X_valid)
#np.save('yvalid_test'+'30'+'.npy',y_valid)
##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!##################

#sys.exit(2)

#model_hist=network.fit_generator(gen_train_data,validation_data=gen_valid_data,use_multiprocessing=False,workers=1,validation_steps=1,steps_per_epoch=1,epochs=10)
#model_hist=network.fit_generator(gtrain,validation_data=gtest,use_multiprocessing=True,workers=16,max_queue_size=10,validation_steps=1,steps_per_epoch=1,epochs=10)

model_hist=network.fit_generator(gtrain,validation_data=(X_valid_out,y_valid_out),use_multiprocessing=True,workers=40,validation_steps=1,steps_per_epoch=1,epochs=50000,callbacks=[CB,tensorboard_callback]) #so that total steps=1+7=8

#####Test57 No the fit_generator doesn't support sample_weight#####
'''
samp_weights=[]
weights=10.0/(np.flipud(np.arange(102)+1))**0.5
for nw in range(BS):
    samp_weights.append(weights)

samp_weights=np.array(samp_weights)

model_hist=network.fit_generator(gtrain,validation_data=(X_valid,y_valid),use_multiprocessing=True,workers=40,validation_steps=1,steps_per_epoch=1,epochs=50000,callbacks=[CB,tensorboard_callback],sample_weight=samp_weights) #so that total steps=1+7=8
'''

#, workers=4
#validation_steps=1,


#Save the model
#tf.keras.models.save_model(network,'./TrainingResults/MSE_MiniB_D1024_D1024_LSTM1024_D1024_D512_Drop0.5_scaleY.h5py')
#tf.keras.models.save_model(network,'./TrainingResults2/MSE_B%d_D%d_D%d_LSTM%d_D%d_D%d_Drop%3.1f_scaleY.h5py'%(BS,HP[0],HP[1],HP[2],HP[3],HP[4],HP[5]))
#tf.keras.models.save_model(network,'./TrainingResults2/MSE_B%d_D%d_D%d_Drop%3.1f_LSTM%d_D%d_D%d_Drop%3.1f_scaleY.h5py'%(BS,HP[0],HP[1],Drops[0],HP[2],HP[3],HP[4],Drops[1]))
#tf.keras.models.save_model(network,'./TrainingResults2/MSE_B%d_D%d_D%d_LSTM%d_D%d_D%d_Drop%3.1f_scaleY_NF5_5stn.h5py'%(BS,HP[0],HP[1],HP[2],HP[3],HP[4],Drops[0]))
#2-LSTM
#tf.keras.models.save_model(network,'./TrainingResults2/MSE_B%d_LSTM%d_LSTM%d_LSTM%d_Drop%3.1f_scaleY_NF5_5stn.h5py'%(BS,HP[0],HP[1],HP[2],Drops[0]))
tf.keras.models.save_model(network,'./TrainingResults2/Test'+Testnum+'.h5py')


#save the training history
#np.save('./TrainingResults/MSE_miniB_D1024_D1024_LSTM1024_D1024_D512_Drop0.5_scaleY.hist.npy',model_hist.history)
#np.save('./TrainingResults2/MSE_B%d_D%d_D%d_LSTM%d_D%d_D%d_Drop%3.1f_scaleY.hist.npy'%(BS,HP[0],HP[1],HP[2],HP[3],HP[4],HP[5]),model_hist.history)
#np.save('./TrainingResults2/MSE_B%d_D%d_D%d_Drop%3.1f_LSTM%d_D%d_D%d_Drop%3.1f_scaleY.hist.npy'%(BS,HP[0],HP[1],Drops[0],HP[2],HP[3],HP[4],Drops[1]),model_hist.history)
#np.save('./TrainingResults2/MSE_B%d_D%d_D%d_LSTM%d_D%d_D%d_Drop%3.1f_scaleY_NF5_5stn.hist.npy'%(BS,HP[0],HP[1],HP[2],HP[3],HP[4],Drops[0]),model_hist.history)
#2-LSTM
#np.save('./TrainingResults2/MSE_B%d_LSTM%d_LSTM%d_LSTM%d_Drop%3.1f_scaleY_NF5_5stn.hist.npy'%(BS,HP[0],HP[1],HP[2],Drops[0]),model_hist.history)
np.save('./TrainingResults2/Test'+Testnum+'.npy',model_hist.history)


