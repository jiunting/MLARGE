#Make individual STFs and save them in directory
import numpy as np
from mudpy import viewFQ
from scipy.integrate import cumtrapz



def M02Mw(M0):
    Mw=(2.0/3)*np.log10(M0*1e7)-10.7 #M0 input is dyne-cm, convert to N-M by 1e7
    return(Mw)


def get_accM0(ruptID,T=np.arange(5,515,5)):
    if type(ruptID)==int:
        ruptID='%06d'%(ruptID)
    rupt=home+project_name+'/'+'output/ruptures/'+run_name+'.'+ruptID+'.rupt'
    t,Mrate=viewFQ.source_time_function(rupt,epicenter=None,dt=0.05,t_total=520,stf_type='dreger',plot=False)
    #get accummulation of Mrate (M0)
    sumM0=cumtrapz(Mrate,t)
    sumMw=M02Mw(sumM0)
    sumMw=np.hstack([0,sumMw])
    interp_Mw=np.interp(T,t,sumMw)
    return T,interp_Mw


#------setting in the .fq.py------#
home='/projects/tlalollin/jiunting/Fakequakes/'
#project_name='Chile_full'
project_name='Chile_full_new'
run_name='subduction'
out_dir='Chile_full_new_STF'  #save the STF here
sampleT=np.arange(5,515,5)
#---------------------------------#

n_senarios=27200
EQids=np.array(['%06d'%(i) for i in range(n_senarios)])
for ID in EQids:
    T,sumMw=get_accM0(ID,T=sampleT)
    #np.save('./Chile_27200_STF/'+project_name+'.'+ID+'.npy',sumMw)
    np.save('./'+out_dir+'/'+project_name+'.'+ID+'.npy',sumMw)

