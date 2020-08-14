#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 14:55:20 2020

@author: timlin
"""
import numpy as np

def rdata_ENZ(home,project_name,run_name,Sta_ordering,tcs_samples=np.arange(5,515,5),outdir='Tmpout_X'):
    #read data and output E,N,Z time serirs in .npy for the desired sampling rate
    import glob
    import obspy
    import numpy as np
    import os
    
    if not(os.path.exists(outdir)):
        os.makedirs(outdir)
    
    ruptures=glob.glob(home+project_name+'/'+'output/waveforms/'+run_name+'*')
    ruptures.sort()
    for nrupt,rupt in enumerate(ruptures):    
        eqid=rupt.split('/')[-1].split('.')[-1]
        #logf=home+project_name+'/'+'output/ruptures/'+run_name+'.'+eqid+'.log'
        #Mw=readMw(logf)
        #print(rupt,eqid,logf,Mw)
        print(rupt,eqid)
        #A_Mw.append(Mw) #save Mw in a list
        all_stname=np.genfromtxt(Sta_ordering,'S') #careful! might have issue. use .decode() of all stations
        all_stname=[i.decode() for i in all_stname]
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
                E_interp=np.interp(tcs_samples,time,E)
                N_interp=np.interp(tcs_samples,time,N)
                Z_interp=np.interp(tcs_samples,time,Z)
            else:
                #no data because the station is too far away
                E_interp=np.zeros(len(tcs_samples))
                N_interp=np.zeros(len(tcs_samples))
                Z_interp=np.zeros(len(tcs_samples))

            sav_E_sta.append(E_interp) #n points(102) for each station by m stations(121)
            sav_N_sta.append(N_interp)
            sav_Z_sta.append(Z_interp)
        sav_E_sta=np.array(sav_E_sta)
        sav_N_sta=np.array(sav_N_sta)
        sav_Z_sta=np.array(sav_Z_sta)
        #save data individually
        np.save(outdir+'/'+project_name+'.'+eqid+'.E.npy',sav_E_sta)
        np.save(outdir+'/'+project_name+'.'+eqid+'.N.npy',sav_N_sta)
        np.save(outdir+'/'+project_name+'.'+eqid+'.Z.npy',sav_Z_sta)


def rSTF(home,project_name,run_name,tcs_samples=np.arange(5,515,5),outdir='Tmpout_y'):
    import numpy as np
    from mudpy import viewFQ
    from scipy.integrate import cumtrapz
    import os,glob
    
    def M02Mw(M0):
        Mw=(2.0/3)*np.log10(M0*1e7)-10.7 #M0 input is dyne-cm, convert to N-M by 1e7
        return(Mw)
     
    def get_accM0(ruptfile,T=np.arange(5,515,5)):
        t,Mrate=viewFQ.source_time_function(ruptfile,epicenter=None,dt=0.05,t_total=520,stf_type='dreger',plot=False)
        #get accummulation of Mrate (M0)
        sumM0=cumtrapz(Mrate,t)
        sumMw=M02Mw(sumM0)
        sumMw=np.hstack([0,sumMw])
        interp_Mw=np.interp(T,t,sumMw)
        return T,interp_Mw
    
    if not(os.path.exists(outdir)):
        os.makedirs(outdir)
    ruptures=glob.glob(home+project_name+'/'+'output/ruptures/'+run_name+'*rupt')
    ruptures.sort()
    for rupt in ruptures:
        T,sumMw=get_accM0(ruptfile=rupt,T=tcs_samples)
        eqid=rupt.split('/')[-1].split('.')[1]
        np.save(outdir+'/'+project_name+'.'+eqid+'.npy',sumMw)
        
    


def gen_Xydata_list(X_dirs,y_dirs,outname='Datalist'):
    #make data list for MLARGE training
    #dirs can be multipath
    import glob
    #X_dirs=['/projects/tlalollin/jiunting/Fakequakes/run/Chile_full_new_ENZ']
    if not (type(X_dirs) is list):
        X_dirs=[X_dirs]
    if not (type(y_dirs) is list):
        y_dirs=[y_dirs]
    EE=[]
    NN=[]
    ZZ=[]
    yy=[]
    for Xdir,ydir in zip(X_dirs,y_dirs):
        E_files=glob.glob(Xdir+'/*.E.npy')
        E_files.sort()
        N_files=glob.glob(Xdir+'/*.N.npy')
        N_files.sort()
        Z_files=glob.glob(Xdir+'/*.Z.npy')
        Z_files.sort()
        y_files=glob.glob(ydir+'/*.npy')
        y_files.sort()
        EE=EE+E_files
        NN=NN+N_files
        ZZ=ZZ+Z_files
        yy=yy+y_files
    OUTE=open(outdir+'_E'+'.txt','w')
    OUTN=open(outdir+'_N'+'.txt','w')
    OUTZ=open(outdir+'_Z'+'.txt','w')
    OUTy=open(outdir+'_y'+'.txt','w')
    for line in range(len(EE)):
        OUTE.write('%s\n'%(EE[line]))
        OUTN.write('%s\n'%(NN[line]))
        OUTZ.write('%s\n'%(ZZ[line]))
        OUTy.write('%s\n'%(yy[line]))
    OUTE.close()
    OUTN.close()
    OUTZ.close()
    OUTy.close()
    
    
def get_EQinfo(home,project_name,run_name,outname='EQinfo'):
    import glob
    import numpy as np
    
    def get_source_Info(logfile):
        #Input log file path from the rupture directory
        #output hypo lon,lat
        IN1=open(logfile,'r')
        for line in IN1.readlines():
            #if 'Target magnitude' in line:
            #    Tmw.append(float(line.split()[3]))
            #    continue
            if 'Actual magnitude' in line:
                Mw=float(line.split()[3])
                #mw.append(float(line.split()[3]))
                continue
            if 'Hypocenter (lon,lat,z[km])' in line:
                Hypo_xyz=line.split(':')[-1].replace('(','').replace(')','').strip().split(',')
                #print Hypo_xyz
                #Hypo.append([ float(Hypo_xyz[0]),float(Hypo_xyz[1]),float(Hypo_xyz[2]) ])
                continue
                #break
            if 'Centroid (lon,lat,z[km])' in line:
                Cent_xyz=line.split(':')[-1].replace('(','').replace(')','').strip().split(',')
                #Cent.append([ float(Cent_xyz[0]),float(Cent_xyz[1]),float(Cent_xyz[2]) ])
                break
        IN1.close()
        return Mw,float(Hypo_xyz[0]),float(Hypo_xyz[1]),float(Hypo_xyz[2]),float(Cent_xyz[0]),float(Cent_xyz[1]),float(Cent_xyz[2])
    
    def get_hyposlip(rupt_file,eqlon,eqlat):
        A=np.genfromtxt(rupt_file)
        lon=A[:,1]
        lat=A[:,2]
        SS=A[:,8]
        DS=A[:,9]
        Slip=(SS**2.0+DS**2.0)**0.5
        hypo_misft=np.abs(lon-eqlon)**2.0+np.abs(lat-eqlat)**2.0
        hypoidx=np.where(hypo_misft==np.min(hypo_misft))[0][0]
        hyposlip=Slip[hypoidx]
        return hyposlip, Slip.max()
    
    OUT1=open(outname+'.EQinfo')
    logs=glob.glob(home+project_name+'/'+'output/ruptures/'+run_name+'*log')
    logs.sort()
    rupts=glob.glob(home+project_name+'/'+'output/ruptures/'+run_name+'*rupt')
    rupts.sort()
    for n,logfile in enumerate(logs):
        if n==0:
            OUT1.write('#Mw Hypo_lon Hypo_lat Hypo_dep Cen_lon Cen_lat Cen_dep hypo_slip max_slip\n')
        ID='%06d'%(n)
        Mw,eqlon,eqlat,eqdep,cenlon,cenlat,cendep=get_source_Info(logfile)
        hypo_slip,max_slip=get_hyposlip(rupts[n],eqlon,eqlat)
        OUT1.write('%s  %.4f  %.6f %.6f %.2f   %.6f %.6f %.2f %f %f\n'%(ID,Mw,eqlon,eqlat,eqdep,cenlon,
                                                                        cenlat,cendep,hypo_slip,max_slip))
    OUT1.close()
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    