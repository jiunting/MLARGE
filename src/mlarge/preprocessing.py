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

    #another functon for M02Mw
    #def M02Mw(M0):
    #    Mw=(np.log10(M0)-9.1)/1.5 
    #    return(Mw)
     
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
        np.save(outdir+'/'+project_name+'.'+eqid+'.STF.npy',sumMw * 0.1) #scale the Mw = 0.1Mw
        

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
    OUTE=open(outname+'_E'+'.txt','w')
    OUTN=open(outname+'_N'+'.txt','w')
    OUTZ=open(outname+'_Z'+'.txt','w')
    OUTy=open(outname+'_y'+'.txt','w')
    for line in range(len(EE)):
        OUTE.write('%s\n'%(EE[line]))
        OUTN.write('%s\n'%(NN[line]))
        OUTZ.write('%s\n'%(ZZ[line]))
        OUTy.write('%s\n'%(yy[line]))
    OUTE.close()
    OUTN.close()
    OUTZ.close()
    OUTy.close()


def gen_multi_Xydata_list(X_dirs,y_dirs,y_type=['STF','Lon','Lat','Dep','Length','Width'],outname='Datalist'):
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
    yy={}
    for Xdir,ydir in zip(X_dirs,y_dirs):
        E_files=glob.glob(Xdir+'/*.E.npy')
        E_files.sort()
        N_files=glob.glob(Xdir+'/*.N.npy')
        N_files.sort()
        Z_files=glob.glob(Xdir+'/*.Z.npy')
        Z_files.sort()
        for yt in y_type:
            i_y_files=glob.glob(ydir+'/*.'+yt+'.npy')
            i_y_files.sort()
            try:
                yy[yt]+=i_y_files
            except:
                yy[yt]=i_y_files[:]

        EE=EE+E_files #joint files from different directories if any
        NN=NN+N_files
        ZZ=ZZ+Z_files
    OUTE=open(outname+'_E'+'.txt','w')
    OUTN=open(outname+'_N'+'.txt','w')
    OUTZ=open(outname+'_Z'+'.txt','w')
    for line in range(len(EE)):
        OUTE.write('%s\n'%(EE[line]))
        OUTN.write('%s\n'%(NN[line]))
        OUTZ.write('%s\n'%(ZZ[line]))
    OUTE.close()
    OUTN.close()
    OUTZ.close()
    #output y
    for ik in yy.keys():
        OUT_ik=open(outname+'_'+ik+'.txt','w')
        for line in yy[ik]:
            OUT_ik.write('%s\n'%(line))
        OUT_ik.close()



    
def get_EQinfo(home,project_name,run_name,outname='EQinfo',fmt='short'):
    import glob
    import numpy as np
    # fmt: short or long
    def get_source_Info(logfile,fmt='short'):
        #Input log file path from the rupture directory
        #output hypo lon,lat
        IN1=open(logfile,'r')
        for line in IN1.readlines():
            #if 'Target magnitude' in line:
            #    Tmw.append(float(line.split()[3]))
            #    continue
            if 'Lmax' in line:
                L = float(line.split()[3])
                continue
            if 'Wmax' in line:
                W = float(line.split()[3])
                continue
            if 'Target magnitude' in line:
                tar_Mw = float(line.split()[3])
                continue
            if 'Actual magnitude' in line:
                Mw = float(line.split()[3])
                #mw.append(float(line.split()[3]))
                continue
            if 'Hypocenter (lon,lat,z[km])' in line:
                Hypo_xyz = line.split(':')[-1].replace('(','').replace(')','').strip().split(',')
                #print Hypo_xyz
                #Hypo.append([ float(Hypo_xyz[0]),float(Hypo_xyz[1]),float(Hypo_xyz[2]) ])
                continue
                #break
            if 'Centroid (lon,lat,z[km])' in line:
                Cent_xyz=line.split(':')[-1].replace('(','').replace(')','').strip().split(',')
                #Cent.append([ float(Cent_xyz[0]),float(Cent_xyz[1]),float(Cent_xyz[2]) ])
                break
        IN1.close()
        if fmt=='short':
            return Mw,float(Hypo_xyz[0]),float(Hypo_xyz[1]),float(Hypo_xyz[2]),float(Cent_xyz[0]),float(Cent_xyz[1]),float(Cent_xyz[2])
        elif fmt=='long':
            return Mw,float(Hypo_xyz[0]),float(Hypo_xyz[1]),float(Hypo_xyz[2]),float(Cent_xyz[0]),float(Cent_xyz[1]),float(Cent_xyz[2]), tar_Mw, L, W
        else:
            print('undefined fmt=%s [short/long]'%(fmt))
            return
    
    def get_slip_Info(rupt_file,eqlon,eqlat,fmt='short'):
        A = np.genfromtxt(rupt_file)
        lon = A[:,1]
        lat = A[:,2]
        SS = A[:,8]
        DS = A[:,9]
        Rise = A[:,7]
        Slip = (SS**2.0+DS**2.0)**0.5
        idx_slip = np.where(Slip!=0)[0] #index of slip (some subfaults are zero slip)
        hypo_misft=np.abs(lon-eqlon)**2.0+np.abs(lat-eqlat)**2.0
        hypoidx=np.where(hypo_misft==np.min(hypo_misft))[0][0]
        hyposlip=Slip[hypoidx]
        if fmt=='short':
            return hyposlip, Slip.max()
        elif fmt=='long':
            return hyposlip, Slip.max(), Slip[idx_slip].mean(), Slip[idx_slip].std(), Rise[idx_slip].max(), Rise[idx_slip].mean(), Rise[idx_slip].std()
        else:
            print('undefined fmt=%s [short/long]'%(fmt))
            return
    
    OUT1=open(outname+'.EQinfo','w')
    logs=glob.glob(home+project_name+'/'+'output/ruptures/'+run_name+'*log')
    logs.sort()
    rupts=glob.glob(home+project_name+'/'+'output/ruptures/'+run_name+'*rupt')
    rupts.sort()
    for n,logfile in enumerate(logs):
        if n==0:
            if fmt=='short':
                OUT1.write('#Mw Hypo_lon Hypo_lat Hypo_dep Cen_lon Cen_lat Cen_dep hypo_slip max_slip\n')
            elif fmt=='long':
                OUT1.write('#Mw Hypo_lon Hypo_lat Hypo_dep Cen_lon Cen_lat Cen_dep hypo_slip max_slip mean_slip std_slip max_rise mean_rise std_rise\n')
            else:
                print('undefined fmt=%s [short/long]'%(fmt))
                return

        ID='%06d'%(n)
        Mw, eqlon, eqlat, eqdep, cenlon, cenlat, cendep, tar_Mw, L, W = get_source_Info(logfile,fmt='long')
        hypo_slip, max_slip, mean_slip, std_slip, max_rise, mean_rise, std_rise = get_slip_Info(rupts[n],eqlon,eqlat,fmt='long')
        if fmt=='short':
            OUT1.write('%s  %.4f  %.6f %.6f %.2f   %.6f %.6f %.2f %f %f\n'%(ID,Mw,eqlon,eqlat,eqdep,cenlon,
                                                                        cenlat,cendep,hypo_slip,max_slip))
        elif fmt=='long':
            OUT1.write('%s  %.4f  %.6f %.6f %.2f   %.6f %.6f %.2f %f %f %f %f %f %f %f\n'%(ID,Mw,eqlon,eqlat,eqdep,cenlon,
                                                                            cenlat,cendep,hypo_slip,max_slip, mean_slip, std_slip, max_rise, mean_rise, std_rise))
        else:
            print('undefined fmt=%s [short/long]'%(fmt))
            return

    OUT1.close()
        

    
def get_fault_LW_cent(rupt_file,dist_strike,dist_dip,center_fault,tcs_samples=np.arange(5,515,5),find_center=False):
    #get fault Length/Width and centroid location one-by-one
    #rupt_file : path of .rupt file
    #dist_strike : path of distance matrix for strike
    #center_fault :index for the center subfault
    #plot fault and check where is the center(only needs to be done once)
    dip=np.load(dist_dip)
    strike=np.load(dist_strike)
    new_x=dip[:,center_fault]
    new_y=strike[:,center_fault]
    rupt=np.genfromtxt(rupt_file)
    rupt_time=rupt[:,-2]
    slip=(rupt[:,8]**2 + rupt[:,9]**2)**0.5
    mu=rupt[:,-1]
    Area=rupt[:,10]*rupt[:,11]
    if find_center:
        order_x = np.zeros(len(dip)) #x=along D; y=along S
        order_y = np.zeros(len(strike))
        for i in range(len(dip)):
            #for each i, what you know is points are larger(left) or smaller(right)
            large_idx = np.where(dip[i,:]>0)[0]
            small_idx = np.where(dip[i,:]<0)[0]
            order_x[large_idx] -= 1
            order_x[small_idx] += 1
        for j in range(len(strike)):
            large_idx = np.where(strike[j,:]>0)[0]
            small_idx = np.where(strike[j,:]<0)[0]
            order_y[large_idx] += 1
            order_y[small_idx] -= 1
        dist_xy=(order_x**2+order_y**2)**0.5
        cent_idx=np.where(dist_xy==np.min(dist_xy))[0][0]
        print('center index=',cent_idx)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(1,2,1)
        plt.scatter(rupt[:,1],rupt[:,2])
        for nf in range(len(rupt)):
            plt.text(rupt[nf,1],rupt[nf,2],rupt[nf,0])
        plt.subplot(1,2,2)
        plt.scatter(order_x,order_y,c=np.arange(len(order_x)),cmap=plt.cm.jet)
        for k in range(len(order_x)):
            plt.text(order_x[k],order_y[k],int(rupt[k,0]),fontsize=6)
        plt.plot(order_x[cent_idx],order_y[cent_idx],'r^')
        plt.show()
    ##start calculate L/W/cent
    rupt_L=[]
    rupt_W=[]
    cen_lon=[]
    cen_lat=[]
    cen_dep=[]
    for t in tcs_samples:
        ind=np.where( (slip!=0) & (rupt_time<=t) )[0]
        rupt_W.append(new_x[ind].max() - new_x[ind].min())
        rupt_L.append(new_y[ind].max() - new_y[ind].min())
        curr_M0=mu[ind]*Area[ind] * slip[ind]
        cen_lon.append( np.sum(rupt[ind,1] * curr_M0 / curr_M0.sum()) )
        cen_lat.append( np.sum(rupt[ind,2] * curr_M0 / curr_M0.sum()) )
        cen_dep.append( np.sum(rupt[ind,3] * curr_M0 / curr_M0.sum()) )
    return rupt_L,rupt_W,cen_lon,cen_lat,cen_dep


    


def get_fault_LW_cent_batch(home,project_name,run_name,center_fault,tcs_samples=np.arange(5,515,5),outdir='Tmpout_y'):
    #get all fault Length/Width and centroid location
    #make sure there is only one distance matrix for strike/dip
    import glob
    import os
    dist_dip=glob.glob(home+project_name+'/data/distances/'+'*'+run_name+'*.dip.npy')[0]
    dist_strike=glob.glob(home+project_name+'/data/distances/'+'*'+run_name+'*.strike.npy')[0]
    print('Loading dip distance matrix:',dist_dip)
    print('Loading strike distance matrix:',dist_strike)
    dip=np.load(dist_dip)
    strike=np.load(dist_strike)
    new_x=dip[:,center_fault]
    new_y=strike[:,center_fault]
    if not(os.path.exists(outdir)):
        os.makedirs(outdir)
    
    ruptures=glob.glob(home+project_name+'/'+'output/ruptures/'+run_name+'*.rupt')
    ruptures.sort()
    for rupt_file in ruptures:
        rupt=np.genfromtxt(rupt_file)
        eqid=rupt_file.split('/')[-1].split('.')[-2]
        rupt_time=rupt[:,-2]
        slip=(rupt[:,8]**2 + rupt[:,9]**2)**0.5
        mu=rupt[:,-1]
        Area=rupt[:,10]*rupt[:,11]
        ##start calculate L/W/cent
        rupt_L=[]
        rupt_W=[]
        cen_lon=[]
        cen_lat=[]
        cen_dep=[]
        for t in tcs_samples:
            ind=np.where( (slip!=0) & (rupt_time<=t) )[0]
            rupt_W.append(new_x[ind].max() - new_x[ind].min())
            rupt_L.append(new_y[ind].max() - new_y[ind].min())
            curr_M0=mu[ind]*Area[ind] * slip[ind]
            cen_lon.append( np.sum(rupt[ind,1] * curr_M0 / curr_M0.sum()) )
            cen_lat.append( np.sum(rupt[ind,2] * curr_M0 / curr_M0.sum()) )
            cen_dep.append( np.sum(rupt[ind,3] * curr_M0 / curr_M0.sum()) )
        rupt_W=np.array(rupt_W)
        rupt_L=np.array(rupt_L)
        cen_lon=np.array(cen_lon)
        cen_lat=np.array(cen_lat)
        cen_dep=np.array(cen_dep)
        #save the result individually
        np.save(outdir+'/'+project_name+'.'+eqid+'.Width.npy',rupt_W * 0.01)
        np.save(outdir+'/'+project_name+'.'+eqid+'.Length.npy',rupt_L * 0.01)
        np.save(outdir+'/'+project_name+'.'+eqid+'.Lon.npy',cen_lon * -0.1)
        np.save(outdir+'/'+project_name+'.'+eqid+'.Lat.npy',cen_lat * -0.1)
        np.save(outdir+'/'+project_name+'.'+eqid+'.Dep.npy',cen_dep * 0.1)

    
    
    
    
    
    
