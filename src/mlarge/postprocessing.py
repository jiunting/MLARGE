#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 22:05:12 2020

@author: timlin
"""

class fault_tool:
    '''
        define a fault object
    '''
    
    def __init__(self,Mw,center,strike,dip,length,width,fout,rupt_path=None,dist_strike=None,dist_dip=None):
        import numpy as np
        import pyproj
        self.Mw = Mw
        self.center = center
        self.strike = strike
        self.dip = dip
        self.length = length
        self.width = width
        self.fout = fout
        self.rupt_path = rupt_path
        self.dist_strike = dist_strike
        self.dist_dip = dist_dip
    
    def set_default_params(self,case_name):
        import numpy as np
        import pyproj
        print('case_name=',case_name)
        if case_name.upper() == "Chile".upper():
            self.center_fault = 1519
            self.tcs_samples = np.arange(5,515,5)
    
    def gen_param_from_rupt(self):
        from mudpy import viewFQ
        from scipy.integrate import cumtrapz
        import numpy as np
        if self.rupt_path==None or self.dist_strike==None or self.dist_dip==None:
            print('rupt_path, dist_strike, dist_dip cannot be None!')
            return
        from mlarge.preprocessing import get_fault_LW_cent
        # set center_fault=1519 for Chile case
        self.length,self.width,cen_lon,cen_lat,cen_dep = get_fault_LW_cent(self.rupt_path,self.dist_strike,self.dist_dip,center_fault=self.center_fault,tcs_samples=self.tcs_samples,find_center=False)
        self.center = [[x,y,z] for x,y,z in zip(cen_lon,cen_lat,cen_dep)]
        # based on the center, find the closest subfault and its strike and dip
        def find_strike_dip(F,lon,lat):
            dist = ((F[:,1]-lon)**2.0 + (F[:,2]-lat)**2.0)**0.5
            idx = np.where(dist==np.min(dist))[0][0]
            return F[idx,4],F[idx,5]

        F = np.genfromtxt(self.rupt_path)
        strike_dip = np.array([find_strike_dip(F,i_center[0],i_center[1]) for i_center in self.center])
        self.strike = list(strike_dip[:,0])
        self.dip = list(strike_dip[:,1])

        # calculate Mw
        def M02Mw(M0):
            Mw=(2.0/3)*np.log10(M0*1e7)-10.7 #M0 input is dyne-cm, convert to N-M by 1e7
            return(Mw)

        def get_accMw(ruptfile,T=np.arange(5,515,5)):
            t,Mrate=viewFQ.source_time_function(ruptfile,epicenter=None,dt=0.05,t_total=520,stf_type='dreger',plot=False)
            #get accummulation of Mrate (M0)
            sumM0=cumtrapz(Mrate,t)
            sumMw=M02Mw(sumM0)
            sumMw=np.hstack([0,sumMw])
            interp_Mw=np.interp(T,t,sumMw)
            return interp_Mw
        self.Mw = get_accMw(self.rupt_path,T=self.tcs_samples)


    '''
    Some work to be done for the below functions
    '''
    def makefault(fout,strike,dip,nstrike,dx_dip,dx_strike,epicenter,num_updip,num_downdip,rise_time):
        '''
        Original function copied from Mudpy:  https://github.com/dmelgarm/MudPy
        Make a planar fault

        strike - Strike angle (degs)
        dip - Dip angle (degs)200/5
        '''
        from numpy import arange,sin,cos,deg2rad,r_,ones,arctan,rad2deg,zeros,isnan,unique,where,argsort
        import pyproj

        proj_angle=180-strike #Angle to use for sin.cos projection (comes from strike)
        y=arange(-nstrike/2+1,nstrike/2+1)*dx_strike
        x=arange(-nstrike/2+1,nstrike/2+1)*dx_strike
        z=ones(x.shape)*epicenter[2]
        y=y*cos(deg2rad(strike))
        x=x*sin(deg2rad(strike))
        #Save teh zero line
        y0=y.copy()
        x0=x.copy()
        z0=z.copy()
        #Initlaize temp for projection up/down dip
        xtemp=x0.copy()
        ytemp=y0.copy()
        ztemp=z0.copy()
        #Get delta h and delta z for up/ddx_dip=1own dip projection
        if num_downdip>0 and num_updip>0:
            dh=dx_dip*cos(deg2rad(dip))
            dz=dx_dip*sin(deg2rad(dip))
            #Project updip lines
            for k in range(num_updip):
                xtemp=xtemp+dh*cos(deg2rad(proj_angle))
                ytemp=ytemp+dh*sin(deg2rad(proj_angle))
                ztemp=ztemp-dz
                x=r_[x,xtemp]
                y=r_[y,ytemp]
                z=r_[z,ztemp]
            #Now downdip lines
            xtemp=x0.copy()
            ytemp=y0.copy()
            ztemp=z0.copy()
            for k in range(num_downdip):
                xtemp=xtemp-dh*cos(deg2rad(proj_angle))
                ytemp=ytemp-dh*sin(deg2rad(proj_angle))
                ztemp=ztemp+dz
                x=r_[x,xtemp]
                y=r_[y,ytemp]
                z=r_[z,ztemp]
        #Now use pyproj to dead reckon anf get lat/lon coordinates of subfaults
        g = pyproj.Geod(ellps='WGS84')
        #first get azimuths of all points, go by quadrant
        az=zeros(x.shape)
        for k in range(len(x)):
            if x[k]>0 and y[k]>0:
                az[k]=rad2deg(arctan(x[k]/y[k]))
            if x[k]<0 and y[k]>0:
                az[k]=360+rad2deg(arctan(x[k]/y[k]))
            if x[k]<0 and y[k]<0:
                az[k]=180+rad2deg(arctan(x[k]/y[k]))
            if x[k]>0 and y[k]<0:
                az[k]=180+rad2deg(arctan(x[k]/y[k]))
        #Quadrant correction
        #Now horizontal distances
        d=((x**2+y**2)**0.5)*1000
        #Now reckon
        lo=zeros(len(d))
        la=zeros(len(d))
        for k in range(len(d)):
            if isnan(az[k]): #No azimuth because I'm on the epicenter
                print('Point on epicenter')
                lo[k]=epicenter[0]
                la[k]=epicenter[1]
            else:
                lo[k],la[k],ba=g.fwd(epicenter[0],epicenter[1],az[k],d[k])
        #Sort them from top right to left along dip
        zunique=np.unique(z)
        for k in range(len(zunique)):
            i=where(z==zunique[k])[0] #This finds all faults at a certain depth
            isort=argsort(la[i]) #This sorths them south to north
            if k==0: #First loop
                laout=la[i][isort]
                loout=lo[i][isort]
                zout=z[i][isort]
            else:
                laout=r_[laout,la[i][isort]]
                loout=r_[loout,lo[i][isort]]
                zout=r_[zout,z[i][isort]]
        #Write to file
        strike=ones(loout.shape)*strike
        dip=ones(loout.shape)*dip
        tw=ones(loout.shape)*0.5
        rise=ones(loout.shape)*rise_time
        L=ones(loout.shape)*dx_strike*1000
        W=ones(loout.shape)*dx_dip*1000
        f=open(fout,'w')
        for k in range(len(x)):
            out='%i\t%.6f\t%.6f\t%.3f\t%.2f\t%.2f\t%.1f\t%.1f\t%.2f\t%.2f\n' % (k+1,loout[k],laout[k],zout[k],strike[k],dip[k],tw[k],rise[k],L[k],W[k])
            f.write(out)
        f.close()

    def gen_fault(fout,Mw,center,strike,dip,length,width):
        dx_strike = 10
        dx_dip = 8
        num_columns = int(np.max([length//10,1]))   #minimum 1
        num_updip = int((width//8)//2)
        num_downdip = num_updip
        # call makefault function
        #fout = 'test_finite001.txt'
        #print('fout,strike,dip,num_columns,dx_dip,dx_strike,center,num_updip,num_downdip,1.0=',fout,strike,dip,num_columns,dx_dip,dx_strike,center,num_updip,num_downdip,1.0)
        makefault(fout,strike,dip,num_columns,dx_dip,dx_strike,center,num_updip,num_downdip,1.0)


