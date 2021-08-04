#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 04 12:08:12 2021

@author: timlin
"""

import numpy as np
import glob
import obspy

#home='/projects/tlalollin/jiunting/Fakequakes/'
#project_name='Chile_full_new'
#run_name='subduction'


def make_tcs(home,project_name,run_name,GF_list,ID,T,outName):
    '''
        make GMT timeseries input from FQ data
        ID: for example '027189'
        T: sampling time
    '''
    #home='/projects/tlalollin/jiunting/Fakequakes/'
    #project_name='Chile_full_new'
    #run_name='subduction'
    staloc = np.genfromtxt(home+project_name+'/data/station_info/'+GF_list) #station file (not include name)
    staloc_name = np.genfromtxt(home+project_name+'/data/station_info/'+GF_list,'S')
    staloc_name_idx = {i[0].decode():ii for ii,i in enumerate(staloc_name)}
    # find all time series sac files
    D_E = glob.glob(home+project_name+'/output/waveforms/'+run_name+'.'+ID+'/*.LYE.sac')
    D_N = glob.glob(home+project_name+'/output/waveforms/'+run_name+'.'+ID+'/*.LYN.sac')
    D_Z = glob.glob(home+project_name+'/output/waveforms/'+run_name+'.'+ID+'/*.LYZ.sac')
    assert len(D_E)==len(D_N)==len(D_Z), "missing component"
    D_E.sort()
    D_N.sort()
    D_Z.sort()
    outNameID = outName+'.'+ID+'.txt'
    OUT1 = open(outNameID,'w')
    OUT1.write('#lon lat name time E N Z\n')
    for d_E,d_N,d_Z in zip(D_E,D_N,D_Z):
        assert d_E.split('/')[-1].split('.')[0]==d_N.split('/')[-1].split('.')[0]==d_Z.split('/')[-1].split('.')[0],"station name inconsistent"
        staName = d_E.split('/')[-1].split('.')[0]
        staLon = staloc[staloc_name_idx[staName]][1]
        staLat = staloc[staloc_name_idx[staName]][2]
        d1 = obspy.read(d_E)
        d2 = obspy.read(d_N)
        d3 = obspy.read(d_Z)
        d1 = np.interp(T,d1[0].times(),d1[0].data)
        d2 = np.interp(T,d2[0].times(),d2[0].data)
        d3 = np.interp(T,d3[0].times(),d3[0].data)
        for i,t in enumerate(T):
            OUT1.write('%f %f %s %d %f %f %f\n'%(staLon,staLat,staName,t,d1[i],d2[i],d3[i]))

    OUT1.close()






class fault():
    '''
        usage example: f=make_GMT.fault([-123,52,30],8.0,120,35,100,30,'test.out')
                       f.gen_fault()
                       check the result in output file or print(f.fault)
    '''
    def __init__(self,centroid,Mw,strike,dip,length,width,fout):
        self.centroid = centroid
        self.Mw = Mw
        self.strike = strike
        self.dip = strike
        self.length = strike
        self.width = strike
        self.fout = fout
        self.fault = None

    def makefault(self,fout,strike,dip,nstrike,dx_dip,dx_strike,epicenter,num_updip,num_downdip,rise_time):
        '''
            Copied from Mudpy
            Make a planar fault
            strike - Strike angle (degs)
            dip - Dip angle (degs)200/5
        '''
        from numpy import arange,sin,cos,deg2rad,r_,ones,arctan,rad2deg,zeros,isnan,unique,where,argsort
        import pyproj
        #fout,strike,dip,nstrike,dx_dip,dx_strike,epicenter,num_updip,num_downdip,rise_time =self.fout,self.strike,self.dip,self.nstrike,self.dx_dip,self.dx_strike,self.epicenter,self.num_updip,self.num_downdip,self.rise_time

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
        # fault file in array
        A = []
        # write output
        f=open(fout,'w')
        for k in range(len(x)):
            out='%i\t%.6f\t%.6f\t%.3f\t%.2f\t%.2f\t%.1f\t%.1f\t%.2f\t%.2f\n' % (k+1,loout[k],laout[k],zout[k],strike[k],dip[k],tw[k],rise[k],L[k],W[k])
            A.append([k+1,loout[k],laout[k],zout[k],strike[k],dip[k],tw[k],rise[k],L[k],W[k]])
            f.write(out)
        f.close()
        self.fault = np.array(A).copy()

    def gen_fault(self):
        #default params
        dx_strike = 10
        dx_dip = 8
        fout = self.fout
        Mw = self.Mw
        centroid = self.centroid
        strike = self.strike
        dip = self.dip
        length = self.length
        width = self.width
        num_columns = int(np.max([length//10,1]))   #minimum 1
        num_updip = int((width//8)//2)
        num_downdip = num_updip
        # call makefault function
        self.makefault(fout,strike,dip,num_columns,dx_dip,dx_strike,centroid,num_updip,num_downdip,1.0)







