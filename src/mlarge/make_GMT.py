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
    staloc = np.genfromtxt(home+project_name+'/data/'+GF_list) #station file (not include name)
    #staloc_name = np.genfromtxt(home+'/data/'+GF_list,'S')
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
    OUT1 = open(outName,'w')
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




