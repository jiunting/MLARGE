#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 12:55:51 2020

@author: timlin
"""

#control file for MLARGE
from mlarge import preprocessing
import numpy as np


#============data preparation===============
save_data_from_FQ=True
#these paths are same as FQs path in Mudpy
home='/projects/tlalollin/jiunting/Fakequakes/'
project_name='Chile_full_new'
run_name='subduction'
tcs_samples=np.arange(5,515,5)
outdir_X='Chile_full_ENZ'
outdir_y='Chile_full_y'

GFlist='Chile_GNSS.gflist'
Sta_ordering='ALL_staname_order.txt'


if save_data_from_FQ:
    preprocessing.rdata_ENZ(home,project_name,run_name,Sta_ordering,tcs_samples=np.arange(5,515,5),outdir=outdir_X)
    preprocessing.rSTF(home,project_name,run_name,tcs_samples=np.arange(5,515,5),outdir=outdir_y)
    


'''
info_files={
        'GFlist':'Chile_GNSS.gflist', #GFlist for Fakequakes
        'Sta_ordering':'ALL_staname_order.txt', #ordering for features X
        'E':'E_full_newEQlist.txt', #list of E data
        'N':'N_full_newEQlist.txt',
        'Z':'Z_full_newEQlist.txt',
        'y':None, #read y directly in the generator
        }

#the structure in default: Dense+Dense+Drop+LSTM+Dense+Dense+Dense+Dense+Drop+Output
train_params={
        'Neurons':[256,256,128,128,64,32,8],
        'Drops':[0.2,0.2],
        'BS':128, #Batch Size for training
        'BS_valid':1024, ######CHANGE it later!!!!! 1024
        'BS_test':819, #batch size for testing
        'scales':[0,1], #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by log10(X)
        'Testnum':'00',
        'FlatY':False, #using flat labeling?
        'NoiseP':0.0, #possibility of noise event
        'Noise_level':[1,10,20,30,40,50,60,70,80,90], #GNSS noise level
        'rm_stans':[0,115], #remove number of stations from 0~115
        'Min_stan_dist':[4,3], #minumum 4 stations within 3 degree
        'Loss_func':'mse', #can be default loss function string or self defined loss
        }

'''
