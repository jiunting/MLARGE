#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 12:55:51 2020

@author: timlin
"""

#control file for MLARGE
from mlarge import preprocessing
from mlarge import mlarge_model
import numpy as np


#============data preparation===============
save_data_from_FQ = 0    #read .txt data and generate .npy data from FQ folders
gen_list = 0             #generate abspath file list for MLARGE training
gen_EQinfo = 0           #generate EQinfo file
train_model = 0
test_model = 1

#---these paths are same as FQs path in Mudpy---
home = '/projects/tlalollin/jiunting/Fakequakes/'
project_name = 'Chile_full_new'
run_name = 'subduction'

#---set the sampled time and output name---
tcs_samples = np.arange(5,515,5)
outdir_X = 'Chile_full_ENZ'
outdir_y = 'Chile_full_y'
out_list = 'Chile_full_Xylist'
out_EQinfo = 'Chile_full_SRC'

#---input file for station loc and ordering for training---
GFlist = 'Chile_GNSS.gflist'
Sta_ordering = 'ALL_staname_order.txt'



if save_data_from_FQ:
    preprocessing.rdata_ENZ(home,project_name,run_name,Sta_ordering,tcs_samples=np.arange(5,515,5),outdir=outdir_X)
    preprocessing.rSTF(home,project_name,run_name,tcs_samples=np.arange(5,515,5),outdir=outdir_y)
    center_fault=1519 #1519 for Chile
    preprocessing.get_fault_LW_cent_batch(home,project_name,run_name,center_fault,tcs_samples=np.arange(5,515,5),outdir=outdir_y)

if gen_list:
    preprocessing.gen_Xydata_list(outdir_X,outdir_y,outname=out_list)

if gen_EQinfo:
    preprocessing.get_EQinfo(home,project_name,run_name,outname=out_EQinfo)


files={
        'GFlist':'Chile_GNSS.gflist', #GFlist for Fakequakes
        'Sta_ordering':'ALL_staname_order.txt', #ordering for features X
        'EQinfo':'Chile_full_SRC.EQinfo',
        'E':'Chile_full_Xylist_E.txt', #list of E data
        'N':'Chile_full_Xylist_N.txt',
        'Z':'Chile_full_Xylist_Z.txt',
        'y':'Chile_full_Xylist_y.txt', #read y directly in the generator
        }

#the structure in default: Dense+Dense+Drop+LSTM+Dense+Dense+Dense+Dense+Drop+Output
train_params={
        'Neurons':[256,256,128,128,64,32,8],
        'epochs':50000,
        'Drops':[0.2,0.2],
        'BS':128, #Batch Size for training
        'BS_valid':1024, #validation batch size
        'BS_test':8192, #batch size for testing
        'scales':[0,1], #(x-scaels[0])/scales[1] #Not scale here, but scale in the function by log10(X)
        'Testnum':'00',
        'FlatY':False, #using flat labeling?
        'NoiseP':0.0, #possibility of noise event
        'Noise_level':[1,10,20,30,40,50,60,70,80,90], #GNSS noise level
        'rm_stans':[0,115], #remove number of stations from 0~115
        'Min_stan_dist':[4,3], #minumum 4 stations within 3 degree
        'Loss_func':'mse', #can be default loss function string or self defined loss
        }


#Train MLARGE
if train_model:
    mlarge_model.train(files,train_params)


#Test MLARGE
if test_model:
    import mlarge.scaling as scale
    Model_path='Lin2020'
    X=np.load('Xtest00.npy')
    y=np.load('ytest00.npy')
    f=lambda a : a
    M=mlarge_model.Model(Model_path,X,y,f,scale.back_scale_X,f,scale.back_scale_y)
    M.predict()
    print(M.predictions) # predicted Mw time series
    print(M.real) #the real Mw
    #calculate model accuracy with 0.3 threshold
    M.accuracy(0.3,True)
    print('Mean model accuracy is {}'.format(M.sav_acc.mean())) #model accuracy 
    #plot the accuracy as a function of time
    M.plot_acc(T=np.arange(102)*5+5,show=True)


'''
#Continue MLARGE model
Model_path='Lin2020'
mlarge_model.train(files,train_params,'Lin2020')  #

'''
