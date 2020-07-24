#make data list for MLARGE training
import numpy as np
import glob

#dirs=['/projects/tlalollin/jiunting/Fakequakes/run/Chile_27200_ENZ','/projects/tlalollin/jiunting/Fakequakes/run/Chile_small_ENZ','/projects/tlalollin/jiunting/Fakequakes/run/Chile_small_2_ENZ']

dirs=['/projects/tlalollin/jiunting/Fakequakes/run/Chile_full_new_ENZ']
EE=[]
NN=[]
ZZ=[]
for Dir in dirs:
    E_files=glob.glob(Dir+'/*.E.npy')
    E_files.sort()
    N_files=glob.glob(Dir+'/*.N.npy')
    N_files.sort()
    Z_files=glob.glob(Dir+'/*.Z.npy')
    Z_files.sort()
    EE=EE+E_files
    NN=NN+N_files
    ZZ=ZZ+Z_files

OUTE=open('E_full_newEQlist.txt','w')
OUTN=open('N_full_newEQlist.txt','w')
OUTZ=open('Z_full_newEQlist.txt','w')
for line in range(len(EE)):
    OUTE.write('%s\n'%(EE[line]))
    OUTN.write('%s\n'%(NN[line]))
    OUTZ.write('%s\n'%(ZZ[line]))


OUTE.close()
OUTN.close()
OUTZ.close()
