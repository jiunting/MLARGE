#get source parameters and output in .txt

import numpy as np


'''
Target magnitude: Mw 7.8000
Actual magnitude: Mw 7.8810
Hypocenter (lon,lat,z[km]): (-70.886786,-19.373478,27.36)
Hypocenter time: 2016-09-07T14:42:26.000000Z
Centroid (lon,lat,z[km]): (-71.016474,-19.507540,22.85)
'''

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




#IDs=['%06d'%(i) for i in range(27200)]
IDs=['%06d'%(i) for i in range(50000)]
for n,ID in enumerate(IDs):
    if n==0:
        #open the file and write header
        #OUT1=open('Chile_full_27200_source.txt','w')
        #OUT1=open('Chile_full_27200_source_slip.txt','w')
        #OUT1=open('Chile_large2_source_slip.txt','w')
        OUT1=open('Chile_full_new_source_slip.txt','w')
        #OUT1.write('#Mw Hypo_lon Hypo_lat Hypo_dep Cen_lon Cen_lat Cen_dep\n')
        OUT1.write('#Mw Hypo_lon Hypo_lat Hypo_dep Cen_lon Cen_lat Cen_dep hypo_slip max_slip\n')
    #logfile='/projects/tlalollin/jiunting/Fakequakes/Chile_full/output/ruptures/subduction.'+ID+'.log'
    #logfile='/projects/tlalollin/jiunting/Fakequakes/Chile_large_2/output/ruptures/subduction.'+ID+'.log'
    logfile='/projects/tlalollin/jiunting/Fakequakes/Chile_full_new/output/ruptures/subduction.'+ID+'.log'
    Mw,eqlon,eqlat,eqdep,cenlon,cenlat,cendep=get_source_Info(logfile)
    #ruptfile='/projects/tlalollin/jiunting/Fakequakes/Chile_full/output/ruptures/subduction.'+ID+'.rupt'
    #ruptfile='/projects/tlalollin/jiunting/Fakequakes/Chile_large_2/output/ruptures/subduction.'+ID+'.rupt'
    ruptfile='/projects/tlalollin/jiunting/Fakequakes/Chile_full_new/output/ruptures/subduction.'+ID+'.rupt'
    hypo_slip,max_slip=get_hyposlip(ruptfile,eqlon,eqlat)
    #OUT1.write('%s  %.4f  %.6f %.6f %.2f   %.6f %.6f %.2f\n'%(ID,Mw,eqlon,eqlat,eqdep,cenlon,cenlat,cendep)) #only source params
    OUT1.write('%s  %.4f  %.6f %.6f %.2f   %.6f %.6f %.2f %f %f\n'%(ID,Mw,eqlon,eqlat,eqdep,cenlon,cenlat,cendep,hypo_slip,max_slip)) #
    #print(ID)
    #print(Mw,eqlon,eqlat,eqdep,cenlon,cenlat,cendep)
    #break

OUT1.close()


