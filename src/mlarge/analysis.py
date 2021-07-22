# some data analysis tools not related to NN
import numpy as np
import obspy


def gen_STA_from_file(sta_loc_file,sta_order_file):
    '''
        generate STA for find_sta_hypo or find_sta_fault input
    '''
    STAINFO = np.genfromtxt(sta_loc_file,'S12',skip_header=1)
    STAINFO = {ista[0].decode():[float(ista[1]),float(ista[2])]  for ista in STAINFO}
    STA = np.genfromtxt(sta_order_file,'S6')
    STA = {ista:STAINFO[sta.decode()] for ista,sta in enumerate(STA)}
    return STA




def find_sta_hypo(Data,STA,nsta,hypo,dist_thres):
    '''
        return station index within dist_thres from the hypo
    '''
    assert Data.ndim==2,"make sure Data.ndim==2"
    sav_idx = [] # index of the station within dist_thres
    for i in range(nsta):
        # look for station existence code(always appended last), if no code, use the feature value(E,N,U,or PGD)
        if np.any(Data[:,int(-1*nsta+i)]!=0):  # if any data for all time epochs at ith station has value
            # check if this station is close enough
            stlon,stlat = STA[i]
            dist = obspy.geodetics.locations2degrees(lat1=hypo[1],long1=hypo[0],lat2=stlat,long2=stlon)
            if dist<=dist_thres:
                sav_idx.append(i)
    sav_idx = np.array(sav_idx)
    return sav_idx
         



#def find_sta_fault():
