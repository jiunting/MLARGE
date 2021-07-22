# some data analysis tools not related to NN
import numpy as np
import obspy


def gen_STA_from_file(sta_loc_file,sta_order_file):
    '''
        generate STA for find_sta_hypo or find_sta_fault input
        IN: 
          sta_loc_file: station location file. GFlist from Mudpy is acceptable
          sta_order_file: station (name) sorting file used by model training. i.e. so that model know X[0] is staXXX
        OUT:
          STA: a dict that uses sta index as the key, stlon,stlat as value. sta index same sorting with sta_order_file
    '''
    STAINFO = np.genfromtxt(sta_loc_file,'S12',skip_header=1)
    STAINFO = {ista[0].decode():[float(ista[1]),float(ista[2])]  for ista in STAINFO}
    STA = np.genfromtxt(sta_order_file,'S6')
    STA = {ista:STAINFO[sta.decode()] for ista,sta in enumerate(STA)}
    return STA




def find_sta_hypo(Data,STA,nsta,hypo,dist_thres):
    '''
        return station index within dist_thres from the hypo
        IN: 
          Data: feature X with ndim=2. format = [time_points,features]
          STA: output from gen_STA_from_file
          nsta: number of stations (same as the Data.shape[1] if there's only PGD and no station existence "code")
          hypo: hypoloc [eqlon,eqlat,(optional eqdep)]
          dist_thres: degree threshold to select stations
        OUT:
          station index
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
         



def dist_sta_fault(Data,STA,nsta,rupt_file):
    '''
        return max (subfaults to stations) distance
        this metric reveals if rupture grows to an area without any staion around
    '''
    # find all available station index
    cent_hypo = [np.mean([STA[k][0] for k in STA.keys()]), np.mean([STA[k][1] for k in STA.keys()])]
    idx_sta = find_sta_hypo(Data,STA,nsta,cent_hypo,999) #return all stations


    # load fault file
    rupt = np.genfromtxt(rupt_file)
    SS = rupt[:,8]
    DS = rupt[:,9]
    rupt_time = rupt[:,12]
    slip = (SS * 2 + DS * 2)**0.5


    idx_rupt =  np.where(slip!=0)[0] #index of subfault that has value
    sav_dist = [] # for each subfault, get closest station distance
    for i in idx_rupt:
        # for each subfault, get the min distance to all stations
        min_fault_sta = 999
        for j in idx_sta:
            dist = obspy.geodetics.locations2degrees(lat1=rupt[i,2],long1=rupt[i,1],lat2=STA[j][1],long2=STA[j][0])
            if dist<min_fault_sta:
                min_fault_sta = dist
        sav_dist.append(min_fault_sta)

    sav_dist = np.array(sav_dist)
    return sav_dist







    
