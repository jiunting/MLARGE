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
        return all distances (from subfaults to stations), slip, and indexes that slip
        this metric reveals if rupture grows to an area without any staion around
        Input:
            Data:         single data with shape [time_points, features].
            STA:          STA format from analysis.gen_STA_from_file
            nsta:         number of stations
            rupt_file:    rupture file (.rupt) from Mudpy/Fakequake
        Output:
            sav_dist:     distance array from each subfault to the closest station
            slip:         slip array sqrt(SS^2+DS^2) for the subfault that slips
            idx_rupt:     indexes of the subfault that slips
    '''
    # find all available station index
    cent_hypo = [np.mean([STA[k][0] for k in STA.keys()]), np.mean([STA[k][1] for k in STA.keys()])]
    idx_sta = find_sta_hypo(Data,STA,nsta,cent_hypo,999) #return all stations


    # load fault file
    rupt = np.genfromtxt(rupt_file)
    SS = rupt[:,8]
    DS = rupt[:,9]
    rupt_time = rupt[:,12]
    slip = (SS ** 2 + DS ** 2)**0.5


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
    return sav_dist, slip[idx_rupt], idx_rupt
#    if weight:
#        return slip[idx_rupt] * (1.0/sav_dist)**2 # slip propragate to station follows 1/r (or 1/r^2)
#    else:
#        return sav_dist


def SR(slip,dist,k=2):
    '''
    Slip recovery calculation
        Input:
            slip:    slip array from .rupt or from the output of analysis.dist_sta_fault()
            dist:    distance array from the output of analysis.dist_sta_fault()
        Output:
            SR:      slip recovery array
    '''
    return (np.sum(slip*(1.0/(1+dist))**k)/np.sum(slip))*100 # in %





#-------GNSS noise--------
def get_psd(level):
    from mudpy.forward import gnss_psd
    return  gnss_psd(level=level,return_as_frequencies=True,return_as_db=False)
    #return f,Epsd,Npsd,Zpsd

def make_noise(n_steps,f,Epsd,Npsd,Zpsd,PGD=False):
    from mudpy.hfsims import windowed_gaussian,apply_spectrum
    #define sample rate
    dt=1.0 #make this 1 so that the length of PGD can be controlled by duration
    #noise model
    #level='median'  #this can be low, median, or high
    #duration of time series
    duration=n_steps
    # get white noise
    E_noise=windowed_gaussian(duration,dt,window_type=None)
    N_noise=windowed_gaussian(duration,dt,window_type=None)
    Z_noise=windowed_gaussian(duration,dt,window_type=None)
    noise=windowed_gaussian(duration,dt,window_type=None)
    #get PSDs
    #f,Epsd,Npsd,Zpsd=gnss_psd(level=level,return_as_frequencies=True,return_as_db=False)
    #control the noise level
    #scale=np.abs(np.random.randn()) #normal distribution
    #Covnert PSDs to amplitude spectrum
    Epsd = Epsd**0.5
    Npsd = Npsd**0.5
    Zpsd = Zpsd**0.5
    #apply the spectrum
    E_noise=apply_spectrum(E_noise,Epsd,f,dt,is_gnss=True)[:n_steps]
    N_noise=apply_spectrum(N_noise,Npsd,f,dt,is_gnss=True)[:n_steps]
    Z_noise=apply_spectrum(Z_noise,Zpsd,f,dt,is_gnss=True)[:n_steps]
    if PGD:
        GD=(np.real(E_noise)**2.0+np.real(N_noise)**2.0+np.real(Z_noise)**2.0)**0.5
        PGD=D2PGD(GD)
        return PGD
    else:
        return np.real(E_noise),np.real(N_noise),np.real(Z_noise)         




#-------GNSS noise--------

