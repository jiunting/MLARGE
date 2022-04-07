# some visualization toolds

import numpy as np
import matplotlib.pyplot as plt


def view_sources(EQinfo,save_fig=None):
    from matplotlib.ticker import LogLocator
    from matplotlib.ticker import MultipleLocator
    import matplotlib
    #source params scatter plot modify from Mudpy
    #EQinfo: source information file from preprocessing.get_EQinfo(fmt='long')

    # load EQinfo
    A = np.genfromtxt(EQinfo)
    id = A[:,0]
    Mw = A[:,1]
    hypo_lon = A[:,2]
    hypo_lat = A[:,3]
    hypo_dep = A[:,4]
    cent_lon = A[:,5]
    cent_lat = A[:,6]
    cent_dep = A[:,7]
    hypo_slip = A[:,8]
    max_slip = A[:,9]
    mean_slip = A[:,10]
    std_slip = A[:,11]
    max_rise = A[:,12]
    mean_rise = A[:,13]
    std_rise = A[:,14]
    tar_Mw = A[:,15]
    Len = A[:,16]
    Wid = A[:,17]

    # get Mw range from target Mw
    tmp_Mw_range = np.arange(5.0,10.2,0.2)
    idx_closest_lo = np.where(np.abs(Mw.min()-tmp_Mw_range)==np.min(np.abs(Mw.min()-tmp_Mw_range)))[0][0]
    idx_closest_hi = np.where(np.abs(Mw.max()-tmp_Mw_range)==np.min(np.abs(Mw.max()-tmp_Mw_range)))[0][0]
    if tmp_Mw_range[idx_closest_lo]>Mw.min():
        idx_closest_lo -= 1
    if tmp_Mw_range[idx_closest_hi]<Mw.max():
        idx_closest_hi += 1

    Mw_range = [tmp_Mw_range[idx_closest_lo], tmp_Mw_range[idx_closest_hi]] #magnitude range suitable for plotting
    Mw_synth = np.arange(Mw_range[0],Mw_range[1]+0.1,0.1)

    #scaling for L and W
    scal_L = 10**(-2.37+0.57*Mw_synth)
    scal_W = 10**(-1.86+0.46*Mw_synth)

    # create figure
#    fig = plt.figure(figsize=(18,10.5))
    fig = plt.figure(figsize=(12,6.3))
#    fig = plt.figure()

    # first subplot
    ax1 = fig.add_subplot(331)
    ax1.plot(Mw, Len,'+',markersize=3,alpha=1)
    ax1.plot(Mw_synth,scal_L,'k-')
    ax1.set_xlim(Mw_range)
    # use default value to define y range. [50,100,200,400,600,800,1000,1200,1500,2000,2500,3000]
    tmp_yrange = np.array([0.1,5,10,20,30,40,50,100,200,400,600,800,1000,1200,1500,2000,2500,3000])
    # use the 1%th and 99%th value instead of the largest to prevent outlier
    tmp_y = Len.copy()
    tmp_y.sort()
    N = len(tmp_y)
    Q_min = tmp_y[round(N*0.01)]
    Q_max = tmp_y[round(N*0.99)]
    min_idx = np.where(np.abs(tmp_yrange-Q_min)==np.min(np.abs(tmp_yrange-Q_min)))[0][0]
    max_idx = np.where(np.abs(tmp_yrange-Q_max)==np.min(np.abs(tmp_yrange-Q_max)))[0][0]
    print('tmp_yrange[min_idx] tmp_yrange[max_idx]=',tmp_yrange[min_idx],tmp_yrange[max_idx])
    if tmp_yrange[min_idx]>Q_min and min_idx!=0 :
        min_idx -= 1
    if tmp_yrange[max_idx]<Q_max and max_idx!=N-1 :
        max_idx += 1
    print('Qmin max=',Q_min,Q_max)
    print('yrange=',tmp_yrange[min_idx],tmp_yrange[max_idx])
    #plt.yticks([1,10,100,500,1000])
    ax1.set_ylim([tmp_yrange[min_idx],tmp_yrange[max_idx]])
    # get ylim after log scale for plotting
    log_ylim = [np.log10(tmp_yrange[min_idx]),np.log10(tmp_yrange[max_idx])]
    ax1.text(Mw_range[0]+(Mw_range[1]-Mw_range[0])*0.3,10**(log_ylim[0]+(log_ylim[1]-log_ylim[0])*0.08),r'$\log (L)=-2.37+0.57M_w$',fontsize=12) #text plot in original scale
    ax1.set_yscale('log')
    ymajorLocator = LogLocator(base=10.0,numticks = 5)
    yminorLocator = LogLocator(base=10.0,subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    ax1.yaxis.set_major_locator(ymajorLocator)
    ax1.yaxis.set_minor_locator(yminorLocator)
    ax1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax1.tick_params(which='major',length=5,width=0.8)
    ax1.tick_params(which='minor',length=2.5,width=0.8)
    ax1.set_ylabel('Fault length (km)',labelpad=-2,size=14)
    ax1.tick_params(axis='x',labelbottom=False)
    ax1.tick_params(axis='y',pad=-1,labelsize=12)

    # second subplot
    ax2 = fig.add_subplot(332)
    ax2.plot(Mw, Wid,'+',markersize=3,alpha=0.9)
    ax2.plot(Mw_synth,scal_W,'k-')
    ax2.set_xlim(Mw_range)
    ax2.set_yscale('log')
    # use default value to define y range. [50,100,200,400,600,800,1000,1200,1500,2000,2500,3000]
    tmp_yrange = np.array([0.1,5,10,20,30,40,50,100,200,400,600,800,1000,1200,1500,2000,2500,3000])
    # use the 1%th and 99%th value instead of the largest to prevent outlier
    tmp_y = Wid.copy()
    tmp_y.sort()
    N = len(tmp_y)
    Q_min = tmp_y[round(N*0.01)]
    Q_max = tmp_y[round(N*0.99)]
    min_idx = np.where(np.abs(tmp_yrange-Q_min)==np.min(np.abs(tmp_yrange-Q_min)))[0][0]
    max_idx = np.where(np.abs(tmp_yrange-Q_max)==np.min(np.abs(tmp_yrange-Q_max)))[0][0]
    print('tmp_yrange[min_idx] tmp_yrange[max_idx]=',tmp_yrange[min_idx],tmp_yrange[max_idx])
    if tmp_yrange[min_idx]>Q_min and min_idx!=0 :
        min_idx -= 1
    if tmp_yrange[max_idx]<Q_max and max_idx!=N-1 :
        max_idx += 1
    print('Qmin max=',Q_min,Q_max)
    print('yrange=',tmp_yrange[min_idx],tmp_yrange[max_idx])
    #plt.yticks([1,10,100,500,1000])
    ax2.set_ylim([tmp_yrange[min_idx],tmp_yrange[max_idx]])
    # get ylim after log scale for plotting
    log_ylim = [np.log10(tmp_yrange[min_idx]),np.log10(tmp_yrange[max_idx])]
    ax2.text(Mw_range[0]+(Mw_range[1]-Mw_range[0])*0.3,10**(log_ylim[0]+(log_ylim[1]-log_ylim[0])*0.08),r'$\log (W)=-1.86+0.46M_w$',fontsize=12) #text plot in original scale
    ax2.set_yscale('log')
    ymajorLocator = LogLocator(base=10.0,numticks = 5)
    yminorLocator = LogLocator(base=10.0,subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    ax2.yaxis.set_major_locator(ymajorLocator)
    ax2.yaxis.set_minor_locator(yminorLocator)
    ax2.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.tick_params(which='major',length=5,width=0.8)
    ax2.tick_params(which='minor',length=2.5,width=0.8)
    ax2.set_ylabel('Fault width (km)',labelpad=-2,size=14)
    ax2.tick_params(axis='x',labelbottom=False)
    ax2.tick_params(axis='y',pad=-1,labelsize=12)

    # third subplot
    ax3 = fig.add_subplot(333)
    ax3.plot(Mw, tar_Mw,'+',markersize=3,alpha=0.9)
    ax3.set_xlim(Mw_range)
    ax3.tick_params(which='major',length=5,width=0.8)
    ax3.tick_params(which='minor',length=2.5,width=0.8)
    ax3.set_ylabel('Target Mw',labelpad=-2,size=14)
    ax3.tick_params(axis='x',labelbottom=False)
    ax3.tick_params(axis='y',pad=-1)

    # 4th subplot
    ax4 = fig.add_subplot(334)
    ax4.plot(Mw, mean_slip,'+',markersize=3,alpha=0.9)
    ax4.set_xlim(Mw_range)
    ax4.set_yscale('log')
    ymajorLocator = LogLocator(base=10.0,numticks = 5)
    yminorLocator = LogLocator(base=10.0,subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    ax4.yaxis.set_major_locator(ymajorLocator)
    ax4.yaxis.set_minor_locator(yminorLocator)
    ax4.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    #Allen Hayes scaling lines
    '''
    ax4.plot([7.0,9.6],[0.4,21.1],c='k',lw=2)
    ax4.plot([7.0,9.6],[0.4,7.87],c='r',lw=2)
    ax4.plot([7.0,9.6],[0.55,10.71],c='orange',lw=2)
    ax4.plot([7.0,9.6],[1.20,23.33],c='g',lw=2)
    ax4.plot([7.0,9.27],[1.08,39.16],c='violet',lw=2)
    '''
    ax4.set_xlim(Mw_range)
    ax4.tick_params(which='major',length=5,width=0.8)
    ax4.tick_params(which='minor',length=2.5,width=0.8)
    ax4.set_ylabel('Mean slip (m)',labelpad=-2,size=14)
    ax4.tick_params(axis='x',labelbottom=False)
    ax4.tick_params(axis='y',pad=-1,labelsize=12)

    # 5th subplot
    ax5 = fig.add_subplot(335)
    ax5.plot(Mw, max_slip,'+',markersize=3,alpha=0.9)
    ax5.set_xlim(Mw_range)
    ax5.set_yscale('log')
    ax5.tick_params(which='major',length=5,width=0.8)
    ax5.tick_params(which='minor',length=2.5,width=0.8)
    ax5.set_ylabel('Max. slip (m)',labelpad=-2,size=14)
    ax5.tick_params(axis='x',labelbottom=False)
    ax5.tick_params(axis='y',pad=-1,labelsize=12)

    # 6th subplot
    ax6 = fig.add_subplot(336)
    ax6.plot(Mw, std_slip,'+',markersize=3,alpha=0.9)
    ax6.set_xlim(Mw_range)
    #ax6.set_ylim([tmp_yrange[min_idx],tmp_yrange[max_idx]])
    ax6.set_ylim([0.05,np.max(std_slip)])
    ax6.set_yscale('log')
    ax6.tick_params(which='major',length=5,width=0.8)
    ax6.tick_params(which='minor',length=2.5,width=0.8)
    ax6.set_ylabel('Slip std. dev. (m)',labelpad=-2,size=14)
    ax6.tick_params(axis='x',labelbottom=False)
    ax6.tick_params(axis='y',pad=-1,labelsize=12)


    # 7th subplot
    ax7 = fig.add_subplot(337)
    ax7.plot(Mw, mean_rise,'+',markersize=3,alpha=0.9)
    ax7.set_xlim(Mw_range)
    ax7.set_xticks(np.arange(7.0,9.5+0.5,0.5))
    ax7.set_yscale('log')
    ax7.tick_params(which='major',length=5,width=0.8)
    ax7.tick_params(which='minor',length=2.5,width=0.8)
    if mean_rise.max()<10:
        ymajorLocator = LogLocator(base=2,numticks = 5)
        yminorLocator = LogLocator(base=2,subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    else:
        ymajorLocator = LogLocator(base=10,numticks = 5)
        yminorLocator = LogLocator(base=10,subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    ax7.yaxis.set_major_locator(ymajorLocator)
    ax7.yaxis.set_minor_locator(yminorLocator)
    ax7.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax7.set_ylabel('Mean rise time (s)',labelpad=-2,size=14)
    ax7.tick_params(axis='y',pad=-1,labelsize=12)
    ax7.set_xlabel('Actual Mw',labelpad=0,size=14)
    ax7.tick_params(axis='x',pad=0,labelsize=12)
    
    # 8th subplot
    ax8 = fig.add_subplot(338)
    ax8.plot(Mw, max_rise,'+',markersize=3,alpha=0.9)
    ax8.set_xlim(Mw_range)
    ax8.set_xticks(np.arange(7.0,9.5+0.5,0.5))
    ax8.set_yscale('log')
    ax8.tick_params(which='major',length=5,width=0.8)
    ax8.tick_params(which='minor',length=2.5,width=0.8)
    ax8.set_ylabel('Max. rise time (s)',labelpad=-2,size=14)
    ax8.tick_params(axis='y',pad=-1,labelsize=12)
    ax8.set_xlabel('Actual Mw',labelpad=0,size=14)
    ax8.tick_params(axis='x',pad=0,labelsize=12)

    # 9th subplot
    ax9 = fig.add_subplot(339)
    ax9.plot(Mw, std_rise,'+',markersize=3,alpha=0.9)
    ax9.set_xlim(Mw_range)
    ax9.set_xticks(np.arange(7.0,9.5+0.5,0.5))
    ax9.set_ylim([0.2,np.max(std_rise)])
    ax9.set_yscale('log')
    ax9.tick_params(which='major',length=5,width=0.8)
    ax9.tick_params(which='minor',length=2.5,width=0.8)
    ax9.set_ylabel('Rise time std. dev. (s)',labelpad=-2,size=14)
    ax9.tick_params(axis='y',pad=-1,labelsize=12)
    ax9.set_xlabel('Actual Mw',labelpad=0,size=14)
    ax9.tick_params(axis='x',pad=0,labelsize=12)
    
    # final adjustment
    plt.subplots_adjust(left=0.05,top=0.97,right=0.97,bottom=0.08,wspace=0.2,hspace=0.1)

    if save_fig:
        fig.savefig(save_fig)
        fig.show()
    else:
        fig.show()


    
def make_hist(train,valid,test,save_fig=None):
    '''
    Input:
        Give the path of train.valid,test .npy file generated automatically during model training
    Output:
        A magnitude figure
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

#    train = np.load('EQinfo_05_1_train.npy')
#    valid = np.load('EQinfo_05_1_valid.npy')
#    test = np.load('EQinfo_05_1_test.npy')
    train = np.load(train)
    valid = np.load(valid)
    test = np.load(test)

    train = np.array([list(i)[:-1] for i in train])
    valid = np.array([list(i)[:-1] for i in valid])
    test = np.array([list(i)[:-1] for i in test])

    sns.set()
    sns.set_context("poster")
    plt.figure(figsize=(6,5))

    plt.hist(train[:,1],20,alpha=0.8)
    plt.hist(valid[:,1],20,alpha=0.9)
    plt.hist(test[:,1],20,alpha=0.9)

    plt.legend(['Training','Validation','Testing'],fontsize=14)
    plt.grid(which='major',axis='x')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax = plt.gca()
    ax.tick_params(pad=1.0,length=0,size=0)

    plt.text(8.0,800,'70%',fontsize=15)
    plt.text(8.0,270,'20%',fontsize=15)
    plt.text(8.0,70,'10%',fontsize=15)
    plt.xticks(np.arange(7.0,9.6,0.5))
    plt.xlabel('Mw',fontsize=15,labelpad=0.5)
    plt.ylabel('count',fontsize=15,labelpad=0.5)

    if save_fig:
        plt.savefig(save_fig,dpi=300)
        plt.show()
    else:
        plt.show()



def train_valid_curve(train_valid,check_point_epo=None,save_fig=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    '''
        train_valid: loss .npy file generated after the training is finished
        check_point_epo can be generated by the following
        #ls -lrt | awk -Fweights. '{print($2)}' | awk -F- '{print($1)}' > steps.txt
    '''
    sns.set()
    A = np.load(train_valid,allow_pickle=True)
    A = A.item()

    plt.plot(np.arange(len(A['loss']))+1,np.log10(A['loss']),color=[0.6,0.6,0.6],linewidth=1.5)
    plt.plot(np.arange(len(A['val_loss']))+1,np.log10(A['val_loss']),'k',linewidth=1.5)
    if check_point_epo:
        epos = np.genfromtxt(check_point_epo)
        epos = [int(i)-1 for i in epos]
        plt.plot((np.arange(len(A['val_loss']))+1)[epos],np.log10(A['val_loss'])[epos],'r.',markersize=10,mew=0.8,markeredgecolor=[0,0,0])
        plt.plot((np.arange(len(A['val_loss']))+1)[epos[-1]],np.log10(A['val_loss'])[epos[-1]],'r*',markersize=16,markeredgecolor=[0,0,0],mew=0.8)
        plt.legend(['training','validation','checkpoint','final model'],fontsize=14,frameon=True)
    else:
        plt.legend(['training','validation'],fontsize=15,frameon=True)
    plt.xlabel('Training steps',fontsize=14,labelpad=1)
    plt.ylabel('log(MSE)',fontsize=14,labelpad=1)
    ax1=plt.gca()
    ax1.tick_params(pad=1,labelsize=12)
    plt.ylim([-3.9,-3.0])
    plt.grid(True)

    if save_fig:
        plt.savefig(save_fig,dpi=300)
        plt.show()
    else:
        plt.show()



def plot_tcs(Data,ncomp,STA,nsta,rupt=None,sort_type='lat',save_fig=None):
    '''
        Data: [time,features((ncomps+(1 existence code))*nsta)]
        ncomp: how many component do you have (existence code not included )
        STA: from mlarge.analysis.gen_STA_from_file()
        rupt: .rupt file or no file
        sort_type: choose from 'lat','dist'(rupt!=None)
    '''
    BMap_flag = False #basemap
    if rupt:
        try:
            from mpl_toolkits.basemap import Basemap
            BMap_flag = True
        except:
            print('cannot import Basemap!')
    if sort_type=='dist':
        assert type(rupt)==str, "Can not find hypo info, rupt should not empty!"
    colors = ['k','r','b']
    sav_D = {} #data for each component
    for n_comp in range(ncomp):
        sav_D[n_comp] = {'data':[],'stlat':[],'stlon':[]}
        for i in range(nsta):
            if np.any(Data[:,int(-1*nsta+i)]!=0): # for ith station,if any data for all time has value/or code
                stlon,stlat = STA[i]
                sav_D[n_comp]['data'].append(Data[:,i+n_comp*nsta])
                sav_D[n_comp]['stlat'].append(stlat)
                sav_D[n_comp]['stlon'].append(stlon)
        sav_D[n_comp]['data'] = np.array(sav_D[n_comp]['data'])
        sav_D[n_comp]['stlat'] = np.array(sav_D[n_comp]['stlat'])
        sav_D[n_comp]['stlon'] = np.array(sav_D[n_comp]['stlon'])
    #--- plot result, scale to max=D deg---
#    LON = [-77,-66]
#    LAT = [-45,-17]
#    tmpLAT = [max([LAT[0],sav_D[0]['stlat'].min()]),min([LAT[1],sav_D[0]['stlat'].max()])]
#    if BMap_flag:
#        plt.subplot(1,2,1)
#        # get the map boundary
#        map = Basemap(projection='cyl',resolution='f',llcrnrlon=LON[0],llcrnrlat=tmpLAT[0],urcrnrlon=LON[1],urcrnrlat=tmpLAT[1])
#        map.arcgisimage(service='ESRI_Imagery_World_2D', xpixels = 2000, verbose= True)
#        map.drawstates()
#        map.drawcountries(linewidth=0.1)
#        map.drawcoastlines(linewidth=0.1)
#        #Lon,Lat lines
#        lats = map.drawparallels(np.arange(-90,90,2),labels=[1,0,0,1],color='w',linewidth=0.5)
#        lons = map.drawmeridians(np.arange(-180,180,2),labels=[1,0,0,1],color='w',linewidth=0.5)
#        # continue to tcs data plot
#        plt.subplot(1,2,2) # for data plot
    if BMap_flag:
        plt.subplot(1,2,1)
        plt.subplot(1,2,2)
    D = 2.0 # let the maximum data(PGD or E,N,Z) to be this value on plot
    max_val = 0 # max val for all the component
    for n_comp in range(ncomp):
        if np.max(np.abs(sav_D[n_comp]['data']))>max_val:
            max_val = np.max(np.abs(sav_D[n_comp]['data']))
    mul = D/max_val
    maxY_tcs = float("-inf")
    minY_tcs = float("inf")
    T = np.arange(102)*5 + 5
    T = np.hstack([0,T]) # have time starts from 0 sec
    for n_comp in range(ncomp):
        tcs = mul*sav_D[n_comp]['data'].T+sav_D[n_comp]['stlat'] #shape=(102*nstans)
        # add zeros at 0 sec for all tcs
        tcs = np.vstack([np.ones(tcs.shape[1])*sav_D[n_comp]['stlat'],tcs])
        plt.plot(T,tcs,color=colors[n_comp])
        maxY_tcs = max([maxY_tcs,np.max(mul*sav_D[n_comp]['data'].T+sav_D[n_comp]['stlat'])])
        minY_tcs = min([minY_tcs,np.min(mul*sav_D[n_comp]['data'].T+sav_D[n_comp]['stlat'])])

    d_Y_tcs = maxY_tcs-minY_tcs
    #plt.plot([420,420],[sav_D[n_comp]['stlat'].min(),sav_D[n_comp]['stlat'].min()+0.5*D],'m',linewidth=2.0)
    # scale = (X_scal * D)/mul), find a best X so that scale can be 1,1.5,2,...
    match = np.array([0.5,1,2,3,5,10,15]) # or define your own scale
    #grid search X_scal
    idx = np.where(np.abs(max_val-match)==np.min(np.abs(max_val-match)))[0][0]
    X_scal = match[idx]*mul/D

    #plt.plot([420,420],[minY_tcs+0.1*d_Y_tcs,minY_tcs+0.1*d_Y_tcs+0.5*D],'m',linewidth=2.0)
    if not BMap_flag:
        plt.plot([420,420],[minY_tcs+0.05*d_Y_tcs,minY_tcs+0.05*d_Y_tcs+X_scal*D],'m',linewidth=2.0)
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        if ((X_scal*D)/mul).is_integer():
            plt.text(435,minY_tcs+0.05*d_Y_tcs+X_scal*D*0.5,'%d (m)'%((X_scal*D)/mul),bbox=props)
        else:
            plt.text(435,minY_tcs+0.05*d_Y_tcs+X_scal*D*0.5,'%.1f (m)'%((X_scal*D)/mul),bbox=props)
#    if ((0.5*D)/mul).is_integer():
#        plt.text(430,sav_D[n_comp]['stlat'].min()+0.2*D,'%d (m)'%((0.5*D)/mul),bbox=props)
#    else:
#        plt.text(430,sav_D[n_comp]['stlat'].min()+0.2*D,'%.2f (m)'%((0.5*D)/mul),bbox=props)
    plt.xlim([0,510])
#plt.ylim([sav_D[0]['stlat'].min()-0.5,sav_D[0]['stlat'].max()+0.5])
    #plt.ylim(tmpLAT)
    minY_tcs = minY_tcs-0.05*d_Y_tcs
    maxY_tcs = maxY_tcs+0.05*d_Y_tcs
    plt.ylim([minY_tcs,maxY_tcs])
    plt.xlabel('Time (s)',fontsize=14)
    plt.ylabel('Lat.',fontsize=14)

    #=====plot map=====
    if BMap_flag:
        LON = [-78,-67]
        LAT = [-45,-16.5] #do not plot outside of this range
        # re-adjust tcs's ylim to match the map
        minY_tcs = max([minY_tcs,LAT[0]])
        maxY_tcs = min([maxY_tcs,LAT[1]])
        plt.ylim([minY_tcs,maxY_tcs])
        #tmpLAT = [max([LAT[0],sav_D[0]['stlat'].min()]),min([LAT[1],sav_D[0]['stlat'].max()])]
        #-------plotting map------------
        plt.subplot(1,2,1)
        # get the map boundary
        #print("LAT:",minY_tcs,maxY_tcs)
        #map = Basemap(projection='cyl',resolution='f',llcrnrlon=LON[0],llcrnrlat=tmpLAT[0],urcrnrlon=LON[1],urcrnrlat=tmpLAT[1])
        map = Basemap(projection='cyl',resolution='f',llcrnrlon=LON[0],llcrnrlat=minY_tcs,urcrnrlon=LON[1],urcrnrlat=maxY_tcs,fix_aspect=False)
        fig = map.arcgisimage(service='ESRI_Imagery_World_2D', xpixels = 2000, verbose= True)
        fig.set_alpha(0.8)
        map.drawstates()
        map.drawcountries(linewidth=0.1)
        map.drawcoastlines(linewidth=0.5)
        dLON = LON[1]-LON[0]
        dLAT = maxY_tcs-minY_tcs
        #map.drawmapscale(LON[0]+0.1*dLON,minY_tcs+0.05*dLAT,np.mean(LON),np.mean([maxY_tcs,minY_tcs]),500) #500km
        #Lon,Lat lines
        if maxY_tcs-minY_tcs<5:
            dn = 1
        elif 5<=maxY_tcs-minY_tcs<10:
            dn = 2
        else:
            dn = 5
        lats = map.drawparallels(np.arange(-90,90,dn),labels=[1,0,0,1],color='w',linewidth=0.5)
        lons = map.drawmeridians(np.arange(-180,180,5),labels=[1,0,0,1],color='w',linewidth=0.5)
        #plot stations on map
        plt.plot(sav_D[0]['stlon'],sav_D[0]['stlat'],'r^',markeredgecolor='k')
        # tcs data plot, remove ylabel and ticks
        plt.subplot(1,2,2) # for data plot
        plt.plot([390,390],[minY_tcs+0.05*d_Y_tcs,minY_tcs+0.05*d_Y_tcs+X_scal*D],'m',linewidth=2.0)
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        if ((X_scal*D)/mul).is_integer():
            plt.text(410,minY_tcs+0.05*d_Y_tcs+X_scal*D*0.5,'%d (m)'%((X_scal*D)/mul),bbox=props)
        else:
            plt.text(410,minY_tcs+0.05*d_Y_tcs+X_scal*D*0.5,'%.1f (m)'%((X_scal*D)/mul),bbox=props)
        ax=plt.gca()
        ax.tick_params(labelleft=False)
        plt.ylabel('')
        plt.subplots_adjust(left=0.08,top=0.95,right=0.97,bottom=0.12,wspace=0.05)


    if save_fig:
        plt.savefig(save_fig)
        plt.show()
        plt.close()
    else:
        plt.show()




def plot_y_scatter(Model_path,X,y,r_yscale,use_final=False,mark_range=None,save_fig=None):
    '''
    scatter plot of y v.s. p_pred at every epoch
    Input:
        Model_path:     path of the preferred model
        X:              feature input [N,epoch,features]
        y:              true labels [N,epoch,1 or multiple outputs]
        use_final:      use final parameter instead of time-dependent parameter
        mark_range:     plot the +- error range in mark_range of possible values from labels
        r_yscale:       a list of function(s) which reverts y to the original sense
        save_fig:       directory to save the plots
    Output:
        Save figures or show on screen if save_fig==None
    '''
    import tensorflow as tf
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()

    # create dir if not exist and save_fig is not None
    if save_fig!=None:
        import os
        if not(os.path.exists(save_fig)):
            os.makedirs(save_fig)

    # load the model
    model_loaded = tf.keras.models.load_model(Model_path,compile=False)

    # make predictions
    y_pred = model_loaded.predict(X)

    # how many output params
    N_p = y.shape[2]
    assert N_p == len(r_yscale), "size of y and r_yscale does not match!"

    # convert y, y_pred to original unit
    import copy
    y_pred_rscale = copy.deepcopy(y_pred)
    y_rscale = copy.deepcopy(y)
    for ip in range(N_p):
        y_pred_rscale[:,:,ip] = r_yscale[ip](y_pred[:,:,ip])
        y_rscale[:,:,ip] = r_yscale[ip](y[:,:,ip])

    #====== start plotting ======
    # color-coded by Mw
    vmin, vmax = 6.9,9.6 #set the Mw from this range, don't want it starts from 0 at 0 s for example.
    cm = plt.cm.magma(  plt.Normalize(vmin,vmax)(y_rscale[:,-1,0]) )
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap='magma')
    cmap.set_array([])


    idx = np.arange(len(y)) # what indexes are you plotting? add any filtering here
    for epo in range(102):
        plt.figure(figsize=(9.5,5.5))
        #epo = 30
        if use_final:
            epo_y = -1
        else:
            epo_y = epo
        #=============
        plt.subplot(2,3,1)
        #plt.plot(sav_mft[(0,epo)],sav_c,'k.')
        plt.scatter(y_rscale[idx,epo_y,0],y_pred_rscale[idx,epo,0],c=cm[idx],cmap='magma',s=10,vmin=vmin,vmax=vmax,alpha=0.9)
        plt.plot([vmin,vmax],[vmin,vmax],'m')
        if mark_range:
            YRange = np.max(y_rscale[:,:,0])-np.min(y_rscale[:,:,0])
            thresh = YRange*mark_range
            thresh = 0.3 # manually fix the Mw error to be 0.3!!!
            print('Add error range at figure 1. Range=%f'%(thresh))
            plt.plot([vmin,vmax],[vmin-thresh,vmax-thresh],'m--')
            plt.plot([vmin,vmax],[vmin+thresh,vmax+thresh],'m--')
            acc = len(np.where( np.abs(y_rscale[idx,epo_y,0]-y_pred_rscale[idx,epo,0])<=thresh )[0])/len(y_rscale[idx,epo_y,0])
            acc *= 100 #percentage
        #plt.scatter(sav_mft[(0,epo)][idx]/R[0],sav_SNR_mean[idx],c=cm[idx],cmap='magma',s=20,vmin=7.4,vmax=9.6,alpha=0.9)
        plt.ylabel('Prediction',fontsize=14,labelpad=0)
        #plt.xlim([y_rscale[:,:,0].min(),y_rscale[:,:,0].max()])
        #plt.ylim([y_rscale[:,:,0].min(),y_rscale[:,:,0].max()])
        plt.xlim([vmin,vmax])
        plt.ylim([vmin,vmax])
        ax1=plt.gca()
        ax1.tick_params(direction='out', pad=0,labelsize=12,length=0)
        ax1.annotate('Mw',xy=(0.94,0.95),xycoords='axes fraction',size=14, ha='right', va='top',bbox=dict(boxstyle='round', fc='w',alpha=0.7))
        if mark_range:
            ax1.annotate('%.1f %%'%(acc),xy=(0.06,0.95),xycoords='axes fraction',size=14, ha='left', va='top',bbox=dict(boxstyle='round', fc='w',alpha=0.7))
        #ax1.set_yscale('log')
        # add colorbar
        #These two lines mean put the bar inside the plot
        fig = plt.gcf()
        cbaxes = fig.add_axes([0.25, 0.62, 0.074, 0.012 ])
        clb = plt.colorbar(cmap,cax=cbaxes,ticks=[7.0, 8.0, 9.0], orientation='horizontal',label='Mw')
        clb.set_label('Mw', rotation=0,labelpad=-2,size=12)
        ax1=plt.gca()
        ax1.tick_params(pad=0,length=0.5)
        #plt.legend(['Mw'],frameon=True)
        #=============
        plt.subplot(2,3,2)
        plt.title('%d s'%(epo*5+5))
        #plt.plot(sav_mft[(1,epo)],sav_c,'k.')
        plt.scatter(y_rscale[idx,epo_y,1],y_pred_rscale[idx,epo,1],c=cm[idx],cmap='magma',s=10,vmin=vmin,vmax=vmax,alpha=0.9)
        #plt.scatter(sav_mft[(1,epo)][idx]/R[1],sav_SNR_mean[idx],c=cm[idx],cmap='magma',s=20,vmin=7.4,vmax=9.6,alpha=0.9)
        plt.plot([y_rscale[:,:,1].min(),y_rscale[:,:,1].max()],[y_rscale[:,:,1].min(),y_rscale[:,:,1].max()],'m')
        if mark_range:
            YRange = np.max(y_rscale[:,:,1])-np.min(y_rscale[:,:,1])
            thresh = YRange*mark_range
            print('Add error range at figure 2. Range=%f'%(thresh))
            plt.plot([y_rscale[:,:,1].min(),y_rscale[:,:,1].max()],[y_rscale[:,:,1].min()-thresh,y_rscale[:,:,1].max()-thresh],'m--')
            plt.plot([y_rscale[:,:,1].min(),y_rscale[:,:,1].max()],[y_rscale[:,:,1].min()+thresh,y_rscale[:,:,1].max()+thresh],'m--')
            acc = len(np.where( np.abs(y_rscale[idx,epo_y,1]-y_pred_rscale[idx,epo,1])<=thresh )[0])/len(y_rscale[idx,epo_y,1])
            acc *= 100 #percentage
        plt.xlim([y_rscale[:,:,1].min(),y_rscale[:,:,1].max()])
        plt.ylim([y_rscale[:,:,1].min(),y_rscale[:,:,1].max()])
        ax1=plt.gca()
        ax1.tick_params(pad=0,labelsize=12,length=0)
        #ax1.set_yscale('log')
        ax1.annotate('Lon${\degree}$',xy=(0.94,0.95),xycoords='axes fraction',size=14, ha='right', va='top',bbox=dict(boxstyle='round', fc='w',alpha=0.7))
        if mark_range:
            ax1.annotate('%.1f %%'%(acc),xy=(0.06,0.95),xycoords='axes fraction',size=14, ha='left', va='top',bbox=dict(boxstyle='round', fc='w',alpha=0.7))
        #=============
        plt.subplot(2,3,3)
        #plt.plot(sav_mft[(2,epo)],sav_c,'k.')
        plt.scatter(y_rscale[idx,epo_y,2],y_pred_rscale[idx,epo,2],c=cm[idx],cmap='magma',s=10,vmin=vmin,vmax=vmax,alpha=0.9)
        #plt.scatter(sav_mft[(2,epo)][idx]/R[2],sav_SNR_mean[idx],c=cm[idx],cmap='magma',s=20,vmin=7.4,vmax=9.6,alpha=0.9)
        plt.plot([y_rscale[:,:,2].min(),y_rscale[:,:,2].max()],[y_rscale[:,:,2].min(),y_rscale[:,:,2].max()],'m')
        if mark_range:
            YRange = np.max(y_rscale[:,:,2])-np.min(y_rscale[:,:,2])
            thresh = YRange*mark_range
            print('Add error range at figure 3. Range=%f'%(thresh))
            plt.plot([y_rscale[:,:,2].min(),y_rscale[:,:,2].max()],[y_rscale[:,:,2].min()-thresh,y_rscale[:,:,2].max()-thresh],'m--')
            plt.plot([y_rscale[:,:,2].min(),y_rscale[:,:,2].max()],[y_rscale[:,:,2].min()+thresh,y_rscale[:,:,2].max()+thresh],'m--')
            acc = len(np.where( np.abs(y_rscale[idx,epo_y,2]-y_pred_rscale[idx,epo,2])<=thresh )[0])/len(y_rscale[idx,epo_y,2])
            acc *= 100 #percentage
        plt.xlim([y_rscale[:,:,2].min(),y_rscale[:,:,2].max()])
        plt.ylim([y_rscale[:,:,2].min(),y_rscale[:,:,2].max()])
        ax1=plt.gca()
        ax1.tick_params(pad=0,labelsize=12,length=0)
        #ax1.set_yscale('log')
        ax1.annotate('Lat${\degree}$',xy=(0.94,0.95),xycoords='axes fraction',size=14, ha='right', va='top',bbox=dict(boxstyle='round', fc='w',alpha=0.7))
        if mark_range:
            ax1.annotate('%.1f %%'%(acc),xy=(0.06,0.95),xycoords='axes fraction',size=14, ha='left', va='top',bbox=dict(boxstyle='round', fc='w',alpha=0.7))
        #=============
        plt.subplot(2,3,4)
        #plt.plot(sav_mft[(3,epo)],sav_c,'k.')
        plt.scatter(y_rscale[idx,epo_y,3],y_pred_rscale[idx,epo,3],c=cm[idx],cmap='magma',s=10,vmin=vmin,vmax=vmax,alpha=0.9)
        #plt.scatter(sav_mft[(3,epo)][idx]/R[3],sav_SNR_mean[idx],c=cm[idx],cmap='magma',s=20,vmin=7.4,vmax=9.6,alpha=0.9)
        plt.plot([y_rscale[:,:,3].min(),y_rscale[:,:,3].max()],[y_rscale[:,:,3].min(),y_rscale[:,:,3].max()],'m')
        if mark_range:
            YRange = np.max(y_rscale[:,:,3])-np.min(y_rscale[:,:,3])
            thresh = YRange*mark_range
            print('Add error range at figure 4. Range=%f'%(thresh))
            plt.plot([y_rscale[:,:,3].min(),y_rscale[:,:,3].max()],[y_rscale[:,:,3].min()-thresh,y_rscale[:,:,3].max()-thresh],'m--')
            plt.plot([y_rscale[:,:,3].min(),y_rscale[:,:,3].max()],[y_rscale[:,:,3].min()+thresh,y_rscale[:,:,3].max()+thresh],'m--')
            acc = len(np.where( np.abs(y_rscale[idx,epo_y,3]-y_pred_rscale[idx,epo,3])<=thresh )[0])/len(y_rscale[idx,epo_y,3])
            acc *= 100 #percentage
        #plt.ylabel('Avg. SNR',fontsize=14,labelpad=0)
        #plt.xlabel('|| y$_{pred}$ - y ||',fontsize=14,labelpad=0)
        plt.xlim([y_rscale[:,:,3].min(),y_rscale[:,:,3].max()])
        plt.ylim([y_rscale[:,:,3].min(),y_rscale[:,:,3].max()])
        plt.ylabel('Prediction',fontsize=14,labelpad=0)
        plt.xlabel('True',fontsize=14,labelpad=0)
        #plt.xlabel('%',fontsize=14,labelpad=0)
        ax1=plt.gca()
        ax1.tick_params(pad=0,labelsize=12,length=0)
        #ax1.set_yscale('log')
        ax1.annotate('Depth (km)',xy=(0.94,0.95),xycoords='axes fraction',size=14, ha='right', va='top',bbox=dict(boxstyle='round', fc='w',alpha=0.7))
        if mark_range:
            ax1.annotate('%.1f %%'%(acc),xy=(0.06,0.95),xycoords='axes fraction',size=14, ha='left', va='top',bbox=dict(boxstyle='round', fc='w',alpha=0.7))
        #=============
        plt.subplot(2,3,5)
        #plt.plot(sav_mft[(4,epo)],sav_c,'k.')
        plt.scatter(y_rscale[idx,epo_y,4],y_pred_rscale[idx,epo,4],c=cm[idx],cmap='magma',s=10,vmin=vmin,vmax=vmax,alpha=0.9)
        #plt.scatter(sav_mft[(4,epo)][idx]/R[4],sav_SNR_mean[idx],c=cm[idx],cmap='magma',s=20,vmin=7.4,vmax=9.6,alpha=0.9)
        plt.plot([y_rscale[:,:,4].min(),y_rscale[:,:,4].max()],[y_rscale[:,:,4].min(),y_rscale[:,:,4].max()],'m')
        if mark_range:
            YRange = np.max(y_rscale[:,:,4])-np.min(y_rscale[:,:,4])
            thresh = YRange*mark_range
            print('Add error range at figure 5. Range=%f'%(thresh))
            plt.plot([y_rscale[:,:,4].min(),y_rscale[:,:,4].max()],[y_rscale[:,:,4].min()-thresh,y_rscale[:,:,4].max()-thresh],'m--')
            plt.plot([y_rscale[:,:,4].min(),y_rscale[:,:,4].max()],[y_rscale[:,:,4].min()+thresh,y_rscale[:,:,4].max()+thresh],'m--')
            acc = len(np.where( np.abs(y_rscale[idx,epo_y,4]-y_pred_rscale[idx,epo,4])<=thresh )[0])/len(y_rscale[idx,epo_y,4])
            acc *= 100 #percentage
        plt.xlim([min(y_rscale[:,:,4].min(),-50),y_rscale[:,:,4].max()]) # this min(.min(),-100) makes better plotting
        plt.ylim([min(y_rscale[:,:,4].min(),-50),y_rscale[:,:,4].max()])
        plt.xlabel('True',fontsize=14,labelpad=0)
        #plt.xlabel('%',fontsize=14,labelpad=0)
        ax1=plt.gca()
        ax1.tick_params(pad=0,labelsize=12,length=0)
        ax1.ticklabel_format(style='sci', axis='y',scilimits=(0,0))
        #ax1.set_yscale('log')
        ax1.annotate('Length (km)',xy=(0.94,0.95),xycoords='axes fraction',size=14, ha='right', va='top',bbox=dict(boxstyle='round', fc='w',alpha=0.7))
        if mark_range:
            ax1.annotate('%.1f %%'%(acc),xy=(0.06,0.95),xycoords='axes fraction',size=14, ha='left', va='top',bbox=dict(boxstyle='round', fc='w',alpha=0.7))
        #=============
        plt.subplot(2,3,6)
        #plt.plot(sav_mft[(5,epo)],sav_c,'k.')
        plt.scatter(y_rscale[idx,epo_y,5],y_pred_rscale[idx,epo,5],c=cm[idx],cmap='magma',s=10,vmin=vmin,vmax=vmax,alpha=0.9)
        #plt.scatter(sav_mft[(5,epo)][idx]/R[5],sav_SNR_mean[idx],c=cm[idx],cmap='magma',s=20,vmin=7.4,vmax=9.6,alpha=0.9)
        plt.plot([y_rscale[:,:,5].min(),y_rscale[:,:,5].max()],[y_rscale[:,:,5].min(),y_rscale[:,:,5].max()],'m')
        if mark_range:
            YRange = np.max(y_rscale[:,:,5])-np.min(y_rscale[:,:,5])
            thresh = YRange*mark_range
            print('Add error range at figure 6. Range=%f'%(thresh))
            plt.plot([y_rscale[:,:,5].min(),y_rscale[:,:,5].max()],[y_rscale[:,:,5].min()-thresh,y_rscale[:,:,5].max()-thresh],'m--')
            plt.plot([y_rscale[:,:,5].min(),y_rscale[:,:,5].max()],[y_rscale[:,:,5].min()+thresh,y_rscale[:,:,5].max()+thresh],'m--')
            acc = len(np.where( np.abs(y_rscale[idx,epo_y,5]-y_pred_rscale[idx,epo,5])<=thresh )[0])/len(y_rscale[idx,epo_y,5])
            acc *= 100 #percentage
        plt.xlim([min(y_rscale[:,:,5].min(),-5),max(y_rscale[:,:,5].max(),y_pred_rscale[:,:,5].max())])
        plt.ylim([min(y_rscale[:,:,5].min(),-5),max(y_rscale[:,:,5].max(),y_pred_rscale[:,:,5].max())])
        plt.xlabel('True',fontsize=14,labelpad=0)
        #plt.xlabel('%',fontsize=14,labelpad=0)
        ax1=plt.gca()
        ax1.tick_params(pad=0,labelsize=12,length=0)
        ax1.ticklabel_format(style='sci', axis='y',scilimits=(0,0))
        #ax1.set_yscale('log')
        ax1.annotate('Width (km)',xy=(0.94,0.95),xycoords='axes fraction',size=14, ha='right', va='top',bbox=dict(boxstyle='round', fc='w',alpha=0.7))
        if mark_range:
            ax1.annotate('%.1f %%'%(acc),xy=(0.06,0.95),xycoords='axes fraction',size=14, ha='left', va='top',bbox=dict(boxstyle='round', fc='w',alpha=0.7))
        #adjust subplots width/length
        plt.subplots_adjust(left=0.05,top=0.92,right=0.97,bottom=0.1,wspace=0.07,hspace=0.14)
        #plt.show()
        #break
        if save_fig:
            plt.subplots_adjust(left=0.05,top=0.95,right=0.99,bottom=0.07,wspace=0.14,hspace=0.14)
            plt.savefig(save_fig+'/fig_%03d.png'%(epo))
            plt.close()
        else:
            plt.show()
        #plt.savefig('./misfit_meanSNR_figs/fig_%03d.png'%(epo))
        #plt.close()




def plot_y_scatter5(Model_path,X,y,r_yscale,use_final=False,idx=None,mark_range=None,save_fig=None):
    '''
    scatter plot of y v.s. y_pred at every epoch
    Input:
        Model_path:     path of the preferred model
        X:              feature input [N,epoch,features]
        y:              true labels [N,epoch,multiple outputs(Mw, Lon, Lat, Length, Width)]
        r_yscale:       a list of function(s) which reverts y to the original sense
        use_final:      use final parameter instead of time-dependent parameter
        idx:            idx to be plotted [np array]
        mark_range:     plot the +- error range in mark_range of possible values from labels
        save_fig:       directory to save the plots
    Output:
        Save figures or show on screen if save_fig==None
    #=====Modified on 4/6=====
    still making y v.s. y_pred for 5 params at every epoch, but add accuracy plot (so 6 subplots)
    '''
    import tensorflow as tf
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    from mlarge.scaling import make_linear_scale

    # create dir if not exist and save_fig is not None
    if save_fig!=None:
        import os
        if not(os.path.exists(save_fig)):
            os.makedirs(save_fig)

    # load the model
    model_loaded = tf.keras.models.load_model(Model_path,compile=False)

    # make predictions
    y_pred = model_loaded.predict(X)

    # how many output params
    N_p = y.shape[2]
    assert N_p == len(r_yscale), "size of y and r_yscale does not match!"

    # convert y, y_pred to original unit
    import copy
    y_pred_rscale = copy.deepcopy(y_pred)
    y_rscale = copy.deepcopy(y)
    for ip in range(N_p):
        y_pred_rscale[:,:,ip] = r_yscale[ip](y_pred[:,:,ip])
        y_rscale[:,:,ip] = r_yscale[ip](y[:,:,ip])

    #====== start plotting ======
    # color-coded by Mw
    vmin, vmax = 6.9,9.6 #set the Mw from this range, don't want it starts from 0 at 0 s for example.
    alpha = 0.6
    cm = plt.cm.magma_r(  plt.Normalize(vmin,vmax)(y_rscale[:,-1,0]) )
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap='magma_r')
    cmap.set_array([])

    if idx is None:
        idx = np.arange(len(y)) # what indexes are you plotting? add any filtering here

    # marker size
    ms = make_linear_scale(min(y_rscale[idx,-1,0]),max(y_rscale[idx,-1,0]),target_min=2,target_max=50)
    ms = ms(y_rscale[idx,-1,0])

    sav_acc_current = {}
    sav_acc_final = {}
    for epo in range(102):
        plt.figure(figsize=(9.5,5.5))
        fig, axes = plt.subplots(2,3, figsize=(9.5,5.5))
        #plt.subplots_adjust(left=0.05,top=0.95,right=0.99,bottom=0.07,wspace=0.14,hspace=0.14)
        #axes[1][2].set_visible(False)
        #tmp0 = axes[0][0].get_position()
        #tmp = axes[1][0].get_position()
        #axes[1][0].set_position([0.24,tmp.y0,tmp0.x1,tmp0.y1])
        #axes[1][1].set_position([0.55,tmp.y0,tmp0.x1,tmp0.y1])
        #epo = 30
        if use_final:
            epo_y = -1
        else:
            epo_y = epo
        #=============
        ##plt.subplot(2,3,1)
        #plt.plot(sav_mft[(0,epo)],sav_c,'k.')
        axes[0][0].scatter(y_rscale[idx,epo_y,0],y_pred_rscale[idx,epo,0],s=ms,c=cm[idx],edgecolor='k',linewidth=0.5,vmin=vmin,vmax=vmax,alpha=alpha)
        axes[0][0].plot([vmin,vmax],[vmin,vmax],color=[1,0,1])
        if mark_range:
            YRange = np.max(y_rscale[:,:,0])-np.min(y_rscale[:,:,0])
            thresh = YRange*mark_range
            thresh = 0.3 # manually fix the Mw error to be 0.3!!!
            print('Add error range at figure 1. Range=%f'%(thresh))
            axes[0][0].plot([vmin,vmax],[vmin-thresh,vmax-thresh],'--',color=[1,0,1])
            axes[0][0].plot([vmin,vmax],[vmin+thresh,vmax+thresh],'--',color=[1,0,1])
            acc = len(np.where( np.abs(y_rscale[idx,epo_y,0]-y_pred_rscale[idx,epo,0])<=thresh )[0])/len(y_rscale[idx,epo_y,0])
            acc *= 100 #percentage
        #---calculate acc for both final y and current y---
        acc_current = len(np.where( np.abs(y_rscale[idx,epo,0]-y_pred_rscale[idx,epo,0])<=thresh )[0])/len(y_rscale[idx,epo,0])*100
        acc_final = len(np.where( np.abs(y_rscale[idx,-1,0]-y_pred_rscale[idx,epo,0])<=thresh )[0])/len(y_rscale[idx,-1,0])*100
        if 0 not in sav_acc_current:
            sav_acc_current[0] = [acc_current]
            sav_acc_final[0] = [acc_final]
        else:
            sav_acc_current[0].append(acc_current) # the 0-st parameter
            sav_acc_final[0].append(acc_final)
        #---calculateion done and will be used later---------
        #plt.scatter(sav_mft[(0,epo)][idx]/R[0],sav_SNR_mean[idx],c=cm[idx],cmap='magma',s=20,vmin=7.4,vmax=9.6,alpha=0.9)
        axes[0][0].set_ylabel('Prediction',fontsize=14,labelpad=0)
        axes[0][0].set_xlabel('True',fontsize=14,labelpad=0)
        #plt.xlim([y_rscale[:,:,0].min(),y_rscale[:,:,0].max()])
        #plt.ylim([y_rscale[:,:,0].min(),y_rscale[:,:,0].max()])
        axes[0][0].set_xlim([vmin,vmax])
        axes[0][0].set_ylim([vmin,vmax])
        #ax1=plt.gca()
        axes[0][0].tick_params(direction='out', pad=0.2,labelsize=12,length=0.2)
        axes[0][0].annotate('Mw',xy=(0.94,0.95),xycoords='axes fraction',size=14, ha='right', va='top',bbox=dict(boxstyle='round', fc='w',alpha=0.7))
        if mark_range:
            axes[0][0].annotate('%.1f %%'%(acc),xy=(0.06,0.95),xycoords='axes fraction',size=14, ha='left', va='top',bbox=dict(boxstyle='round', fc='w',alpha=0.7))
        #ax1.set_yscale('log')
        # add colorbar
        #These two lines mean put the bar inside the plot
        fig = plt.gcf()
        cbaxes = fig.add_axes([0.22, 0.62, 0.074, 0.012 ])
        clb = plt.colorbar(cmap,cax=cbaxes,ticks=[7.0, 8.0, 9.0], orientation='horizontal',label='Mw')
        clb.set_label('Mw', rotation=0,labelpad=-2,size=12)
        clb.solids.set(alpha=alpha)
        ax1=plt.gca()
        ax1.tick_params(pad=0.1,length=0.5)
        #plt.legend(['Mw'],frameon=True)
        #=============
        ##plt.subplot(2,3,2)
        axes[0][1].set_title('%d s'%(epo*5+5))
        #plt.plot(sav_mft[(1,epo)],sav_c,'k.')
        axes[0][1].scatter(y_rscale[idx,epo_y,1],y_pred_rscale[idx,epo,1],s=ms,c=cm[idx],edgecolor='k',linewidth=0.5,vmin=vmin,vmax=vmax,alpha=alpha)
        #plt.scatter(sav_mft[(1,epo)][idx]/R[1],sav_SNR_mean[idx],c=cm[idx],cmap='magma',s=20,vmin=7.4,vmax=9.6,alpha=0.9)
        axes[0][1].plot([y_rscale[:,:,1].min(),y_rscale[:,:,1].max()],[y_rscale[:,:,1].min(),y_rscale[:,:,1].max()],color=[1,0,1])
        if mark_range:
            YRange = np.max(y_rscale[:,:,1])-np.min(y_rscale[:,:,1])
            thresh = YRange*mark_range
            print('Add error range at figure 2. Range=%f'%(thresh))
            axes[0][1].plot([y_rscale[:,:,1].min(),y_rscale[:,:,1].max()],[y_rscale[:,:,1].min()-thresh,y_rscale[:,:,1].max()-thresh],'--',color=[1,0,1])
            axes[0][1].plot([y_rscale[:,:,1].min(),y_rscale[:,:,1].max()],[y_rscale[:,:,1].min()+thresh,y_rscale[:,:,1].max()+thresh],'--',color=[1,0,1])
            acc = len(np.where( np.abs(y_rscale[idx,epo_y,1]-y_pred_rscale[idx,epo,1])<=thresh )[0])/len(y_rscale[idx,epo_y,1])
            acc *= 100 #percentage
        #---calculate acc for both final y and current y---
        acc_current = len(np.where( np.abs(y_rscale[idx,epo,1]-y_pred_rscale[idx,epo,1])<=thresh )[0])/len(y_rscale[idx,epo,1])*100
        acc_final = len(np.where( np.abs(y_rscale[idx,-1,1]-y_pred_rscale[idx,epo,1])<=thresh )[0])/len(y_rscale[idx,-1,1])*100
        if 1 not in sav_acc_current:
            sav_acc_current[1] = [acc_current]
            sav_acc_final[1] = [acc_final]
        else:
            sav_acc_current[1].append(acc_current) # the 0-st parameter
            sav_acc_final[1].append(acc_final)
        #---calculateion done and will be used later---------
        axes[0][1].set_ylabel('Prediction',fontsize=14,labelpad=0)
        axes[0][1].set_xlabel('True',fontsize=14,labelpad=0)
        axes[0][1].set_xlim([y_rscale[:,:,1].min(),y_rscale[:,:,1].max()])
        axes[0][1].set_ylim([y_rscale[:,:,1].min(),y_rscale[:,:,1].max()])
        #ax1=plt.gca()
        axes[0][1].tick_params(pad=0.2,labelsize=12,length=0.2)
        #ax1.set_yscale('log')
        axes[0][1].annotate('Lon${\degree}$',xy=(0.94,0.95),xycoords='axes fraction',size=14, ha='right', va='top',bbox=dict(boxstyle='round', fc='w',alpha=0.7))
        if mark_range:
            axes[0][1].annotate('%.1f %%'%(acc),xy=(0.06,0.95),xycoords='axes fraction',size=14, ha='left', va='top',bbox=dict(boxstyle='round', fc='w',alpha=0.7))
        #=============
        ##plt.subplot(2,3,3)
        #plt.plot(sav_mft[(2,epo)],sav_c,'k.')
        axes[0][2].scatter(y_rscale[idx,epo_y,2],y_pred_rscale[idx,epo,2],s=ms,c=cm[idx],edgecolor='k',linewidth=0.5,vmin=vmin,vmax=vmax,alpha=alpha)
        #plt.scatter(sav_mft[(2,epo)][idx]/R[2],sav_SNR_mean[idx],c=cm[idx],cmap='magma',s=20,vmin=7.4,vmax=9.6,alpha=0.9)
        axes[0][2].plot([y_rscale[:,:,2].min(),y_rscale[:,:,2].max()],[y_rscale[:,:,2].min(),y_rscale[:,:,2].max()],color=[1,0,1])
        if mark_range:
            YRange = np.max(y_rscale[:,:,2])-np.min(y_rscale[:,:,2])
            thresh = YRange*mark_range
            print('Add error range at figure 3. Range=%f'%(thresh))
            axes[0][2].plot([y_rscale[:,:,2].min(),y_rscale[:,:,2].max()],[y_rscale[:,:,2].min()-thresh,y_rscale[:,:,2].max()-thresh],'--',color=[1,0,1])
            axes[0][2].plot([y_rscale[:,:,2].min(),y_rscale[:,:,2].max()],[y_rscale[:,:,2].min()+thresh,y_rscale[:,:,2].max()+thresh],'--',color=[1,0,1])
            acc = len(np.where( np.abs(y_rscale[idx,epo_y,2]-y_pred_rscale[idx,epo,2])<=thresh )[0])/len(y_rscale[idx,epo_y,2])
            acc *= 100 #percentage
        #---calculate acc for both final y and current y---
        acc_current = len(np.where( np.abs(y_rscale[idx,epo,2]-y_pred_rscale[idx,epo,2])<=thresh )[0])/len(y_rscale[idx,epo,2])*100
        acc_final = len(np.where( np.abs(y_rscale[idx,-1,2]-y_pred_rscale[idx,epo,2])<=thresh )[0])/len(y_rscale[idx,-1,2])*100
        if 2 not in sav_acc_current:
            sav_acc_current[2] = [acc_current]
            sav_acc_final[2] = [acc_final]
        else:
            sav_acc_current[2].append(acc_current) # the 0-st parameter
            sav_acc_final[2].append(acc_final)
        #---calculateion done and will be used later---------
        axes[0][2].set_ylabel('Prediction',fontsize=14,labelpad=0)
        axes[0][2].set_xlabel('True',fontsize=14,labelpad=0)
        axes[0][2].set_xlim([y_rscale[:,:,2].min(),y_rscale[:,:,2].max()])
        axes[0][2].set_ylim([y_rscale[:,:,2].min(),y_rscale[:,:,2].max()])
        #ax1=plt.gca()
        axes[0][2].tick_params(pad=0.2,labelsize=12,length=0.2)
        #ax1.set_yscale('log')
        axes[0][2].annotate('Lat${\degree}$',xy=(0.94,0.95),xycoords='axes fraction',size=14, ha='right', va='top',bbox=dict(boxstyle='round', fc='w',alpha=0.7))
        if mark_range:
            axes[0][2].annotate('%.1f %%'%(acc),xy=(0.06,0.95),xycoords='axes fraction',size=14, ha='left', va='top',bbox=dict(boxstyle='round', fc='w',alpha=0.7))
        #=============
        #plt.subplot(2,3,4)
        #plt.plot(sav_mft[(3,epo)],sav_c,'k.')
        axes[1][0].scatter(y_rscale[idx,epo_y,3],y_pred_rscale[idx,epo,3],s=ms,c=cm[idx],edgecolor='k',linewidth=0.5,vmin=vmin,vmax=vmax,alpha=alpha)
        #plt.scatter(sav_mft[(3,epo)][idx]/R[3],sav_SNR_mean[idx],c=cm[idx],cmap='magma',s=20,vmin=7.4,vmax=9.6,alpha=0.9)
        axes[1][0].plot([y_rscale[:,:,3].min(),y_rscale[:,:,3].max()],[y_rscale[:,:,3].min(),y_rscale[:,:,3].max()],color=[1,0,1])
        if mark_range:
            YRange = np.max(y_rscale[:,:,3])-np.min(y_rscale[:,:,3])
            thresh = YRange*mark_range
            print('Add error range at figure 4. Range=%f'%(thresh))
            axes[1][0].plot([y_rscale[:,:,3].min(),y_rscale[:,:,3].max()],[y_rscale[:,:,3].min()-thresh,y_rscale[:,:,3].max()-thresh],'--',color=[1,0,1])
            axes[1][0].plot([y_rscale[:,:,3].min(),y_rscale[:,:,3].max()],[y_rscale[:,:,3].min()+thresh,y_rscale[:,:,3].max()+thresh],'--',color=[1,0,1])
            acc = len(np.where( np.abs(y_rscale[idx,epo_y,3]-y_pred_rscale[idx,epo,3])<=thresh )[0])/len(y_rscale[idx,epo_y,3])
            acc *= 100 #percentage
        #---calculate acc for both final y and current y---
        acc_current = len(np.where( np.abs(y_rscale[idx,epo,3]-y_pred_rscale[idx,epo,3])<=thresh )[0])/len(y_rscale[idx,epo,3])*100
        acc_final = len(np.where( np.abs(y_rscale[idx,-1,3]-y_pred_rscale[idx,epo,3])<=thresh )[0])/len(y_rscale[idx,-1,3])*100
        if 3 not in sav_acc_current:
            sav_acc_current[3] = [acc_current]
            sav_acc_final[3] = [acc_final]
        else:
            sav_acc_current[3].append(acc_current) # the 0-st parameter
            sav_acc_final[3].append(acc_final)
        #---calculateion done and will be used later---------
        #plt.ylabel('Avg. SNR',fontsize=14,labelpad=0)
        #plt.xlabel('|| y$_{pred}$ - y ||',fontsize=14,labelpad=0)
        axes[1][0].set_ylabel('Prediction',fontsize=14,labelpad=0)
        axes[1][0].set_xlabel('True',fontsize=14,labelpad=0)
        axes[1][0].set_xlim([min(y_rscale[:,:,3].min(),-50),y_rscale[:,:,3].max()])
        axes[1][0].set_ylim([min(y_rscale[:,:,3].min(),-50),y_rscale[:,:,3].max()])
        #plt.xlabel('%',fontsize=14,labelpad=0)
        #ax1=plt.gca()
        axes[1][0].tick_params(pad=0.2,labelsize=12,length=0.2)
        axes[1][0].ticklabel_format(style='sci', axis='y',scilimits=(0,0))
        #ax1.set_yscale('log')
        axes[1][0].annotate('Length (km)',xy=(0.94,0.95),xycoords='axes fraction',size=14, ha='right', va='top',bbox=dict(boxstyle='round', fc='w',alpha=0.7))
        if mark_range:
            axes[1][0].annotate('%.1f %%'%(acc),xy=(0.06,0.95),xycoords='axes fraction',size=14, ha='left', va='top',bbox=dict(boxstyle='round', fc='w',alpha=0.7))
        #=============
        ##plt.subplot(2,3,5)
        #plt.plot(sav_mft[(4,epo)],sav_c,'k.')
        axes[1][1].scatter(y_rscale[idx,epo_y,4],y_pred_rscale[idx,epo,4],s=ms,c=cm[idx],edgecolor='k',linewidth=0.5,vmin=vmin,vmax=vmax,alpha=alpha)
        #plt.scatter(sav_mft[(4,epo)][idx]/R[4],sav_SNR_mean[idx],c=cm[idx],cmap='magma',s=20,vmin=7.4,vmax=9.6,alpha=0.9)
        axes[1][1].plot([y_rscale[:,:,4].min(),y_rscale[:,:,4].max()],[y_rscale[:,:,4].min(),y_rscale[:,:,4].max()],color=[1,0,1])
        if mark_range:
            YRange = np.max(y_rscale[:,:,4])-np.min(y_rscale[:,:,4])
            thresh = YRange*mark_range
            print('Add error range at figure 5. Range=%f'%(thresh))
            axes[1][1].plot([y_rscale[:,:,4].min(),y_rscale[:,:,4].max()],[y_rscale[:,:,4].min()-thresh,y_rscale[:,:,4].max()-thresh],'--',color=[1,0,1])
            axes[1][1].plot([y_rscale[:,:,4].min(),y_rscale[:,:,4].max()],[y_rscale[:,:,4].min()+thresh,y_rscale[:,:,4].max()+thresh],'--',color=[1,0,1])
            acc = len(np.where( np.abs(y_rscale[idx,epo_y,4]-y_pred_rscale[idx,epo,4])<=thresh )[0])/len(y_rscale[idx,epo_y,4])
            acc *= 100 #percentage
        #---calculate acc for both final y and current y---
        acc_current = len(np.where( np.abs(y_rscale[idx,epo,4]-y_pred_rscale[idx,epo,4])<=thresh )[0])/len(y_rscale[idx,epo,4])*100
        acc_final = len(np.where( np.abs(y_rscale[idx,-1,4]-y_pred_rscale[idx,epo,4])<=thresh )[0])/len(y_rscale[idx,-1,4])*100
        if 4 not in sav_acc_current:
            sav_acc_current[4] = [acc_current]
            sav_acc_final[4] = [acc_final]
        else:
            sav_acc_current[4].append(acc_current) # the 0-st parameter
            sav_acc_final[4].append(acc_final)
        #---calculateion done and will be used later---------
        axes[1][1].set_xlim([min(y_rscale[:,:,4].min(),-5),y_rscale[:,:,4].max()]) # this min(.min(),-100) makes better plotting
        axes[1][1].set_ylim([min(y_rscale[:,:,4].min(),-5),y_rscale[:,:,4].max()])
        axes[1][1].set_ylabel('Prediction',fontsize=14,labelpad=0)
        axes[1][1].set_xlabel('True',fontsize=14,labelpad=0)
        #plt.xlabel('%',fontsize=14,labelpad=0)
        #ax1=plt.gca()
        axes[1][1].tick_params(pad=0.2,labelsize=12,length=0.2)
        axes[1][1].ticklabel_format(style='sci', axis='y',scilimits=(0,0))
        #ax1.set_yscale('log')
        axes[1][1].annotate('Width (km)',xy=(0.94,0.95),xycoords='axes fraction',size=14, ha='right', va='top',bbox=dict(boxstyle='round', fc='w',alpha=0.7))
        if mark_range:
            axes[1][1].annotate('%.1f %%'%(acc),xy=(0.06,0.95),xycoords='axes fraction',size=14, ha='left', va='top',bbox=dict(boxstyle='round', fc='w',alpha=0.7))
        #=============
        #---make accuracy plot---
        #plt.subplot(2,3,6)
        line_colors = [[1,0,0],[0,0,0],[0,1,0],[0,0,1],[1,0,1]]
        line_styles_current = ['o--','v--','^--','*--','s--']
        line_styles_final = ['o-','v-','^-','*-','s-']
        acc_time = np.arange(102)*5+5
        for isrc in range(5):
            #for the i-th source get the results from sav_acc_current & sav_acc_final
            # plot two accuracy is way toooo busy, just one accuracy instead
            if use_final:
                axes[1][2].plot(acc_time[:epo+1],sav_acc_final[isrc],line_styles_final[isrc],color=line_colors[isrc],markeredgecolor='k',markeredgewidth=0.1,linewidth=0.1,markersize=5)
            else:
                axes[1][2].plot(acc_time[:epo+1],sav_acc_current[isrc],line_styles_current[isrc],color=line_colors[isrc],markeredgecolor='k',markeredgewidth=0.1,linewidth=0.1,markersize=5)
        axes[1][2].legend(['Mw','Lon','Lat','Length','Width'],fontsize=12,loc=4,frameon=True,shadow=True)
        axes[1][2].set_xlim([0,515])
        axes[1][2].set_ylim([50,101])
        axes[1][2].set_xlabel('Time (s)',fontsize=14,labelpad=0)
        axes[1][2].set_ylabel('Accuracy (%)',fontsize=14,labelpad=0)
        axes[1][2].tick_params(pad=0.2,labelsize=12,length=0.2)
        '''
        plt.xlim([min(y_rscale[:,:,5].min(),-5),max(y_rscale[:,:,5].max(),y_pred_rscale[:,:,5].max())])
        plt.ylim([min(y_rscale[:,:,5].min(),-5),max(y_rscale[:,:,5].max(),y_pred_rscale[:,:,5].max())])
        plt.xlabel('True',fontsize=14,labelpad=0)
        #plt.xlabel('%',fontsize=14,labelpad=0)
        ax1=plt.gca()
        ax1.tick_params(pad=0,labelsize=12,length=0)
        ax1.ticklabel_format(style='sci', axis='y',scilimits=(0,0))
        #ax1.set_yscale('log')
        ax1.annotate('Width (km)',xy=(0.94,0.95),xycoords='axes fraction',size=14, ha='right', va='top',bbox=dict(boxstyle='round', fc='w',alpha=0.7))
        if mark_range:
            ax1.annotate('%.1f %%'%(acc),xy=(0.06,0.95),xycoords='axes fraction',size=14, ha='left', va='top',bbox=dict(boxstyle='round', fc='w',alpha=0.7))
        '''
        #adjust subplots width/length
        #plt.subplots_adjust(left=0.05,top=0.92,right=0.97,bottom=0.1,wspace=0.07,hspace=0.14)
        #plt.show()
        #break
        plt.subplots_adjust(left=0.05,top=0.95,right=0.99,bottom=0.07,wspace=0.24,hspace=0.24)
        #-----add these two lines to center the bottom two subplots-----
        #tmp0 = axes[0][0].get_position()
        #tmp = axes[1][0].get_position()
        #axes[1][0].set_position([0.21,tmp.y0,tmp0.x1-tmp0.x0,tmp0.y1-tmp0.y0])
        #axes[1][1].set_position([0.555,tmp.y0,tmp0.x1-tmp0.x0,tmp0.y1-tmp0.y0])
        if save_fig:
            #plt.subplots_adjust(left=0.05,top=0.95,right=0.99,bottom=0.07,wspace=0.14,hspace=0.14)
            plt.savefig(save_fig+'/fig_%03d.png'%(epo),dpi=300)
            plt.close()
        else:
            plt.show()
        #plt.savefig('./misfit_meanSNR_figs/fig_%03d.png'%(epo))
        #plt.close()





def plot_rupt_retc(rupt,min_slip,max_time,rect_fault,fix_vmax=[0,10],save_fig=None):
    '''
    plot rupt file v.s. rectangular fault file
    rectangular fault can be generated by postprocessing.fault_tool() object
    e.g.
        F = postprocessing.fault_tool(Mw,center,strike,dip,length,width,fout,rupt_path,dist_strike,dist_dip)
        # Note that Mw, center, strike, dip, length and width will be reset when calling F.gen_param_from_rupt()
        F.set_default_params('Chile')
        F.gen_param_from_rupt()
        F.gen_fault(dx_strike=10,dx_dip=8)
    Input:
        rupt: rupture file path
        min_slip: only plot slip greater than this value
        max_time: only plot slip occur before "or equal" this time
        rect_fault: rectangular fault file path
    '''
    rupt = np.genfromtxt(rupt)
    slip = (rupt[:,8]**2 + rupt[:,9]**2)**0.5
    rupt_time = rupt[:,-2]
    idx = np.where( (slip>0) & (rupt_time<=max_time))[0]
    idx_full = np.where(slip>0)[0] # to get plotting boundary
    XLIM1 = [rupt[idx_full,1].min(),rupt[idx_full,1].max()]
    YLIM1 = [rupt[idx_full,2].min(),rupt[idx_full,2].max()]
    # plot rupt
    #plt.plot(rupt[:,1],rupt[:,2],'.',color=[0.8,0.8,0.8])
    plt.scatter(rupt[idx,1],rupt[idx,2],c=slip[idx],vmin=fix_vmax[0],vmax=fix_vmax[1],s=20,cmap='magma')
    plt.colorbar()
    # plot rect_fault
    fault = np.genfromtxt(rect_fault)
    if fault.ndim==1:
        plt.plot(fault[1],fault[2],'bs',alpha=0.1)
        XLIM2 = [fault[1],fault[1]]
        YLIM2 = [fault[2],fault[2]]
    else:
        plt.plot(fault[:,1],fault[:,2],'bs',alpha=0.1)
        XLIM2 = [fault[:,1].min(),fault[:,1].max()]
        YLIM2 = [fault[:,2].min(),fault[:,2].max()]
    XLIM = [min(XLIM1[0],XLIM2[0])-0.5,max(XLIM1[1],XLIM2[1])+0.5]
    YLIM = [min(YLIM1[0],YLIM2[0])-0.5,max(YLIM1[1],YLIM2[1])+0.5]
    plt.xlim(XLIM)
    plt.ylim(YLIM)
    if save_fig:
        plt.savefig(save_fig)
        plt.close()
    else:
        plt.show()


def plot_sum_rupt(rupt_dirs,save_fig=None):
    '''
        Plot sum of all ruptures based on the subfault participation (rupture +1 or not +0)
        Input:
            rupt_dirs: directory of Mudpy ruptures output, can be multiple directories
        Output:
            A map
    '''
    import glob
    rupts = []
    for rupt_dir in rupt_dirs:
        rupts += glob.glob(rupt_dir+'/'+'*.rupt')
    #initial the rupt_part based on the first rupt file
    tmp = np.genfromtxt(rupts[0])
    sum_rupt = np.zeros(len(tmp))
    for i_rupt,rupt in enumerate(rupts):
        if i_rupt%50==0:
            print('%d out of %d'%(i_rupt,len(rupts)))
        A = np.genfromtxt(rupt)
        #SS = A[:,8]
        #DS = A[:,9]
        idx = np.where((A[:,8]!=0) | (A[:,9]!=0))[0]
        sum_rupt[idx] += 1
    # make figure
    BMap_flag = False #basemap
    try:
        from mpl_toolkits.basemap import Basemap
        BMap_flag = True
    except:
        print('cannot import Basemap!')
    if BMap_flag:
        map = Basemap(projection='cyl',resolution='f',llcrnrlon=min(tmp[:,1])-1,llcrnrlat=min(tmp[:,2])-1,urcrnrlon=max(tmp[:,1])+1,urcrnrlat=max(tmp[:,2])+1,fix_aspect=False)
        fig = map.arcgisimage(service='ESRI_Imagery_World_2D', xpixels = 2000, verbose= True)
        fig.set_alpha(0.8)
        map.drawstates()
        map.drawcountries(linewidth=0.1)
        map.drawcoastlines(linewidth=0.5)
        #Lon,Lat lines
        dn = 5
        lats = map.drawparallels(np.arange(-90,90,dn),labels=[1,0,0,1],color='w',linewidth=0.5)
        lons = map.drawmeridians(np.arange(-180,180,2),labels=[1,0,0,1],color='w',linewidth=0.5)
        plt.scatter(tmp[:,1],tmp[:,2],c=sum_rupt,cmap='jet',s=10)
        plt.colorbar()
        #plot stations on map
        #plt.plot(sav_D[0]['stlon'],sav_D[0]['stlat'],'r^',markeredgecolor='k')
    else:
        plt.scatter(tmp[:,1],tmp[:,2],c=sum_rupt,cmap='jet',s=10)
        plt.colorbar()
    if save_fig:
        plt.savefig(save_fig)
    else:
        plt.show()
    plt.close()












