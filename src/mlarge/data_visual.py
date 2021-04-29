# some visualization toolds

import numpy as np
import matplotlib.pyplot as plt


def view_sources(EQinfo):
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
    fig = plt.figure(figsize=(18,10.5))
    
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
    ax1.text(Mw_range[0]+(Mw_range[1]-Mw_range[0])*0.4,10**(log_ylim[0]+(log_ylim[1]-log_ylim[0])*0.08),r'$\log (L)=-2.37+0.57M_w$',fontsize=12) #text plot in original scale
    ax1.set_yscale('log')
    ymajorLocator = LogLocator(base=10.0,numticks = 5)
    yminorLocator = LogLocator(base=10.0,subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    ax1.yaxis.set_major_locator(ymajorLocator)
    ax1.yaxis.set_minor_locator(yminorLocator)
    ax1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax1.tick_params(which='major',length=5,width=0.8)
    ax1.tick_params(which='minor',length=2.5,width=0.8)
    ax1.set_ylabel('Fault length (km)',labelpad=-1)
    ax1.tick_params(axis='x',labelbottom=False)
    ax1.tick_params(axis='y',pad=-1)

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
    ax2.text(Mw_range[0]+(Mw_range[1]-Mw_range[0])*0.4,10**(log_ylim[0]+(log_ylim[1]-log_ylim[0])*0.08),r'$\log (W)=-1.86+0.46M_w$',fontsize=12) #text plot in original scale
    ax2.set_yscale('log')
    ymajorLocator = LogLocator(base=10.0,numticks = 5)
    yminorLocator = LogLocator(base=10.0,subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    ax2.yaxis.set_major_locator(ymajorLocator)
    ax2.yaxis.set_minor_locator(yminorLocator)
    ax2.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.tick_params(which='major',length=5,width=0.8)
    ax2.tick_params(which='minor',length=2.5,width=0.8)
    ax2.set_ylabel('Fault width (km)',labelpad=-1)
    ax2.tick_params(axis='x',labelbottom=False)
    ax2.tick_params(axis='y',pad=-1)

    # third subplot
    ax3 = fig.add_subplot(333)
    ax3.plot(Mw, tar_Mw,'+',markersize=3,alpha=0.9)
    ax3.set_xlim(Mw_range)
    ax3.tick_params(which='major',length=5,width=0.8)
    ax3.tick_params(which='minor',length=2.5,width=0.8)
    ax3.set_ylabel('Target Mw',labelpad=-1)
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
    ax4.set_ylabel('Mean slip (m)',labelpad=-1)
    ax4.tick_params(axis='x',labelbottom=False)
    ax4.tick_params(axis='y',pad=-1)

    # 5th subplot
    ax5 = fig.add_subplot(335)
    ax5.plot(Mw, max_slip,'+',markersize=3,alpha=0.9)
    ax5.set_xlim(Mw_range)
    ax5.set_yscale('log')
    ax5.tick_params(which='major',length=5,width=0.8)
    ax5.tick_params(which='minor',length=2.5,width=0.8)
    ax5.set_ylabel('Max. slip (m)',labelpad=-1)
    ax5.tick_params(axis='x',labelbottom=False)
    ax5.tick_params(axis='y',pad=-1)

    # 6th subplot
    ax6 = fig.add_subplot(336)
    ax6.plot(Mw, std_slip,'+',markersize=3,alpha=0.9)
    ax6.set_xlim(Mw_range)
    #ax6.set_ylim([tmp_yrange[min_idx],tmp_yrange[max_idx]])
    ax6.set_ylim([0.05,np.max(std_slip)])
    ax6.set_yscale('log')
    ax6.tick_params(which='major',length=5,width=0.8)
    ax6.tick_params(which='minor',length=2.5,width=0.8)
    ax6.set_ylabel('Slip std. dev. (m)',labelpad=-1)
    ax6.tick_params(axis='x',labelbottom=False)
    ax6.tick_params(axis='y',pad=-1)


    # 7th subplot
    ax7 = fig.add_subplot(337)
    ax7.plot(Mw, mean_rise,'+',markersize=3,alpha=0.9)
    ax7.set_xlim(Mw_range)
    ax7.set_yscale('log')
    if mean_rise.max()<10:
        ymajorLocator = LogLocator(base=2,numticks = 5)
        yminorLocator = LogLocator(base=2,subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    else:
        ymajorLocator = LogLocator(base=10,numticks = 5)
        yminorLocator = LogLocator(base=10,subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    
    ax7.yaxis.set_major_locator(ymajorLocator)
    ax7.yaxis.set_minor_locator(yminorLocator)
    ax7.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax7.tick_params(which='major',length=5,width=0.8)
    ax7.tick_params(which='minor',length=2.5,width=0.8)
    ax7.set_ylabel('Mean rise time (s)',labelpad=-1)
    ax7.tick_params(axis='y',pad=-1)
    ax7.set_xlabel('Actual Mw',labelpad=0)
    ax7.tick_params(axis='x',pad=0)
    
    # 8th subplot
    ax8 = fig.add_subplot(338)
    ax8.plot(Mw, max_rise,'+',markersize=3,alpha=0.9)
    ax8.set_xlim(Mw_range)
    ax8.set_yscale('log')
    ax8.tick_params(which='major',length=5,width=0.8)
    ax8.tick_params(which='minor',length=2.5,width=0.8)
    ax8.set_ylabel('Max. rise time (s)',labelpad=-1)
    ax8.tick_params(axis='y',pad=-1)
    ax8.set_xlabel('Actual Mw',labelpad=0)
    ax8.tick_params(axis='x',pad=0)

    # 9th subplot
    ax9 = fig.add_subplot(339)
    ax9.plot(Mw, std_rise,'+',markersize=3,alpha=0.9)
    ax9.set_xlim(Mw_range)
    ax9.set_ylim([0.2,np.max(std_rise)])
    ax9.set_yscale('log')
    ax9.tick_params(which='major',length=5,width=0.8)
    ax9.tick_params(which='minor',length=2.5,width=0.8)
    ax9.set_ylabel('Rise time std. dev. (s)',labelpad=-1)
    ax9.tick_params(axis='y',pad=-1)
    ax9.set_xlabel('Actual Mw',labelpad=0)
    ax9.tick_params(axis='x',pad=0)
    
    # final adjustment
    plt.subplots_adjust(left=0.08,top=0.88,right=0.97,bottom=0.1,wspace=0.15,hspace=0.06)
    plt.show()


    

