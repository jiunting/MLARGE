#Script that test the RNN model

import numpy as np
from scipy.integrate import cumtrapz
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import matplotlib
from tensorflow.keras import models
from tensorflow.keras import layers
import seaborn as sns
import obspy
from obspy.taup import TauPyModel
import sys,os,shutil,glob,datetime
import imageio
import glob
#GFAST
from eew_data_engine_synthetic import data_engine_pgd



#parameters for analyzing result
home='/Users/timlin/Documents/Project/MLARGE'         #Home directory
#model_name='Test80_weights.49160-0.000127.hdf5'
model_name='Test81_weights.49475-0.000131.hdf5'
#model_name='Test73_weights.37255-0.000129.hdf5'
run_name='Test81'


#Model path
model_loaded=tf.keras.models.load_model(home+'/'+'models/'+model_name,compile=False) #
X1=np.load(home+'/data/'+'Xtest81.npy') #IMPORTANT! Make sure use the same scale that's used for the ML training
y1=np.load(home+'/data/'+'ytest81.npy') #
real_EQID=np.load(home+'/data/'+'Run81_test_EQID.npy') #real EQID for testing dataset
#ALLEQ=np.genfromtxt('../Chile_full_27200_source.txt')
ALLEQ=np.genfromtxt(home+'/data/'+'Chile_full_new_source_slip.txt')

#plot ML structure
plot_architecture=True

#Define what kind of misfit you want to use
Misfit_current=False #True: use the (pred-real current label)<0.3,  False: use the (pred-final label)<0.3
#Misfit_current=True #True: for the real EQ, so that makes more sense


def back_scale_X(X,half=True):
    #Backward scaling
    if half:
        nstan=int(X.shape[-1]/2)
        X1=X.copy()
#        X1[:,:,:nstan]=(X1[:,:,:nstan]*10.0)**2.0  #change this part
        X1[:,:,:nstan]=10**X1[:,:,:nstan]  #change this part
        return X1
    else:
        return (X*10.0)**2.0 #change this part
        #return 10**X

def scale_X(X,half=True):
    #Forward scaling
    if half:
        nstan=int(X.shape[-1]/2)
        X1=X.copy()
        #X1[:,:,:nstan]=(X1[:,:,:nstan]**0.5)/10.0  #change this part
        #or a different scaling, !!!remember to filter the 0 data
        X2=X1[:,:,:nstan].copy()
        X2=np.where(X2>=0.01,X2,0.01)
        X1[:,:,:nstan]=np.log10(X2)  #change this part
        return X1
    else:
        #return X**0.5/10.0
        X1=X.copy()
        X1=np.where(X1>=0.01,X1,0.01)
        return np.log10(X1)


def back_scale_y(y):
    return y*10.0


def valid_loss_timeMw(y1,predictions):
    #weighted Mw, Time, MSE
    idx=np.arange(1,y1.shape[1]+1)
    idx =  1 - (np.exp(-1*idx/2))
    idx = idx.reshape(-1,1)
    diff_M=(predictions[:,:,0]-y1[:,:,0])**2.0 * y1[:,:,0]
#    tmp_sum=0
#    for tmp_diff in diff_M:
#        tmp_sum = tmp_sum + np.sum(tmp_diff*idx[:,0])
    return np.mean(np.dot(diff_M,idx))


def get_accuracy(pred_Mw,real_Mw,tolorance=0.3,NoiseMw=False,tolorance_noise=False):
    #Make accuracy calculation
    #tolorance: constant +- tolorance consider as corrected prediction
    #NoiseMw: what are the noise event magnitude? or False if no noise events
    #tolorance_noise can be +- tolorance for noise magnitude, or False (do not count noise event)
    if NoiseMw==False and tolorance_noise==False:
        #only EQ events
        #accuracy for EQ
        n_success_EQ=np.where( np.abs(pred_Mw-real_Mw)<=tolorance )[0] #Mw is the Mw prediction at certain time t
        return (len(n_success_EQ)/len(pred_Mw))*100, None
    else:
        #noise and EQ events
        EQidx=np.where(real_Mw!=NoiseMw)[0]
        Noiseidx=np.where(real_Mw==NoiseMw)[0]
        #accuracy for EQ
        n_success_EQ=np.where(np.abs(pred_Mw[EQidx]-real_Mw[EQidx])<=tolorance)[0] #Mw is the Mw prediction at certain time t
        #accuracy for Noise
        n_success_Noise=np.where(np.abs(pred_Mw[Noiseidx]-real_Mw[Noiseidx])<=tolorance_noise )[0] #Mw is the Mw prediction at certain time t
        return (len(n_success_EQ)/len(pred_Mw))*100, (len(n_success_Noise)/len(Noiseidx))*100




#Plot model archetrcture
if plot_architecture:
    tf.keras.utils.plot_model(model_loaded,to_file='model_'+run_name+'.png',show_shapes=True,expand_nested=True,dpi=200)

###Make predictions#####
predictions=model_loaded.predict(X1)
#scale the labels back to the real sense
predictions=back_scale_y(predictions)
y1=back_scale_y(y1)


###Accuracy funtion(Time only)
sav_accEQ_3=[] #accuracy function for threshold=0.3
sav_accEQ_2=[]
sav_accEQ_1=[]
for i_epoch in range(102):
    acc_EQ,acc_noise=get_accuracy(predictions[:,i_epoch],y1[:,i_epoch],0.3)
    sav_accEQ_3.append(acc_EQ)
    acc_EQ,acc_noise=get_accuracy(predictions[:,i_epoch],y1[:,i_epoch],0.2)
    sav_accEQ_2.append(acc_EQ)
    acc_EQ,acc_noise=get_accuracy(predictions[:,i_epoch],y1[:,i_epoch],0.1)
    sav_accEQ_1.append(acc_EQ)


plt.figure()
plt.plot(np.arange(102)*5+5,sav_accEQ_3,'bo-')
plt.plot(np.arange(102)*5+5,sav_accEQ_2,'yo-')
plt.plot(np.arange(102)*5+5,sav_accEQ_1,'ro-')
plt.xlabel('Time (sec)',fontsize=14)
plt.ylabel('Accuracy(%)',fontsize=14)
plt.grid(True)
plt.show()
###------------END------------###

###Example scatter plot snapshot###
#plt.plot(y1[:,2],predictions[:,2],'b.',markersize=5)
#plt.plot([6.5,8.5],[6.5,8.5],'r--',linewidth=2)
#plt.plot([6.5,8.5],[6.5-0.3,8.5-0.3],'r--',linewidth=2)
#plt.plot([6.5,8.5],[6.5+0.3,8.5+0.3],'r--',linewidth=2)
#plt.xlim([6.5,8.5])
#plt.ylim([6.5,8.5])
#plt.xlabel('Real Mw',fontsize=14)
#plt.ylabel('Pred Mw',fontsize=14)
#plt.show()
###---------END---------###


###Get All the T/F point at Time,Mw. This will then used by accuracy function(Time,Mw) later###
sav_Mw_TF=[]
sav_Time_Mw_TF=[] #in X(Time),Y(Pred_Mw),Z(1,or 0)
for i_epoch in range(102):
    if Misfit_current:
        T_idx=np.where(np.abs(predictions[:,i_epoch,0]-y1[:,i_epoch,0])<=0.3)[0]
    else:
        T_idx=np.where(np.abs(predictions[:,i_epoch,0]-y1[:,-1,0])<=0.3)[0]
    Mw_TF=np.zeros(len(predictions))    #Mw prediction is True=1 or False=0?
    Mw_TF[T_idx]=1
    for i in range(len(Mw_TF)):
        if Misfit_current:
            sav_Time_Mw_TF.append([i_epoch*5+5,predictions[i,i_epoch,0],Mw_TF[i],predictions[i,i_epoch,0]-y1[i,i_epoch,0]])
        else:
            sav_Time_Mw_TF.append([i_epoch*5+5,predictions[i,i_epoch,0],Mw_TF[i],predictions[i,i_epoch,0]-y1[i,-1,0]])

sav_Time_Mw_TF=np.array(sav_Time_Mw_TF)
#plot those T/F points
plt.figure()
idx_T=np.where(sav_Time_Mw_TF[:,2]==1)[0]
idx_F=np.where(sav_Time_Mw_TF[:,2]==0)[0]
plt.plot(sav_Time_Mw_TF[:,0][idx_T],sav_Time_Mw_TF[:,1][idx_T],'r.',alpha=0.6,markersize=5)
plt.plot(sav_Time_Mw_TF[:,0][idx_F],sav_Time_Mw_TF[:,1][idx_F],'b.',alpha=0.6,markersize=5)
plt.legend(['True','False'])
plt.xlabel('Time(sec)',fontsize=14)
plt.ylabel('Predicted Mw',fontsize=14)
plt.show()
###---------END-------------###


###Get accuracy funtion as a function of Time,Mw###
#Group them by a nearby Mw, Time (i.e. Time and Mw smoothing)
GP_mw=np.arange(7.0,9.6,0.01)
Time=np.arange(102)*5+5
neighbor_mw=0.1 #center on target magnitude and +- this value
neighbor_time=5 #center on target time and +- this value
MovingAcc_Time_Mw=[]
sav_misfit_distribute=[]
for i_time in Time:
    #for each time, calculate accuracy for each group
    for gp_mw in GP_mw:
        tmp_gp_idxT=np.where( (np.abs(sav_Time_Mw_TF[:,0]-i_time)<=neighbor_time) & (np.abs(sav_Time_Mw_TF[:,1]-gp_mw)<=neighbor_mw) & (sav_Time_Mw_TF[:,2]==1) )[0]
        tmp_gp_idxF=np.where( (np.abs(sav_Time_Mw_TF[:,0]-i_time)<=neighbor_time) & (np.abs(sav_Time_Mw_TF[:,1]-gp_mw)<=neighbor_mw) & (sav_Time_Mw_TF[:,2]==0) )[0]
        if len(tmp_gp_idxT)==0 and len(tmp_gp_idxF)==0:
            tmpAcc=np.nan
            tmp_avgmis=np.nan
            tmp_distribution=np.nan
        else:
            misfit_T=sav_Time_Mw_TF[:,3][tmp_gp_idxT] #misfits for accurate determined event
            misfit_F=sav_Time_Mw_TF[:,3][tmp_gp_idxF] #misfits for accurate determined event
            misfit_all=np.hstack([misfit_T,misfit_F])
            tmpAcc=len(tmp_gp_idxT)/(len(tmp_gp_idxT)+len(tmp_gp_idxF))
            tmp_avgmis=np.mean(np.abs(misfit_all))
            tmp_distribution=misfit_all #this is an array
        sav_misfit_distribute.append(tmp_distribution)
        MovingAcc_Time_Mw.append([i_time,gp_mw,tmpAcc,tmp_avgmis])
    print('Finished:%f'%(i_time))

MovingAcc_Time_Mw=np.array(MovingAcc_Time_Mw)
#sav_misfit_distribute=np.array(sav_misfit_distribute)
#np.save('MovingAcc_Time_current.npy',MovingAcc_Time_Mw) #save the misfit function defined by current(not final) label
#np.save('sav_misfit_distribute_current.npy',sav_misfit_distribute)

#plt.scatter(MovingAcc_Time_Mw[:,0],MovingAcc_Time_Mw[:,1],c=MovingAcc_Time_Mw[:,2],cmap='jet') #This is scatter plot sense
def get_XYZ(x,y,z,stype,thres):
    #x,y,z are array-like and can be plotted by plt.scatter(x,y,c=z)
    #convert the format of x,y,z to X,Y,Z so that you can use plt.pcolor(X,Y,Z)
    #stype=sort type, fix x first and loop through y? stype=1, otherwise stype=2
    #threshold of how large as considered different point?
    X=[]
    for ix in x:
        if X==[]:
            X.append(ix)
            X=np.array(X)
        else:
            diff=np.abs(ix-X) #the difference between this point to the previous array
            if len(np.where(diff<thres)[0]>=1):
                continue
            else:
                X=np.hstack([X,ix])
    Y=[]
    for iy in y:
        if Y==[]:
            Y.append(iy)
            Y=np.array(Y)
        else:
            diff=np.abs(iy-Y) #the difference between this point to the previous array
            if len(np.where(diff<thres)[0]>=1):
                continue
            else:
                Y=np.hstack([Y,iy])
    #X=np.unique(x) #you can use unique if the data are perfectly gridded e.g. x=[0.000001 0.000001.....0.000002]; not [0.0000012 0.0000011 0.000001102], though they are close
    #Y=np.unique(y)
    z=np.array(z)
    if stype=='auto':
        if  (( (x[1]-x[0])<=thres) & ( (y[1]-y[0])>thres ) ) :
            stype=1
        elif (( (x[1]-x[0])>thres) & ( (y[1]-y[0])>thres) ):
            stype=2
        else:
            print('fail to determine the order!!, please check')
            if (x[1]-x[0])<(y[1]-y[0]):
                stype=1
            else:
                stype=2
            print('determined stype=%d?'%(stype))
    if stype==1:
        Z=z.reshape(len(X),len(Y))
        Z=Z.transpose()
    elif stype==2:
        Z=z.reshape(len(Y),len(X))
    else:
        print('please specific sort type of your Z by plot(x) or plot(y)')
    return(X,Y,Z)

def get_box_distribution(MovingAcc_Time_Mw,sav_misfit_distribute,target_Time=50,target_Mw=8.7):
    diff_T=np.abs(MovingAcc_Time_Mw[:,0]-target_Time)
    diff_Mw=np.abs(MovingAcc_Time_Mw[:,1]-target_Mw)
    idx=np.where( (diff_T==diff_T.min()) & (diff_Mw==diff_Mw.min()) )[0][0] #just return a constant
    return idx,sav_misfit_distribute[idx]

def Tr(Mw):
    #Half duration from Zacharie Duputel (EPSL 2013)
    M0=10**((Mw+10.7)*(3/2))  #[unit in dyne-cm]. From Mw=(2/3)*np.log10(M0)-10.7
    return(1.2*10**-8*M0**(1/3))

###Accuracy function(Time,Mw)
sns.set()
X,Y,Z=get_XYZ(MovingAcc_Time_Mw[:,0],MovingAcc_Time_Mw[:,1],MovingAcc_Time_Mw[:,2]*100.0,stype=1,thres=0.005  )
#plt.pcolor(X,Y,Z,vmin=0.0,vmax=1,cmap='jet')
fig=plt.figure(figsize=(10,8))
plt.subplot(1,2,1)
plt.pcolor(X,Y,Z,vmin=0,vmax=100,cmap='magma')
plt.plot(Tr(np.arange(7.0,9.6,0.1))*2.0,np.arange(7.0,9.6,0.1),'k--')
plt.text(200,9.0,r'duration=2$\tau_{half}$',fontsize=16)
#clb.set_label('Accuracy(%)',fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Time (sec)',fontsize=16)
plt.ylabel('Predicted Mw',fontsize=16)
#cbaxes = fig.add_axes([0.65, 0.21, 0.2, 0.028])
cbaxes = fig.add_axes([0.13, 0.16, 0.2, 0.022])
#clb=plt.colorbar(cmap,cax=cbaxes, orientation='horizontal')
clb=plt.colorbar(cax=cbaxes,orientation='horizontal')
plt.xticks(fontsize=13)
ax1=plt.gca()
ax1.tick_params(pad=1.5)
clb.set_label('Accuracy(%)', rotation=0,labelpad=-1,fontsize=14)
#plt.show()

###Misfit function(Time,Mw)
plt.subplot(1,2,2)
X,Y,Z=get_XYZ(MovingAcc_Time_Mw[:,0],MovingAcc_Time_Mw[:,1],MovingAcc_Time_Mw[:,3],stype=1,thres=0.005  )
#plt.pcolor(X,Y,Z,vmin=0.0,vmax=0.3,cmap='afmhot_r')
plt.pcolor(X,Y,Z,vmin=0.0,vmax=0.3,cmap='jet')
plt.plot(Tr(np.arange(7.0,9.6,0.1))*2.0,np.arange(7.0,9.6,0.1),'k--')
#plt.text(200,9.0,r'duration=2$\tau_r$',fontsize=16)
#clb=plt.colorbar()
#clb.set_label('Avg misfit',fontsize=14)
plt.xlabel('Time (sec)',fontsize=16)
plt.xticks(fontsize=14)
plt.yticks([])
#plt.ylabel('Predicted Mw',fontsize=14)
cbaxes = fig.add_axes([0.6, 0.16, 0.2, 0.022])
clb=plt.colorbar(cax=cbaxes,orientation='horizontal',ticks=[0, 0.15, 0.3])
#clb.ax.set_yticks([0,0.15,0.3])
plt.xticks(fontsize=13)
ax1=plt.gca()
ax1.tick_params(pad=1.5)
#plt.xtikcs([0,0.15,0.3])
clb.set_label('Avg. misfit', rotation=0,labelpad=-1,fontsize=14)
#clb=plt.colorbar(cmap,cax=cbaxes, orientation='horizontal')
#clb=plt.colorbar(cax=cbaxes,orientation='horizontal')
plt.subplots_adjust(left=0.08,top=0.88,right=0.97,bottom=0.1,wspace=0.07)
plt.savefig('distribution_Testing.png')
plt.show()


#
#sav_Mw_TF=np.array(sav_Mw_TF)
#sav_Time=np.array(sav_Time)

def groupMw(y_real,y_pred,G,minMw=7.2):
    sav_avgMw=[]
    sav_std=[]
    for i in range(len(G)-1):
        idx=np.where((y_real>=G[i]) & (y_real<G[i+1]) & (y_pred>=7.2))[0]
        sav_avgMw.append(np.mean(y_pred[idx]))
        sav_std.append(np.std(y_pred[idx]))
    sav_avgMw=np.array(sav_avgMw)
    sav_std=np.array(sav_std)
    return sav_avgMw,sav_std


def make_epoch_fitting(real,pred,err_range,Zoomed=True,save_dir='Tmp',savegif=False):
    import seaborn as sns
    import matplotlib
    sns.set()
    if not(os.path.exists(save_dir)):
        os.makedirs(save_dir)
    else:
        print('Directory %s already exist, overwritting existing files'%(save_dir))
    y1=real
    predictions=pred
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    #err_range=0.3
    for epoch in range(len(y1[0][:,0])):
        fig=plt.figure(1,figsize=(5.8,4.75))
        ax1=fig.add_axes((0.16,0.18,0.80,0.80))
        plt.grid(True)
        #if current mw is the final mw, plot as different color
#        idx_finalM=np.where( np.abs(y1[:,epoch,0] - y1[:,-1,0])<0.05 )[0]
#        idx_NfinalM=np.where( np.abs(y1[:,epoch,0] - y1[:,-1,0])>=0.05 )[0]
#        if len(idx_finalM)>0:
#            plt.plot(y1[idx_finalM,epoch,0],predictions[idx_finalM,epoch,0],'r.')
#        if len(idx_NfinalM)>0:
#            plt.plot(y1[idx_NfinalM,epoch,0],predictions[idx_NfinalM,epoch,0],'bo',markerfacecolor='None',mew=0.5,ms=3)
#        plt.plot(y1[:,epoch,0],predictions[:,epoch,0],'o',markerfacecolor=[0.6,0.6,0.6],markeredgecolor='k',mew=0.25,ms=3.5) #X is target Mw
        plt.plot(y1[:,-1,0],predictions[:,epoch,0],'o',markerfacecolor=[0.65,0.65,0.65],markeredgecolor='k',mew=0.25,ms=3.5) # X is final Mw
        #plt.scatter(filt_y1[:,-1,0],filt_misfit[:,int(nsub*t_step),0],c=sav_lon,s=10,cmap=plt.cm.bwr)
        plt.plot([y1.min(),y1.max()],[y1.min(),y1.max()],'k--',linewidth=2)
        plt.plot([y1.min(),y1.max()],[y1.min()-err_range,y1.max()-err_range],'k--',linewidth=0.5)
        plt.plot([y1.min(),y1.max()],[y1.min()+err_range,y1.max()+err_range],'k--',linewidth=0.5)
        plt.fill_between([y1.min(),y1.max()],[y1.min()-err_range,y1.max()-err_range],[y1.min()+err_range,y1.max()+err_range],facecolor='k',alpha=0.25)
        #Plot Grouped/Avg Mw
        #G=np.arange(7.5,9.6,0.2)
        #avgMw,Mwstd=groupMw(y1[:,epoch,0],predictions[:,epoch,0],G,7.0)
        #plt.errorbar((G[1:]+G[:-1])/2,avgMw,Mwstd, marker='s', mfc=[0.9,0.9,0],mec=[0,0.9,0], ms=12, mew=2,elinewidth=3,capsize=15,color=[0,0.9,0])
        #        plt.errorbar((G[1:]+G[:-1])/2,avgMw,Mwstd,color=[0,0.5,0])
        #-----plot real data predictions (run the later part to get the predictions_real first!)----------
        EQMw=[8.3,8.1,8.8,7.6,7.7]
        EQs=['Illapel2015','Iquique2014','Maule2010','Melinka2016','Iquique_aftershock2014']
        smbs=['^','s','*','p','D']
#        smbs_size=[20,17,23,20,17]
        smbs_size=[16,13,19,16,13]
        colrs=[[0,0,1],[0,1,0],[1,0,0],[1,1,0],[1,0,1]]
        hans=[]
        for neq in range(len(predictions_real)):
            han=plt.plot(EQMw[neq],predictions_real[neq,epoch,0],smbs[neq],mew=0.25,ms=smbs_size[neq],markerfacecolor=colrs[neq],markeredgecolor='k')
            hans.append(han)
        #---------END real data predictions---------------------------------------------------------------
        if Zoomed:
            plt.xlim([7.4,9.52])
            plt.ylim([7.4,9.52])
        Xlim=plt.xlim()
        Ylim=plt.ylim()
        Xpos=(Xlim[1]-Xlim[0])*0.06+Xlim[0]
        Ypos=(Ylim[1]-Ylim[0])*0.9+Ylim[0]
        plt.text(Xpos,Ypos,'%03d sec'%(epoch*5+5),bbox=props,fontsize=17)
        plt.ylabel('Predicted Mw',fontsize=19)
#        plt.xlabel('Target Mw',fontsize=19)
        plt.xlabel('Final Mw',fontsize=19)
        plt.xticks(fontsize=17, rotation=30)
        #ax1.set_aspect('equal')
        ax1.tick_params(pad=0.3)
        plt.yticks(fontsize=17, rotation=0)
#        plt.legend((hans[0][0],hans[1][0],hans[2][0],hans[3][0],hans[4][0]),('Illapel','Iquique','Maule','Melinka','Iquique aft.'),loc=0,fontsize=25)
        if epoch==11:
            #60 sec
            plt.legend((hans[3][0],hans[4][0],hans[1][0],hans[0][0],hans[2][0]),('Melinka','Iquique aft.','Iquique','Illapel','Maule'),loc=4,fontsize=14,frameon=True)
#            break
        if save_dir:
            plt.savefig(save_dir+'/%03d.png'%(epoch*5+5),dpi=300)
        #plt.close()
        plt.clf()
    if savegif:
        images=[]
        filenames=glob.glob(save_dir+'/*.png')
        filenames.sort()
        for filename in filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave(save_dir+'/movie.gif', images)

#save all the predictions snapshot
make_epoch_fitting(y1,predictions,0.3,Zoomed=True,save_dir='%s_figs'%(run_name),savegif=False)



#------------Analysis the misfit to see if there're overfitting/underfitting------------#
#misft=y1[:,:,0]-predictions[:,:,0]
#misft=predictions[:,:,0]-y1[:,:,0] #current misfit
if Misfit_current:
    misft=predictions[:,:,0]-y1[:,:]  # label Mw misfit
else:
    misft=predictions[:,:,0]-y1[:,-1]  # final Mw misfit

#Earthquake's ID that used in different category
#ID_train=np.load('../EQID_train_73.npy')
#ID_valid=np.load('../EQID_valid_73.npy')
#ID_test=np.load('../EQID_test_73.npy')
#ALLEQ

#plot all of the misfit timeseries, color coded by Mw
#c_map=plt.cm.jet(y1[s_id,-1,0])
#minD=np.min(y1[:,-1,0])
#maxD=np.max(y1[:,-1,0])
minD=7.5
maxD=9.5
c_map=plt.cm.jet(plt.Normalize(minD,maxD)(y1[:,-1,0]))
fig, ax = plt.subplots()
idx_large=np.where(misft[:,30]<-0.8)[0] #find the large underestimation (manually!) and plot it with a different linewidth
print('Real ID for idx_large is',real_EQID[idx_large])
#idx_small=np.where(misft[:,24]>0.5)[0] #find the large overestimation and plot it by different line
sns.set()
for ii,i_line in enumerate(misft[:,:]):
    plt.plot(np.arange(102)*5+5,i_line,color=c_map[ii],linewidth=0.1,alpha=0.8)
    if (ii in idx_large):
        plt.plot(np.arange(102)*5+5,i_line,color=c_map[ii],linewidth=2)
        print('ii=',ii)

plt.xlabel('Time (s)',fontsize=16)
plt.ylabel('Predicted-Final (M$_w$)',fontsize=16)
plt.xlim([0,515])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
norm = matplotlib.colors.Normalize(vmin=minD, vmax=maxD)
cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap='jet')
cmap.set_array([])
cbaxes = fig.add_axes([0.6, 0.22, 0.24, 0.028])
clb=plt.colorbar(cmap,cax=cbaxes,orientation='horizontal')
clb.set_label('M$_w$', rotation=0,labelpad=0,fontsize=14)
clb.set_ticks(np.arange(7.5,9.6,0.5))
ax1=plt.gca()
ax1.tick_params(pad=0)
plt.xticks(fontsize=12)
plt.savefig('Misfit_testing_finalMw.pdf')
plt.show()


#locating the large misfit and find any example


#------------Analysis the misfit to see if there're overfitting/underfitting  END------------#

#----------------plot STF and grouped STF for all the 27200 scenarios----------------------#
#STF=np.load('../data/STF/STF_Chile_27200_new.npy') #unit are dyne-cm
#STF_T=np.arange(0,512,0.1) #define in the function that generate STFs

STF=np.load('../data/STF/STF_Chile_27200_new_long.npy') #unit are dyne-cm
STF_T=np.arange(0,2048,0.5) #define in the function that generate STFs
sns.set()
def M02Mw(M0):
    Mw=(2.0/3)*np.log10(M0*1e7)-10.7 #Mudpy input is N-M, convert to dyne-cm by 1e7,
    return(Mw)

def Mw2M0(Mw):
    M0=10**((Mw+10.7)*(3/2))*1e-7 #make the unit to be N-M
    return(M0)

def group_STF(STF_T,STFs,G):
    G_idx={}
    for i,stf in enumerate(STFs):
        sumM0=cumtrapz(stf,STF_T)
        Mw=M02Mw(sumM0[-1]) #just use the last one (assume last one is the final Mw)
        #assign the right group for Mw
        idx=np.where( np.abs(Mw-G)==np.min(np.abs(Mw-G)))[0][0]
        try:
            G_idx[idx].append(i)
        except:
            G_idx[idx]=[idx]
    avg_STF=[]
    for i_key in range(len(G)):
        avg_STF.append(np.mean(STFs[G_idx[i_key]],axis=0))
    return np.array(avg_STF)

minD=7.5
maxD=9.5
c_map=plt.cm.jet(plt.Normalize(minD,maxD)(ALLEQ[:,1]))
fig, ax = plt.subplots(figsize=(5.2,4.8))
for i,st in enumerate(STF[::-1,:]):
    if i%100==0:
        print('Now at %d out of %d'%(i,len(STF)))
    plt.plot(STF_T,st,color=c_map[::-1,:][i],linewidth=0.08,alpha=0.8)

#Plot grouped STF
gp_STF=group_STF(STF_T,STFs=STF,G=np.arange(7.5,9.7,0.3))
c_map_gp=plt.cm.jet(plt.Normalize(7.5,9.5)(np.arange(7.5,9.7,0.3)))
for i,stf in enumerate(gp_STF):
    plt.plot(STF_T,stf,color=c_map_gp[i],linewidth=3)

plt.xlabel('Time (sec)',fontsize=16)
plt.ylabel('$\dot \mathrm{M}$ (N-m/sec)',fontsize=16,labelpad=-5) #moment rate
plt.xlim([-5,515])
plt.grid(True)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
ax1=plt.gca()
ax1.tick_params(axis='y',pad=0)
#plot box region
y_max=6e20
plt.plot([0,50],[0,0],'k--')
plt.plot([0,50],[y_max,y_max],'k--')
plt.plot([0,0],[0,y_max],'k--')
plt.plot([50,50],[0,y_max],'k--')

#add colorbar
norm = matplotlib.colors.Normalize(vmin=minD, vmax=maxD)
cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap='jet')
cmap.set_array([])
cbaxes = fig.add_axes([0.16, 0.8, 0.23, 0.028])
clb=plt.colorbar(cmap,cax=cbaxes,orientation='horizontal')
clb.set_label('M$_w$', rotation=0,labelpad=0,fontsize=14)
clb.set_ticks(np.arange(7.5,9.6,0.5))
ax1=plt.gca()
ax1.tick_params(pad=0.5,rotation=30)
plt.xticks(fontsize=14)

#ax1=fig.add_axes((0.57,0.35,0.3,0.3),fc='k')
ax1=fig.add_axes((0.47,0.42,0.34,0.4),fc='k',alpha=0.5)
for i,stf in enumerate(gp_STF):
    plt.plot(STF_T,stf,color=c_map_gp[i],linewidth=3)

ax1.yaxis.tick_right()
ax1.yaxis.label_position='right'
plt.xlim(0,50)
plt.xlabel('Time (sec)',fontsize=14,labelpad=0)
#plt.tick_params(axis='y', which='right', labelleft=False, labelright=True)
plt.ylabel('$\dot \mathrm{M}$ (N-m/sec)',fontsize=15,labelpad=25) #moment rate
ax1=plt.gca()
ax1.tick_params(pad=0.5)
plt.xticks([0,20,40],fontsize=14)
#plt.yticks([0e21,0.5e21,1e21],fontsize=14)
plt.yticks(fontsize=14)
plt.ylim([0,y_max])
plt.savefig('STF_gpSTF.png',dpi=300)
#plt.savefig('STF_gpSTF.pdf',dpi=300)
plt.show()
#-----------plot STF and grouped STF for all the 27200 scenarios END----------------------#


#----------Example plot for STF's duration, half duration and time-to-corrected Mw--------------
def find_tau(STF_T,stf,realMw):
    #tau_d is the time to final magnitude
    #tau_r is half duration from Duputal 2013
    #tau_p is the time to peak moment rate
    #realMw is the magnitude from log file, but here just assume the 515 sec is the final Mw
    sumM0=cumtrapz(stf,STF_T)
#    idx=np.where(np.abs(sumM0-Mw2M0(realMw))<=0.05*Mw2M0(realMw))[0][0] #the first one
#    idx=np.where(np.abs(sumM0-sumM0[-1])<=0.05*sumM0[-1])[0][0] #the first one
    idx=np.where(np.abs(sumM0-sumM0[-1])<=1e-9)[0][0] #duration (the first one)
    idx2=np.where(np.abs(sumM0-0.5*sumM0[-1]) == np.min(np.abs(sumM0-0.5*sumM0[-1])) )[0][0]
    tau_d=STF_T[idx]
    tau_cent=STF_T[idx2]
    tau_r=Tr(realMw)
    idx_p=np.where(stf==stf.max())[0][0]
    tau_p=STF_T[idx_p]
    return tau_r,tau_d,tau_p,tau_cent

def time2correct(misft,threshold=0.3):
    #find time to correct Mw by the given Mw misfit timeseries and threshold
    #not only find the first one, but also check if the prediction is stable
    idx_st=np.where(np.abs(misft)<=threshold)[0] #starting index
    if (idx_st is []) or (np.abs(misft[-1])>threshold):
        #cannot converge
        return False
    #determine if this first index is valid
    for id in idx_st:
        if (idx_st[-1]-id+1)==len(range(id,idx_st[-1]+1)):
            return id


#get the final Mw from STF at 515 sec
STF_test=[STF[int(id)] for id in real_EQID] #only testing dataset
#STF_test=[STF[int(id)] for id in range(27200)] #all 27200 dataset

STF_test=np.array(STF_test) #STF for testing dataset
Mw_testing_final=M02Mw(cumtrapz(STF_test[:],STF_T))[:,-1]

n_c=6
n_r=6
props = dict(boxstyle='round', facecolor='white', alpha=1) #box for plt.text
plt.figure()
#plt.figure(figsize=(12,10))
plt.subplot(n_c,n_r,1)
#sns.set_style("dark")
#For testing dataset
for n_msft,msft in enumerate(misft):
    plt.subplot(n_c,n_r,n_msft+1)
    ax=plt.gca()
    ax.set_facecolor([0.93,0.93,0.93])
    expid=int(real_EQID[n_msft]) #example id (real id)
    tau_r,tau_d,tau_p,tau_cent=find_tau(STF_T,STF[expid],ALLEQ[expid,1])
    plt.plot(STF_T,STF[expid],color='k',linewidth=0.1)
    plt.fill_between(STF_T,STF[expid],np.zeros(len(STF_T)),facecolor=[0.6,0.6,0.6])
    plt.plot([tau_d,tau_d],[0,STF[expid].max()],'r--',linewidth=1)
    plt.plot([tau_p,tau_p],[0,STF[expid].max()],'g--',linewidth=1)
    plt.plot([tau_cent,tau_cent],[0,STF[expid].max()],'k--',linewidth=1)
    idx=time2correct(msft,threshold=0.3) #
    if idx:
        tau_det=(np.arange(102)*5+5)[idx]   #time to corrected Mw
        plt.plot([tau_det,tau_det],[0,STF[expid].max()],'--',color=[0,0,1],linewidth=1)
    plt.ylim([0,STF[expid].max()])
    plt.xlim([0,np.min([515,tau_d*1.2])])
    #plt.xlabel('Time (s)',fontsize=16)
    #plt.ylabel('$\dot \mathrm{M}$ (N-M)',fontsize=16,labelpad=0) #moment rate
    plt.xticks([],[])
    plt.yticks([],[])
    plt.text(np.min([515,tau_d*1.2])*0.7,STF[expid].max()*0.75,'%3.1f'%(y1[n_msft,-1,0]),fontsize=8.5,bbox=props)
    if n_msft==int(n_c*n_r-1):
        break


#For all 27200 where misft haven't calculated (preliminary test here)
'''
Rand_eqid=np.arange(27200)
np.random.shuffle(Rand_eqid)
for n_id,expid in enumerate(Rand_eqid):
    plt.subplot(n_c,n_r,n_id+1)
    ax=plt.gca()
    ax.set_facecolor([0.93,0.93,0.93])
    tau_r,tau_d,tau_p,tau_cent=find_tau(STF_T,STF[expid],ALLEQ[expid,1])
    plt.plot(STF_T,STF[expid],color='k',linewidth=0.1)
    plt.fill_between(STF_T,STF[expid],np.zeros(len(STF_T)),facecolor=[0.6,0.6,0.6])
    plt.plot([tau_d,tau_d],[0,STF[expid].max()],'r--',linewidth=1)
    plt.plot([tau_p,tau_p],[0,STF[expid].max()],'g--',linewidth=1)
    plt.plot([tau_cent,tau_cent],[0,STF[expid].max()],'b--',linewidth=1)
    plt.ylim([0,STF[expid].max()])
    plt.xlim([0,np.min([515,tau_d*1.2])])
    #plt.xlabel('Time (s)',fontsize=16)
    #plt.ylabel('$\dot \mathrm{M}$ (N-M)',fontsize=16,labelpad=0) #moment rate
    plt.xticks([],[])
    plt.yticks([],[])
    tmp_Mw=M02Mw( cumtrapz(STF[expid],STF_T)[-1] )
    plt.text(np.min([515,tau_d*1.2])*0.75,STF[expid].max()*0.75,'%3.1f'%(tmp_Mw),fontsize=12,bbox=props)
    if n_id==int(n_c*n_r-1):
        break
'''


plt.subplots_adjust(left=0.08,top=0.97,right=0.97,bottom=0.08,wspace=0.08,hspace=0.06)
ax1=plt.gcf()
ax1=ax1.add_axes((0.08,0.08,0.89,0.89),fc='none')
plt.xticks([],[])
plt.yticks([],[])
plt.xlabel('Time',fontsize=16)
plt.ylabel('Source time function',fontsize=16)
plt.savefig('Example_td_tr_tc_.png',dpi=300)
plt.show()
#----------Example plot for STF's duration, half duration and time-to-corrected Mw  END--------------



#----------Single example plot for STF's duration, half duration and time-to-corrected Mw-------------
n_msft=3
#n_msft=10 #tau_c earlier
#n_msft=2598 #tau_c later
#n_msft=23
n_msft=29 #close to symmetric
n_msft=1734 #cannot converge #467,  914,  976,  978, 1010, 1090, 1471, 1734, 1739, 1766, 1801 np.where(np.abs(misft[:,-1])>0.3)
n_msft=6617 #np.where(real_EQID==real_EQID[1734])
n_msft=6 #np.where(real_EQID==real_EQID[1734])
n_msft=3533 #[1203, 2880, 3533, 4388] large misfit for 3533
msft=misft[n_msft]
plt.figure()
ax=plt.gca()
ax.set_facecolor([0.93,0.93,0.93])
expid=int(real_EQID[n_msft]) #example id (real id)
tau_r,tau_d,tau_p,tau_cent=find_tau(STF_T,STF[expid],ALLEQ[expid,1])
plt.plot(STF_T,STF[expid],color='k',linewidth=0.1)
plt.fill_between(STF_T,STF[expid],np.zeros(len(STF_T)),facecolor=[0.6,0.6,0.6])
plt.plot([tau_d,tau_d],[0,STF[expid].max()],'r--',linewidth=2)
plt.plot([tau_cent,tau_cent],[0,STF[expid].max()],'k--',linewidth=2)
#plt.plot([tau_p,tau_p],[0,STF[expid].max()],'y--',linewidth=2)
idx=time2correct(msft,threshold=0.3) #
if idx:
    tau_det=(np.arange(102)*5+5)[idx]   #time to corrected Mw
    plt.plot([tau_det,tau_det],[0,STF[expid].max()],'--',color=[0,0,1],linewidth=2)
    plt.text(tau_det+5,STF[expid].max()*0.05,r'$\tau_c$',fontsize=18)

plt.ylim([0,STF[expid].max()])
plt.xlim([0,np.min([515,tau_d*1.2])])
#plt.xlabel('Time (s)',fontsize=16)
#plt.ylabel('$\dot \mathrm{M}$ (N-M)',fontsize=16,labelpad=0) #moment rate
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax1=plt.gca()
ax1.tick_params(pad=2)
plt.text(np.min([515,tau_d*1.2])*0.85,STF[expid].max()*0.9,'M$_w$%3.1f'%(y1[n_msft,-1,0]),fontsize=16)
plt.text(tau_d+5,STF[expid].max()*0.05,r'$\tau_d$',fontsize=18)
plt.text(tau_r+5,STF[expid].max()*0.05,r'$\tau_r$',fontsize=18)
plt.xlabel('Time (s)',fontsize=16,labelpad=1)
plt.ylabel('Moment rate (N-M/sec)',fontsize=16)
#Add secondary figure shows the Mw
ax1=plt.gca()
ax2 = ax1.twinx()
ax2.set_xlim(ax1.get_xlim())
plt.plot(np.arange(102)*5+5,predictions[n_msft],color=[0,0.5,0])
plt.plot([0,510],[y1[n_msft,-1,0],y1[n_msft,-1,0]],color='m')
plt.yticks(fontsize=14)
plt.ylim([0,y1[n_msft,-1,0]*1.15])
plt.ylabel('M$_w$',fontsize=18)
plt.grid(False)
plt.show()
plt.savefig('Example_td_tr_tc_single_delay2.png')
plt.show()

#quck_plot_data(b_X1[1203],real_EQID[1734],ALLEQ,staloc_sort)
#quck_plot_data(b_X1[6617],real_EQID[6617],ALLEQ,staloc_sort)

#output .txt data for GMT plot
b_X1=back_scale_X(X1,half=True)
#Load station order file
station_order_file='../Data/ALL_staname_order.txt' #this is the order in training(PGD_Chile_3400.npy)
STA=np.genfromtxt(station_order_file,'S6')
STA=np.array([sta.decode() for sta in STA])
STA_info=np.genfromtxt('../data/Chile_GNSS.gflist',usecols=[0],skip_header=1,dtype='S6')
STA_idx={sta.decode():nsta for nsta,sta in enumerate(STA_info)}
staloc=np.genfromtxt('../data/Chile_GNSS.gflist') #station file (not include name)
#Final result! this is the same order in training
staloc_sort=staloc[np.array([STA_idx[i_STA] for i_STA in STA])][:,1:3]

expid1=3533
expid2=2880
use_idx1=np.where(b_X1[expid1,-1,121:]!=0)[0]
use_idx2=np.where(b_X1[expid2,-1,121:]!=0)[0]
np.savetxt('/Users/timlin/Documents/Project/GMTplot/Chile/Example_rupt_PGD/example_station1.txt',staloc_sort[use_idx1],fmt='%f',delimiter=' ')
np.savetxt('/Users/timlin/Documents/Project/GMTplot/Chile/Example_rupt_PGD/example_station2.txt',staloc_sort[use_idx2],fmt='%f',delimiter=' ')
np.savetxt('/Users/timlin/Documents/Project/GMTplot/Chile/Example_rupt_PGD/example_STF.txt',np.hstack([STF_T.reshape(-1,1),STF[int(real_EQID[expid1])].reshape(-1,1)]),fmt='%f',delimiter=' ')
exp_pred=np.hstack([(np.arange(102)*5+5).reshape(-1,1),predictions[expid1].reshape(-1,1),\
                    predictions[expid2].reshape(-1,1),y1[expid1].reshape(-1,1) \
                    ] ) #example prediction
np.savetxt('/Users/timlin/Documents/Project/GMTplot/Chile/Example_rupt_PGD/example_prediction.txt',exp_pred,fmt='%f',delimiter=' ')
np.savetxt('/Users/timlin/Documents/Project/GMTplot/Chile/Example_rupt_PGD/example_PGD1.txt',b_X1[expid1],fmt='%f',delimiter=' ')
np.savetxt('/Users/timlin/Documents/Project/GMTplot/Chile/Example_rupt_PGD/example_PGD2.txt',b_X1[expid2],fmt='%f',delimiter=' ')

#----------Single example plot for STF's duration, half duration and time-to-corrected Mw  END-------------





#---------Merge 4 plots into one plot, example plot for STF's duration, half duration and time-to-corrected Mw-----------
n_msft=3
#n_msft=10 #tau_c earlier
#n_msft=2598 #tau_c later
#n_msft=23
n_msft=29 #close to symmetric
n_msft=1090 #cannot converge #467,  914,  976,  978, 1010, 1090, 1471, 1734, 1739, 1766, 1801 np.where(np.abs(misft[:,-1])>0.3)
#n_msft=1190 #np.where(real_EQID==real_EQID[1090])
sns.set()
#plt.figure(figsize=(8,6))
plt.figure(figsize=(9,7))
#n_msfts=[10,2598,29,1734] #6617 the same source as #1734 but converge  29,53,56
#n_msfts=[10,29,53,56] #6617 the same source as #1734 but converge
n_msfts=[202,221,108,8172] #6617 the same source as #1734 but converge   202late,93,100,221,225early?,65,69,94,138,174two asperities,78,108symmetric
for i,n_msft in enumerate(n_msfts):
    msft=misft[n_msft]
    plt.subplot(2,2,i+1)
    plt.grid(False)
    expid=int(real_EQID[n_msft]) #example id (real id)
    tau_r,tau_d,tau_p,tau_cent=find_tau(STF_T,STF[expid],ALLEQ[expid,1])
    plt.plot(STF_T,STF[expid],color='k',linewidth=0.1)
    plt.fill_between(STF_T,STF[expid],np.zeros(len(STF_T)),facecolor=[0.6,0.6,0.6])
    plt.plot([tau_d,tau_d],[0,STF[expid].max()],'r--',linewidth=2)
    plt.plot([tau_cent,tau_cent],[0,STF[expid].max()],'y--',linewidth=2)
#    plt.plot([tau_p,tau_p],[0,STF[expid].max()],'y--',linewidth=2)
    idx=time2correct(msft,threshold=0.3) #
    if idx:
        tau_det=(np.arange(102)*5+5)[idx]   #time to corrected Mw
        plt.plot([tau_det,tau_det],[0,STF[expid].max()],'--',color=[0,0,1],linewidth=2)
        #plt.text(tau_det+5,STF[expid].max()*0.05,r'$\tau_c$',fontsize=18)
    plt.ylim([0,STF[expid].max()])
    plt.xlim([0,np.min([515,tau_d*1.2])])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax1=plt.gca()
    ax1.tick_params(pad=2)
    ax1.tick_params(axis='x',pad=5)
    if i==0:
        plt.text(tau_d,STF[expid].max()*0.05,r'$\tau_{dur}$',fontsize=18)
        plt.text(tau_cent,STF[expid].max()*0.05,r'$\tau_{cent}$',fontsize=18)
#        plt.text(tau_p-20,STF[expid].max()*0.05,r'$\tau_p$',fontsize=18)
        if idx:
            plt.text(tau_det-4,STF[expid].max()*0.05,r'$\tau_c$',fontsize=18)
    if i in [2,3]:
        plt.xlabel('Time (s)',fontsize=16,labelpad=1)
    plt.ylabel('$\dot \mathrm{M}$ (N-M/s)',fontsize=16,labelpad=1)
    #Add secondary figure shows the Mw
    ax1=plt.gca()
    ax2 = ax1.twinx()
    ax2.set_xlim(ax1.get_xlim())
    plt.plot(np.arange(102)*5+5,predictions[n_msft],color=[0,0.5,0])
    plt.plot([0,510],[y1[n_msft,-1,0],y1[n_msft,-1,0]],color='m')
    plt.yticks(fontsize=14)
    ax2.tick_params(pad=0)
    plt.ylim([0,y1[n_msft,-1,0]*1.07])
    plt.ylabel('M$_w$',fontsize=18,labelpad=1)
    plt.grid(False)
    textprops = dict(boxstyle='square', facecolor='white', alpha=0.9,pad=0.1)
    plt.text(np.min([515,tau_d*1.1])*0.85,y1[n_msft,-1,0],'M$_w$%3.1f'%(y1[n_msft,-1,0]),fontsize=14,bbox=textprops)

plt.subplots_adjust(left=0.09,top=0.97,right=0.94,bottom=0.08,wspace=0.32,hspace=0.22)
#plt.show()

plt.savefig('Example_4figs_td_tr_tc_100percent.png',dpi=300)
plt.show()
#---------Merge 4 plots into one plot, example plot for STF's duration, half duration and time-to-corrected Mw END-----------



#------------Calculate tau_c, tau_d, tau_r WRS to Mw, scatter plot and boxplot-----------------------
sav_mw=[]
sav_tau_r=[]
sav_tau_d=[]
sav_tau_p=[]
sav_tau_c=[]
sav_tau_cent=[]
sav_tau_M3real=[] #time when real Mw reaches the lower boundary (Mw-0.3)
sav_ID=[]
Threshold=0.3
#Threshold=0.2

#For testing dataset
for n_msft,msft in enumerate(misft):
    expid=int(real_EQID[n_msft]) #example id (real id)
    tau_r,tau_d,tau_p,tau_cent=find_tau(STF_T,STF[expid],ALLEQ[expid,1])
    idx=time2correct(msft,threshold=Threshold) #
    if idx:
        tau_det=(np.arange(102)*5+5)[idx]   #time to corrected Mw
        sav_mw.append(y1[n_msft,-1,0])
        sav_tau_r.append(tau_r)
        sav_tau_d.append(tau_d)
        sav_tau_p.append(tau_p)
        sav_tau_c.append(tau_det)
        sav_tau_cent.append(tau_cent)
        #find the time when Mw reaches to the lowest boundary of real Mw
        tmpidx=np.where(y1[n_msft,:,0] >= (y1[n_msft,-1,0]-Threshold) )[0][0]
        sav_tau_M3real.append((np.arange(102)*5+5)[tmpidx])
        sav_ID.append(n_msft)



#For all 27200 dataset
'''
Rand_eqid=np.arange(27200)
np.random.shuffle(Rand_eqid)
for n_id,r_eqid in enumerate(Rand_eqid):
    tmp_Mw=M02Mw( cumtrapz(STF[r_eqid],STF_T)[-1] )
    tau_r,tau_d,tau_p,tau_cent=find_tau(STF_T,STF[r_eqid],tmp_Mw)
    idx=True #
    if idx:
        tau_det=(np.arange(102)*5+5)[idx]   #time to corrected Mw
        sav_mw.append(tmp_Mw)
        sav_tau_r.append(tau_r)
        sav_tau_d.append(tau_d)
        sav_tau_p.append(tau_p)
        sav_tau_cent.append(tau_cent)
        #        sav_tau_c.append(tau_det)
        sav_ID.append(r_eqid)
'''



'''
plt.scatter(sav_tau_cent,sav_tau_c,c=sav_mw,s=20,cmap=plt.cm.jet)
plt.plot([0,510],[0,510],'k--')
plt.xlabel('Centroid (s)',fontsize=16)
plt.ylabel('Corrected (s)',fontsize=16)
clb=plt.colorbar()
clb.set_label('Mw')
plt.show()

plt.scatter(sav_tau_p,sav_tau_c,c=sav_mw,s=20,cmap=plt.cm.jet)
plt.plot([0,510],[0,510],'k--')
plt.xlabel('Peak (s)',fontsize=16)
plt.ylabel('Corrected (s)',fontsize=16)
clb=plt.colorbar()
clb.set_label('Mw')
plt.show()
'''

#-------plot CC matrix -----------
'''
#put all the taus in pd format
import pandas as pd
taus=np.array([sav_tau_c,sav_tau_p,sav_tau_cent,sav_tau_d])
df=pd.DataFrame(data=taus.T,columns=["corrected","peak","centroid","duration"])
corr = df.corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))

f, ax = plt.subplots(figsize=(8., 6))
heatmap = sns.heatmap(corr,
                      square = True,
#                      mask=mask,
                      linewidths = .5,
                      cmap = 'coolwarm',
                      cbar_kws = {'shrink': .4,
#                      'ticks' : [-1, -.5, 0, 0.5, 1]},
                      'ticks' : [0.5, 0.6, 0.7, 0.8, 0.9,1]},
                      vmin = 0.5,
                      vmax = 1,
                      annot = True,
                      annot_kws = {'size': 14}
                      )
#add the column names as labels
ax.set_yticklabels(corr.columns, rotation = 0,fontsize=16)
ax.set_xticklabels(corr.columns,fontsize=16)
sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
'''
#-------plot CC matrix END-----------

sns.set()
plt.figure(figsize=(8.3,4.7))
plt.subplot(1,2,1)
plt.plot(sav_mw,sav_tau_d,'rx',mew=0.3,markersize=6)
plt.plot(sav_mw,sav_tau_p,'g.',markersize=4)
#plt.plot(sav_mw,sav_tau_c,'b.',markersize=5)
#plt.plot(np.arange(np.min(sav_mw),np.max(sav_mw),0.1),Tr(np.arange(np.min(sav_mw),np.max(sav_mw),0.1)),'k-')
#lg=plt.legend([r'$\tau_d$',r'$\tau_p$',r'$\tau_c$',r'$\tau_r$'],fontsize=15)
lg=plt.legend([r'$\tau_d$',r'$\tau_p$'],fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim([7.4,9.6])
plt.ylim([-10,510])
ax1=plt.gca()
ax1.tick_params(pad=2)
plt.xlabel('M$_w$',fontsize=18,labelpad=0.5)
plt.ylabel('Time(s)',fontsize=18,labelpad=0.5)
#plt.text(8.0,460,'misfit=%3.1f'%(Threshold),fontsize=16)
plt.grid(True)
#plt.savefig('scatter_tau_misfit%3.1f.png'%(Threshold),dpi=300)
#plt.show()

#Plot boxes
#group tau_d, tau_c by Mw For testing dataset
def group_tau(tau_c,tau_d,tau_p,tau_M3real,Mws,G):
    #tau_c:time to corrected Mw, [1-d array]
    #tau_d:duration, [1-d array]
    #tau_p:time to peak M rate, [1-d array]
    #tau_M3real:time to real Mw-0.3
    #Mws:magnitude corresponded to tau_c,tau_d [1-d array]
    #All these above should be the same shape
    #G: grouped Mw, [1-d array]
    tau_c=np.array(tau_c);tau_d=np.array(tau_d);tau_p=np.array(tau_p);Mws=np.array(Mws);G=np.array(G)
    G_tau_c={}
    G_tau_d={}
    G_tau_p={}
    G_tau_M3real={}
    for i,Mw in enumerate(Mws):
        #assign the right Mw group for tau_c (and tau_d, the same)
        idx=np.where( np.abs(Mw-G)==np.min(np.abs(Mw-G)))[0][0] #the group
        try:
            G_tau_c[idx].append(tau_c[i])
            G_tau_d[idx].append(tau_d[i])
            G_tau_p[idx].append(tau_p[i])
            G_tau_M3real[idx].append(tau_M3real[i])
        except:
            G_tau_c[idx]=[tau_c[i]]
            G_tau_d[idx]=[tau_d[i]]
            G_tau_p[idx]=[tau_p[i]]
            G_tau_M3real[idx]=[tau_M3real[i]]
    return G_tau_c,G_tau_d,G_tau_p,G_tau_M3real


#group tau_d, tau_c by Mw
'''
def group_tau(tau_d,tau_p,Mws,G):
    #tau_c:time to corrected Mw, [1-d array]
    #tau_d:duration, [1-d array]
    #tau_p:time to peak M rate, [1-d array]
    #Mws:magnitude corresponded to tau_c,tau_d [1-d array]
    #All these above should be the same shape
    #G: grouped Mw, [1-d array]
    tau_d=np.array(tau_d);tau_p=np.array(tau_p);Mws=np.array(Mws);G=np.array(G)
    G_tau_d={}
    G_tau_p={}
    for i,Mw in enumerate(Mws):
        #assign the right Mw group for tau_c (and tau_d, the same)
        idx=np.where( np.abs(Mw-G)==np.min(np.abs(Mw-G)))[0][0] #the group
        try:
            G_tau_d[idx].append(tau_d[i])
            G_tau_p[idx].append(tau_p[i])
        except:
            G_tau_d[idx]=[tau_d[i]]
            G_tau_p[idx]=[tau_p[i]]
    return G_tau_d,G_tau_p
'''

#import matplotlib
#matplotlib.rc_file_defaults()

#plt.figure()
Gp_mw=np.arange(7.5,9.7,0.3)
G_tau_c,G_tau_d,G_tau_p,G_tau_M3real=group_tau(sav_tau_c,sav_tau_d,sav_tau_p,sav_tau_M3real,sav_mw,G=Gp_mw)
#G_tau_d,G_tau_p=group_tau(sav_tau_d,sav_tau_p,sav_mw,G=Gp_mw)
box_tau_c=[]
box_tau_d=[]
box_tau_p=[]
box_tau_c_d=[] #tau_c/tau_d
box_tau_c_p=[] #tau_c/tau_p
box_tau_p_d=[] #tau_p/tau_d
box_tau_c_M3real=[]
for ig in range(len(Gp_mw)):
    box_tau_c.append(G_tau_c[ig])
    box_tau_d.append(G_tau_d[ig])
    box_tau_p.append(G_tau_p[ig])
    box_tau_c_d.append(np.array(G_tau_c[ig])/np.array(G_tau_d[ig]))
    box_tau_c_p.append(np.array(G_tau_c[ig])/np.array(G_tau_p[ig]))
    box_tau_p_d.append(np.array(G_tau_p[ig])/np.array(G_tau_d[ig])) #the original
    box_tau_c_M3real.append(np.array(G_tau_c[ig])/np.array(G_tau_M3real[ig]) )



#plt.subplot(1,2,2)
fig=plt.figure(figsize=(6,4.8))
#ax=plt.gca()
#ax.set_position([0.1, 0.11, 0.9, 0.88])
wd=0.07
#wd=0.1
max_whiskers1=[]
max_whiskers2=[]
max_whiskers3=[]
out_dots = dict(markerfacecolor=[0.,0.0,0.0],markeredgecolor=[1.,1.0,1.0],mew=0.1, marker='d',markersize=0)
#Just plot the tau_c, tau_dur and their ratio
#bp=plt.boxplot(box_tau_c_d,positions=Gp_mw-wd*0.5,widths=wd,patch_artist=True,flierprops=out_dots)
bp=plt.boxplot(box_tau_c,positions=Gp_mw-wd*0.5,widths=wd,patch_artist=True,flierprops=out_dots)
whiskerloc=[item.get_ydata() for item in bp['whiskers']] #get the upper value of whiskers for plotting text
for nw in range(int(len(whiskerloc)/2)):
    max_whiskers1.append( np.max(whiskerloc[nw*2+1]) )

for patch in bp['boxes']:
    patch.set_facecolor([1,0,0.])
    patch.set_edgecolor([0,0.,0])
    patch.set_linewidth(0.5)
    patch.set_alpha(0.8)

for whisker in bp['whiskers']:
    whisker.set_color([1,0,0.])
    whisker.set_linewidth(1.5)

for cap in bp['caps']:
    cap.set_color([1,0,0.])
    cap.set_linewidth(1.5)

for medn in bp['medians']:
    medn.set_color([0,0,0])

#bp=plt.boxplot(box_tau_c_p,positions=Gp_mw+wd*0.5,widths=wd,patch_artist=True,flierprops=out_dots)
#bp=plt.boxplot(box_tau_c_M3real,positions=Gp_mw+wd*0.5,widths=wd,patch_artist=True,flierprops=out_dots)
bp=plt.boxplot(box_tau_d,positions=Gp_mw-wd*0.5,widths=wd,patch_artist=True,flierprops=out_dots)    #duration
whiskerloc=[item.get_ydata() for item in bp['whiskers']] #get the upper value of whiskers for plotting text
for nw in range(int(len(whiskerloc)/2)):
    max_whiskers2.append( np.max(whiskerloc[nw*2+1]) )

for patch in bp['boxes']:
    patch.set_facecolor([0.2,0.50588,0.867])
#    patch.set_facecolor([0.,0.7,0.])
#    patch.set_facecolor([1,0,0.])
    patch.set_edgecolor([0,0.,0])
    patch.set_linewidth(0.5)
    patch.set_alpha(0.8)


for whisker in bp['whiskers']:
    whisker.set_color([0.2,0.50588,0.867])
    whisker.set_linewidth(1.5)

for cap in bp['caps']:
    cap.set_color([0.2,0.50588,0.867])
    cap.set_linewidth(1.5)

for medn in bp['medians']:
    medn.set_color([0,0,0])

plt.ylim([0,800])
plt.yticks([0,200,400,600,800],fontsize=14)
plt.xlim([7.3,9.8])
plt.ylabel('Time (sec)',fontsize=16,labelpad=0)
tmpax=plt.gca()
tmpax.tick_params(axis='y',pad=0)
plt.xticks([7.5,7.8,8.1,8.4,8.7,9.0,9.3,9.6],[7.5,7.8,8.1,8.4,8.7,9.0,9.3,9.6],fontsize=15)
plt.xlabel('Mw',fontsize=16)
#plot their ratio
ax1=plt.gca()
sav_ticks=ax1.get_xticks()
sav_ticks_labels=ax1.get_xticklabels()
ax2 = ax1.twinx()
bp=ax2.boxplot(box_tau_c_d,positions=Gp_mw+wd*0.5,widths=wd,patch_artist=True,flierprops=out_dots)
whiskerloc=[item.get_ydata() for item in bp['whiskers']] #get the upper value of whiskers for plotting text
for nw in range(int(len(whiskerloc)/2)):
    max_whiskers3.append( np.max(whiskerloc[nw*2+1]) )

for patch in bp['boxes']:
    patch.set_facecolor([0,0.6,0.])
    patch.set_edgecolor([0,0.,0])
    patch.set_linewidth(0.5)
    patch.set_alpha(0.8)

for whisker in bp['whiskers']:
    whisker.set_color([0.0,0.6,0.])
    whisker.set_linewidth(1.5)

for cap in bp['caps']:
    cap.set_color([0.0,0.6,0.])
    cap.set_linewidth(1.5)

for medn in bp['medians']:
    medn.set_color([0,0,0])

plt.ylim([0,1])
y1_max=ax1.get_ylim()[1]
plt.yticks(ax1.get_yticks()/y1_max,fontsize=15) #assume ax2.get_ylim=1
tmpax=plt.gca()
tmpax.tick_params(axis='y',pad=0)
plt.ylabel('Ratio',fontsize=16,labelpad=0)
plt.xticks(sav_ticks,sav_ticks_labels,fontsize=15)

ax1.set_ylim([0,800*0.9])
ax2.set_ylim([0,1*0.9])
ax2.grid(False)
#ax2.set_xlabel('Mw',fontsize=16)

#Add text for # in each groups
props = dict(boxstyle='round', facecolor='white', alpha=0.5,pad=0.05)
for i in range(len(Gp_mw)):
    if i==0:
        plt.text(Gp_mw[i],max_whiskers3[i]+0.028,'n=%d'%(len(box_tau_c_p[i])),va='bottom',ha='center',bbox=props,fontsize=15)
    else:
        if i==5:
            plt.text(Gp_mw[i],0.58,'%d'%(len(box_tau_c_p[i])),va='bottom',ha='center',bbox=props,fontsize=15 )
        elif i==6:
            plt.text(Gp_mw[i],0.8,'%d'%(len(box_tau_c_p[i])),va='bottom',ha='center',bbox=props,fontsize=15 )
        elif i==7:
            plt.text(Gp_mw[i],0.75,'%d'%(len(box_tau_c_p[i])),va='bottom',ha='center',bbox=props,fontsize=15 )
        else:
            plt.text(Gp_mw[i],max_whiskers3[i]+0.03,'%d'%(len(box_tau_c_p[i])),va='bottom',ha='center',bbox=props,fontsize=15 )


'''
bp=plt.boxplot(box_tau_p_d,positions=Gp_mw+wd,widths=wd,patch_artist=True,flierprops=out_dots)
whiskerloc=[item.get_ydata() for item in bp['whiskers']] #get the upper value of whiskers for plotting text
for nw in range(int(len(whiskerloc)/2)):
    max_whiskers3.append( np.max(whiskerloc[nw*2+1]) )

for patch in bp['boxes']:
    patch.set_facecolor([0.,0.7,0.])
    #    patch.set_facecolor([1,0,0.])
    patch.set_edgecolor([0,0.,0])
    patch.set_linewidth(0.5)
#    patch.set_alpha(0.7)

for medn in bp['medians']:
    medn.set_color([0,0,0])
'''

#max_whiskers=np.vstack([max_whiskers1,max_whiskers2])
#max_whiskers=np.max(max_whiskers,axis=0)
#max_whiskers=np.max(max_whiskers3,axis=0)

#wd=0.07
#bp=plt.boxplot(box_tau_c,positions=Gp_mw-wd*0.5,widths=wd,patch_artist=True,flierprops=out_dots)
#for patch in bp['boxes']:
#    patch.set_facecolor([0.2,0.50588,0.867])
#    patch.set_edgecolor([0,0.,0])
#    patch.set_linewidth(0.5)

#bp=plt.boxplot(box_tau_d,positions=Gp_mw+wd*0.5,widths=wd,patch_artist=True,flierprops=out_dots)
#for patch in bp['boxes']:
#    patch.set_facecolor([1,0,0])
#    patch.set_edgecolor([0,0.,0])
#    patch.set_linewidth(0.5)
#
#for medn in bp['medians']:
#    medn.set_color([0,0,0])

#----make legned manually------
ax1=fig.add_axes((0.145,0.78,0.33,0.08),fc=[1,1,1]) #x0,y0,lengthX,lengthY

out_dots = dict(markerfacecolor=[0.,0.0,0.0],markeredgecolor=[0.,0.0,0.0],mew=0.1, marker='d',markersize=0)
#tmpy=np.random.randn(1000)*0.02
tmpy=np.random.normal(0, 0.02, 10000)
idx=np.where(np.abs(tmpy) < np.std(tmpy))[0]
tmpy=tmpy[idx]
#tmpy=np.random.normal(0,0.1,1000)
#bp=plt.boxplot([tmpy+3.8],positions=[7.45],widths=wd,patch_artist=True,flierprops=out_dots)
#bp=plt.boxplot([tmpy+0.65],positions=[7.45],widths=wd,patch_artist=True,flierprops=out_dots)
bp=plt.boxplot([tmpy+0.675],positions=[7.3],widths=wd,patch_artist=True,flierprops=out_dots)
for patch in bp['boxes']:
    patch.set_facecolor([1,0,0])
    patch.set_edgecolor([0,0.,0])
    patch.set_linewidth(0.5)

for whisker in bp['whiskers']:
    whisker.set_color([1,0,0])
    whisker.set_linewidth(1.)

for cap in bp['caps']:
    cap.set_color([1,0,0])
    cap.set_linewidth(1.)

for medn in bp['medians']:
    medn.set_color([0,0,0])


#bp=plt.boxplot([tmpy+3.1],positions=[7.45],widths=wd,patch_artist=True,flierprops=out_dots)
#bp=plt.boxplot([tmpy+0.7],positions=[7.5],widths=wd,patch_artist=True,flierprops=out_dots)
bp=plt.boxplot([tmpy+0.675],positions=[7.65],widths=wd,patch_artist=True,flierprops=out_dots)
for patch in bp['boxes']:
    patch.set_facecolor([0.2,0.50588,0.867])
    patch.set_edgecolor([0,0.,0])
    patch.set_linewidth(0.5)

for whisker in bp['whiskers']:
    whisker.set_color([0.2,0.50588,0.867])
    whisker.set_linewidth(1.)

for cap in bp['caps']:
    cap.set_color([0.2,0.50588,0.867])
    cap.set_linewidth(1.)

for medn in bp['medians']:
    medn.set_color([0,0,0])


#bp=plt.boxplot([tmpy+0.68],positions=[7.55],widths=wd,patch_artist=True,flierprops=out_dots)
bp=plt.boxplot([tmpy+0.675],positions=[8.0],widths=wd,patch_artist=True,flierprops=out_dots)
for patch in bp['boxes']:
    patch.set_facecolor([0,0.6,0.])
    patch.set_edgecolor([0,0.,0])
    patch.set_linewidth(0.5)

for whisker in bp['whiskers']:
    whisker.set_color([0,0.6,0.])
    whisker.set_linewidth(1.)

for cap in bp['caps']:
    cap.set_color([0,0.6,0.])
    cap.set_linewidth(1.)

for medn in bp['medians']:
    medn.set_color([0,0,0])


#plt.text(7.28,0.69,r'$\tau_{d}$',fontsize=20,rotation=90)
#plt.text(7.28,0.64,r'$\tau_{c}$',fontsize=20,rotation=90)
#plt.text(7.61,0.645,r'$\tau_{c}$/$\tau_{d}$',fontsize=20,rotation=90)
plt.text(7.325,0.66,r'$\tau_{c}$',fontsize=20,rotation=0)
plt.text(7.675,0.66,r'$\tau_{d}$',fontsize=20,rotation=0)
plt.text(8.025,0.66,r'$\tau_{c}$/$\tau_{d}$',fontsize=20,rotation=0)

plt.xlim([7.22,8.49])
plt.ylim([0.675-0.03,0.675+0.03])
plt.xticks([])
plt.yticks([])

plt.show()
plt.savefig('tau_ratio_misfit0.3_100percent_new.png',dpi=450)
#plt.savefig('tau_100percent.png',dpi=300)

#bp=plt.boxplot([tmpy+3.4],positions=[7.45],widths=wd,patch_artist=True,flierprops=out_dots)
#for patch in bp['boxes']:
#    patch.set_facecolor([0.,0.7,0.])
#    patch.set_edgecolor([0,0.,0])
#    patch.set_linewidth(0.5)
#
#for medn in bp['medians']:
#    medn.set_color([0,0,0])

#plt.text(7.5,4.4,r'$\tau_c$/$\tau_d$',fontsize=18)
#plt.text(7.5,3.7,r'$\tau_c$/$\tau_p$',fontsize=18)
#plt.text(7.5,3.0,r'$\tau_p$/$\tau_d$',fontsize=18)
#plt.text(7.5,3.7,r'$\tau_{correct}$/$\tau_{duration}$',fontsize=16)
#plt.text(7.5,3.0,r'$\tau_{correct}$/$\tau_{(real-0.3)}$',fontsize=16)
#plt.text(7.5,3.3,r'$\tau_{peak}$/$\tau_{duration}$',fontsize=16)
#plt.text(7.5,3.0,r'$\tau_{duration}$/$\tau_{peak}$',fontsize=16)
#----make legned manually END------

#Add text for # in each groups
#props = dict(boxstyle='round', facecolor='white', alpha=0.5,pad=0.2)
#for i in range(len(Gp_mw)):
#    if i==0:
#        plt.text(Gp_mw[i],max_whiskers3[i]+0.05,'n=%d'%(len(box_tau_c_p[i])),va='bottom',ha='center',bbox=props)
#    else:
#        plt.text(Gp_mw[i],max_whiskers3[i]+0.05,'%d'%(len(box_tau_c_p[i])),va='bottom',ha='center',bbox=props )


#plt.xticks(Gp_mw,['%3.1f'%(i) for i in Gp_mw ],fontsize=14)
#plt.yticks(fontsize=14)
#ax1=plt.gca()
#ax1.tick_params(pad=0.8)
#plt.xlim([7.3,9.8])
##plt.ylim([-0.05,2.47])
##plt.ylim([-0.05,5.3])
#plt.ylim([-0.05,4.2])
##plt.text(7.45,plt.ylim()[1]*0.87,'misfit=%3.1f'%(Threshold),fontsize=16)
#plt.xlabel('M$_w$',fontsize=18,labelpad=0.5)
#plt.ylabel('Ratio',fontsize=18,labelpad=0.5)
#plt.ylabel(r'$\tau_c$/$\tau_d$',fontsize=18,labelpad=0.5)

#plt.subplots_adjust(left=0.095,top=0.97,right=0.96,bottom=0.092,wspace=0.19,hspace=0.25)
#plt.show()
#plt.savefig('tau_ratio_misfit0.3_100percent.png',dpi=300)
#plt.savefig('tau_100percent.png',dpi=300)

#-----------------Calculate tau_c, tau_d, tau_r WRS to Mw END---------------------------------


###Make scatter plot for all taus#####
plt.figure(figsize=(10,4.3))
plt.subplot(1,3,1)
plt.plot(sav_tau_p,sav_tau_d,'k.')
#regression
#tmp_G=np.matrix(np.array(sav_tau_p).reshape(-1,1))
#GTG=np.multiply(tmp_G.T,tmp_G)
plt.xlim([0,510]);plt.ylim([0,510])
plt.xlabel(r'$\tau_{peak}$',fontsize=16,labelpad=0.5)
plt.ylabel(r'$\tau_{duration}$',fontsize=16,labelpad=0.5)
ax1=plt.gca()
ax1.tick_params(pad=1)
plt.subplot(1,3,2)
plt.plot(sav_tau_c,sav_tau_d,'k.')
plt.xlim([0,510]);plt.ylim([0,510])
plt.xlabel(r'$\tau_{correct}$',fontsize=16,labelpad=0.5)
plt.ylabel(r'$\tau_{duration}$',fontsize=16,labelpad=0.5)
ax1=plt.gca()
ax1.tick_params(pad=0.8)
plt.subplot(1,3,3)
plt.plot(sav_tau_p,sav_tau_c,'k.')
plt.xlim([0,510]);plt.ylim([0,510])
plt.xlabel(r'$\tau_{peak}$',fontsize=16,labelpad=0.5)
plt.ylabel(r'$\tau_{correct}$',fontsize=16,labelpad=0.5)
ax1=plt.gca()
ax1.tick_params(pad=0.8)
plt.subplots_adjust(left=0.095,top=0.97,right=0.96,bottom=0.092,wspace=0.22,hspace=0.25)
plt.show()
####Make scatter plot for all taus END############






close_id=np.where(np.abs(ALLEQ[int(real_EQID[idx_large]),1] - ALLEQ[:,1])<0.01)[0] #find some similar event (i.e. similar Mw)
max_STF=np.max(STF[close_id,:],axis=1)
plt.plot(STF_T,STF[close_id,:].T,linewidth=0.5,alpha=0.5)
plt.plot(STF_T,STF[int(real_EQID[idx_large])],'r')#STF for the id_large
plt.show()

plt.plot(STF_T,STF[int(real_EQID[idx_small])],'b')
plt.show()

#####plot the epi-centroid separation#####
#dist_sepr=obspy.geodetics.locations2degrees(lat1=ALLEQ[:,3],long1=ALLEQ[:,2],lat2=ALLEQ[:,6],long2=ALLEQ[:,5])
#plt.scatter(range(len(dist_sepr)),dist_sepr,c=ALLEQ[:,1],s=20,cmap='jet')
#plt.xlim([0,27200])
#plt.ylim([0,dist_sepr.max()+0.3])
#plt.xlabel('run number',fontsize=16)
#plt.ylabel('epi-centroid separation($^\circ$)',fontsize=16)
#clb=plt.colorbar()
#clb.set_label('M$_w$',fontsize=14)
#plt.savefig('Epi_centroid_sep.png',dpi=300)
#####plot the epi-centroid separation END#####

b_X1=back_scale_X(X1,half=True)

#Load station order file
station_order_file='../Data/ALL_staname_order.txt' #this is the order in training(PGD_Chile_3400.npy)
STA=np.genfromtxt(station_order_file,'S6')
STA=np.array([sta.decode() for sta in STA])
STA_info=np.genfromtxt('../data/Chile_GNSS.gflist',usecols=[0],skip_header=1,dtype='S6')
STA_idx={sta.decode():nsta for nsta,sta in enumerate(STA_info)}
staloc=np.genfromtxt('../data/Chile_GNSS.gflist') #station file (not include name)
#Final result! this is the same order in training
staloc_sort=staloc[np.array([STA_idx[i_STA] for i_STA in STA])][:,1:3]

def quck_plot_data(X,ruptID,SRCinfo=ALLEQ,staloc_sort=staloc_sort,rupt_path='../ruptures'):
    import matplotlib
    matplotlib.rc_file_defaults()
    #function that quick plot the source, and stations
    #X: PGD data[time_steps,channels(nsta*2)], station existance code included
    #rupture ID: should be string i.e. '006081'
    #SRCinfo: source information
    #staloc_sort: station loc already sorted with the PGD data
    #Dealing with map
    coast1=np.genfromtxt('/Users/timlin/Documents/Project/NASA/LSTM_training/Chile/chile_coastline/chile_coast1.xy')#load coast
    coast2=np.genfromtxt('/Users/timlin/Documents/Project/NASA/LSTM_training/Chile/chile_coastline/chile_coast2.xy')#load coast
    coast3=np.genfromtxt('/Users/timlin/Documents/Project/NASA/LSTM_training/Chile/chile_coastline/chile_coast3.xy')#load coast
    A=np.genfromtxt(rupt_path+'/'+'subduction.'+ruptID+'.rupt')
    sta_T=np.where(X[-1,121:]!=0)[0] #station is exist
    lon=A[:,1]
    lat=A[:,2]
    SS=A[:,8]
    DS=A[:,9]
    Slip=(SS**2.0+DS**2.0)**0.5
    idx=np.where(Slip>0.01)[0]
    plt.subplot(1,2,1)
    plt.plot(coast1[:,0],coast1[:,1],'k',linewidth=0.5)
    plt.plot(coast2[:,0],coast2[:,1],'k',linewidth=0.5)
    plt.plot(coast3[:,0],coast3[:,1],'k',linewidth=0.5)
    plt.scatter(lon[idx],lat[idx],c=Slip[idx],s=8,cmap='Oranges')
    plt.plot(SRCinfo[int(ruptID),2],SRCinfo[int(ruptID),3],'r*',markersize=15,markeredgecolor='k')
    PGD=X[-1,sta_T]
    plt.scatter(staloc_sort[sta_T,0],staloc_sort[sta_T,1],c=PGD,s=50,marker='^',edgecolor='k',cmap='cool')
    plt.axis('equal')
    plt.xlim([-74.5,-68.5])
    plt.ylim([-32,-19.8])
    #Dealing with PGD data
    plt.subplot(1,2,2)
    plt.plot(np.arange(102)*5+5,X[:,sta_T])
    plt.xlabel('Time(sec)',fontsize=16)
    plt.ylabel('PGD(m)',fontsize=16)
    plt.xlim([0,515])
    plt.show()
    return 1

quck_plot_data(b_X1[idx_large[0]],'022265',ALLEQ,staloc_sort)
quck_plot_data(b_X1[5812],'022265',ALLEQ,staloc_sort) #different data

quck_plot_data(b_X1[idx_small[0]],real_EQID[idx_small[0]],ALLEQ,staloc_sort) #different data

'''
#--------------plot 4-pannels plot---------------------
plt.subplot(2,2,1)
plt.plot(y1[:,6,0],predictions[:,6,0],'b.')
plt.plot([6.5,9.5],[6.5,9.5],'r--')
plt.plot([6.5,9.5],[6.5-0.3,9.5-0.3],'r--')
plt.plot([6.5,9.5],[6.5+0.3,9.5+0.3],'r--')
plt.xlim([7.0,9.6])
plt.ylim([7.0,9.6])
plt.subplot(2,2,2)
plt.plot(y1[:,12,0],predictions[:,12,0],'b.')
plt.plot([6.5,9.5],[6.5,9.5],'r--')
plt.plot([6.5,9.5],[6.5-0.3,9.5-0.3],'r--')
plt.plot([6.5,9.5],[6.5+0.3,9.5+0.3],'r--')
plt.xlim([7.0,9.6])
plt.ylim([7.0,9.6])
plt.subplot(2,2,3)
plt.plot(y1[:,24,0],predictions[:,24,0],'b.')
plt.plot([6.5,9.5],[6.5,9.5],'r--')
plt.plot([6.5,9.5],[6.5-0.3,9.5-0.3],'r--')
plt.plot([6.5,9.5],[6.5+0.3,9.5+0.3],'r--')
plt.xlim([7.0,9.6])
plt.ylim([7.0,9.6])
plt.subplot(2,2,4)
plt.plot(y1[:,48,0],predictions[:,48,0],'b.')
plt.plot([6.5,9.5],[6.5,9.5],'r--')
plt.plot([6.5,9.5],[6.5-0.3,9.5-0.3],'r--')
plt.plot([6.5,9.5],[6.5+0.3,9.5+0.3],'r--')
plt.xlim([7.0,9.6])
plt.ylim([7.0,9.6])
plt.show()
#--------------plot 4-pannels plot END---------------------


plt.plot(y1[:,-1,0],predictions[:,-1,0],'b.')
plt.plot([6.5,9.5],[6.5,9.5],'r--')
plt.plot([6.5,9.5],[6.5-0.3,9.5-0.3],'r--')
plt.plot([6.5,9.5],[6.5+0.3,9.5+0.3],'r--')
plt.xlim([7.0,9.6])
plt.ylim([7.0,9.6])
'''


####################Test the model on real data#######################
def D2PGD(data):
    PGD=[]
    if np.ndim(data)==2:
        for i in range(data.shape[0]):
            PGD.append(np.max(data[:i+1,:],axis=0))
    elif np.ndim(data)==1:
        for i in range(len(data)):
            PGD.append(np.max(data[:i+1]))
    
    PGD=np.array(PGD)
    return(PGD)


data_path='/Users/timlin/Documents/Project/NASA/LSTM_training/Chile/chile_GNSS'
EQs=['Illapel2015','Iquique2014','Maule2010','Melinka2016','Iquique_aftershock2014']
EQs_lb=['Illapel 2015 (M$_w$8.3)','Iquique 2014 (M$_w$8.1)','Maule 2010 (M$_w$8.8)','Melinka 2016 (M$_w$7.6)','Iquique_aft 2014 (M$_w$7.7)'] #label for plotting
STFs=['Illapel.mr.txt','Iquique.mr.txt','Maule.mr.txt','Melinka.mr.txt','Iquique_aft.mr.txt']
EQt=[obspy.UTCDateTime(2015,9,16,22,54,33),obspy.UTCDateTime(2014,4,1,23,46,47),obspy.UTCDateTime(2010,2,27,6,34,14),obspy.UTCDateTime(2016,12,25,14,22,26),obspy.UTCDateTime(2014,4,3,2,43,13)]
Add_travel_time=False #remove travel time latency?

EQMw=[8.3,8.1,8.8,7.6,7.7]
EQloc=[(-71.654,-31.57,29),
       (-70.769,-19.61,25),
       (-72.733,-35.909,35),
       (-74.391,-43.517,30),
       (-70.493,-20.571,22.4),
       ]
sampl=5 #5 seconds of sampling rate

station_order_file=home+'/data/'+'ALL_staname_order.txt' #this is the order in training (PGD_Chile_3400.npy)
STA=np.genfromtxt(station_order_file,'S6')
STA=np.array([sta.decode() for sta in STA]) #PGD should sorted in this order

#Load GF_list to get staloc
STA_info=np.genfromtxt(home+'/data/'+'Chile_GNSS.gflist',usecols=[0],skip_header=1,dtype='S6')
STA_idx={sta.decode():nsta for nsta,sta in enumerate(STA_info)}
staloc=np.genfromtxt(home+'/data/'+'Chile_GNSS.gflist') #station file (not include name)
tmp_order=np.array([STA_idx[tmpsta] for tmpsta in STA])
staloc_sorted=staloc[tmp_order,1:3]

sav_X_data=[]
sav_delay_sec=np.zeros(len(EQs))
all_save_hypodist=[]
for ieq,eq in enumerate(EQs):
    closest_dist=1e9
    #plt.figure(1)
    #plt.title(eq,fontsize=15,fontweight='bold')
    Alldata=glob.glob(data_path+'/'+eq+'/GPS/'+'*.sac')
    stanames=[]
    sav_PGD=[]
    sav_t=[]
    save_hypodist=[]
    for data in Alldata:
        sta=data.split('/')[-1].split('.')[0].upper()
        if not(sta in stanames):
            if sta in STA:
                stanames.append(sta)
                #get lon,lat of the station
                tmpidx=STA_idx[sta]
                stlo=staloc[tmpidx][1]
                stla=staloc[tmpidx][2]
                hypodist=obspy.geodetics.locations2degrees(lat1=EQloc[ieq][1],long1=EQloc[ieq][0],lat2=stla,long2=stlo)
                save_hypodist.append(hypodist)
                if hypodist<closest_dist:
                    closest_dist=hypodist
                    closest_STA=sta
                    closest_stlo=stlo
                    closest_stla=stla
            else:
                print('Found station:%s not in the training dataset'%(sta))
    print('Closest station of %s is %s'%(EQs[ieq],closest_STA))
    all_save_hypodist.append(save_hypodist)
    #P-wave travel time to the cloest station
    model = TauPyModel(model="ak135")
    P=model.get_travel_times(source_depth_in_km=EQloc[ieq][2], distance_in_degree=closest_dist, phase_list=('P','p'), receiver_depth_in_km=0)
    Pt=P[0].time #or P[1].time
    sav_delay_sec[ieq]=Pt
    stanames=np.array(stanames)
    print('The closest station is:%s, whth hypodist:%f, P-travel time:%f'%(closest_STA,closest_dist,Pt))
    print('For %s, Total of %d stations\n'%(eq,len(stanames)))
    print('Stations:%s\n'%(stanames))
    #Remove the travel time effect
    if Add_travel_time:
        print('Before:',EQt[ieq])
        EQt[ieq]=EQt[ieq]+datetime.timedelta(seconds=Pt) #move the event origin time later, means shift PGD earlier
        print('After:',EQt[ieq])
    #find 3-component of the data and calculate PGD
    for sta in stanames:
        comps3=glob.glob(data_path+'/'+eq+'/GPS/'+sta+'*.sac')
        if len(comps3)==0:
            comps3=glob.glob(data_path+'/'+eq+'/GPS/'+sta.lower()+'*.sac')
        if len(comps3)!=3:
            break #not enought components for PGD
        #start calculate PGD from 3-comps, dealing with time
        D1=obspy.read(comps3[0])
        DD1=D1.copy()
        DD1.trim(starttime=EQt[ieq],endtime=EQt[ieq]+510,pad=True,fill_value=0)
        D2=obspy.read(comps3[1])
        DD2=D2.copy()
        DD2.trim(starttime=EQt[ieq],endtime=EQt[ieq]+510,pad=True,fill_value=0)
        D3=obspy.read(comps3[2])
        DD3=D3.copy()
        DD3.trim(starttime=EQt[ieq],endtime=EQt[ieq]+510,pad=True,fill_value=0)
        #Dsum=(D1[0].data**2+D2[0].data**2+D3[0].data**2)**0.5
        #make the data=0 at 0 second
        DD1[0].data=DD1[0].data-DD1[0].data[0]
        DD2[0].data=DD2[0].data-DD2[0].data[0]
        DD3[0].data=DD3[0].data-DD3[0].data[0]
        Dsum=(DD1[0].data**2+DD2[0].data**2+DD3[0].data**2)**0.5
        PGD=D2PGD(Dsum)
        sav_PGD.append(PGD)
        t=DD1[0].times()
        sav_t.append(t)
        #plt.plot(t,Dsum)
        #plt.plot(t,PGD,'--',color=[0.5,0.5,0.5])
    #plt.xlabel('sec',fontsize=15)
    #plt.ylabel('Displacement(m)',fontsize=15)
    #plt.show()
    T=np.arange(0+sampl,t.max()+sampl,sampl)
    #Make matrix for training
    #X_data=[]
    X_data=np.zeros([len(T),len(STA)*2])
    nsta=0
    for n,sta in enumerate(STA):
        idx=np.where(sta == stanames)[0]
        if len(idx)!=0:
            print('sta=',sta,'station found, idx=',idx)
            nsta+=1
            #PGD_resampled=np.interp(T,t,sav_PGD[idx[0]])
            PGD_resampled=np.interp(T,sav_t[idx[0]],sav_PGD[idx[0]])
            X_data[:,n]=PGD_resampled.copy()
            X_data[:,n+len(STA)]=np.ones(len(T)) * 0.5 #Station existence code is 0.5 instead of 1!!!!!!!!!!!!!!!
#            X_data[:,n+len(STA)]=np.ones(len(T))  #Station existence code is 1 for Test80 instead of 0.5!!!!!!!!!!!!!!!
        #X_data.append(PGD_resampled)
        #plt.plot(T,PGD_resampled,'--',color=[0.5,0.5,0.5])
        #plt.show()
        else:
            print('Station not found, do not fill values')
    #X_data.append(np.zeros(len(T)))
    sav_X_data.append(X_data)

sav_X_data=np.array(sav_X_data) #This is the raw PGD timeseries data without feature scaling
sav_X_data_orig=sav_X_data.copy()
#Now, rember to scale the data for ML input
#print('-------------Now, scale the X by sqrt(X)/10.0-----------------')
#sav_X_data[:,:,:121]=((sav_X_data[:,:,:121])**0.5)/10.0

print('-------------Now, scale the X by log10(X)-----------------')
sav_X_data=scale_X(sav_X_data,half=True)


########Prediction on the REAL data###########
predictions_real=model_loaded.predict(sav_X_data)
predictions_real=back_scale_y(predictions_real)
#convert PGD back to the original scale!!!
#sav_X_data[:,:,:121]=((sav_X_data[:,:,:121])*10.0)**2
#convert PGD back to the original scale!!!
sav_X_data=back_scale_X(sav_X_data,half=True)


#savefig=True
savefig=False
sns.set()
props = dict(boxstyle='round', facecolor='white', alpha=0.95) #box for plt.text
for neq in range(len(predictions_real)):
    plt.figure()
    plt.plot(T,predictions_real[neq,:],'b.-')
    plt.plot([T.min(),T.max()],[EQMw[neq],EQMw[neq]],'r') #Mw8.3
    maxPGD=sav_X_data[neq][:,:121].max()
    mul_scale=(EQMw[neq]-1.5)/maxPGD
    for nsta in range(121):
        if sav_X_data[neq][-1,nsta]!=0:
            plt.plot(T,sav_X_data[neq][:,nsta]*mul_scale,'--',color=[0.5,0.5,0.5],linewidth=0.8) #pad zero for the 0 sec
    #plt.plot(T,sav_X_data[neq][:,:121]*mul_scale,color=[0.5,0.5,0.5],linewidth=0.5) #this also plot the zeros
    plt.text(380,EQMw[neq]-1.2,'PGD$_m$$_a$$_x$=%3.2f m'%(maxPGD),fontsize=12,bbox=props)
    plt.grid(True)
    plt.xlabel('Time(s)',fontsize=15)
    plt.ylabel('Pred. Mw',fontsize=15)
    plt.title(EQs_lb[neq],fontsize=15)
    if savefig:
        plt.savefig('RealEQ_'+run_name+'_EQ%d.png'%(neq+1),dpi=200)
        plt.close()
    else:
        plt.show()

#EQs=['Illapel2015','Iquique2014','Maule2010','Melinka2016','Iquique_aftershock2014']
#manually get the P-arrival time
sav_delay_sec=[7.5882,14.8001, 16.5069,12.373970928914572, 9.8072]

#Prediction box plot

#time_scale=np.arange(102)*5+5
#box_scale=np.array(range(len(time_scale)))
out_dots = dict(markerfacecolor=[0.5,1.0,0.0],markeredgecolor=[0.5,1.0,0.0],mew=0.1, marker='.',markersize=1.5)
props = dict(boxstyle='round', facecolor='white', alpha=0.7) #box for plt.text
PGD_ticks=[[0,0.5,1,1.5],[0,0.5,1],[0,2,4,6],[0,0.2,0.4],[0,0.2,0.4] ]
mul_PGD_scale=[3,3,3,5,4.5] #try these parameters and see
for neq in range(len(predictions_real)):
    sav_X_box=[]
    sav_Y_box=[]
    T_all=np.arange(102)*5+5
    for i,tmp_mw in enumerate(predictions_real[neq,:,0]):
        tmpidx,tmpbox=get_box_distribution(MovingAcc_Time_Mw,sav_misfit_distribute,target_Time=T_all[i],target_Mw=tmp_mw)
        sav_X_box.append(T_all[i])
        #sav_Y_box.append(tmpbox+tmp_mw)
        sav_Y_box.append(tmp_mw-tmpbox) #so that the distribution y is the
    fig=plt.figure(1)
    ax1 = fig.add_subplot(111)
    #plot only after P-arrival
    sav_X_box=np.array(sav_X_box)
    sav_Y_box=np.array(sav_Y_box)
    p_idx=np.where(sav_X_box>=sav_delay_sec[neq])[0]
    bp=plt.boxplot(sav_Y_box[p_idx],positions=sav_X_box[p_idx],widths=1,patch_artist=True,flierprops=out_dots) #flierprops=red_square
    plt.plot(sav_X_box[p_idx],np.array([np.median(md) for md in sav_Y_box[p_idx]]),'r-',markersize=8,markeredgecolor=[1,0,0],markerfacecolor='None',mew=0.1)
    #plot the prediction only
#    plt.plot(np.arange(102)*5.0+5,predictions_real[neq,:,0],'b')
    ax1.plot([T_all.min(),T_all.max()],[EQMw[neq],EQMw[neq]],'k--')
    #ax1.plot([T_all.min(),T_all.max()],[EQMw[neq]-0.3,EQMw[neq]-0.3],'b--',linewidth=0.5)
    #ax1.plot([T_all.min(),T_all.max()],[EQMw[neq]+0.3,EQMw[neq]+0.3],'b--',linewidth=0.5)
    ax1.fill_between([T_all.min(),T_all.max()],[EQMw[neq]-0.3,EQMw[neq]-0.3],[EQMw[neq]+0.3,EQMw[neq]+0.3],facecolor='k',alpha=0.25)
    for patch in bp['boxes']:
#        patch.set_facecolor([1,0.5,0])
#        patch.set_edgecolor([1,0.5,0])
        patch.set_facecolor([1,0.,0])
        patch.set_edgecolor([1,0.,0])
        patch.set_linewidth(0.05)
    for wisker in bp['whiskers']:
        wisker.set_color([1,1,0])
        wisker.set_linewidth(1.0)
        wisker.set_markeredgecolor([1,0,0])
    for cap in bp['caps']:
        cap.set_color([1,1,0])
    for medn in bp['medians']:
        medn.set_color('None')
    plt.xticks([100,200,300,400,500],['100','200','300','400','500'])
    plt.xlim([0,510])
    plt.ylim([6.5,9.6])
    if neq==2:
#        plt.text(20,9.3,'%s'%(EQs_lb[neq]),fontsize=14,bbox=props)
#        plt.xlabel('Seconds since origin',fontsize=14)
#        plt.ylabel('Predicted Mw',fontsize=14)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        ax1.tick_params(pad=2)
        plt.text(20,9.3,'%s'%(EQs_lb[neq]),fontsize=17,bbox=props)
        plt.xlabel('Seconds since origin',fontsize=17,labelpad=-1)
        plt.ylabel('Predicted Mw',fontsize=17)
    else: #small subplots
#        plt.xticks(fontsize=16)
#        plt.yticks(fontsize=16)
#        ax1.tick_params(pad=2)
#        plt.text(20,9.3,'%s'%(EQs_lb[neq]),fontsize=20,bbox=props)
#        plt.xlabel('Seconds since origin',fontsize=20,labelpad=-1)
#        plt.ylabel('Predicted Mw',fontsize=20)
        plt.xticks(fontsize=19)
        plt.yticks(fontsize=19)
        ax1.tick_params(pad=1)
        plt.text(20,9.3,'%s'%(EQs_lb[neq]),fontsize=20,bbox=props)
        plt.xlabel('Seconds since origin',fontsize=23,labelpad=-2.5)
        plt.ylabel('Predicted Mw',fontsize=23)
    #####Mark the area before the P-wave arrival#####
    ax1.fill_between([0,sav_delay_sec[neq]],[6.0,6.0],[10.0,10.0],edgecolor='k',facecolor=[0.5,0.5,0.5],hatch="/")
    #############Make GFast prediction##############
    evlo=EQloc[neq][0]
    evla=EQloc[neq][1]
    evdp=EQloc[neq][2]
    use_idx=np.where(sav_X_data_orig[neq][-1,:121]!=0)[0]  #The last PGD point is not zero that means the station is operating
    stlo=staloc_sorted[:,0][use_idx].reshape(-1,1)
    stla=staloc_sorted[:,1][use_idx].reshape(-1,1)
    nbuff=sav_X_data_orig[neq][:,use_idx].T #nbuff is the pgd already, others should be zeros. shape=[reduced_Nstan by nepoch]
    ebuff=np.zeros_like(nbuff)
    ubuff=np.zeros_like(nbuff)
    tbuff=np.arange(102)*5+5  #shape should be [Ndata by 1]
    sav_GFAST_Mw=[]
    sav_GFAST_time=[]
#    for runtime in np.arange(102)*5+5:
    for runtime in np.arange(0,515,30):
        GFMw,VR,NS = data_engine_pgd(evla,evlo,evdp,0,stla,stlo,np.zeros_like(stlo),nbuff,ebuff,ubuff,tbuff,runtime)
        try:
            sav_GFAST_Mw.append(GFMw.max())
        except:
            sav_GFAST_Mw.append(0) #just a constant
        sav_GFAST_time.append(runtime)
    sav_GFAST_Mw=np.array(sav_GFAST_Mw)
    sav_GFAST_time=np.array(sav_GFAST_time)
    idx_plot1=np.where(sav_GFAST_Mw>0)[0]
    plt.plot(sav_GFAST_time[idx_plot1],sav_GFAST_Mw[idx_plot1],'b*-',markersize=10,markeredgecolor=[1,1,1],markerfacecolor=[0,0,1],mew=0.5)
    #############GFast prediction END##############
#    plt.title(EQs_lb[neq],fontsize=14)
#    plt.text(30,9.2,'%s'%(EQs_lb[neq]),fontsize=14,bbox=props)
    ax2 = ax1.twiny()
    ######Second x axis (shifted time)######
    ax2.set_xlim(ax1.get_xlim())
    orig_ticks=ax2.get_xticks()
    shifted_ticks=orig_ticks-sav_delay_sec[neq]
#    ax2.set_xticks(orig_ticks,['%.1f'%(i) for i in shifted_ticks])
#    ax2.set_xticks(orig_ticks,['1','2','3','4','5','6','7'])
    #plt.xticks(orig_ticks,['%.1f'%(i) for i in shifted_ticks]) #Use the original tick
    #####or add the new tick starting from 0#####
    d_shft=-shifted_ticks[0]
#    plt.xticks(orig_ticks+d_shft,['%.1f'%(i) for i in orig_ticks])
    plt.xticks(orig_ticks+d_shft,['%d'%(i) for i in orig_ticks])
#    plt.xticks([0,100,200,300,400,500])
    plt.xlim([0,510])
#    ax2.set_xlabel('Shifted time(sec)',fontsize=14)
    ax2.tick_params(direction='out', length=2.5, width=1.2, colors='k',pad=-3)
    ax2.grid(False)
    if neq==2:
#        ax2.set_xlabel('Seconds since p-wave arrival',fontsize=15)
        ax2.set_xlabel('Seconds since p-wave arrival',fontsize=18)
        plt.xticks(fontsize=16)
    else:#small subplots
#        ax2.set_xlabel('Seconds since p-wave arrival',fontsize=20)
#        plt.xticks(fontsize=16)
        ax2.set_xlabel('Seconds since p-wave arrival',fontsize=23,labelpad=1.5)
        plt.xticks(fontsize=19)
    ###Third axis (y-axis) (PGD)###
    ax3 = ax1.twinx()
    ax3.set_xlim(ax1.get_xlim())
    ###plot PGDs###
    maxPGD=sav_X_data_orig[neq][:,:121].max()
    for nsta in range(121):
        if sav_X_data_orig[neq][-1,nsta]!=0:
            plt.plot(T,sav_X_data_orig[neq][:,nsta]*1,'--',color=[0.5,0.5,0.5],linewidth=0.8) #pad zero for the 0 sec
    if neq==2:
#        plt.text(370,maxPGD+maxPGD*0.04,'PGD$_m$$_a$$_x$=%3.2f m'%(maxPGD),fontsize=12)
        plt.text(340,maxPGD+maxPGD*0.04,'PGD$_m$$_a$$_x$=%3.2f m'%(maxPGD),fontsize=15)
    else: #small subplots
#        plt.text(350,maxPGD+maxPGD*0.04,'PGD$_m$$_a$$_x$=%3.2f m'%(maxPGD),fontsize=15)
        plt.text(310,maxPGD+maxPGD*0.04,'PGD$_m$$_a$$_x$=%3.2f m'%(maxPGD),fontsize=18)
    plt.ylim([0,maxPGD*mul_PGD_scale[neq]])
    plt.yticks([])
#    plt.yticks(PGD_ticks[neq])
#    if neq==2:
#        plt.yticks(fontsize=12)
#        ax3.tick_params(pad=1.5)
#    else:
#        plt.yticks(fontsize=12)
#        ax3.tick_params(pad=2.5)
    plt.grid(False)
    ###END PGD###
    #Also plot STF
    stf=np.genfromtxt(home+'/data/STF_realEQs/'+STFs[neq])
    t_interp=np.arange(0,T_all.max(),1)
    stf_interp=np.interp(t_interp,stf[:,0],stf[:,1])
    idx_plot=np.where(stf_interp>0)[0]
    scal_stf=maxPGD*mul_PGD_scale[neq]*0.2/stf_interp.max()
    plt.plot(t_interp[idx_plot],stf_interp[idx_plot]*scal_stf,color=[1,0,1])
    plt.savefig('EQ_range_%s.png'%(neq+1),dpi=600)
    plt.show()
    plt.clf()


####Plot PGD-Mw relationship
#for istan in range(len(stlo)):
#    hypodist=obspy.geodetics.locations2degrees(lat1=evla,long1=evlo,lat2=stla[istan],long2=stlo[istan])
#    plt.plot(np.arange(102)*5+5,nbuff[istan]+hypodist*0.1)








