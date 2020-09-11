#scaling function for features
import numpy as np

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


scale_y  = lambda y : y * 0.1
back_scale_y  = lambda y : y * 10.0

