#scaling function for features
import numpy as np
import glob

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


#=====scaling functions for multiple X, y========

def get_ENZ_range(Xpath):
    #load the data and have a sense of the value range
    E_files=glob.glob(Xpath+'/*.E.npy')
    E_files.sort()
    N_files=glob.glob(Xpath+'/*.N.npy')
    N_files.sort()
    Z_files=glob.glob(Xpath+'/*.Z.npy')
    Z_files.sort()
    sav_min=np.inf
    sav_max=-np.inf
    for E_file in E_files:
        E=np.load(E_file)
        Emin=E.min()
        Emax=E.max()
        if Emin<sav_min:
            sav_min=Emin
        if Emax>sav_min:
            sav_max=Emax

    for N_file in N_files:
        N=np.load(N_file)
        Nmin=N.min()
        Nmax=N.max()
        if Nmin<sav_min:
            sav_min=Nmin
        if Nmax>sav_min:
            sav_max=Nmax

    for Z_file in Z_files:
        Z=np.load(Z_file)
        Zmin=Z.min()
        Zmax=Z.max()
        if Zmin<sav_min:
            sav_min=Zmin
        if Zmax>sav_min:
            sav_max=Zmax
    return sav_min,sav_max


def make_linear_scale(min_value,max_value,target_min=0,target_max=1):
    r=(target_max-target_min)/(max_value-min_value)
    shft=(max_value+min_value)*0.5
    #create function
    #def f(x):
    #    x = (x-min_value)*r + target_min
    return lambda x:(x-min_value)*r + target_min   # cannot be saved by pickle
    #return f

'''
A way to save functions in npy, however, the function input in M-LARGE will be complicated
class make_linear_scale():
    def __init__(self,min_value,max_value,target_min=0,target_max=1):
        self.min_value = min_value
        self.max_value = max_value
        self.target_min = target_min
        self.target_max = target_max
        
    def scale(self):
        r=(self.target_max-self.target_min)/(self.max_value-self.min_value)
        shft=(self.max_value+self.min_value)*0.5
        return lambda x:(x-self.min_value)*r + self.target_min

class make_X_scale():
    #make_X_scale(np.arctan,0,0.1)
    def __init__(self,function,add,mul):
        self.function = function
        self.add = add
        self.mul = mul
    
    def scale(self):
        Xscale = lambda x : self.function((x+self.add)*self.mul)
        return Xscale
'''



#Note of parameters range for Chilean fakequakes
#ENZ=(-44.00901794433594,10.02457046508789)
#Mw=(7.3577,9.6254)
#hypo_Lon=(-75.68104,-69.73932)
#hypo_Lat=(-43.91879,-18.028)
#hypo_dep=(4.3,53.69)
#cent_Lon=(-75.23913,-69.86168)
#cent_Lat=(-43.5795,-18.19313)
#cent_dep=(8.6,50.2900)

#Xscale=make_linear_scale(-15,10,target_min=0,target_max=1) #displacement reaches -44 is very unusual
#yscale=[
#        make_linear_scale(7.5,9.5,target_min=0,target_max=1), #Mw
#        make_linear_scale(-75.5,-69.5,target_min=0,target_max=1), #cent_lon
#        make_linear_scale(-43.5,-18.5,target_min=0,target_max=1), #cent_lat
#        make_linear_scale(8.5,50,target_min=0,target_max=1), #cent_dep
#        make_linear_scale(0,1200,target_min=0,target_max=1), #length
#        make_linear_scale(0,150,target_min=0,target_max=1), #width
#]






