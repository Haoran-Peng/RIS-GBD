# -*- coding: utf-8 -*-

import sys
import numpy as np
import numpy.matlib
import math as ma
import random



def userpara(Num_User,Tx_antBS,RIS_Lnum,Tx_antRIS, BS_antenna_gain, RIS_antenna_gain,seed):
    
    np.random.seed(seed)
    
    Bandwidth=1 #MHz
    Noise_o=-104 #dBm  -174dBm/Hz
    # Noise_o=-100 #dBm  -174dBm/Hz
    Noise=pow(10, (Noise_o/10))/pow(10,3) #watts
    
    # BS_antenna_gain = pow(10, (BS_antenna_gain/10))/pow(10,3)
    # RIS_antenna_gain = pow(10, (RIS_antenna_gain/10))/pow(10,3)
    
    
    #Area: Square
    lengA = -500 
    lengB = 60 # m
    
    #power limits
    
    mu=5/4                                   #---------power amplifier efficiency ^ -1---------#
    
    P_max_o=50 #dBm
    P_max=pow(10, (P_max_o/10))/pow(10,3)    #------------Maximum transmitted power------------#
    P_k_o=10  #dBm
    P_k=pow(10, (P_k_o/10))/pow(10,3)        #----------User static power comsumption----------#
    
    
    
    #------------BS location-----------------#

    BS_loc=np.array([lengA/2,lengA/2, 10])
    


    #-----------RIS location-----------------#

    RISloc=np.zeros([RIS_Lnum, 3])

    

    for i in range(0,RIS_Lnum):   # i = l_RIS, i從0開始

        RISloc[i] = [0, 0, 20]
        
        

    #---------------User location-----------------#

    User_loc = np.zeros([Num_User, 3])
    
    #--------------Ramdom or Fixed user location------------#
    
    if seed != 15:
        
        User_loc[0] =[lengB*np.random.rand(1), lengB*np.random.rand(1), 1]
    else:
        User_loc[0] =[20, 20, 1]
   

    for i in range(0,Num_User):   #i = userloc
            User_loc[i] = User_loc[0]
            
    dismin = 10
    
    PL_0 = ma.pow(10, (-30/10))


    #--------------------------------Pathloss between User and BS------------------------------------#   
     
    #--------Compute distance---------#
    Distance_UserBS = np.maximum(np.sqrt(np.sum(np.square(np.matlib.repmat(BS_loc, Num_User, 1) - User_loc), axis=1)), dismin).reshape(2,1)
    

    #----------------PathLoss between BS and User large scale----------------#
    
    PathLoss_UserBS_value = ma.sqrt(PL_0*ma.pow(Distance_UserBS[0], -4)) * BS_antenna_gain
    

    
    #----------------PathLoss BS User small scale----------------#  
    
    PathLoss_UserBS = np.matlib.repmat(PathLoss_UserBS_value, Num_User, Tx_antBS)
    PathLoss_UserBS = np.multiply(PathLoss_UserBS, ma.sqrt(1/2)*(np.random.randn(Num_User,Tx_antBS)+1j*np.random.randn(Num_User,Tx_antBS)))


    
    #--------------------------------Pathloss between User and RIS-------------------------------------#
    
    Distance_UserRIS = np.zeros([Num_User,RIS_Lnum])

    #--------Compute distance---------#
    for i in range(0,RIS_Lnum):
        Distance_UserRIS[:,i] = np.maximum(np.sqrt(np.sum(np.square(np.matlib.repmat(RISloc[i,:], Num_User, 1) - User_loc), axis=1)), dismin)
        

    

    #---------------Pathloss between User and RIS large scale-----------------#      

    PathLoss_UserRIS_value = ma.sqrt(PL_0*ma.pow(Distance_UserRIS[0], -2)) * RIS_antenna_gain
    


    
    PathLoss_UserRIS = np.zeros([Tx_antRIS,Num_User,RIS_Lnum], dtype = 'complex_')


    #----------------PathLoss RIS User small scale----------------#
    
    for i in range(0,Num_User):       # i = k_user
        for j in range(0,RIS_Lnum):   # j = l_RIS
            PathLoss_UserRIS[:,i,j] = (np.multiply(np.matlib.repmat(PathLoss_UserRIS_value, Tx_antRIS, 1), ma.sqrt(1/2)*(np.random.randn(Tx_antRIS, 1)+1j*np.random.randn(Tx_antRIS, 1)))).reshape(Tx_antRIS,)

    



    #-------------------------------------Pathloss between BS and RIS-------------------------------------------#

    Distance_RISBS = np.zeros([RIS_Lnum,1])


    for i in range(0,RIS_Lnum):
        Distance_RISBS[i,0] = np.maximum(np.sqrt(np.sum(np.square(RISloc[i,:] - BS_loc))), dismin)
        

    #---------------Pathloss between BS and RIS Large Scale-----------------#   

    PathLoss_RISBS_value = ma.sqrt(PL_0*ma.pow(Distance_RISBS[0], -2.2)) * BS_antenna_gain**2
    

    
    PathLoss_RISBS=np.zeros([Tx_antBS,RIS_Lnum,Tx_antRIS], dtype = 'complex_')


    #----------------PathLoss RIS BS small scale----------------#
    
    for i in range(0,RIS_Lnum):
        
        PathLoss_RISBS[:,i,:] = (np.multiply(np.matlib.repmat(PathLoss_RISBS_value, Tx_antBS, Tx_antRIS), ma.sqrt(1/2)*(np.random.randn(Tx_antBS, Tx_antRIS)+1j*np.random.randn(Tx_antBS, Tx_antRIS))))


    
    return Bandwidth, Noise, P_max, P_k, mu, PathLoss_UserBS, PathLoss_UserRIS, PathLoss_RISBS