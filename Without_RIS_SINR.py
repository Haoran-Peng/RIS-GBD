#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 13:52:04 2023

@author: dddd
"""

#!/usr/bin/env python3

#----GBD primal problem----#

# import sys
import numpy as np
import numpy.matlib
import math as ma
import time
# import random
from pyomo.environ import *
from pyomo.util.infeasible import log_infeasible_constraints

import pyutilib.subprocess.GlobalData
pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False


def Without_RIS_exh(Bandwidth, Noise, P_max, P_k, mu, PathLoss_UserBS, PathLoss_UserRIS, PathLoss_RISBS,
                   Num_User, Tx_antBS, RIS_Lnum, Tx_antRIS, Power_PU):
    
    #--------------------------------array setting-----------------------------------#
    
    glvec_1 = np.zeros([Tx_antBS,1], dtype = 'complex_')
    userris_1 = np.zeros([Tx_antRIS,1], dtype = 'complex_')
    Ulmar_1 = np.zeros([Tx_antRIS,Tx_antBS], dtype = 'complex_')
    # xini = xonoff[0]

    risbs = np.zeros([Tx_antRIS, Tx_antBS], dtype = 'complex_')
    U_diag = np.eye(Tx_antRIS)

    glvec_1[:,0] = PathLoss_UserBS[0,:]                        #--------------------direct link channel-------------------#

    for j in range(0, Tx_antBS):
        risbs[:,j] = PathLoss_RISBS[j,:,:]                     #--------------------RIS to BS channel---------------------#

    flag = 0

    for i in range(0,RIS_Lnum):
        userris_1[:,0] = PathLoss_UserRIS[:,0,:].reshape(Tx_antRIS,)  #-------------------RIS to User channel---------------------#
        
        #------------combine two channel, called matrix U----------------#
        
        Ulmar_1[flag*Tx_antRIS : (flag+1)*Tx_antRIS,:] = np.dot(U_diag*(userris_1.conj().T), risbs)
        flag = flag+1
        
        
    Ulmar_1 = Ulmar_1.conj().T
    
    WithoutRIS_SINR = np.zeros([31,1])
    
    temp_power_primary = Power_PU
    temp_power_secondary = 20
    temp_power_secondary_withoutRIS = 0.0001
    Best_SINR_secondary_withoutRIS = 0
    Best_power_secondary_withoutRIS = 0
    SINR_secondary_withoutRIS = 0
    
    while(temp_power_secondary < 51):
        
        temp_power_secondary_withoutRIS =pow(10, (temp_power_secondary/10))/pow(10,3)
    
        SINR_secondary_withoutRIS = temp_power_secondary_withoutRIS * (glvec_1.conj().T @ glvec_1)/ (temp_power_primary * (glvec_1.conj().T @ glvec_1) + Noise)
        WithoutRIS_SINR[temp_power_secondary - 20] = SINR_secondary_withoutRIS
        
        if(SINR_secondary_withoutRIS > Best_SINR_secondary_withoutRIS):
            Best_SINR_secondary_withoutRIS = SINR_secondary_withoutRIS
            Best_power_secondary_withoutRIS = temp_power_secondary_withoutRIS
            
        temp_power_secondary += 1
        
    return Best_SINR_secondary_withoutRIS, Best_power_secondary_withoutRIS, WithoutRIS_SINR

    
def SINR_withRIS(Noise, P_max, P_k, mu, PathLoss_UserBS, PathLoss_UserRIS, PathLoss_RISBS,
                   Num_User, Tx_antBS, RIS_Lnum, Tx_antRIS, Power_PU, power_SU, RIS_phase):
    
    #--------------------------------array setting-----------------------------------#
    
    glvec_1 = np.zeros([Tx_antBS,1], dtype = 'complex_')
    userris_1 = np.zeros([Tx_antRIS,1], dtype = 'complex_')
    Ulmar_1 = np.zeros([Tx_antRIS,Tx_antBS], dtype = 'complex_')
    # xini = xonoff[0]

    risbs = np.zeros([Tx_antRIS, Tx_antBS], dtype = 'complex_')
    U_diag = np.eye(Tx_antRIS)
    userris_1[:,0] = PathLoss_UserRIS[:,0,:].reshape(Tx_antRIS,)

    glvec_1[:,0] = PathLoss_UserBS[0,:]                        #--------------------direct link channel-------------------#

    for j in range(0, Tx_antBS):
        risbs[:,j] = PathLoss_RISBS[j,:,:]                     #--------------------RIS to BS channel---------------------#

    flag = 0

    for i in range(0,RIS_Lnum):
        userris_1[:,0] = PathLoss_UserRIS[:,0,:].reshape(Tx_antRIS,)  #-------------------RIS to User channel---------------------#
        
        #------------combine two channel, called matrix U----------------#
        
        Ulmar_1[flag*Tx_antRIS : (flag+1)*Tx_antRIS,:] = np.dot(U_diag*(userris_1.conj().T), risbs)
        flag = flag+1
        
        
    Ulmar_1 = Ulmar_1.conj().T
    
    temp_power_primary = Power_PU
    temp_power_secondary = power_SU
    SINR_secondary_withRIS = 0
    temp_power_secondary_withRIS = 0
    Best_SINR_secondary_withRIS = 0
    Best_power_secondary_withRIS = 0
    
    Total_link = glvec_1.conj().T + userris_1.conj().T @ With_phase_quantized(RIS_phase) @ risbs
    
    temp_power_secondary_withRIS =pow(10, (temp_power_secondary/10))/pow(10,3)
    # print("SU power dbm : ", temp_power_secondary, "\nSU power W : ", temp_power_secondary_withRIS, "PU power dbm : ", Power_PU)
    SINR_secondary_withRIS = temp_power_secondary_withRIS * (Total_link @ Total_link.conj().T)/ (temp_power_primary * (Total_link @ Total_link .conj().T) + Noise)
   
    if(SINR_secondary_withRIS > Best_SINR_secondary_withRIS) :
        Best_SINR_secondary_withRIS = SINR_secondary_withRIS
        Best_power_secondary_withRIS = temp_power_secondary_withRIS
    
    # while(temp_power_secondary < 50):
        
    #     temp_power_secondary_withRIS =pow(10, (temp_power_secondary/10))/pow(10,3)
       
    #     SINR_secondary_withRIS = temp_power_secondary_withRIS * (Total_link @ Total_link.conj().T)/ (temp_power_primary * (Total_link @ Total_link .conj().T) + Noise)
       
    #     if(SINR_secondary_withRIS > Best_SINR_secondary_withRIS) :
    #         Best_SINR_secondary_withRIS = SINR_secondary_withRIS
    #         Best_power_secondary_withRIS = temp_power_secondary_withRIS
       
    #     temp_power_secondary += 1
      
    # Total_link_ver = glvec_1 + Ulmar_1 @ (np.exp(1j*(2*ma.pi*RIS_phase/(4))).conj()).reshape(Tx_antRIS, 1)
    
    # print("T veri: ", (glvec_1).shape)
    # print("T link gain : ", Total_link @ Total_link.conj().T, "\nT link veri : ", Total_link_ver.conj().T @ Total_link_ver)
    
    return Best_SINR_secondary_withRIS

    

       
#----------primal problem objective function----------#

def With_phase_quantized(phase_int):
    
    phase_shift = np.diag(np.exp(1j*(2*ma.pi*phase_int/(4))))
    
    return phase_shift


def With_phase_single_quantized(phase_int):
    
    phase_shift = np.exp(1j*(2*ma.pi*phase_int/(4)))
    
    return phase_shift
# def exh_withRIS(NumRISEle):
    
#     phase_stage = 4
    
#     for idx_1 in range(NumRISEle):
#         for idx_2 in range(NumRISEle):
#             for idx_3 in range(NumRISEle):
#                 for idx_4 in range(NumRISEle):
    
#     return


def create_withoutRIS( depth: int, max_depth: int, password, best, best_word, Noise, P_max, P_k, mu, PathLoss_UserBS, PathLoss_UserRIS, PathLoss_RISBS,
                   Num_User, Tx_antBS, RIS_Lnum, Tx_antRIS, Power_PU, power_SU):
    # print("first : ", password)
    # AA = []
    # temp_i=-5
    if (depth == max_depth):
        # print(password)
        Temp_SINR = SINR_withRIS(Noise, P_max, P_k, mu, PathLoss_UserBS, PathLoss_UserRIS, PathLoss_RISBS, Num_User, Tx_antBS, RIS_Lnum, Tx_antRIS, Power_PU, power_SU, password)
        if( Temp_SINR > best):
            best = Temp_SINR
            best_word = password
        # print("Obj : ", best, "\n Password : ", best_word)
        return best, best_word

    for i in range(-2,2):
    # while( temp_i < 5)
        
        if (depth < Tx_antRIS):
            password[depth] = i
            # password[depth] = With_phase_single_quantized(i)
        # idx = 0
        # print("AA in : ", password)
        # idx += 1
        
        bbest, bbest_word = create_withoutRIS(depth + 1, max_depth, password, best, best_word, Noise, P_max, P_k, mu, PathLoss_UserBS, PathLoss_UserRIS, PathLoss_RISBS,
                           Num_User, Tx_antBS, RIS_Lnum, Tx_antRIS, Power_PU, power_SU)
        # print("AA back : ", password, "BB word : ", bbest_word)
        # idx -= 1
    
    return bbest, bbest_word



def find_all_withoutRIS(self: int, digits, best_test_min, best_word, Noise, P_max, P_k, mu, PathLoss_UserBS, PathLoss_UserRIS, PathLoss_RISBS,
                   Num_User, Tx_antBS, RIS_Lnum, Tx_antRIS, Power_PU, power_SU):
    AAA = np.zeros(digits, dtype=np.float128)
    
    A, AA = create_withoutRIS(0, digits, AAA, best_test_min, best_word, Noise, P_max, P_k, mu, PathLoss_UserBS, PathLoss_UserRIS, PathLoss_RISBS,
                       Num_User, Tx_antBS, RIS_Lnum, Tx_antRIS, Power_PU, power_SU)

    return A, AA
        
    
    
    
