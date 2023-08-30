#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import sys
import numpy as np
import math as ma
import time

import scipy.io as sio 
import os
import json

# from sklearn import preprocessing, discriminant_analysis, linear_model, svm
import joblib
# from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, roc_curve, auc 

from EE_RIS_BB import RIS_Bonmin_BB
from EE_RIS_OA import RIS_Bonmin_OA

from EE_RIS_GBD_parameters import userpara
# import EE_RIS_GBD_Single_Multi as gbd_realMC
import EE_RIS_GBD_Single_Multi_v2 as gbd_realMC
# import EE_RIS_GBD_Single_Multi_v2_pro as gbd_realMC
from EE_main_train_run_v2 import gbd_train_run

import EE_without_RIS_v2 as WithoutRIS
import Without_RIS_SINR as WithoutRIS_SINR
import EE_with_RIS_ipopt as WithRIS_ipopt
# from EE_RIS_SCA import RIS_SCAIPOPT_run


#-----------------input parameters------------------#

Num_User = 2                                #-----------Number of user(or receiver)-------------#
RIS_Lnum = 1                                #-----------Number of RIS-------------#
Tx_antBS = 8                                #-----------Number of BS antena-------------#
Tx_antRIS = 32                              #-----------Number of RIS element-------------#

# power_P0 = 24                               #-----------BS(primary transmitter) tansmitted power (dbm)-------------#
# power_P0 =pow(10, (power_P0/10))/pow(10,3)  #-----------BS(primary transmitter) tansmitted power (watts)-------------#
power_Sini = 0
PU_SINR_min = 10
numsol = 2                                  #-----------number of cut-------------#
threshold = 1                               #--------------classified model threshold----------------#
data_num = 1
seed = 15

BS_antenna_gain = 5
RIS_antenna_gain = 5

xonoff = np.ones([RIS_Lnum,1])
thetamarini = np.ones([Tx_antRIS,1], dtype = 'complex_')

#-----------------generate other parameters------------------#

[Bandwidth0, Noise0, P_max0, P_k0, mu0, PathLoss_UserBS0, PathLoss_UserRIS0, PathLoss_RISBS0]\
    = userpara(Num_User, Tx_antBS, RIS_Lnum, Tx_antRIS, BS_antenna_gain, RIS_antenna_gain, seed)

#*****************************************************************Data correction test**********************************************************************#

# ===================================================================Without RIS======================================================================= #

power_withoutRIS_ini = 0.0005

power_PU = 40
power_PU =pow(10, (power_PU/10))/pow(10,3)

power_SU = 47
# power_SU =pow(10, (power_PU/10))/pow(10,3)

# WithoutRIS_powerSU, primal_prob_status, prob_withous_obj, PP_solvertime, PP_time = WithoutRIS.Without_RIS_ipopt(Bandwidth0, Noise0, P_max0, P_k0, mu0,
#                 PathLoss_UserBS0, PathLoss_UserRIS0, PathLoss_RISBS0, Num_User, Tx_antBS, RIS_Lnum, Tx_antRIS, power_PU, PU_SINR_min, power_withoutRIS_ini)

# print("Without Obj : ", prob_withous_obj, "\nWithout power", WithoutRIS_powerSU)



# WithoutRIS_SINR_primary, WithoutRIS_power_primary, Total_secondary_SINR_without = WithoutRIS_SINR.Without_RIS_exh(Bandwidth0, Noise0, P_max0, P_k0, mu0,
#                 PathLoss_UserBS0, PathLoss_UserRIS0, PathLoss_RISBS0, Num_User, Tx_antBS, RIS_Lnum, Tx_antRIS, power_PU)

# print("Without SINR Obj : ", WithoutRIS_SINR_primary, "\nWithout power : ", WithoutRIS_power_primary)

Total_secondary_SINR_with = np.zeros([31,1])

# for idx in range(Tx_antRIS):
#     RIS_phase[idx] = np.random.randint(-2, 1)

# WithoutRIS_SINR_primary, WithoutRIS_power_primary = WithoutRIS_SINR.SINR_withRIS(Bandwidth0, Noise0, P_max0, P_k0, mu0,
#                 PathLoss_UserBS0, PathLoss_UserRIS0, PathLoss_RISBS0, Num_User, Tx_antBS, RIS_Lnum, Tx_antRIS, power_PU, RIS_phase)

# digits = Tx_antRIS
# best_test_min = -100
# vector_varibles = np.zeros(digits, dtype=np.float128)

# for idx_supow in range(20, 51):

#     Best_SINR_withRIS, Best_power_withRIS = WithoutRIS_SINR.find_all_withoutRIS(0, digits, best_test_min, vector_varibles, Noise0, P_max0, P_k0, mu0, PathLoss_UserBS0, 
#                         PathLoss_UserRIS0, PathLoss_RISBS0, Num_User, Tx_antBS, RIS_Lnum, Tx_antRIS, power_PU, idx_supow)
     
#     Total_secondary_SINR_with[idx_supow - 20] = Best_SINR_withRIS

# print("With RIS SINR : ", Best_SINR_withRIS, "\nWith RIS power : ", Best_power_withRIS)

# ===================================================================BB======================================================================= #

# idx_b = 33
# power_P0 = idx_b
# power_P0 =pow(10, (power_P0/10))/pow(10,3)

# BB_power_ini = 0.191             #-------------power variable initial point---------------#
# BB_thereal_ini = 3.45             #-----------theta real variable initial point------------#
# BB_theimag_ini = 0             #-----------theta imag variable initial point------------#
# BB_onoff_ini = 0                   #--------------onoff variable initial point--------------#


# BB_time_start =time.process_time() #----------------BB start time----------------------#

# #-----------------BB solve problem------------------#

# BB_status, BB_objective, model_z_fun_BB, BB_solver_timeuse= \
#     RIS_Bonmin_BB(Bandwidth0, Noise0, P_max0, P_k0, mu0, PathLoss_UserBS0, PathLoss_UserRIS0, PathLoss_RISBS0,
#                     Num_User,Tx_antBS,RIS_Lnum,Tx_antRIS, power_P0, PU_SINR_min, BB_thereal_ini, BB_theimag_ini, BB_power_ini, BB_onoff_ini)


# BB_time_use =time.process_time()   #----------------BB over time----------------------#

# print("BB fun obj: ", BB_objective)
# print("BB fun status: ", BB_status)
# print("BB fun powerSU: ", model_z_fun_BB)
# print("BB fun time: ", BB_solver_timeuse + (BB_time_use - BB_time_start))

# ===================================================================Multi GBD======================================================================= #

# idx_p1 = 23

# power_P0 = idx_p1
# power_P0 =pow(10, (power_P0/10))/pow(10,3)

# GBD_power_ini = 3.5                 #-------------power variable initial point---------------#
# GBD_thereal_ini = 3.5                 #-----------theta real variable initial point------------#
# GBD_theimag_ini = 0                 #-----------theta imag variable initial point------------#

# GBD_time_start_re =time.process_time()  #----------------multi or single GBD start time----------------------#

# #-----------------Multi or Single cut GBD solve problem------------------#

# fin_upper_bound_re, fin_lower_bound_re, power_SUopt_re, theta_opt_re, rho_opt_re, rho_re, convergence_flag_re, iter_num_re, UB_iter_store_re,\
#     LB_iter_store_re, PP_total_time_gbd= \
#   gbd_realMC.gbd_remulti_run(Tx_antRIS, Tx_antBS, RIS_Lnum, Num_User, power_P0, Bandwidth0, Noise0, P_max0, P_k0, mu0, PathLoss_UserBS0,
#             PathLoss_UserRIS0, PathLoss_RISBS0, PU_SINR_min, numsol, GBD_thereal_ini, GBD_theimag_ini, GBD_power_ini, data_num)
    
  
# GBD_time_use_re = time.process_time()  #----------------multi or single GBD over time----------------------#

# print("fin_SINRP_GBD", (fin_upper_bound_re + fin_lower_bound_re)/2)
# # # print("GBD upper bound", fin_upper_bound_re)
# # # print("GBD lower bound", fin_lower_bound_re)
# # # # print("GBD power SU opt: ", power_SUopt_re)
# # # # print("GBD on off opt: ", rho_opt_re)
# # # # print("GBD PP time use: ", PP_total_time_gbd)
# print("GBD without solver: ", GBD_time_use_re - GBD_time_start_re)
# # print("GBD time use: ", (GBD_time_use_re - GBD_time_start_re) + PP_total_time_gbd)
# # print("GBD iter num: ", iter_num_re)

#=================================================================GBD V2=======================================================================#

# idx_p1 = 35

# power_P0 = idx_p1
# power_P0 =pow(10, (power_P0/10))/pow(10,3)

# GBD_power_ini = 0.32               #-------------power variable initial point---------------#
# GBD_thereal_ini = 3.              #-----------theta real variable initial point------------#
# GBD_theimag_ini = 0                 #-----------theta imag variable initial point------------#

# GBD_time_start_re =time.process_time()  #----------------multi or single GBD start time----------------------#

# #-----------------Multi or Single cut GBD solve problem------------------#

# fin_upper_bound_re, fin_lower_bound_re, power_SUopt_re, theta_opt_re, rho_opt_re, rho_re, convergence_flag_re, iter_num_re, UB_iter_store_re,\
#     LB_iter_store_re, PP_total_time_gbd, Calculation_time_gbd= \
#   gbd_realMC.gbd_remulti_run(Tx_antRIS, Tx_antBS, RIS_Lnum, Num_User, power_P0, Bandwidth0, Noise0, P_max0, P_k0, mu0, PathLoss_UserBS0,
#             PathLoss_UserRIS0, PathLoss_RISBS0, PU_SINR_min, numsol, GBD_thereal_ini, GBD_theimag_ini, GBD_power_ini, data_num)
    
  
# GBD_time_use_re = time.process_time()  #----------------multi or single GBD over time----------------------#

# print("fin_SINRP_GBD", (fin_upper_bound_re + fin_lower_bound_re)/2)
# # # # print("GBD upper bound", fin_upper_bound_re)
# # # # print("GBD lower bound", fin_lower_bound_re)
# print("GBD power SU opt: ", power_SUopt_re)
# # # # print("GBD on off opt: ", rho_opt_re)
# # # # # print("GBD PP time use: ", PP_total_time_gbd)
# print("GBD without solver: ", GBD_time_use_re - GBD_time_start_re)
# print("GBD Calcution time", Calculation_time_gbd)
# print("GBD on off opt: ", rho_opt_re)


#===================================================================Acc GBD=======================================================================#

# scaler_filename = 'model_1/scaler_' + str(Num_User) + '_' +str(RIS_Lnum) + '.save'   #----------------scaler filename---------------------#
# model_path_svm = 'model_1/model_svm_'+str(Num_User) + '_' +str(RIS_Lnum) + '.m'      #----------------model filename---------------------#

# scaler = joblib.load(scaler_filename)  #----------------load scaler----------------------#
# clf = joblib.load(model_path_svm)      #----------------load classified model----------------------#
# model = clf                            #----------------classified model----------------------#

# idx_p=230

# power_P0 = idx_p
# power_P0 =pow(10, (power_P0/10))/pow(10,3)

# GBD_power_ini = 0.015                 #-------------power variable initial point---------------#
# GBD_thereal_ini = 2.95             #-----------theta real variable initial point------------#
# GBD_theimag_ini = 0                 #-----------theta imag variable initial point------------#

# acc_gbd_start = time.process_time()   #----------------acc GBD start time----------------------#

# #-----------------Multi or Single cut GBD solve problem------------------#

# fin_upper_bound, fin_lower_bound, power_SUopt, theta_opt, rho_opt, rho, convergence_flag, iter_num, UB_iter_store, LB_iter_store, \
#     PP_total_time_acc, ML_total_time_acc, Acc_calculation_time = \
#     gbd_train_run(Tx_antRIS, Tx_antBS, RIS_Lnum, Num_User, power_P0, Bandwidth0, Noise0, P_max0, P_k0, mu0, PathLoss_UserBS0, PathLoss_UserRIS0,
#                   PathLoss_RISBS0, PU_SINR_min, numsol, data_num, data_num, model, 'test', threshold, scaler, GBD_thereal_ini, GBD_theimag_ini, GBD_power_ini)

# acc_gbd_use = time.process_time()     #----------------acc GBD over time----------------------#

# print("final ans: ", (fin_upper_bound + fin_lower_bound)/2)
# # # # print("PP time use: ",  PP_total_time_acc)
# # # # print("classifi time: ", ML_total_time_acc)
# # # # print("acc GBD time: ", acc_gbd_use - acc_gbd_start)
# print("acc GBD time class: ", (acc_gbd_use - acc_gbd_start) - ML_total_time_acc)
# # print("acc GBD time use: ", (acc_gbd_use - acc_gbd_start) + PP_total_time_acc)
# print("acc GBD time total: ", (acc_gbd_use - acc_gbd_start) + PP_total_time_acc - ML_total_time_acc)
# print("onoff: ", rho_opt)
# print("power opt: ", power_SUopt)
# print("acc GBD iter: ", iter_num)
# print("acc GBD cal time: ", Acc_calculation_time - ML_total_time_acc)


#===================================================================OA=======================================================================#

# idx_oa = 40
# power_P0 = idx_oa
# power_P0 =pow(10, (power_P0/10))/pow(10,3)

# OA_power_ini = 1.0                 #-------------power variable initial point---------------#
# OA_thereal_ini = 3.7           #-----------theta real variable initial point------------#
# OA_theimag_ini = 0               #-----------theta imag variable initial point------------#
# OA_onoff_ini = 0                   #--------------onoff variable initial point--------------#

# OA_time_start =time.process_time() #----------------OA start time----------------------#

# #-----------------OA solve problem------------------#

# OA_status, OA_objective, model_z_fun_OA, OA_solver_timeuse = \
#     RIS_Bonmin_OA(Bandwidth0, Noise0, P_max0, P_k0, mu0, PathLoss_UserBS0, PathLoss_UserRIS0, PathLoss_RISBS0,
#                         Num_User,Tx_antBS,RIS_Lnum,Tx_antRIS, power_P0, PU_SINR_min, OA_thereal_ini, OA_theimag_ini, OA_power_ini, OA_onoff_ini)



# OA_time_use =time.process_time()   #----------------OA over time----------------------#


# print("OA fun obj: ", OA_objective)
# print("OA fun status: ", OA_status)
# print("OA fun powerSU: ", model_z_fun_OA)
# print("OA fun time: ", OA_solver_timeuse + (OA_time_use - OA_time_start))

#==========================================================Interior Optimizer=========================================================================#


idx_op1 = 40

power_P0 = idx_op1
power_P0 =pow(10, (power_P0/10))/pow(10,3)

GBD_power_ini = 34               #-------------power variable initial point---------------#
GBD_thereal_ini = 0.2              #-----------theta real variable initial point------------#
GBD_theimag_ini = 0                 #-----------theta imag variable initial point------------#


With_optimizer_status, With_optimizer_obj = WithRIS_ipopt.With_RIS_ipopt(Bandwidth0, Noise0, P_max0, P_k0, mu0, PathLoss_UserBS0, PathLoss_UserRIS0, 
                PathLoss_RISBS0, xonoff, Num_User, Tx_antBS, RIS_Lnum, Tx_antRIS, power_P0, PU_SINR_min, 
                   GBD_thereal_ini, GBD_power_ini)

print("With RIS optimizer Obj: ", With_optimizer_obj)

#==========================================================SCA=========================================================================#

# SCA_results = np.zeros([21, 1])

# # for idx_sca in range(20, 41):
# SINRP_diff = 0
# SINRP_last = 0
# SCA_iter = 0
# Max_iter = 1

# idx_sca = 23

# power_P0 = idx_sca
# power_P0 =pow(10, (power_P0/10))/pow(10,3)

# SCA_time_start_on = time.process_time()

# xonoff = np.ones([RIS_Lnum,1])

# #------------------SCA solve problem------------------#

# while (SCA_iter < Max_iter):
    
    
#     SINRP_SCA_last, theta_SCA_SINPP_ipopt_last, SINRP_SCA_power_SU_ipopt_last, SCA_status_last, SCA_solver_timeuse_last\
#           = RIS_SCAIPOPT_run(thetamarini, xonoff, Bandwidth0, Noise0, P_max0, P_k0, mu0, PathLoss_UserBS0,
#                           PathLoss_UserRIS0, PathLoss_RISBS0, Num_User, Tx_antBS, RIS_Lnum, Tx_antRIS, power_P0, PU_SINR_min)
          
#     SINRP_diff = SINRP_SCA_last - SINRP_last
#     if(abs(SINRP_diff)/SINRP_last < 1e-4):
#         break
    
#     thetamarini = theta_SCA_SINPP_ipopt_last
    
#     SINRP_last = SINRP_SCA_last
#     SCA_iter += 1

# SCA_results[idx_sca - 20] = SINRP_SCA_last

# SCA_time_use_on = time.process_time()

# print("SCA iter num : ", SCA_iter)
# print("SCA status on: ", SCA_status_last)
# print("SINRP obj: ", SINRP_SCA_last)
# print("SCA time use on: ", (SCA_time_use_on - SCA_time_start_on) + SCA_solver_timeuse_last)
# print("SINRP power SU", SINRP_SCA_power_SU_ipopt_last)
# print("SCA results :", SCA_results)







#**********************************************************************************************************************************************************#

#-----------------load ML model file------------------#

# scaler_filename = 'model_1/scaler_' + str(Num_User) + '_' +str(RIS_Lnum) + '.save'   #----------------scaler filename---------------------#
# model_path_svm = 'model_1/model_svm_'+str(Num_User) + '_' +str(RIS_Lnum) + '.m'      #----------------model filename---------------------#

# scaler = joblib.load(scaler_filename)  #----------------load scaler----------------------#
# clf = joblib.load(model_path_svm)      #----------------load classified model----------------------#
# model = clf                            #----------------classified model----------------------#


#------------------------------------------------------------Run Accelerating GBD--------------------------------------------------------------------------#

# acc_GBD_results = np.zeros([21, 1])

# GBD_ini_point = np.array([[0.1, 3, 0], [0.1, 2.9, 0], [0.1, 2.9, 0], [0.1, 3.3, 0], [0.1, 3.9, 0], [0.1, 2.9, 0], [0.04, 3.2, 0], \
#                           [0.1, 3.7, 0], [0.1, 3.6, 0], [0.1, 3.3, 0], [0.1, 3.3, 0], [0.1, 2.9, 0], [0.1, 2.8, 0], [0.8, 1.2, 0], \
#                               [0.1, 3, 0], [0.1, 3.4, 0], [0.1, 2.7, 0], [0.1, 3.6, 0], [0.1, 3.6, 0], [0.8, 0.6, 0], [0.1, 2.7, 0] ] )

# for idx_p in range(20, 41):
    
#     power_P0 = idx_p
#     power_P0 =pow(10, (power_P0/10))/pow(10,3)
    
#     GBD_power_ini = GBD_ini_point[idx_p - 20][0]                   #-------------power variable initial point---------------#
#     GBD_thereal_ini = GBD_ini_point[idx_p - 20][1]                 #-----------theta real variable initial point------------#
#     GBD_theimag_ini = GBD_ini_point[idx_p - 20][2]                 #-----------theta imag variable initial point------------#
    
#     acc_gbd_start = time.process_time()   #----------------acc GBD start time----------------------#
    
#     #-----------------Multi or Single cut GBD solve problem------------------#
    
#     fin_upper_bound, fin_lower_bound, power_SUopt, theta_opt, rho_opt, rho, convergence_flag, iter_num, UB_iter_store, LB_iter_store, PP_total_time_acc, ML_total_time_acc = \
#         gbd_train_run(Tx_antRIS, Tx_antBS, RIS_Lnum, Num_User, power_P0, Bandwidth0, Noise0, P_max0, P_k0, mu0, PathLoss_UserBS0, PathLoss_UserRIS0,
#                       PathLoss_RISBS0, PU_SINR_min, numsol, data_num, data_num, model, 'test', threshold, scaler, GBD_thereal_ini, GBD_theimag_ini, GBD_power_ini)
    
#     acc_GBD_results[idx_p - 20] = (fin_upper_bound + fin_lower_bound)/2
    
#     acc_gbd_use = time.process_time()     #----------------acc GBD over time----------------------#

# # print("final ans: ", (fin_upper_bound + fin_lower_bound)/2)
# # print("PP time use: ",  PP_total_time_acc)
# # print("classifi time: ", ML_total_time_acc)
# # print("acc GBD time: ", acc_gbd_use - acc_gbd_start)
# # print("acc GBD time class: ", (acc_gbd_use - acc_gbd_start) - ML_total_time_acc)
# # print("acc GBD time use: ", (acc_gbd_use - acc_gbd_start) + PP_total_time_acc)
# # print("acc GBD time total: ", (acc_gbd_use - acc_gbd_start) + PP_total_time_acc - ML_total_time_acc)
# print("onoff: ", rho_opt)
# # print("power opt: ", power_SUopt)
# # print("acc GBD iter: ", iter_num)
# print("acc_GBD :", acc_GBD_results)
# print("phase shift:", theta_opt)

#-----------------------------------------------------------------Multi-cut or Single-cut GBD--------------------------------------------------------------#

# GBD_results = np.zeros([21, 1])

# GBD_ini_point = np.array([[0.1, 3, 0], [0.1, 2.9, 0], [0.1, 2.9, 0], [0.1, 3.3, 0], [0.1, 3.9, 0], [0.1, 2.9, 0], [0.04, 3.2, 0], \
#                           [0.1, 3.7, 0], [0.1, 3.6, 0], [0.1, 3.3, 0], [0.1, 3.3, 0], [0.1, 2.9, 0], [0.1, 2.8, 0], [0.8, 1.2, 0], \
#                               [0.1, 3, 0], [0.1, 3.4, 0], [0.1, 2.7, 0], [0.1, 3.6, 0], [0.1, 3.6, 0], [0.8, 0.6, 0], [0.1, 2.7, 0] ] )

# for idx_p1 in range(20, 41):
    
#     power_P0 = idx_p1
#     power_P0 =pow(10, (power_P0/10))/pow(10,3)
    
#     GBD_power_ini = GBD_ini_point[idx_p1 - 20][0]                   #-------------power variable initial point---------------#
#     GBD_thereal_ini = GBD_ini_point[idx_p1 - 20][1]                 #-----------theta real variable initial point------------#
#     GBD_theimag_ini = GBD_ini_point[idx_p1 - 20][2]                 #-----------theta imag variable initial point------------#
    
#     GBD_time_start_re =time.process_time()  #----------------multi or single GBD start time----------------------#
    
#     #-----------------Multi or Single cut GBD solve problem------------------#
    
#     fin_upper_bound_re, fin_lower_bound_re, power_SUopt_re, theta_opt_re, rho_opt_re, rho_re, convergence_flag_re, iter_num_re, UB_iter_store_re,\
#         LB_iter_store_re, PP_total_time_gbd= \
#       gbd_realMC.gbd_remulti_run(Tx_antRIS, Tx_antBS, RIS_Lnum, Num_User, power_P0, Bandwidth0, Noise0, P_max0, P_k0, mu0, PathLoss_UserBS0,
#                 PathLoss_UserRIS0, PathLoss_RISBS0, PU_SINR_min, numsol, GBD_thereal_ini, GBD_theimag_ini, GBD_power_ini, data_num)
      
#     GBD_results[idx_p1 - 20] = (fin_upper_bound_re + fin_lower_bound_re)/2   
      
#     GBD_time_use_re = time.process_time()  #----------------multi or single GBD over time----------------------#

# # print("fin_SINRP_GBD", (fin_upper_bound_re + fin_lower_bound_re)/2)
# # print("GBD power SU opt: ", power_SUopt_re)
# # print("GBD on off opt: ", rho_opt_re)
# # print("GBD PP time use: ", PP_total_time_gbd)
# # print("GBD without solver: ", GBD_time_use_re - GBD_time_start_re)
# # print("GBD time use: ", (GBD_time_use_re - GBD_time_start_re) + PP_total_time_gbd)
# # print("GBD iter num: ", iter_num_re)
# # print(UB_iter_store_re)
# # print(LB_iter_store_re)
# print("GBD results :", GBD_results)



#---------------------------------------------------------------Branch and Bound----------------------------------------------------------------------------#

# BB_results = np.zeros([21, 1])

# BB_ini_point = np.array([[0.1, 3.3, 0], [0.1, 3.65, 0], [0.1, 2.9, 0], [0.018, 3.65, 0], [0.1, 3.6, 0], [0.1, 3.6, 0], [0.1, 3.7, 0], \
#                           [0.1, 3.5, 0], [0.1, 3.4, 0], [0.1, 2.6, 0], [0.1, 2.9, 0], [0.1, 3.3, 0], [0.1, 2.7, 0], [0.001, 0.7, 0], \
#                               [0.1, 2.7, 0], [0.1, 2.9, 0], [0.1, 3.3, 0], [0.1, 3, 0], [0.1, 3.2, 0], [0.1, 3.2, 0], [0.1, 3.2, 0]])

# for idx_b in range(20, 41):
    
#     power_P0 = idx_b
#     power_P0 =pow(10, (power_P0/10))/pow(10,3)
    
#     BB_power_ini = BB_ini_point[idx_b - 20][0]                #-------------power variable initial point---------------#
#     BB_thereal_ini = BB_ini_point[idx_b - 20][1]              #-----------theta real variable initial point------------#
#     BB_theimag_ini = BB_ini_point[idx_b - 20][2]              #-----------theta imag variable initial point------------#
#     BB_onoff_ini = 0                   #--------------onoff variable initial point--------------#
    
    
#     BB_time_start =time.process_time() #----------------BB start time----------------------#
    
#     #-----------------BB solve problem------------------#
    
#     BB_status, BB_objective, model_z_fun_BB, BB_solver_timeuse= \
#         RIS_Bonmin_BB(Bandwidth0, Noise0, P_max0, P_k0, mu0, PathLoss_UserBS0, PathLoss_UserRIS0, PathLoss_RISBS0,
#                         Num_User,Tx_antBS,RIS_Lnum,Tx_antRIS, power_P0, PU_SINR_min, BB_thereal_ini, BB_theimag_ini, BB_power_ini, BB_onoff_ini)
        
#     BB_results[idx_b - 20] = BB_objective
    
    
#     BB_time_use =time.process_time()   #----------------BB over time----------------------#

# # print("BB fun obj: ", BB_objective)
# # print("BB fun status: ", BB_status)
# # print("BB fun powerSU: ", model_z_fun_BB)
# # print("BB fun time: ", BB_solver_timeuse + (BB_time_use - BB_time_start))
# print(BB_results)
# # print("phase shift BB", phase_shift_BB)


#--------------------------------------------------------------Outer approximation--------------------------------------------------------------------------#

# OA_results = np.zeros([21,1])

# OA_ini_point = np.array([[0.1, 3.4, 0], [0.1, 3.65, 0], [0.1, 2.9, 0], [0.018, 3.65, 0], [0.1, 3.5, 0], [0.1, 3.4, 0], [0.1, 3.7, 0], \
#                   [0.1, 3.6, 0], [0.1, 3.4, 0], [0.1, 2.6, 0], [0.1, 2.6, 0], [0.1, 3.3, 0], [0.1, 3.4, 0], [1.1, 1.1, 0], \
#                   [0.1, 3.4, 0], [0.1, 3.4, 0], [0.1, 3.3, 0], [0.1, 3, 0], [0.1, 3.2, 0], [0.1, 3.2, 0], [0.1, 3.2, 0]])

# for idx_oa in range(20, 41):
    
#     power_P0 = idx_oa
#     power_P0 =pow(10, (power_P0/10))/pow(10,3)
    
#     OA_power_ini = OA_ini_point[idx_oa - 20][0]                 #-------------power variable initial point---------------#
#     OA_thereal_ini = OA_ini_point[idx_oa - 20][1]              #-----------theta real variable initial point------------#
#     OA_theimag_ini = OA_ini_point[idx_oa - 20][2]               #-----------theta imag variable initial point------------#
#     OA_onoff_ini = 0                   #--------------onoff variable initial point--------------#
    
#     OA_time_start =time.process_time() #----------------OA start time----------------------#
    
#     #-----------------OA solve problem------------------#
    
#     OA_status, OA_objective, model_z_fun_OA, OA_solver_timeuse = \
#         RIS_Bonmin_OA(Bandwidth0, Noise0, P_max0, P_k0, mu0, PathLoss_UserBS0, PathLoss_UserRIS0, PathLoss_RISBS0,
#                             Num_User,Tx_antBS,RIS_Lnum,Tx_antRIS, power_P0, PU_SINR_min, OA_thereal_ini, OA_theimag_ini, OA_power_ini, OA_onoff_ini)
    
#     OA_results[idx_oa - 20] = OA_objective
    
#     OA_time_use =time.process_time()   #----------------OA over time----------------------#


# print("OA fun obj: ", OA_objective)
# print("OA fun status: ", OA_status)
# print("OA fun powerSU: ", model_z_fun_OA)
# print("OA fun time: ", OA_solver_timeuse + (OA_time_use - OA_time_start))
# print("OA result: ", OA_results)

#---------------------------------------------------------Successive convex approximation-------------------------------------------------------------------#

# SCA_results = np.zeros([21, 1])

# for idx_sca in range(20, 41):
    
#     power_P0 = idx_sca
#     power_P0 =pow(10, (power_P0/10))/pow(10,3)

#     SCA_time_start_on = time.process_time()
    
#     xonoff = np.ones([RIS_Lnum,1])
    
#     #------------------SCA solve problem------------------#
      
#     SINRP_SCA_last, theta_SCA_SINPP_ipopt_last, SINRP_SCA_power_SU_ipopt_last, SCA_status_last, SCA_solver_timeuse_last\
#           = RIS_SCAIPOPT_run(thetamarini, xonoff, Bandwidth0, Noise0, P_max0, P_k0, mu0, PathLoss_UserBS0,
#                           PathLoss_UserRIS0, PathLoss_RISBS0, Num_User, Tx_antBS, RIS_Lnum, Tx_antRIS, power_P0, PU_SINR_min)
    
#     SCA_results[idx_sca - 20] = SINRP_SCA_last
    
#     SCA_time_use_on = time.process_time()


# # print("SCA status on: ", SCA_status_last)
# # print("SINRP obj: ", SINRP_SCA_last)
# # print("SCA time use on: ", (SCA_time_use_on - SCA_time_start_on) + SCA_solver_timeuse_last)
# # print("SINRP power SU", SINRP_SCA_power_SU_ipopt_last)
# print("SCA results :", SCA_results)



#-----------------------------------------------------------------random user location--------------------------------------------------#

power_P0  = 23
power_P0 =pow(10, (power_P0/10))/pow(10,3)

rand_user_seed = [21, 26, 35, 41, 49, 57, 40, 58, 65, 66]


#------------------SCA solve problem------------------#

# SCA_rand_results = np.zeros([10, 1])

# for idx_sca in range(10):


#     [Bandwidth0, Noise0, P_max0, P_k0, mu0, PathLoss_UserBS0, PathLoss_UserRIS0, PathLoss_RISBS0]\
#         =userpara(Num_User, Tx_antBS, RIS_Lnum, Tx_antRIS, rand_user_seed[idx_sca])
    
#     SCA_time_start_on = time.process_time()
    
#     xonoff = np.ones([RIS_Lnum,1])
    
    
      
#     SINRP_SCA_last, theta_SCA_SINPP_ipopt_last, SINRP_SCA_power_SU_ipopt_last, SCA_status_last, SCA_solver_timeuse_last\
#           = RIS_SCAIPOPT_run(thetamarini, xonoff, Bandwidth0, Noise0, P_max0, P_k0, mu0, PathLoss_UserBS0,
#                           PathLoss_UserRIS0, PathLoss_RISBS0, Num_User, Tx_antBS, RIS_Lnum, Tx_antRIS, power_P0, PU_SINR_min)
    
    
#     SCA_time_use_on = time.process_time()
    
#     SCA_rand_results[idx_sca] = SINRP_SCA_last

# print("SCA status on: ", SCA_status_last)
# print("SINRP obj: ", SINRP_SCA_last)
# print("SCA time use on: ", (SCA_time_use_on - SCA_time_start_on) + SCA_solver_timeuse_last)
# print("SINRP power SU", SINRP_SCA_power_SU_ipopt_last)
# print(SCA_rand_results)

#------------------GBD solve problem------------------#



# GBD_rand_results = np.zeros([10, 1])

# GBD_user_inipoint = np.array([[3, 3, 0], [3.5, 3.5, 0], [1.1, 1.1, 0], [0.7, 0.7, 0], [1.6, 1.6, 0], [3.3, 3.3, 0], [3.5, 3.5, 0], \
#                         [3.4, 3.4, 0], [0.1, 3.3, 0], [3.3, 3.3, 0]])

# for idx_gbd in range(10):
    
#     print("-------idx----------", idx_gbd)
    
#     [Bandwidth0, Noise0, P_max0, P_k0, mu0, PathLoss_UserBS0, PathLoss_UserRIS0, PathLoss_RISBS0]\
#         =userpara(Num_User, Tx_antBS, RIS_Lnum, Tx_antRIS, rand_user_seed[idx_gbd])
    
#     GBD_power_ini = GBD_user_inipoint[idx_gbd][0]                   #-------------power variable initial point---------------#
#     GBD_thereal_ini = GBD_user_inipoint[idx_gbd][1]                #-----------theta real variable initial point------------#
#     GBD_theimag_ini = GBD_user_inipoint[idx_gbd][2]                #-----------theta imag variable initial point------------#
    
#     GBD_time_start_re =time.process_time()  #----------------multi or single GBD start time----------------------#
    
#     #-----------------Multi or Single cut GBD solve problem------------------#
    
#     fin_upper_bound_rand, fin_lower_bound_rand, power_SUopt_rand, theta_opt_rand, rho_opt_rand, rho_rand, convergence_flag_rand, iter_num_rand, UB_iter_store_rand,\
#         LB_iter_store_rand, PP_total_time_gbd= \
#       gbd_realMC.gbd_remulti_run(Tx_antRIS, Tx_antBS, RIS_Lnum, Num_User, power_P0, Bandwidth0, Noise0, P_max0, P_k0, mu0, PathLoss_UserBS0,
#                 PathLoss_UserRIS0, PathLoss_RISBS0, PU_SINR_min, numsol, GBD_thereal_ini, GBD_theimag_ini, GBD_power_ini, data_num)
        
      
#     GBD_time_use_re = time.process_time()

#     GBD_rand_results[idx_gbd] = (fin_upper_bound_rand + fin_lower_bound_rand)/2

# # print("fin_SINRP_GBD", (fin_upper_bound_re + fin_lower_bound_re)/2)
# # print("GBD on off opt: ", rho_opt_re)
# # print("GBD without solver: ", GBD_time_use_re - GBD_time_start_re)
# print(GBD_rand_results)


#------------------Acc GBD solve problem------------------#

# Acc_rand_results = np.zeros([10, 1])

# Acc_rand_inipoints = np.array([[3, 3, 0], [3.5, 3.5, 0], [1.1, 1.1, 0], [0.7, 0.7, 0], [1.6, 1.6, 0], [3.3, 3.3, 0], [3.5, 3.5, 0], \
#                         [3.4, 3.4, 0], [0.1, 3.3, 0], [3.3, 3.3, 0]])


#-----------------load ML model file------------------#

# scaler_filename = 'model_1/scaler_' + str(Num_User) + '_' +str(RIS_Lnum) + '.save'   #----------------scaler filename---------------------#
# model_path_svm = 'model_1/model_svm_'+str(Num_User) + '_' +str(RIS_Lnum) + '.m'      #----------------model filename---------------------#

# scaler = joblib.load(scaler_filename)  #----------------load scaler----------------------#
# clf = joblib.load(model_path_svm)      #----------------load classified model----------------------#
# model = clf                            #----------------classified model----------------------#

#------------------Run Acc GBD ------------------#

# for idx_acc in range(10):
    
#     [Bandwidth0, Noise0, P_max0, P_k0, mu0, PathLoss_UserBS0, PathLoss_UserRIS0, PathLoss_RISBS0]\
#         =userpara(Num_User, Tx_antBS, RIS_Lnum, Tx_antRIS, rand_user_seed[idx_acc])
    
#     GBD_power_ini = Acc_rand_inipoints[idx_acc][0]                   #-------------power variable initial point---------------#
#     GBD_thereal_ini = Acc_rand_inipoints[idx_acc][1]                 #-----------theta real variable initial point------------#
#     GBD_theimag_ini = Acc_rand_inipoints[idx_acc][2]                 #-----------theta imag variable initial point------------#
    
#     acc_gbd_start = time.process_time()   #----------------acc GBD start time----------------------#
    
#     #-----------------Multi or Single cut GBD solve problem------------------#
    
#     fin_upper_bound, fin_lower_bound, power_SUopt, theta_opt, rho_opt, rho, convergence_flag, iter_num, UB_iter_store, LB_iter_store, PP_total_time_acc, ML_total_time_acc = \
#         gbd_train_run(Tx_antRIS, Tx_antBS, RIS_Lnum, Num_User, power_P0, Bandwidth0, Noise0, P_max0, P_k0, mu0, PathLoss_UserBS0, PathLoss_UserRIS0,
#                       PathLoss_RISBS0, PU_SINR_min, numsol, data_num, data_num, model, 'test', threshold, scaler, GBD_thereal_ini, GBD_theimag_ini, GBD_power_ini)
    
   
    
#     acc_gbd_use = time.process_time()     #----------------acc GBD over time----------------------#
    
#     Acc_rand_results[idx_acc] = (fin_upper_bound + fin_lower_bound)/2


# print(Acc_rand_results)

#------------------BB solve problem------------------#

# BB_rand_results = np.zeros([10, 1])

# BB_rand_inipoint = ([[3.7, 3.7, 0], [0.1, 3, 0], [3.8, 3.8, 0], [0.001, 0.7, 0], [2.6, 2.6, 0], [2.3, 2.3, 0], [3.1, 3.1, 0], \
#                       [0.1, 3.4, 0], [0.1, 3.3, 0], [0.1, 3.4, 0]])

   

# for idx_bb in range(10):

    
    
#     [Bandwidth0, Noise0, P_max0, P_k0, mu0, PathLoss_UserBS0, PathLoss_UserRIS0, PathLoss_RISBS0]\
#         =userpara(Num_User, Tx_antBS, RIS_Lnum, Tx_antRIS, rand_user_seed[idx_bb])
    
#     BB_power_ini =  BB_rand_inipoint[idx_bb][0]             #-------------power variable initial point---------------#
#     BB_thereal_ini = BB_rand_inipoint[idx_bb][1]            #-----------theta real variable initial point------------#
#     BB_theimag_ini = BB_rand_inipoint[idx_bb][2]            #-----------theta imag variable initial point------------#
#     BB_onoff_ini = 0                                        #--------------onoff variable initial point--------------#
    
    
#     BB_time_start =time.process_time() #----------------BB start time----------------------#
    
#     #-----------------BB solve problem------------------#
    
#     BB_status, BB_objective, model_z_fun_BB, BB_solver_timeuse = \
#         RIS_Bonmin_BB(Bandwidth0, Noise0, P_max0, P_k0, mu0, PathLoss_UserBS0, PathLoss_UserRIS0, PathLoss_RISBS0,
#                         Num_User,Tx_antBS,RIS_Lnum,Tx_antRIS, power_P0, PU_SINR_min, BB_thereal_ini, BB_theimag_ini, BB_power_ini, BB_onoff_ini)
            
#     BB_time_use =time.process_time()   #----------------BB over time----------------------#
    
#     BB_rand_results[idx_bb] = BB_objective
    

# print("BB fun obj: ", BB_objective)
# print("BB fun status: ", BB_status)
# print("BB fun powerSU: ", model_z_fun_BB)
# print("BB fun time: ", BB_solver_timeuse + (BB_time_use - BB_time_start))
# print(BB_rand_results)

#------------------OA solve problem------------------#

# OA_rand_results = np.zeros([10, 1])

# OA_rand_inipoint = ([[0.1, 3.2, 0], [0.1, 3, 0], [0.1, 3, 0], [1.1, 1.1, 0], [0.1, 3.4, 0], [0.1, 3.8, 0], [0.1, 3.8, 0], \
#                       [0.1, 3.4, 0], [0.1, 3.4, 0], [0.1, 3.4, 0]])

# for idx_oa in range(10):

#     [Bandwidth0, Noise0, P_max0, P_k0, mu0, PathLoss_UserBS0, PathLoss_UserRIS0, PathLoss_RISBS0]\
#         =userpara(Num_User, Tx_antBS, RIS_Lnum, Tx_antRIS, rand_user_seed[idx_oa])    

#     OA_power_ini = OA_rand_inipoint[idx_oa][0]                 #-------------power variable initial point---------------#
#     OA_thereal_ini = OA_rand_inipoint[idx_oa][1]              #-----------theta real variable initial point------------#
#     OA_theimag_ini = OA_rand_inipoint[idx_oa][2]               #-----------theta imag variable initial point------------#
#     OA_onoff_ini = 0                   #--------------onoff variable initial point--------------#
    
#     OA_time_start =time.process_time() #----------------OA start time----------------------#
    
#     #-----------------OA solve problem------------------#
    
#     OA_status, OA_objective, model_z_fun_OA, OA_solver_timeuse = \
#         RIS_Bonmin_OA(Bandwidth0, Noise0, P_max0, P_k0, mu0, PathLoss_UserBS0, PathLoss_UserRIS0, PathLoss_RISBS0,
#                             Num_User,Tx_antBS,RIS_Lnum,Tx_antRIS, power_P0, PU_SINR_min, OA_thereal_ini, OA_theimag_ini, OA_power_ini, OA_onoff_ini)
    
    
#     OA_time_use =time.process_time()   #----------------OA over time----------------------#
    
#     OA_rand_results[idx_oa] = OA_objective


# print("OA fun obj: ", OA_objective)
# print("OA fun status: ", OA_status)
# print("OA fun powerSU: ", model_z_fun_OA)
# print("OA fun time: ", OA_solver_timeuse + (OA_time_use - OA_time_start))
# print(OA_rand_results)