#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#-------------------Function for running Accelerating GBD----------------------#

import sys
import numpy as np
import time
#import matlab
#import matlab.engine
# import math as ma

from EE_RIS_GBD_primal_v2 import RIS_ipopt_primal
from EE_RIS_GBD_infeasible import RIS_ipopt_infeasible

import EE_Acc_master_v2 as master_test

# from master_multi import Master_class
# import scipy.io as sio 
# import os
# import json


# eng = matlab.engine.start_matlab()

def gbd_ml_testacc(Bandwidth0, Noise0, P_max0, P_k0, mu0, RIS_Lnum0, Tx_antBS0, Tx_antRIS0, Num_User0, 
              PathLoss_UserBS0, PathLoss_UserRIS0, PathLoss_RISBS0, xonoffini, power_P0, PU_SINR_min0, numsol, 
              model, threshold, p_i, scaler, GBD_thereal_ini0, GBD_theimag_ini0, GBD_power_ini0):

    global RIS_Lnum, Tx_antBS, Tx_antRIS, power_P, PathLoss_RISBS, PathLoss_UserBS, PathLoss_UserRIS, Bandwidth, Noise,\
    P_k, P_max, power_P, mu, PU_SINR_min
    
    
    PathLoss_RISBS = PathLoss_RISBS0
    PathLoss_UserBS = PathLoss_UserBS0
    PathLoss_UserRIS = PathLoss_UserRIS0
    Num_User = Num_User0
    Tx_antBS = Tx_antBS0
    RIS_Lnum = RIS_Lnum0
    Tx_antRIS = Tx_antRIS0
    Bandwidth = Bandwidth0
    Noise = Noise0
    P_k = P_k0
    P_max = P_max0
    power_P = power_P0    
    mu = mu0
    PU_SINR_min = PU_SINR_min0
    GBD_thereal_ini = GBD_thereal_ini0
    GBD_theimag_ini = GBD_theimag_ini0
    GBD_power_ini = GBD_power_ini0
    

    # log_json = []
    
    
    i_gbd = 1           #--------------set iteration index---------------#
    max_iter_num = 1E4  #-----------set maximum iteration num------------#
    
    xonoff = xonoffini.copy()
    
    UB_iter_store = np.zeros([Tx_antRIS, 1], dtype=np.float128)
    LB_iter_store = np.zeros([Tx_antRIS, 1], dtype=np.float128)

    #set global upper bound and lower bound
    upper_bound = sys.maxsize
    lower_bound = -sys.maxsize
    #master calss initialization
    # master_class = Master_class(K0, L0, h_CD0, h_CB0, h_D0, h_DB0, P_max_D0, R_min_C0, P_max_C0, p_max, a, b)
    
    master_class_test = master_test.Master_class(Bandwidth0, Noise0, P_max0, P_k0, mu0, PathLoss_UserBS0,PathLoss_UserRIS0,PathLoss_RISBS0,
                 xonoffini,Num_User0, Tx_antBS0, RIS_Lnum0, Tx_antRIS0, power_P0, PU_SINR_min0, p_i)
    	
    # alpha = 0
    primal_infeasible_alpha = 0
    # infeasible_theta_var = 0
    # infeasible_power_var = 0

    
    reapeat_count = 0        #-----------avoid local minimum---------------#
    
    #-----------optimal results------------#
    
    rho_opt = np.zeros([RIS_Lnum0,1])
    power_SUopt = 0
    theta_opt = np.zeros([Tx_antRIS0,1])
    
    # time_total = 0
    cut_total = 0
    PP_total_time = 0
    ML_total_time = 0
    Calculation_time = 0
    
    one_feasible_flag_multi = np.zeros((numsol,1))
    one_thetavec_PU_SU_primal_multi = np.zeros((numsol,Tx_antRIS,1), dtype = 'complex_')
    one_thetavec_PU_SU_primal_dual_multi = np.zeros((numsol,Tx_antRIS,1), dtype = 'complex_')
    one_thetavec_PU_SU_primal_lowb_dual_multi = np.zeros((numsol,Tx_antRIS,1), dtype = 'complex_')
    one_power_SU_primal_multi = np.zeros((numsol,1))
    one_power_SU_primal_dual_multi = np.zeros((numsol,2,1))
    # PU_SINR_multi = np.zeros((numsol_real,1))
    one_PU_SINR_primal_dual_multi = np.zeros((numsol,1))
    one_PU_signal_primal_dual_multi = np.zeros((numsol,1))
    # one_infeasible_theta_multi = np.zeros((numsol,1))
    one_infeasible_alpha_multi = np.zeros((numsol,1))

    for idx_inisol in range(numsol):
        
    	
        #-------------------------------first iteration-------------------------------#
        #---------------solve the primal problem and get #numsol cuts-----------------#
                
        theta_PU_SU_primal, theta_PU_SU_primal_dual, power_S_opt, power_SU_dual, PU_SINR_dual, PU_signal_dual,\
            prob_exitflag_power, prob_LowerBound, PP_solvertime, PP_use_time\
            = RIS_ipopt_primal(Bandwidth, Noise, P_max, P_k, mu, PathLoss_UserBS, PathLoss_UserRIS, PathLoss_RISBS, xonoff[idx_inisol],
                            Num_User,Tx_antBS,RIS_Lnum,Tx_antRIS, power_P, PU_SINR_min, GBD_thereal_ini, GBD_theimag_ini, GBD_power_ini)
            
        # print("power SU primal: ", power_S_opt)
        # print("theta SCA primal: ", theta_PU_SU_primal)
        # print("primal status: ", prob_exitflag_power)
        # print("idx_inisol", idx_inisol)
        # print("primal LB: ", prob_LowerBound)
        PP_total_time += PP_solvertime
        Calculation_time += PP_use_time
        
        if prob_exitflag_power == "optimal":
            if lower_bound < prob_LowerBound:
                rho_opt = xonoffini[idx_inisol].copy()
                theta_opt = np.array(theta_PU_SU_primal).copy()
                power_SUopt = power_S_opt.copy()
                lower_bound = prob_LowerBound
                LB_iter_store[i_gbd-1] = lower_bound 
        else:
    		#----------solve feasibilty check problem------------#            
            
            theta_PU_SU_primal, theta_PU_SU_primal_dual, theta_PU_SU_primal_lowb_dual, power_S_opt, \
                power_SU_dual, PU_SINR_dual, PU_signal_dual, primal_infeasible_alpha, PP_inf_solver_time \
                    = RIS_ipopt_infeasible(PathLoss_UserBS, PathLoss_UserRIS, \
                    PathLoss_RISBS, Tx_antBS, Tx_antRIS, RIS_Lnum, xonoffini[idx_inisol], power_P, P_max, Noise, PU_SINR_min, \
                    GBD_thereal_ini, GBD_theimag_ini, GBD_power_ini)
            
            PP_total_time += PP_inf_solver_time
            if primal_infeasible_alpha < 1e-20:
                print("find a feasible point")
                # feasible_flag=1
                # prob_exitflag_theta = "feasible"
                prob_exitflag_power = "optimal"
                if lower_bound < prob_LowerBound:
                    rho_opt = xonoffini[idx_inisol].copy()
                    theta_opt = np.array(theta_PU_SU_primal).copy()
                    power_SUopt = power_S_opt.copy()
                    lower_bound = prob_LowerBound
                    LB_iter_store[i_gbd-1] = lower_bound
                else:
                    print("find an infeasible point")
    
        
        
        if prob_exitflag_power == "optimal":
            feasible_flag = 1
            # print("feasible_flag:" , feasible_flag)
        else:
            feasible_flag = 0
        
        one_feasible_flag_multi[idx_inisol] = np.array(feasible_flag).reshape(1,1)
        
        one_thetavec_PU_SU_primal_multi[idx_inisol,:,:] = np.array(theta_PU_SU_primal).reshape(1,Tx_antRIS,1)
        one_thetavec_PU_SU_primal_dual_multi[idx_inisol,:,:] = np.array(theta_PU_SU_primal_dual).reshape(1,Tx_antRIS,1)
        # one_thetavec_PU_SU_primal_lowb_dual_multi[idx_inisol,:,:] = np.array(theta_PU_SU_primal_lowb_dual).reshape(1,Tx_antRIS,1)
        
        one_power_SU_primal_multi[idx_inisol] = np.array(power_S_opt).reshape(1,1)
        one_power_SU_primal_dual_multi[idx_inisol,:,:] = np.array(power_SU_dual).reshape(1,2,1)
        
        one_PU_SINR_primal_dual_multi[idx_inisol] = np.array(PU_SINR_dual).reshape(1,1)
        one_PU_signal_primal_dual_multi[idx_inisol] = np.array(PU_signal_dual).reshape(1,1)
        
        one_infeasible_alpha_multi[idx_inisol] = np.array(primal_infeasible_alpha).reshape(1,1)
    
    #-------------solve the relaxed master problem----------------#
    # print("Master problem solving:")
    
    MP_start_time = time.process_time()
    
    rho_new, psi, numsol_real, classifi_use_time = master_class_test.master_sol(one_feasible_flag_multi,one_thetavec_PU_SU_primal_multi, one_thetavec_PU_SU_primal_dual_multi,
                   one_power_SU_primal_multi, one_power_SU_primal_dual_multi, one_PU_SINR_primal_dual_multi, one_PU_signal_primal_dual_multi,
                    one_infeasible_alpha_multi, numsol, xonoff, upper_bound, xonoff[1], model, scaler)
	
    MP_end_time = time.process_time()
    MP_time = MP_end_time - MP_start_time
    Calculation_time += MP_time 
    
    ML_total_time += classifi_use_time

    if np.array_equal(rho_new[0],xonoffini[0]) and abs((psi)-upper_bound)<0.000001:
        reapeat_count = reapeat_count+1

	#------------update global lower bound---------------#
    upper_bound = psi
    UB_iter_store[i_gbd-1] = upper_bound
    
    if upper_bound==0:
        upper_bound = 1e-6
    rho = rho_new.copy()
    # thetaini = theta_PU_SU_primal
    # power_Sini = power_S_opt
    
    
	

	#--------------------------from second iteration---------------------------#
	#------------solve #numsol primal problems and get #numsol cuts------------#
    while ((upper_bound-lower_bound)/abs(lower_bound)>1e-7) and (i_gbd<max_iter_num):
        i_gbd = i_gbd + 1
                
        feasible_flag_multi = np.zeros((numsol_real,1))
        thetavec_PU_SU_primal_multi = np.zeros((numsol_real,Tx_antRIS,1), dtype = 'complex_')
        thetavec_PU_SU_primal_dual_multi = np.zeros((numsol_real,Tx_antRIS,1), dtype = 'complex_')
        thetavec_PU_SU_primal_lowb_dual_multi = np.zeros((numsol_real,Tx_antRIS,1), dtype = 'complex_')
        power_SU_primal_multi = np.zeros((numsol_real,1))
        power_SU_primal_dual_multi = np.zeros((numsol_real,2,1))
        # PU_SINR_multi = np.zeros((numsol_real,1))
        PU_SINR_primal_dual_multi = np.zeros((numsol_real,1))
        PU_signal_primal_dual_multi = np.zeros((numsol_real,1))
        # infeasible_theta_multi = np.zeros((numsol_real,1))
        infeasible_alpha_multi = np.zeros((numsol_real,1))
        


        for i_sol in range(numsol_real):
			#------------solve the primal problem----------------#
            
            theta_PU_SU_primal, theta_PU_SU_primal_dual, power_S_opt, power_SU_dual, PU_SINR_dual, PU_signal_dual,\
                prob_exitflag_power, prob_LowerBound, PP_solvertime, PP_use_time\
                = RIS_ipopt_primal(Bandwidth, Noise, P_max, P_k, mu, PathLoss_UserBS, PathLoss_UserRIS, PathLoss_RISBS, rho[i_sol],
                                Num_User, Tx_antBS, RIS_Lnum, Tx_antRIS, power_P, PU_SINR_min, GBD_thereal_ini, GBD_theimag_ini, GBD_power_ini)
            
            # print("i_numsol primal: ", i_sol)
            # print("multi integer",rho[i_sol])
            # print("power SU primal: ", power_S_opt)
            # print("theta SCA primal: ", theta_PU_SU_primal)
            # print("primal status: ", prob_exitflag_power)
            # print("primal LB: ", prob_LowerBound)
            PP_total_time += PP_solvertime
            Calculation_time += PP_use_time 
            
            if prob_exitflag_power == "optimal":
                if lower_bound < prob_LowerBound:
                    rho_opt = rho[i_sol].copy()
                    theta_opt = np.array(theta_PU_SU_primal).copy()
                    power_SUopt = power_S_opt.copy()
                    lower_bound = prob_LowerBound
                    LB_iter_store[i_gbd-1] = lower_bound
            else:
				#---------solve feasibilty check problem-------------#
                                
                theta_PU_SU_primal, theta_PU_SU_primal_dual, theta_PU_SU_primal_lowb_dual, power_S_opt, \
                    power_SU_dual, PU_SINR_dual, PU_signal_dual, primal_infeasible_alpha, PP_inf_solver_time \
                        = RIS_ipopt_infeasible(PathLoss_UserBS, PathLoss_UserRIS, \
                        PathLoss_RISBS, Tx_antBS, Tx_antRIS, RIS_Lnum, rho[i_sol], power_P, P_max, Noise, PU_SINR_min,\
                        GBD_thereal_ini, GBD_theimag_ini, GBD_power_ini)
                
                PP_total_time += PP_inf_solver_time
                if primal_infeasible_alpha < 1e-20:
                    print("find a feasible point")
                    # feasible_flag=1
                    # prob_exitflag_theta = "feasible"
                    prob_exitflag_power = "optimal"
                    if lower_bound < prob_LowerBound:
                        rho_opt = rho[i_sol].copy()
                        theta_opt = np.array(theta_PU_SU_primal).copy()
                        power_SUopt = power_S_opt.copy()
                        lower_bound = prob_LowerBound
                        LB_iter_store[i_gbd-1] = lower_bound
                else:
                    print("find an infeasible point")
            
                        
            if prob_exitflag_power == "optimal":
                feasible_flag = 1
            else:
                feasible_flag = 0
            
            
            feasible_flag_multi[i_sol] = feasible_flag
            thetavec_PU_SU_primal_multi[i_sol,:,:] = np.array(theta_PU_SU_primal).reshape(Tx_antRIS,1).copy()
            thetavec_PU_SU_primal_dual_multi[i_sol,:,:] = np.array(theta_PU_SU_primal_dual).reshape(Tx_antRIS,1).copy()
            # thetavec_PU_SU_primal_lowb_dual_multi[i_sol,:,:] = np.array(theta_PU_SU_primal_lowb_dual).reshape(Tx_antRIS,1).copy()
            
            
            power_SU_primal_multi[i_sol] = power_S_opt
            power_SU_primal_dual_multi[i_sol,:,:] = np.array(power_SU_dual).copy()
            
            
            PU_SINR_primal_dual_multi[i_sol] = np.array(PU_SINR_dual).copy()
            PU_signal_primal_dual_multi[i_sol] = np.array(PU_signal_dual).copy()
            
            # infeasible_theta_multi[i_sol] = np.array(infeasible_theta_var).copy() #---????????????????---#
            infeasible_alpha_multi[i_sol] = np.array(primal_infeasible_alpha).copy() #---????????????????---#


		#---------------------solve the relaxed master problem---------------------#        
        
        MP_start_time = time.process_time()
        
        rho_new, psi, numsol_real_new, classifi_use_time = master_class_test.master_sol(feasible_flag_multi,thetavec_PU_SU_primal_multi, thetavec_PU_SU_primal_dual_multi,
            power_SU_primal_multi, power_SU_primal_dual_multi, PU_SINR_primal_dual_multi, PU_signal_primal_dual_multi,
            infeasible_alpha_multi, numsol, rho, upper_bound, rho[0], model, scaler)
        
        MP_end_time = time.process_time()
        MP_time = MP_end_time - MP_start_time
        Calculation_time += MP_time 
        
        # print("Master problem solved!")
        ML_total_time += classifi_use_time

        if np.array_equal(rho_new[0], rho[0]) and abs(psi-upper_bound)<0.000001:
            reapeat_count = reapeat_count+1

		#----------------update global lower bound-----------------#
        
        upper_bound = psi
        UB_iter_store[i_gbd-1] = upper_bound
        if upper_bound==0:
            upper_bound = 1e-6
        rho = rho_new.copy()
        # thetaini = theta_PU_SU_primal
        # power_Sini = power_S_opt
                        
        numsol_real = numsol_real_new
		

        # print("lower_bound",lower_bound) 
        # print("upper_bound",upper_bound)


	
        if reapeat_count == 2:
            if abs(upper_bound-lower_bound)<1e-3:
                print("---------------------------------", reapeat_count)
                convergence_flag = 1
                
            else:
                convergence_flag = -1
                
            
            # master_class_test.json_output()
            # acc_array = np.array([])


            return power_SUopt, theta_opt, rho_opt, rho, upper_bound, lower_bound, convergence_flag, i_gbd, UB_iter_store, LB_iter_store, \
                PP_total_time, ML_total_time, Calculation_time


    if i_gbd<max_iter_num:
        convergence_flag = 1
        print("Problem solved:")
        # print("xonoff:", rho_opt)
        # print("powerSU:", power_SUopt)
        # print("thetaopt:", theta_opt)
        # print("UB:", upper_bound)
        # print("LB:", lower_bound)
        
    else:
        convergence_flag = 0
        print("Problem cannot converge")
        # print("xonoff:", rho_opt)
        # print("powerSU:", power_SUopt)
        # print("thetaopt:", theta_opt)
        # print("UB:", upper_bound)
        # print("LB:", lower_bound)
        

    # master_class_test.json_output()

    return power_SUopt, theta_opt, rho_opt, rho, upper_bound, lower_bound, convergence_flag, i_gbd, UB_iter_store, LB_iter_store, \
        PP_total_time, ML_total_time, Calculation_time

