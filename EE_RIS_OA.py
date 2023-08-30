#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#-------------------Solve problem by branch and bound algorithm-------------------#

import numpy as np
import pandas as pd
import math as ma
import time
from pyomo.environ import *


def RIS_Bonmin_OA(Bandwidth, Noise, P_max, P_k, mu, PathLoss_UserBS, PathLoss_UserRIS, PathLoss_RISBS,
                   Num_User,Tx_antBS,RIS_Lnum,Tx_antRIS, Power_PU, PU_SINR_min, 
                   OA_thereal_ini, OA_theimag_ini, OA_power_ini, OA_onoff_ini):
    
    #-------------------array setting---------------------#
    
    glvec_1 = np.zeros([Tx_antBS,1], dtype = 'complex_')
    userris_1 = np.zeros([Tx_antRIS,1], dtype = 'complex_')
    Ulmar_1 = np.zeros([Tx_antRIS,Tx_antBS], dtype = 'complex_')

    risbs = np.zeros([Tx_antRIS, Tx_antBS], dtype = 'complex_')
    U_diag = np.eye(Tx_antRIS)

    glvec_1[:,0] = PathLoss_UserBS[0,:]     #-----------------direct link channel-------------------#

    for j in range(0, Tx_antBS):
        risbs[:,j] = PathLoss_RISBS[j,:,:]  #-----------------RIS to BS channel---------------------#

    flag = 0

    for i in range(0,RIS_Lnum):
        userris_1[:,0] = PathLoss_UserRIS[:,0,:].reshape(32,)  #-----------------RIS to User channel---------------------#
                
        Ulmar_1[flag*Tx_antRIS : (flag+1)*Tx_antRIS,:] = np.dot(U_diag*(userris_1.conj().T), risbs)  #------------combine two channel----------------#
        flag = flag+1
        

    #Ulmar_original = Ulmar_1
    Ulmar_1 = Ulmar_1.conj().T   #------------combine two channel will conjugate the matrix----------------#
    
    
    #--------------------------------------------object function-----------------------------------------------#
    
    def object_func(model):
        
        direct_link_sum = sum(abs(glvec_1[idx_c])**2 for idx_c in range(Tx_antBS)) 
        
        return (( direct_link_sum\
            + 2*model.xnf*sum(glvec_1[idx_m].real* RIS_part_real(model, idx_m)\
            + glvec_1[idx_m].imag*RIS_part_imag(model, idx_m) for idx_m in range(Tx_antBS))\
            + model.xnf*RIS_part_square(model))*model.SUPower\
            /((direct_link_sum\
                + 2*sum(glvec_1[idx_m].real*RIS_part_real(model, idx_m)\
                + glvec_1[idx_m].imag*RIS_part_imag(model, idx_m) for idx_m in range(Tx_antBS))\
                + RIS_part_square(model))*Power_PU + Noise))\
                /(mu*model.SUPower + P_k)
                
    #------------------------------------------------objective func part-----------------------------------------------------------#


    def RIS_part_real(model, idx_m):
        
        # return sum((Ulmar_1[idx_m,j].real*model.thereal[j] - Ulmar_1[idx_m,j].imag*model.theimg[j]) for j in range(Tx_antRIS))
        return sum((Ulmar_1[idx_m,j].real*cos(model.theta[j]) - Ulmar_1[idx_m,j].imag*sin(model.theta[j])) for j in range(Tx_antRIS))

    def RIS_part_imag(model, idx_m):
        
        # return sum((Ulmar_1[idx_m,j].real*model.theimg[j]  + Ulmar_1[idx_m,j].imag*model.thereal[j]) for j in range(Tx_antRIS))
        return sum((Ulmar_1[idx_m,j].real*sin(model.theta[j])  + Ulmar_1[idx_m,j].imag*cos(model.theta[j])) for j in range(Tx_antRIS))

    def RIS_part_square(model):
        
        return sum((RIS_part_real(model, idx_s))**2 + (RIS_part_imag(model, idx_s))**2 for idx_s in range(Tx_antBS))

    #------------------------------------------------------------------------------------------------------------------------------#


    def constraint_01(model, i):         #--------------absolute value of phase shift inequality---------------#
        
        # return sqrt(model.thereal[j]**2 + model.theimg[j]**2) == 1
        return sqrt(cos(model.theta[j])**2 + sin(model.theta[j])**2) == 1

    def constraint_02(model):            #--------------power greater than zero inequality---------------#
        return model.SUPower >= 0

    def constraint_03(model):            #--------------power greater than zero inequality---------------#
        return model.SUPower <= P_max

    def constraint_04(model, Power_PU):  #--------------minimum primary receiver inequality---------------#
        
        direct_link_sum = sum(abs(glvec_1[idx_c])**2 for idx_c in range(Tx_antBS))
        
        return ( direct_link_sum \
            + 2*sum(glvec_1[idx_m].real*RIS_part_real(model, idx_m)\
            + glvec_1[idx_m].imag*RIS_part_imag(model, idx_m) for idx_m in range(Tx_antBS))\
            + RIS_part_square(model))*Power_PU\
            /((direct_link_sum \
                + 2*model.xnf*sum(glvec_1[idx_m].real*RIS_part_real(model, idx_m)\
                + glvec_1[idx_m].imag*RIS_part_imag(model, idx_m) for idx_m in range(Tx_antBS))\
                + model.xnf*RIS_part_square(model))*model.SUPower + Noise) >= PU_SINR_min
                
    def constraint_06(model):                   #--------------direct link less than few times through RIS link---------------#
        
        direct_link_sum = sum(abs(glvec_1[idx_c])**2 for idx_c in range(Tx_antBS))
        # print("&&&&&&&&&&&", direct_link_sum)
        
        return 1e13*(direct_link_sum \
            + 2*sum(glvec_1[idx_m].real*RIS_part_real(model, idx_m)\
            + glvec_1[idx_m].imag*RIS_part_imag(model, idx_m) for idx_m in range(Tx_antBS))\
            + RIS_part_square(model)) >= 1e14*(0.5)*direct_link_sum[0]

    #-----------------------------------Use above function setting solver math model----------------------------------------# 
    
    model = ConcreteModel(name="cp_test_OA_1")
    model.dual = Suffix(direction=Suffix.IMPORT)
    
    model.SUPower = Var(bounds=(0,P_max), within=NonNegativeReals, initialize = OA_power_ini)
    # print("OA_thereal_ini", OA_thereal_ini)
    # print("OA_theimag_inii", OA_theimag_ini)
    # model.thereal = Var([i for i in range(32)], bounds=(-1,1), within=Reals, initialize = OA_thereal_ini)
    # model.theimg = Var([i for i in range(32)], bounds=(-1,1), within=Reals, initialize = OA_theimag_ini)
    model.theta = Var([i for i in range(32)], bounds=(0, 2*ma.pi), within=Reals, initialize = OA_thereal_ini)
    model.xnf = Var(bounds=(0,1), within=Binary, initialize = OA_onoff_ini)

    model.cons = ConstraintList()
    for idx in range(Tx_antRIS):
        model.cons.add(constraint_01(model, idx))

    model.cons.add(constraint_02(model))
    model.cons.add(constraint_03(model))
    model.cons.add(constraint_04(model, Power_PU))
    model.cons.add(constraint_06(model))

    model.obj = Objective(expr=object_func, sense=maximize)

    solver_path = '/home/dddd/ris_gbd_solver/Bonmin-1.8.8/build/bin/bonmin'

    opt = SolverFactory('bonmin', executable=solver_path)
    opt.options['bonmin.algorithm'] = 'B-OA'
    # opt.options['bonmin.oa_decomposition'] = 'yes'

    # opt.options['bonmin.iteration_limit'] = int(1)
    # opt.options['bonmin.node_limit'] = int(1)

    # opt.options['oa_log_frequency'] = 1 


    # opt.options['max_iter'] = int(2)

    results = opt.solve(model, tee=True)
    results.write()
    # model.solutions.load_from(results)

    # model.display()
    # model.pprint()
    # output_BB = results.Solver._list
    

    OA_solver_timeuse = results.solver.time    #---------------solver use time-------------------------#
    
    #--------------------get the variable results---------------------------#
    
    model_y_OA = np.array(list(model.theta.get_values().values()))
    # model_x_OA = np.array(list(model.thereal.get_values().values()))
    # model_y_OA = np.array(list(model.theimg.get_values().values()))
    model_z_OA = np.array(list(model.SUPower.get_values().values()))
    model_a_OA = np.array(list(model.xnf.get_values().values()))
    
    # model_y_coef_OA = np.zeros([Tx_antRIS, 1], dtype = 'complex_')
    
    # for idx_y in range(0, Tx_antRIS):
    #     model_y_coef_OA[idx_y] = (cos(model_y_OA [idx_y]) + 1j*sin(model_y_OA [idx_y]))
        
    
    OA_status = results.solver.termination_condition
    OA_objective = model.obj()
    model.cons.display()
    print("OA on off : ", model_a_OA)
    # print("OA theta: ", model_y_OA)
    # print("OA theta coef: ", model_y_coef_OA)
    # # print("abs: ", abs(model_y_coef_OA))
    # print("cos real: ", cos(model_y_OA[0]))
    # print("cos imag: ", sin(model_y_OA[0]))
    # print("real: ", model_y_coef_OA[0].real)
    # print("imag: ", model_y_coef_OA[0].imag)
    # print("real phase shift: ", acos(model_y_coef_OA[0].real))
    # print("imag phase shift: ", asin(model_y_coef_OA[0].imag))
    
    return OA_status, OA_objective, model_z_OA, OA_solver_timeuse