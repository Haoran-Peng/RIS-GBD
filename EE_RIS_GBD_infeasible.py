#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#--------------------GBD infeasible problem--------------------#

import sys
import numpy as np
import numpy.matlib
import math as ma
import random
from pyomo.environ import *

import pyutilib.subprocess.GlobalData
pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False


def RIS_ipopt_infeasible(PathLoss_UserBS, PathLoss_UserRIS, PathLoss_RISBS, Tx_antBS, Tx_antRIS, RIS_Lnum, xonoff,
                         Power_PU, P_max, Noise, PU_SINR_min, inf_GBD_thereal_ini, inf_GBD_theimag_ini, inf_GBD_power_ini):
    
    #--------------------------------array setting---------------------------------#
    
    glvec_1 = np.zeros([Tx_antBS,1], dtype = 'complex_')
    userris_1 = np.zeros([Tx_antRIS,1], dtype = 'complex_')
    Ulmar_1 = np.zeros([Tx_antRIS,Tx_antBS], dtype = 'complex_')
    xini = xonoff[0]

    risbs = np.zeros([Tx_antRIS, Tx_antBS], dtype = 'complex_')
    U_diag = np.eye(Tx_antRIS)

    glvec_1[:,0] = PathLoss_UserBS[0,:]  #--------------------direct link channel-------------------#

    for j in range(0, Tx_antBS):
        risbs[:,j] = PathLoss_RISBS[j,:,:]  #--------------------RIS to BS channel---------------------#

    flag = 0

    for i in range(0,RIS_Lnum):
        userris_1[:,0] = PathLoss_UserRIS[:,0,:].reshape(32,)  #-------------------RIS to User channel---------------------#
        
        #------------combine two channel, called matrix U----------------#
        
        Ulmar_1[flag*Tx_antRIS : (flag+1)*Tx_antRIS,:] = np.dot(U_diag*(userris_1.conj().T), risbs)
        flag = flag+1
        
        
    Ulmar_1 = Ulmar_1.conj().T
    
    
    def object_func(model):
        
           
        return model.alpha
    
    #-------------------------------------------infeasible objctive function------------------------------------------------------------#
    
    def RIS_part_real(model, idx_m):  #------------------U matrix multiply theta variable real part-----------------#
        
        return sum((Ulmar_1[idx_m,j].real*cos(model.theta[j]) - Ulmar_1[idx_m,j].imag*sin(model.theta[j])) for j in range(Tx_antRIS))

    def RIS_part_imag(model, idx_m):  #------------------U matrix multiply theta variable imag part-----------------#
        
        return sum((Ulmar_1[idx_m,j].real*sin(model.theta[j]) + Ulmar_1[idx_m,j].imag*cos(model.theta[j])) for j in range(Tx_antRIS))

    def RIS_part_square(model):       #------------------U matrix multiply theta variable square part-----------------#
        
        return sum((RIS_part_real(model, idx_s))**2 + (RIS_part_imag(model, idx_s))**2 for idx_s in range(Tx_antBS))

    #--------------------------------------------------------------------------------------------------------------#


    def constraint_01(model, i):               #--------------absolute value of phase shift inequality---------------#
        
        # return sqrt(model.thereal[i]**2 + model.theimg[i]**2) <= 1 + model.alpha
        return sqrt(cos(model.theta[i])**2 + sin(model.theta[i])**2) <= 1 + model.alpha

    def constraint_02(model):                  #--------------power greater than zero inequality---------------#
        return model.SUPower + model.alpha >= 0

    def constraint_03(model):                  #--------------power greater than zero inequality---------------#
        return model.SUPower <= P_max +model.alpha 
    
    def constraint_04(model, Power_PU, xini):  #--------------minimum primary receiver inequality---------------#
        
        direct_link_sum = sum(abs(glvec_1[idx_c])**2 for idx_c in range(Tx_antBS))
        
        
        return 1e8*((direct_link_sum \
            + 2*sum(glvec_1[idx_m].real*RIS_part_real(model, idx_m)\
            + glvec_1[idx_m].imag*RIS_part_imag(model, idx_m) for idx_m in range(Tx_antBS))\
            + RIS_part_square(model))*Power_PU)\
                >= 1e8*((PU_SINR_min - model.alpha)*((direct_link_sum \
                + 2*xini*sum(glvec_1[idx_m].real*RIS_part_real(model, idx_m)\
                + glvec_1[idx_m].imag*RIS_part_imag(model, idx_m) for idx_m in range(Tx_antBS))\
                + xini*RIS_part_square(model))*model.SUPower + Noise))
                    
    def constraint_05(model, i):               #--------------absolute value of phase shift inequality---------------#
        
        # return sqrt(model.thereal[i]**2 + model.theimg[i]**2) >= 1
        return sqrt(cos(model.theta[i])**2 + sin(model.theta[i])**2) + model.alpha >= 1
    
    def constraint_06(model):                  #--------------direct link less than few times through RIS link---------------#
        
        direct_link_sum = sum(abs(glvec_1[idx_c])**2 for idx_c in range(Tx_antBS))
        # print("&&&&&&&&&&&", direct_link_sum)
        
        return 1e13*(direct_link_sum \
            + 2*sum(glvec_1[idx_m].real*RIS_part_real(model, idx_m)\
            + glvec_1[idx_m].imag*RIS_part_imag(model, idx_m) for idx_m in range(Tx_antBS))\
            + RIS_part_square(model)) + model.alpha >= 1e14*(0.5)*direct_link_sum[0]
    
    
    #----------Use above function setting solver math model-----------#
    
    model = ConcreteModel(name="cp_primal_infeasible")
    model.dual = Suffix(direction=Suffix.IMPORT)

    model.alpha = Var(within=NonNegativeReals, initialize = 2.242)
    model.SUPower = Var(bounds=(0,P_max), within=NonNegativeReals, initialize = inf_GBD_power_ini)
    # model.thereal = Var([i for i in range(32)], bounds=(-1,1), within=Reals, initialize = inf_GBD_thereal_ini)
    # model.theimg = Var([i for i in range(32)], bounds=(-1,1), within=Reals, initialize = inf_GBD_theimag_ini)
    model.theta = Var([i for i in range(32)], bounds=(0,2*ma.pi), within=Reals, initialize = GBD_thereal_ini)
    
    model.cons = ConstraintList()
    for idx in range(Tx_antRIS):
        model.cons.add(constraint_01(model, idx))

    model.cons.add(constraint_02(model))
    model.cons.add(constraint_03(model))
    model.cons.add(constraint_04(model, Power_PU, xini))
    
    for idx in range(Tx_antRIS):
        model.cons.add(constraint_05(model, idx))
        
    model.cons.add(constraint_06(model))

    model.obj = Objective(expr=object_func, sense=minimize)
    
    solver_path = '/home/dddd/ris_gbd_solver/Bonmin-1.8.8/build/bin/ipopt'
    
    opt = SolverFactory('ipopt', executable=solver_path)
    
    opt.options['max_iter'] = int(1e3)
    opt.options['nlp_scaling_method'] = 'gradient-based'
    opt.options['nlp_scaling_max_gradient'] = int(1e3)
    opt.options['mu_strategy'] = 'adaptive'

    #results = opt.solve(model, tee=True)
    results = opt.solve(model)
    results.write()
    
    # model_x = np.array(list(model.thereal.get_values().values()))
    # model_y = np.array(list(model.theimg.get_values().values()))
    model_z = np.array(list(model.SUPower.get_values().values()))
    model_x = np.array(list(model.theta.get_values().values()))
    
    
    primal_infeasible_phase_shift = np.zeros([Tx_antRIS,1], dtype = 'complex_')
    primal_infeasible_phase_shift_dual = np.zeros([Tx_antRIS,1], dtype = 'complex_')
    primal_infeasible_phase_shift_dual_lowbound = np.zeros([Tx_antRIS,1], dtype = 'complex_')
    
    primal_infeasible_powerSU_dual = np.zeros([2,1], dtype = 'complex_')
    
    for i in range(Tx_antRIS):
        primal_infeasible_phase_shift[i] = cos(model_x[i]) + sin(model_x[i])*1j
        
    for j in range(Tx_antRIS):
        primal_infeasible_phase_shift_dual[j] = model.dual[model.cons[j+1]]
        
    for j in range(36, 36+Tx_antRIS):
        primal_infeasible_phase_shift_dual_lowbound[j-36] = model.dual[model.cons[j]] 
    
    primal_infeasible_powerSU = model_z
    primal_infeasible_powerSU_dual[0] = model.dual[model.cons[33]]
    primal_infeasible_powerSU_dual[1] = model.dual[model.cons[34]]
    
    PU_SINR_min_infeasible_dual = model.dual[model.cons[35]]
    
    PU_signal_min_infeasible_dual = model.dual[model.cons[68]]
    
    primal_infeasible_alpha = model.obj()
    
    
    PP_inf_solver_time = results.solver.time
    
    
    return primal_infeasible_phase_shift, primal_infeasible_phase_shift_dual, primal_infeasible_phase_shift_dual_lowbound,\
        primal_infeasible_powerSU, primal_infeasible_powerSU_dual, PU_SINR_min_infeasible_dual, PU_signal_min_infeasible_dual, primal_infeasible_alpha, PP_inf_solver_time
    
    
    
    
