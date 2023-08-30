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


def RIS_ipopt_primal(Bandwidth, Noise, P_max, P_k, mu, PathLoss_UserBS, PathLoss_UserRIS, PathLoss_RISBS, xonoff,
                   Num_User, Tx_antBS, RIS_Lnum, Tx_antRIS, Power_PU, PU_SINR_min, 
                   GBD_thereal_ini, GBD_theimag_ini, GBD_power_ini):
    
    #--------------------------------array setting-----------------------------------#
    
    glvec_1 = np.zeros([Tx_antBS,1], dtype = 'complex_')
    userris_1 = np.zeros([Tx_antRIS,1], dtype = 'complex_')
    Ulmar_1 = np.zeros([Tx_antRIS,Tx_antBS], dtype = 'complex_')
    xini = xonoff[0]

    risbs = np.zeros([Tx_antRIS, Tx_antBS], dtype = 'complex_')
    U_diag = np.eye(Tx_antRIS)

    glvec_1[:,0] = PathLoss_UserBS[0,:]                        #--------------------direct link channel-------------------#

    for j in range(0, Tx_antBS):
        risbs[:,j] = PathLoss_RISBS[j,:,:]                     #--------------------RIS to BS channel---------------------#

    flag = 0

    for i in range(0,RIS_Lnum):
        userris_1[:,0] = PathLoss_UserRIS[:,0,:].reshape(32,)  #-------------------RIS to User channel---------------------#
        
        #------------combine two channel, called matrix U----------------#
        
        Ulmar_1[flag*Tx_antRIS : (flag+1)*Tx_antRIS,:] = np.dot(U_diag*(userris_1.conj().T), risbs)
        flag = flag+1
        
        
    Ulmar_1 = Ulmar_1.conj().T
    
   

    #----------primal problem objective function----------#
    
    def object_func(model):

        direct_link_sum = sum(abs(glvec_1[idx_c])**2 for idx_c in range(Tx_antBS)) 
        
        return (( direct_link_sum\
            + 2*xini*sum(glvec_1[idx_m].real* RIS_part_real(model, idx_m)\
            + glvec_1[idx_m].imag*RIS_part_imag(model, idx_m) for idx_m in range(Tx_antBS))\
            + xini*RIS_part_square(model))*model.SUPower\
            /((direct_link_sum\
                + 2*sum(glvec_1[idx_m].real*RIS_part_real(model, idx_m)\
                + glvec_1[idx_m].imag*RIS_part_imag(model, idx_m) for idx_m in range(Tx_antBS))\
                + RIS_part_square(model))*Power_PU + Noise))\
                /(mu*model.SUPower + P_k)
    
    
    #-------------------------------------------sub function for objective function------------------------------------------------------------#
          
    def RIS_part_real(model, idx_m):  #-------------theta coefficient complex real---------------#
        
        # return sum((Ulmar_1[idx_m,j].real*model.thereal[j] - Ulmar_1[idx_m,j].imag*model.theimg[j]) for j in range(Tx_antRIS))
        return sum((Ulmar_1[idx_m,j].real*cos(model.theta[j]) - Ulmar_1[idx_m,j].imag*sin(model.theta[j])) for j in range(Tx_antRIS))
    
    def RIS_part_imag(model, idx_m):  #-------------theta coefficient complex imag---------------#
        
        # return sum((Ulmar_1[idx_m,j].real*model.theimg[j] + Ulmar_1[idx_m,j].imag*model.thereal[j]) for j in range(Tx_antRIS))
        return sum((Ulmar_1[idx_m,j].real*sin(model.theta[j]) + Ulmar_1[idx_m,j].imag*cos(model.theta[j])) for j in range(Tx_antRIS))
    
    
    def RIS_part_square(model):       #-------------theta coefficient complex square---------------#
        
        return sum((RIS_part_real(model, idx_s))**2 + (RIS_part_imag(model, idx_s))**2 for idx_s in range(Tx_antBS))

    #------------------------------------------------------------------------------------------------------------------------------------------#

           
    def constraint_01(model, i):                #--------------absolute value of phase shift inequality---------------#
        
        # return sqrt(model.thereal[i]**2 + model.theimg[i]**2) <= 1
        return sqrt(cos(model.theta[i])**2 + sin(model.theta[i])**2) == 1
    
        
    def constraint_02(model):                   #--------------power greater than zero inequality----------------#
        return model.SUPower >= 0
    
    
    def constraint_03(model):                   #--------------power greater than zero inequality----------------#
        return model.SUPower <= P_max
    
        
    def constraint_04(model, Power_PU, xini):   #--------------minimum primary receiver inequality---------------#
        
        direct_link_sum = sum(abs(glvec_1[idx_c])**2 for idx_c in range(Tx_antBS))

        return (1e18*(( direct_link_sum \
            + 2*sum(glvec_1[idx_m].real*RIS_part_real(model, idx_m)\
            + glvec_1[idx_m].imag*RIS_part_imag(model, idx_m) for idx_m in range(Tx_antBS))\
            + RIS_part_square(model))*Power_PU)) >= (1e18*(PU_SINR_min*((direct_link_sum \
                + 2*xini*sum(glvec_1[idx_m].real*RIS_part_real(model, idx_m)\
                + glvec_1[idx_m].imag*RIS_part_imag(model, idx_m) for idx_m in range(Tx_antBS))\
                + xini*RIS_part_square(model))*(model.SUPower) + Noise)))
    
    
    
    # def constraint_05(model, i):                #--------------absolute value of phase shift inequality---------------#
        
    #     # return sqrt(model.thereal[i]**2 + model.theimg[i]**2) >= 1
    #     return sqrt(cos(model.theta[i])**2 + sin(model.theta[i])**2) >= 1
    
        
    def constraint_06(model):                   #--------------direct link less than few times through RIS link---------------#
        
        direct_link_sum = sum(abs(glvec_1[idx_c])**2 for idx_c in range(Tx_antBS))
        # print("&&&&&&&&&&&", direct_link_sum)
        
        return 1e13*(direct_link_sum[0] \
            + 2*sum(glvec_1[idx_m].real*RIS_part_real(model, idx_m)\
            + glvec_1[idx_m].imag*RIS_part_imag(model, idx_m) for idx_m in range(Tx_antBS))\
            + RIS_part_square(model)) >= 1e14*(0.5)*direct_link_sum[0]
 
    
    #--------Use above function setting solver math model---------# 
    
    model = ConcreteModel(name="cp_primal")
    model.dual = Suffix(direction=Suffix.IMPORT)
    
    # intial_power = powerSU_ini
    # intial_thereal = thereal
    # intial_theimag = theimag
    # if Power_PU > 0.5 and Power_PU < 2:
    #     intial_power += 0.015
        
    #----------------setting math model variable-----------------# 
    
    model.SUPower = Var(bounds=(0,P_max), within=NonNegativeReals, initialize = GBD_power_ini)
    # model.thereal = Var([i for i in range(32)], bounds=(-1,1), within=Reals, initialize = GBD_thereal_ini)
    # model.theimg = Var([i for i in range(32)], bounds=(-1,1), within=Reals, initialize = GBD_theimag_ini)
    model.theta = Var([i for i in range(32)], bounds=(0,2*ma.pi), within=Reals, initialize = GBD_thereal_ini)
    
    
    #----------------Add constraint to math model--------------#
    
    model.cons = ConstraintList()
    for idx in range(Tx_antRIS):
        model.cons.add(constraint_01(model, idx))

    model.cons.add(constraint_02(model))
    model.cons.add(constraint_03(model))
    model.cons.add(constraint_04(model, Power_PU, xini))
    # for idx in range(Tx_antRIS):
    #     model.cons.add(constraint_05(model, idx))
        
    model.cons.add(constraint_06(model))
    
    PP_start_time = time.process_time()
    
    model.obj = Objective(expr=object_func, sense=maximize)
    
    #----------------Solve math model by solver--------------#
    
    solver_path = '/home/dddd/ris_gbd_solver/Bonmin-1.8.8/build/bin/ipopt'
    # solver_path = '/home/dddd/Solver_Couenne/Couenne-0.5.8/build/bin/ipopt'
    
    
    
    opt = SolverFactory('ipopt', executable=solver_path)
    
    #----------------solver options --------------#
    
    opt.options['max_iter'] = int(10e2)
    opt.options['nlp_scaling_method'] = 'gradient-based'
    opt.options['nlp_scaling_max_gradient'] = int(1e3)
    opt.options['mu_strategy'] = 'adaptive'
    
    
    
    # results = opt.solve(model, tee=True)
    results = opt.solve(model, tee=False)
    # results.write()
    
    PP_end_time = time.process_time()
    PP_time = PP_end_time - PP_start_time
    
    # log_infeasible_constraints(model)
    
    #----------------Get the solved results--------------#
    
    # model_x = np.array(list(model.thereal.get_values().values()))
    # model_y = np.array(list(model.theimg.get_values().values()))
    model_z = np.array(list(model.SUPower.get_values().values()))
    model_x = np.array(list(model.theta.get_values().values()))
    # print("ipopt primal obj:", model.obj())
    
    # model.dual.display()
    
    
    
    #----------------collate the results--------------#
    
    primal_phase_shift = np.zeros([Tx_antRIS,1], dtype = 'complex_')
    primal_phase_shift_dual = np.zeros([Tx_antRIS,1], dtype = 'complex_')
    primal_phase_shift_dual_lowbound = np.zeros([Tx_antRIS,1], dtype = 'complex_')
    
    primal_powerSU_dual = np.zeros([2,1], dtype = np.float64)
    
    for i in range(Tx_antRIS):
        primal_phase_shift[i] = cos(model_x[i]) + sin(model_x[i])*1j
        
    for j in range(Tx_antRIS):
        primal_phase_shift_dual[j] = model.dual[model.cons[j+1]]
    
    # for j in range(36, 36+Tx_antRIS):
    #     primal_phase_shift_dual_lowbound[j-36] = model.dual[model.cons[j]] 
    
    primal_powerSU = model_z
    primal_powerSU_dual[0] = model.dual[model.cons[33]]
    primal_powerSU_dual[1] = model.dual[model.cons[34]]
    
    PU_SINR_min_dual = model.dual[model.cons[35]]
    
    PU_signal_min_dual = model.dual[model.cons[36]]
    
    prob_lower_bound = model.obj()
    primal_prob_status = results.solver.termination_condition
    
    
    
    PP_solvertime = results.solver.time
    
    PP_time = PP_time + PP_solvertime
    
    # test_real = 0
    # test_imag = 0
    # teat_plus = 0
    # test_square = 0
    # direct_link_sum = sum(abs(glvec_1[idx_c])**2 for idx_c in range(Tx_antBS))
    
    # for idx_m in range(Tx_antBS):
    #     test_real = sum((Ulmar_1[idx_m,j].real*model_x[j] - Ulmar_1[idx_m,j].imag*model_y[j]) for j in range(Tx_antRIS))
    #     test_imag = sum((Ulmar_1[idx_m,j].real*model_y[j] + Ulmar_1[idx_m,j].imag*model_x[j]) for j in range(Tx_antRIS))
    #     teat_plus += glvec_1[idx_m].real*test_real - glvec_1[idx_m].imag*test_imag
    #     test_square += test_real**2 + test_imag**2
            
    # PU_SINR_up = (direct_link_sum \
    #      + 2*teat_plus\
    #      + test_square)*Power_PU
    
    # print("PU_SINR upper", PU_SINR_up)
        
    # PU_SINR_low = PU_SINR_min*((direct_link_sum \
    #      + 2*teat_plus\
    #      + test_square)*model_z + Noise)
    
    # print("PU_SINR lower", PU_SINR_low)
    
    # model.dual.display()
    # model.cons.display()
    
    print("--------Primal v2 Solved--------")
    print("primal status :", primal_prob_status)
    print("primal time :", PP_solvertime)
    
    return primal_phase_shift, primal_phase_shift_dual, primal_powerSU, primal_powerSU_dual,\
        PU_SINR_min_dual, PU_signal_min_dual, primal_prob_status, prob_lower_bound, PP_solvertime, PP_time
    
