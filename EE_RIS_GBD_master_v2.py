#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#-----------------------GBD master problem-------------------------------#

import cplex
from cplex.exceptions import CplexError
from cplex.exceptions import CplexSolverError
import sys
import numpy as np
import copy
import math
import time


class Master_class:
    my_prob = cplex.Cplex()
    init_flag = False

    def __init__(self, Bandwidth_in, Noise_in, P_max_in, P_k_in, mu_in, PathLoss_UserBS_in, PathLoss_UserRIS_in, PathLoss_RISBS_in,
                 xonoff, Num_User_in, Tx_antBS_in, RIS_Lnum_in, Tx_antRIS_in, power_P_in,
                 PU_SINR_min_in):
        
        #-------------------initial setting for master problem-------------------# 
        
        self.my_prob = cplex.Cplex()
        self.init_flag == False

        self.my_prob.objective.set_sense(self.my_prob.objective.sense.maximize)
        global PathLoss_UserBS, PathLoss_UserRIS, PathLoss_RISBS, Num_User, Tx_antBS, RIS_Lnum, \
            Tx_antRIS, Bandwidth, Noise, P_max, power_P, PU_SINR_min, P_k, mu

        PathLoss_UserBS = PathLoss_UserBS_in
        PathLoss_UserRIS = PathLoss_UserRIS_in
        PathLoss_RISBS = PathLoss_RISBS_in
        Num_User = Num_User_in
        Tx_antBS = Tx_antBS_in
        RIS_Lnum = RIS_Lnum_in
        Tx_antRIS = Tx_antRIS_in
        Bandwidth = Bandwidth_in
        Noise = Noise_in
        P_max = P_max_in
        power_P = power_P_in
        PU_SINR_min = PU_SINR_min_in
        P_k = P_k_in
        mu = mu_in
        



    #----------------function for add constraints------------------#

    def populatebyrow(self, my_obj, my_lb, my_ub, my_colnames, my_new_row, my_new_rhs, RIS_Lnum):
        
        if self.init_flag == False:
            self.init_flag = True
            my_ctype = ""
            for index in range(0, RIS_Lnum):
                my_ctype = my_ctype + "I"
            my_ctype = my_ctype + "C"

            self.my_prob.variables.add(
                obj=my_obj, lb=my_lb, ub=my_ub, names=my_colnames, types=my_ctype)
                       
            
        my_new_row = [my_new_row]
        my_sense = "L"

        #print("my_row:",my_new_row)        
        #print("my_sense:",my_sense)
        #print("rhs:",my_new_rhs)
        self.my_prob.linear_constraints.add(
            lin_expr=my_new_row, senses=my_sense, rhs=my_new_rhs)

    #---------------------function for solved problem------------------------#

    def lplex1(self, my_obj, my_lb, my_ub, my_colnames, my_new_row, my_new_rhs, Tx_antRIS, numsol, ret_flag):
        
        try:
            handle = self.populatebyrow(
                my_obj, my_lb, my_ub, my_colnames, my_new_row, my_new_rhs, RIS_Lnum)
        except CplexSolverError:
            print("Exception raised during populate")
            return

        if ret_flag == True:
            # print("test_numsol",numsol)
            self.my_prob.set_warning_stream(None)
            self.my_prob.set_results_stream(None)
            
            self.my_prob.parameters.mip.pool.capacity.set(numsol)
            self.my_prob.parameters.mip.pool.replace.set(1)
            
            self.my_prob.populate_solution_pool()
            
            self.my_prob.write("lplex1.lp")

            num_solution = self.my_prob.solution.pool.get_num()
            # print("The solution pool contains %d solutions." % num_solution)
            # meanobjval = cpx.solution.pool.get_mean_objective_value()
            
            # master_cons = self.my_prob.solution.basis.get_basis()
            # print("master_cons: ", master_cons)

            numsol_real = min(numsol, num_solution)
            sol_pool = []
        
            obj_temp = np.zeros(numsol_real)
            for i in range(numsol_real):
                obj_temp[i] = self.my_prob.solution.pool.get_objective_value(i) 
            new_index = sorted(range(len(obj_temp)), key=lambda k: obj_temp[k], reverse = True)
            # new_index = sorted(range(len(obj_temp)), key = lamba, k: obj_temp[k], reverse = True)
            # print(obj_temp)
            # print(new_index)

            for j in range(numsol_real):
                i = new_index[j]
                objval_i = self.my_prob.solution.pool.get_objective_value(i)
                x_i = self.my_prob.solution.pool.get_values(i)
                nb_vars=len(x_i)
                sol = []
                for k in range(nb_vars):
                    sol.append(x_i[k])
                sol_pool.append(sol)
                
            # print("pools =",sol_pool)

            # Print information about the incumbent
            # print("Objective value of the incumbent  = ",
                # self.my_prob.solution.get_objective_value())

            r_Psi = self.my_prob.solution.get_objective_value()

            r_rho = np.ones([numsol_real, RIS_Lnum], dtype=int)
            
            
            for i in range(0, numsol_real):
                x_i = sol_pool[i]
                for index in range(0, RIS_Lnum):
                    r_rho[i, index] = x_i[index]
                    
            # print("r_rho: ",r_rho)
            # print("r_psi: ",r_Psi)

            return r_rho, r_Psi, numsol_real#, time_use-time_start
        else:
            return -1, -1, -1
    
    #---------------------function for solved master problem progress------------------------#
    
    def master_sol(self,feasible_flag_multi,thetavec_PU_SU_primal_multi, thetavec_SU_primal_dual_multi,
                   power_SU_primal_multi, power_SU_primal_dual_multi, PU_SINR_primal_dual_multi, PU_signal_primal_dual_multi,
                   infeasible_alpha_multi, numsol):
        
        # U_diag = np.eye(Tx_antRIS)
        print("--------Master v2 Solved--------")
        
        my_obj = np.zeros(RIS_Lnum+1, dtype=np.float_)   
        my_obj[RIS_Lnum] = 1.0                              #---------master objective function coefficient---------#
        # print(my_obj)
        my_lb = np.zeros(RIS_Lnum+1, dtype=np.float_)       
        my_ub = np.ones(RIS_Lnum+1, dtype=np.float_)        

        my_lb[RIS_Lnum] = -cplex.infinity                   #---------optimization variable lower bound---------#
        my_ub[RIS_Lnum] = cplex.infinity                    #---------optimization variable upper bound---------#

        my_colnames = []

        index_i = 1
        for index in range(0, RIS_Lnum):
            # str_temp = "RIS_SU_onoff"+repr(index_i)
            str_temp = "rho"
            my_colnames.append(str_temp)                    #---------append discrete variable name to list---------#
            index_i = index_i + 1

        my_colnames.append("Psi")                           #---------append continuous variable name to list---------#

        this_num = feasible_flag_multi.shape[0]
        
        #------------------primal problem solution---------------------#
        
        for num_index in range(this_num):
            
            feasible_flag = feasible_flag_multi[num_index]                        
            thetavec_PU_SU_primal = thetavec_PU_SU_primal_multi[num_index]                 #-------------------theta solution--------------------#
            thetavec_SU_primal_dual = thetavec_SU_primal_dual_multi[num_index]             #-------------------theta less constraint dual--------------------#
            # thetavec_SU_primal_lowb_dual = thetavec_SU_primal_lowb_dual_multi[num_index]   #-----------------theta greater constraint dual-------------------#
            
            power_SU_primal = power_SU_primal_multi[num_index]                             #-------------------power solution--------------------#
            
            power_SU_primal_dual = power_SU_primal_dual_multi[num_index]                   #-------------------power constraint dual--------------------#
            PU_SINR_primal_dual = PU_SINR_primal_dual_multi[num_index]                     #-------------primary receiver SINR constraint dual---------------#
            PU_signal_primal_dual = PU_signal_primal_dual_multi[num_index]                 #----------direct link less than RIS link constraint dual---------#
            
            
            infeasible_alpha = infeasible_alpha_multi[num_index]                           #-------------------infeasible objective value--------------------#

            if feasible_flag:  #-------------primal problem feasible--------------#
                
                print("--------!!Problem Feasible!!--------")
                
                variables_names = copy.deepcopy(my_colnames)             #----------------All master problem variable-------------------#
                variables_coef = np.zeros(RIS_Lnum+1, dtype =np.float_)  #----------------All master problem variable coefficient-----------------# 
                rho_coef = np.zeros([RIS_Lnum, 1], dtype =np.float_)     #----------------discrete variable coefficient-----------------# 
                new_rhs = 0.0                                            #--------------Lagrange dual inequality constant---------------# 
                
                
                
                
                variables_coef[RIS_Lnum] = 1.0                           #-----------continuous variable coefficient----------#
                
                new_rhs -= power_SU_primal_dual[0]*power_SU_primal       #---------power greater than zero constraint---------#
                
                new_rhs += power_SU_primal_dual[1]*power_SU_primal       #---------power less than Max transmitted power constraint---------#
                new_rhs -= power_SU_primal_dual[1]*P_max
                
                for idx_mas in range(Tx_antRIS):                         #---------theta abs less than one constraint---------#
                    
                    new_rhs += (thetavec_SU_primal_dual[idx_mas]).real * np.linalg.norm(thetavec_PU_SU_primal[idx_mas])
                    new_rhs -= (thetavec_SU_primal_dual[idx_mas]).real
                
                # for idx_mas in range(Tx_antRIS):                         #---------theta abs gretear than one constraint---------#
                    
                #     new_rhs -= (thetavec_SU_primal_lowb_dual[idx_mas]).real * np.linalg.norm(thetavec_PU_SU_primal[idx_mas])
                #     new_rhs += (thetavec_SU_primal_lowb_dual[idx_mas]).real
                                
                
                
                glvec_1_total = 0
                glvec_1 = np.zeros([Tx_antBS,1], dtype = 'complex_')
                glvec_1[:,0] = PathLoss_UserBS[0,:]
                
                # Ulmar_1 = np.zeros([Tx_antRIS,Tx_antBS], dtype = 'complex_')
                # U_diag = np.eye(Tx_antRIS)
                
                userris_1 = np.zeros([Tx_antRIS, 1], dtype = 'complex_')
                risbs = np.zeros([Tx_antRIS,Tx_antBS], dtype = 'complex_')
                
                for j in range(0, Tx_antBS):
                    risbs[:,j] = PathLoss_RISBS[j,:,:]
                
                # flag = 0

                # for i in range(0,RIS_Lnum):
                #     userris_1[:,0] = PathLoss_UserRIS[:,0,:].reshape(32,)
                                        
                #     Ulmar_1[flag*Tx_antRIS : (flag+1)*Tx_antRIS,:] = np.dot(U_diag*(userris_1.conj().T), risbs)
                #     flag = flag+1
                    
                    
                # Ulmar_1 = Ulmar_1.conj().T
                
                # U_ris_ref = np.zeros([Tx_antBS,1], dtype = 'complex_')
                # U_ris_ref = np.dot(Ulmar_1, thetavec_PU_SU_primal.reshape(Tx_antRIS, 1))
                
                #-----------direct link sum of square------------#
                
                for index_bs in range(0, Tx_antBS):
                    glvec_1_total += abs(glvec_1[index_bs])**2
                
                #-----------BS-RIS-User channel with primal solution-------------#
                
                ris_ref = np.zeros([Tx_antBS,1], dtype = 'complex_')
                temp_coef = 0
                
                for i in range(0, RIS_Lnum):
                    userris_1[:,0] = PathLoss_UserRIS[:,0,:].reshape(32,)
                    thetal_1 = thetavec_PU_SU_primal[i*Tx_antRIS:(i+1)*Tx_antRIS,0].conj()
                    ris_ref = np.dot(np.dot(userris_1.conj().T,(np.diag(thetal_1))), risbs)
                  
                                
                ris_ref = ris_ref.conj().T
                
                #-----------------BS-RIS-User link part------------------#
                
                for index_ris in range(0,Tx_antBS):
                    
                    rho_coef += 2*(glvec_1[index_ris]*(ris_ref[index_ris].conj())).real + abs(ris_ref[index_ris])**2
                    
                #-----------------Minimum primary receiver SINR constraint--------------------#
                
                new_rhs -= ((PU_SINR_primal_dual)*(glvec_1_total + rho_coef)*power_P).reshape(1,)
                new_rhs += (PU_SINR_primal_dual)*PU_SINR_min*(glvec_1_total*power_SU_primal + Noise)
                
                #-----------------Direct link greter than through RIS link constraint--------------------#
                
                new_rhs += PU_signal_primal_dual*5*glvec_1_total
                new_rhs -= PU_signal_primal_dual*(glvec_1_total + rho_coef).reshape(1,)
                
                #-----------------Objective function in Lagrange constraint--------------------#
                
                func_deno = (((glvec_1_total + rho_coef)*power_P + Noise)*(P_k + mu*power_SU_primal)).reshape(1,)
                                
                new_rhs += (glvec_1_total*power_SU_primal)/func_deno
                
                temp_coef = (rho_coef*power_SU_primal)/func_deno
                
                #-------------------calculate discrete variable solution--------------------#
                
                rho_coef_final = -temp_coef - (PU_SINR_primal_dual)*PU_SINR_min*rho_coef*power_SU_primal 
                variables_coef[0] = rho_coef_final
                
                
                # print("var_coef_optimal = ", variables_coef.tolist())
                # print("optimal_rhs=", new_rhs)
                new_rhs = new_rhs.real 
                # print("***upper bound***: ", (-variables_coef[0]) + new_rhs)
                new_row = [variables_names, variables_coef.tolist()]

                new_rhs=new_rhs.tolist()

                #print("feasible")
                
                MP_start_time = time.process_time()
                
                r_rho, r_Psi, r_solnum = self.lplex1(
                    my_obj, my_lb, my_ub, my_colnames, new_row, new_rhs, Tx_antRIS, numsol, num_index==this_num-1)
                
                MP_end_time = time.process_time()
                MP_time = MP_end_time - MP_start_time


            else:  #-------------primal problem infeasible--------------#
                
                # print("*********************infeasibel master problem cut********************")
                variables_names = copy.deepcopy(my_colnames)                    #----------------All master problem variable-------------------#
                variables_coef = np.zeros(RIS_Lnum+1, dtype = np.float_)        #----------------All master problem variable cofficient-------------------#
                rho_coef = np.zeros([RIS_Lnum, 1], dtype = np.float_)           #----------------discrete variable coefficient-----------------# 
                new_rhs = 0.0                                                   #--------------Lagrange dual inequality constant---------------# 
                               
                
                new_rhs -= power_SU_primal_dual[0]*(power_SU_primal+infeasible_alpha)  #---------power greater than zero constraint---------#
                
                new_rhs += power_SU_primal_dual[1]*power_SU_primal                     #---------power less than Max transmitted power constraint---------#
                new_rhs -= power_SU_primal_dual[1]*P_max
                new_rhs -= power_SU_primal_dual[1]*infeasible_alpha
                
                
                for idx_masinf in range(Tx_antRIS):                                    #---------theta abs less than one constraint---------#
                    
                    new_rhs += (thetavec_SU_primal_dual[idx_masinf]).real * np.linalg.norm(thetavec_PU_SU_primal[idx_masinf])
                    new_rhs -= (thetavec_SU_primal_dual[idx_masinf]).real*(1 + infeasible_alpha)
                
                # for idx_mas in range(Tx_antRIS):                                       #---------theta abs gretear than one constraint---------#
                    
                #     new_rhs -= (thetavec_SU_primal_lowb_dual[idx_mas]).real * np.linalg.norm(thetavec_PU_SU_primal[idx_mas])
                #     new_rhs += (thetavec_SU_primal_lowb_dual[idx_mas]).real*(1 - infeasible_alpha)
                    
                
                
                
                
                glvec_1_total = 0
                glvec_1 = np.zeros([Tx_antBS,1], dtype = 'complex_')
                glvec_1[:,0] = PathLoss_UserBS[0,:]
                
                userris_1 = np.zeros([Tx_antRIS, 1], dtype = 'complex_')
                risbs = np.zeros([Tx_antRIS,Tx_antBS], dtype = 'complex_')
                
                for j in range(0, Tx_antBS):
                    risbs[:,j] = PathLoss_RISBS[j,:,:]
                
                #-----------direct link sum of square------------#
                
                for index_bs in range(0, Tx_antBS):
                    glvec_1_total += abs(glvec_1[index_bs])**2
                
                #-----------BS-RIS-User channel with primal solution-------------#
                
                ris_ref = np.zeros([Tx_antBS,1], dtype = 'complex_')
                
                for i in range(0, RIS_Lnum):
                    userris_1[:,0] = PathLoss_UserRIS[:,0,:].reshape(32,)
                    thetal_1 = thetavec_PU_SU_primal[i*Tx_antRIS:(i+1)*Tx_antRIS,0].conj()
                    ris_ref = np.dot(np.dot(userris_1.conj().T,(np.diag(thetal_1))), risbs)
                    
                
                ris_ref = ris_ref.conj().T    
                
                #-----------------BS-RIS-User link part------------------#
                
                for index_ris in range(0,Tx_antBS):
                    
                    rho_coef += 2*(glvec_1[index_ris]*ris_ref[index_ris].conj()).real + abs(ris_ref[index_ris])**2
                
                
                #-----------------Minimum primary receiver SINR constraint--------------------#
                
                
                new_rhs += PU_SINR_primal_dual*(PU_SINR_min-infeasible_alpha)*(glvec_1_total*power_SU_primal + Noise)
                new_rhs -= (PU_SINR_primal_dual*(glvec_1_total + rho_coef)*power_P).reshape(1,)
                
                #-----------------Direct link greter than through RIS link constraint--------------------#
                
                new_rhs += PU_signal_primal_dual*5*glvec_1_total
                new_rhs -= PU_signal_primal_dual*(glvec_1_total + rho_coef).reshape(1,)
                new_rhs -= PU_signal_primal_dual*infeasible_alpha
                
                #-------------------calculate discrete variable solution--------------------#
                
                variables_coef[0] = -PU_SINR_primal_dual*(PU_SINR_min - infeasible_alpha)*rho_coef*power_SU_primal
                

                
                new_rhs = new_rhs.real
                new_row = [variables_names, variables_coef.tolist()]

                # print("relaxed master infeasible")
                
                MP_start_time = time.process_time()
                
                r_rho, r_Psi, r_solnum = self.lplex1(
                    my_obj, my_lb, my_ub, my_colnames, new_row, new_rhs, Tx_antRIS, numsol, num_index==this_num-1)
                
                MP_end_time = time.process_time()
                MP_time = MP_end_time - MP_start_time
                
        return r_rho, r_Psi, r_solnum, MP_time

