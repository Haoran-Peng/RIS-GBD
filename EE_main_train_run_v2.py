#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#----------------------generate train data and training classified model-------------------#

# import math
import sys
import numpy as np
# import matlab
# import matlab.engine
import os
# import matplotlib.pyplot as plt

import json

from EE_RIS_GBD_parameters import userpara
# import ee_feature_label_calwhole as gbdML
import EE_multi_collect_train_v2 as gbdmultiML
import EE_Acc_GBD_v2 as gbdacc

from sklearn import preprocessing, discriminant_analysis, linear_model, svm
import joblib
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, roc_curve, auc 

def load_data(path):
    with open(path,'r',encoding='utf-8') as f:
        # print("path: ", path)
        # print(type(path))
        if path == 'outputs/problem_3_3/data1.json':
            print("What are tou fucking run????")
        # print("f: ", f)
        data_input = json.loads(f.read())

    features = data_input['features']
    labels = data_input['labels']
    print(len(features))
    print(len(labels))
    if len(labels) != len(features):
        print("Input data error")
        sys.exit()
	
    x = []
    y = []

    for label in labels:
        y.append(label['VALUABLE'])	

    for feature in features:
        temp = {}

        temp[0] = feature['num_cut']
        temp[1] = feature['i_gbd']
        temp[2] = feature['cut_violation']
        temp[3] = feature['cut_type']
        temp[4] = feature['cut_order']

        x.append(temp)
    # print("load_data_x", x)
    # print("load_data_y", y)

    return y,x

def load_all_data(K0,L0,train_flag):
    
    if train_flag == 'train':
        path = 'outputs/problem_'+str(K0)+'_'+str(L0)
    else:
        path = 'test_outputs/problem_'+str(K0)+'_'+str(L0)
    # print(os.listdir(path))
    files = os.listdir(path)
    # print(type(files))
    y = []
    x = []
    for file in files:
        temp_y, temp_x = load_data(path+'/'+file)
        y += temp_y
        x += temp_x
    return y,x


def normalize_data(y,x):
    x_feature = np.zeros([len(y), 5])
    
    for i in range(len(y)):
        for j in range(5):
            x_feature[i,j] = x[i][j]
    
    stand_scaler = preprocessing.StandardScaler()
    x_scaled = stand_scaler.fit_transform(x_feature)
    x_scaled_dict = []
    for i in range(len(y)):
        temp = {}

        temp[0] = x_scaled[i,0]
        temp[1] = x_scaled[i,1]
        temp[2] = x_scaled[i,2]
        temp[3] = x_scaled[i,3]
        temp[4] = x_scaled[i,4]

        x_scaled_dict.append(temp)
    
    return y,x_scaled_dict, stand_scaler
    
def under_sample(y, x, del_num):
    i=0
    while i < del_num:
        del_index = np.random.randint(low=0, high=len(y))
        if y[del_index] == 0:
            del y[del_index]
            del x[del_index]
            i = i + 1
    
    return y,x


def dic_to_array(y,x):
    x_feature = np.zeros([len(y), 5])
    
    for i in range(len(y)):
        for j in range(5):
            x_feature[i,j] = x[i][j]
    
    return x_feature

def num_count(y):
    num1 = 0
    num0 = 0
    
    for i in range(len(y)):
        if y[i] == 1:
            num1 += 1
        else:
            num0+=1
    
    return num1, num0

def normalize_data_scaler(y, x, scaler):
    x_feature = np.zeros([len(y), 5])
    
    for i in range(len(y)):
        for j in range(5):
            x_feature [i,j] = x[i][j]
        
    x_scaled = scaler.transform(x_feature)
    x_scaled_dict = []
    for i in range(len(y)):
        temp = {}
        
        temp[0] = x_scaled[i,0]
        temp[1] = x_scaled[i,1]
        temp[2] = x_scaled[i,2]
        temp[3] = x_scaled[i,3]
        temp[4] = x_scaled[i,4]
        
        x_scaled_dict.append(temp)
        
    return y, x_scaled_dict


def performance_metric(y, x, model):
    
    model = model.fit(x,y)
    predictions = model.predict(x)
    print('Accuracy:', accuracy_score(y, predictions))  
    print('Recall:', recall_score(y, predictions)) 
    cm = confusion_matrix(y, predictions)
    print('Recall negative:', cm[0,0]/(cm[0,0]+cm[0,1])) 
    #print('Confusion matrix:\n', cm)
    y_score = model.decision_function(x)
    fpr,tpr,threshold = roc_curve(y, y_score) #calculate false_positive and true_positive rate
    roc_auc = auc(fpr,tpr) #calcuate auc
    
    return fpr, tpr, roc_auc, model


def gbd_train_run(Tx_antRIS, Tx_antBS, RIS_Lnum0, Num_User0, power_P0, Bandwidth0,Noise0, P_max0, P_k0, mu0, PathLoss_UserBS0, PathLoss_UserRIS0,
                  PathLoss_RISBS0, PU_SINR_min, numsol, data_num, data_num_test, model, train_flag, threshold, scaler, 
                  GBD_thereal_ini, GBD_theimag_ini, GBD_power_ini):
    
    Num_User0 = 2
    RIS_Lnum0 = 1
    Tx_antBS0 = Tx_antBS
    Tx_antRIS0 = Tx_antRIS
    
    
    
    PU_SINR_min0 = PU_SINR_min
    xonoffini = np.zeros([numsol, 1])
    # xonoffini = np.ones([numsol, 1])
    if numsol > 1:
        xonoffini[1] = 1
    thetaini = np.ones([Tx_antRIS0,1], dtype = 'complex_')
    # thetaini = np.zeros([Tx_antRIS0,1], dtype = 'complex_')
    # power_Sini = 0
    
    PP_total_time = 0
    ML_total_time = 0
    # if train_flag == 'test':
        
    ini_start = data_num
    
    for i in range(data_num_test):
        iter_total = 0
        repeat_number = 0
        
        while repeat_number<=1:
            # print("solving problem #%d..."%(i+1))

            if train_flag == 'train':
                
                Calculation_time_acc = 0
                                
                power_SUopt, theta_opt, rho_opt, rho, fin_upper_bound, fin_lower_bound, convergence_flag, iter_num, UB_iter_store, LB_iter_store \
                    = gbdmultiML.gbd_multi_col_train(Bandwidth0, Noise0, P_max0, P_k0, mu0, RIS_Lnum0, Tx_antBS0, Tx_antRIS0, Num_User0,\
                  PathLoss_UserBS0, PathLoss_UserRIS0, PathLoss_RISBS0, xonoffini, power_P0,\
                      PU_SINR_min0, numsol, threshold, i+ini_start, GBD_thereal_ini, GBD_theimag_ini, GBD_power_ini)
                        
            elif train_flag == 'test':

                power_SUopt, theta_opt, rho_opt, rho, fin_upper_bound, fin_lower_bound, convergence_flag, iter_num, UB_iter_store, LB_iter_store,\
                    PP_total_time, ML_total_time, Calculation_time_acc \
                    = gbdacc.gbd_ml_testacc(Bandwidth0, Noise0, P_max0, P_k0, mu0, RIS_Lnum0, Tx_antBS0, Tx_antRIS0, Num_User0, 
                    PathLoss_UserBS0, PathLoss_UserRIS0, PathLoss_RISBS0, xonoffini, power_P0, 
                    PU_SINR_min0, numsol, model, threshold, i, scaler, GBD_thereal_ini, GBD_theimag_ini, GBD_power_ini)

            iter_total = iter_total + iter_num
			
            if convergence_flag == 1:
                print("problem #%d solved!"%(i+1))
                break
            else:
                if convergence_flag == 0:
                    print("problem #%d cannot converge!"%(i+1))
                    break
                else:
                    print("new iteration needed!")
# 					rho_initial = rho[0,:,:].copy()
                repeat_number = repeat_number + 1
        if repeat_number>1:
            print("problem #%d cannot converge!"%(i+1))  
        
    return fin_upper_bound, fin_lower_bound, power_SUopt, theta_opt, rho_opt, rho, convergence_flag, iter_num, UB_iter_store, LB_iter_store,\
        PP_total_time, ML_total_time, Calculation_time_acc 
    
    
def main(argv=None):
    
    Num_User0 = 2
    RIS_Lnum0 = 1
    Tx_antBS0 = 8
    Tx_antRIS0 = 32
    PU_SINR_min = 7
    Power_PU0 = 29
    Power_PU0 =pow(10, (Power_PU0/10))/pow(10,3)
    power_Sini = 0
    
    numsol = 2
    threshold = 1
    data_num = 11
    
    #------test-------#
    
    data_num_solve = 1
    num_sol_test = 1
    
    # rand_seed_array = np.array([18,21,26,35,38,41,49,50,54,57,60])
    rand_seed_array = np.array([26,35,40,49,58,91,103,104,189,214,244])
    rand_location_ini_point = np.array([[0.012, 3.5, 0], [0.012, 3.6, 0], [0.012, 3.6, 0], [0.012, 3.7, 0], [0.012, 3.8, 0], \
                                        [0.012, 2.504, 0], [0.012, 2.8, 0], [0.012, 2.75, 0], [0.012, 2.49, 0], [0.012, 2.95, 0],\
                                        [0.012, 2.8, 0], [0.012, 2.918, 0], [0.012, 2.95, 0]])
    
    train_results = np.zeros([11, 1])
    
    for idx_trainset in range(data_num):
    
        [Bandwidth0, Noise0, P_max0, P_k0, mu0, PathLoss_UserBS0, PathLoss_UserRIS0, PathLoss_RISBS0]\
            =userpara(Num_User0, Tx_antBS0, RIS_Lnum0, Tx_antRIS0, rand_seed_array[idx_trainset])
        
        #---------------collect training data---------------#
        
        GBD_power_ini = rand_location_ini_point[idx_trainset][0]
        GBD_thereal_ini = rand_location_ini_point[idx_trainset][1]
        GBD_theimag_ini = rand_location_ini_point[idx_trainset][2]
        
        fin_upper_bound, fin_lower_bound, power_SUopt, theta_opt, rho_opt, rho, convergence_flag, iter_num, UB_iter_store, LB_iter_store, \
            PP_total_time, ML_total_time = \
        gbd_train_run(Tx_antRIS0, Tx_antBS0, RIS_Lnum0, Num_User0, Power_PU0, Bandwidth0, Noise0, P_max0, P_k0, mu0,
                      PathLoss_UserBS0,PathLoss_UserRIS0, PathLoss_RISBS0, PU_SINR_min, numsol, idx_trainset, data_num_solve, '', 'train', threshold, '',
                      GBD_thereal_ini, GBD_theimag_ini, GBD_power_ini)
        
        train_results[idx_trainset] = (fin_upper_bound + fin_lower_bound)/2
    
    # print("train problem result :", train_results)
    #-------------------file name-----------------#
    
    scaler_filename = 'model_1/scaler_' + str(Num_User0) + '_' +str(RIS_Lnum0) + '.save'
    model_path_svm = 'model_1/model_svm_'+str(Num_User0) + '_' +str(RIS_Lnum0) + '.m'
    
    #---------------load training data---------------#

    y, x = load_all_data(Num_User0, RIS_Lnum0, 'train')
    
    num1, num0 = num_count(y)
    
    y, x = under_sample(y, x, num0-num1)    
    num1, num0 = num_count(y)
    
    y,x_scaled,scaler = normalize_data(y,x)
    x_array = dic_to_array(y,x)
    x_scaled_array = dic_to_array(y,x_scaled)
    
    # print("scaler: ", scaler)
    # print(x_array)
    # print("x scaled: ", x_scaled_array)
    # print("shape: ", x_scaled_array.shape)
    # print("y: ", y)
    # print("y shape: ", len(y))
    
    if not os.path.exists('model_1'):
        os.makedirs('model_1')
        
    joblib.dump(scaler, scaler_filename)
    
    #----------------------SVM training-----------------------------#
    
    clf = svm.SVC(class_weight = {0:1,1:2} )
    fpr1, tpr1, roc_auc1, clf = performance_metric(y, x_scaled_array, clf)
    
    joblib.dump(clf, model_path_svm)
    
    
    
    
if __name__ =="__main__":
    main()
    
    
    
    
