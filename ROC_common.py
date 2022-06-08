"""
Functions defined below are for common ROC calculation used
"""

import os
import math  
import pickle
import warnings
import statistics
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels as sm
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn import metrics,preprocessing
from sklearn.metrics import roc_curve, auc
from calculate_ROC import get_ROC_AUC_V3

def Make_judgement(Num_para,theta_pos=None,theta_neg=None,gene=None,hets=None,depth=None,sigma=None):
    thetas=[theta_pos,theta_neg]
    if thetas.count(None) >0:
        raise Exception('Both thetas could not be None. The number of None from input was {}'.format(thetas.count(None)))
    var=[gene,hets,depth,sigma]
    N = 4-Num_para
    if Num_para == 4:
        if var.count(None) > N:
            raise Exception('None of the variables could be set to None. The number of None from input was {}'.format(var.count(None)))
    elif Num_para == 3:
        if var.count(None) > N:
            raise Exception('One of the variables have to be set to None. The number of None from input was {}'.format(var.count(None)))
    elif Num_para == 2:
        if var.count(None) > N:
            raise Exception('Two variables have to be set to None. The number of None from input was {}'.format(var.count(None)))
            

def Prepare_data_fix(gene, hets, depth, sigma,source,workdir,if_prob,theta_pos,theta_neg,Num_para):
    var=[gene,hets,depth,sigma]
    var_map_np = np.array(['g','h','d','s'])
    var_fullname_map_np = np.array(['gene','hets','depth','sigma'])
    full_var_map_np = np.array([gene, hets, depth, sigma])
    valid_var_np = var_map_np[np.array(var) != None]
    variable_var_np = var_fullname_map_np[np.array(var) == None]
    fixed_var_np = var_fullname_map_np[np.array(var) != None]
    valid_full_var_np = full_var_map_np[np.array(var) != None]
    if hets is not None:
        h=hets
    if sigma is not None:
        s=sigma
    if gene is not None:
        g=gene
    if depth is not None:
        d=depth
        
    path_model= Generate_path(source,workdir=workdir,if_prob=if_prob)
    all_file = sorted(os.listdir(path_model))
    file_dict = {}
    for pkl in all_file:
        if ".pickle" in pkl:
            name=pkl.rsplit(".pickle")[0].rsplit("_")
            file_dict[pkl] = {}
            for each_value in name:
                file_dict[pkl][each_value.split("-")[0]] = float(each_value.split("-")[1])
        else:continue
    file_dict_pd = pd.DataFrame(file_dict).transpose()
    file_dict_pd['file'] = file_dict_pd.index
    if Num_para == 3:
        pos_pd = file_dict_pd[(file_dict_pd[valid_var_np[0]] == valid_full_var_np[0])&(file_dict_pd[valid_var_np[1]] == valid_full_var_np[1])&(file_dict_pd[valid_var_np[2]] == valid_full_var_np[2])&(file_dict_pd['t'] == theta_pos)].sort_values(['d','h','g','s'])
        neg_pd = file_dict_pd[(file_dict_pd[valid_var_np[0]] == valid_full_var_np[0])&(file_dict_pd[valid_var_np[1]] == valid_full_var_np[1])&(file_dict_pd[valid_var_np[2]] == valid_full_var_np[2])&(file_dict_pd['t'] == theta_neg)].sort_values(['d','h','g','s'])
    elif Num_para == 2:
        pos_pd = file_dict_pd[(file_dict_pd[valid_var_np[0]] == valid_full_var_np[0])&(file_dict_pd[valid_var_np[1]] == valid_full_var_np[1])&(file_dict_pd['t'] == theta_pos)].sort_values(['d','h','g','s'])
        neg_pd = file_dict_pd[(file_dict_pd[valid_var_np[0]] == valid_full_var_np[0])&(file_dict_pd[valid_var_np[1]] == valid_full_var_np[1])&(file_dict_pd['t'] == theta_neg)].sort_values(['d','h','g','s'])
    d_group = pos_pd[var_map_np[np.array(var) == None][0]].unique()
    return d_group,var,var_map_np,fixed_var_np,var_fullname_map_np,variable_var_np,pos_pd,neg_pd

def Generate_path(source,workdir,if_prob=None):
    model="BEASTIE3"
    if if_prob != None:
        path_model = "/Users/scarlett/Desktop/HARDAC/scarlett/software/cmdstan/examples/" + str(model)+"/" + str(workdir) + "/output_pkl/model_prob2/"
    else:
        path_model = "/Users/scarlett/Desktop/HARDAC/scarlett/software/cmdstan/examples/" + str(model) + "/" + workdir + "/output_pkl/model_med/"
    return path_model

def Get_file_name(g,h,d,s,theta_pos,theta_neg):
    reduced_file_pos = "g-%s_h-%s_d-%s_t-%g.pickle" % (g, h, d, float(theta_pos))
    reduced_file_neg = "g-%s_h-%s_d-%s_t-%s.pickle" % (g, h, d, int(theta_neg))
    full_file_pos = "g-%s_h-%s_d-%s_t-%g_s-%s.pickle" % (g, h, d, float(theta_pos), s)
    full_file_neg = "g-%s_h-%s_d-%s_t-%s_s-%s.pickle" % (g, h, d, int(theta_neg), s)
    xlabels="Fixed parameters: gene-%s hets-%s depth-%s sigma-%s"%(g,h,d,s)
    return reduced_file_pos,reduced_file_neg,full_file_pos,full_file_neg,xlabels

def get_ROC_AUC_V3(path, file_pos, file_neg, if_prob=None, if_baseline=None,if_print=None,if_AA=None,if_drop=True):
    if if_print==True:
        print(path+file_pos)
    #print(path+file_pos)
    prob1 = pickle.load(open(path+file_pos,"rb"))
    prob2 = pickle.load(open(path+file_neg,"rb"))
    
    # baseline estiamtes/prob
    if if_baseline == True and if_AA !=True:
        if if_prob == True:
            #print("baseline prob")
            fpr,tpr, _ = roc_curve([0 for i in range(len(prob1))] + [1 for i in range(len(prob2))], prob1 + prob2,drop_intermediate=if_drop)
        else:
            #print("baseline esti")
            fpr,tpr, _ = roc_curve([1 for i in range(len(prob1))] + [0 for i in range(len(prob2))], prob1 + prob2,drop_intermediate=if_drop)

    # ADM estimates
    elif if_AA==True and if_baseline!=True:
        if if_prob == True:
            #print("ADM prob")
            fpr, tpr, _ = roc_curve([0 for i in range(len(prob1))] + [1 for i in range(len(prob2))], prob1 + prob2,drop_intermediate=if_drop)
        else:
            #print("ADM esti")
            fpr, tpr, _ = roc_curve([1 for i in range(len(prob1))] + [0 for i in range(len(prob2))], prob1 + prob2,drop_intermediate=if_drop)
    
    # model estimates/prob
    elif if_baseline !=True and if_AA != True:
        if if_prob != True: #estimates
            #print("model esti")
            prob1_t = [abs(x - 1) for x in prob1]
            prob2_t = [abs(x - 1) for x in prob2]
            fpr, tpr, _ = roc_curve([1 for i in range(len(prob1_t))] + [0 for i in range(len(prob2_t))], prob1_t + prob2_t,drop_intermediate=if_drop)
        else:
            #print("model prob")
            fpr, tpr, _ = roc_curve([1 for i in range(len(prob1))] + [0 for i in range(len(prob2))], prob1 + prob2,drop_intermediate=if_drop)
    return(fpr, tpr)