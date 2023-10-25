#!/bin/python
"""
Functions defined below are for model comparison using ROC score, lambda results are shown
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
from ROC_common import get_ROC_AUC_V3,Make_judgement,Generate_path,Get_file_name
from . import ROC_common
# prefix="/Users/scarlett/Desktop/HARDAC/scarlett/"

def Get_tpr_fpr_lambda(path_model, path_lambda1,path_lambda2,path_lambda3,path_lambda4,path_lambda5,path_base, path_base_p,path_AA,reduced_file_pos,reduced_file_neg,full_file_pos,full_file_neg,if_prob):
    fpr_m,tpr_m = get_ROC_AUC_V3(path_model,full_file_pos,full_file_neg,if_prob=if_prob)    
    fpr_1,tpr_1 = get_ROC_AUC_V3(path_lambda1,full_file_pos,full_file_neg,if_prob=if_prob)
    fpr_2,tpr_2 = get_ROC_AUC_V3(path_lambda2,full_file_pos,full_file_neg,if_prob=if_prob)
    fpr_3,tpr_3 = get_ROC_AUC_V3(path_lambda3,full_file_pos,full_file_neg,if_prob=if_prob)
    fpr_4,tpr_4 = get_ROC_AUC_V3(path_lambda4,full_file_pos,full_file_neg,if_prob=if_prob)
    fpr_5,tpr_5 = get_ROC_AUC_V3(path_lambda5,full_file_pos,full_file_neg,if_prob=if_prob)
    #fpr_m2,tpr_m2= get_ROC_AUC_V2(path_model_2,reduced_file_pos,reduced_file_neg,if_prob=if_prob)
    #fpr_b,tpr_b = get_ROC_AUC_V3(path_base,reduced_file_pos,reduced_file_neg,if_prob=if_prob,if_baseline=True)
    #fpr_b_p,tpr_b_p = get_ROC_AUC_V3(path_base_p,reduced_file_pos,reduced_file_neg,if_prob=if_prob,if_baseline=True)
    #fpr_a,tpr_a = get_ROC_AUC_V3(path_AA,reduced_file_pos,reduced_file_neg,if_prob=if_prob,if_AA=True)
    #return fpr_m,tpr_m,fpr_1,tpr_1,fpr_2,tpr_2,fpr_3,tpr_3,fpr_4,tpr_4,fpr_5,tpr_5,fpr_b,tpr_b,fpr_b_p,tpr_b_p,fpr_a,tpr_a
    return fpr_m,tpr_m,fpr_1,tpr_1,fpr_2,tpr_2,fpr_3,tpr_3,fpr_4,tpr_4,fpr_5,tpr_5


def Generate_path_LAMBDA(model,source,workdir,datatype,lamdbda):
    if lamdbda == True:
        path_model = source + str(model)+"/" + workdir + "/output_pkl/model_prob1/"
        path_lambda1 = source + str(model)+"/" + workdir + "_lambda/output_pkl/lambda_1.2/"+str(datatype)+"/"
        path_lambda2 = source + str(model)+"/" + workdir + "_lambda/output_pkl/lambda_1.4/"+str(datatype)+"/"
        path_lambda3 = source + str(model)+"/" + workdir + "_lambda/output_pkl/lambda_1.6/"+str(datatype)+"/"
        path_lambda4 = source + str(model)+"/" + workdir + "_lambda/output_pkl/lambda_1.8/"+str(datatype)+"/"
        path_lambda5 = source + str(model)+"/" + workdir + "_lambda/output_pkl/lambda_2/"+str(datatype)+"/"
    elif lamdbda!=True:
        path_model = source + str(model)+"/" + workdir + "/output_pkl/model_prob1/"
        path_lambda1 = None
        path_lambda2 = None
        path_lambda3 = None
        path_lambda4 = None
        path_lambda5 = None      
    path_base = source +"binomial/output_pkl/" + workdir + "/prob3/"
    path_base_p = source +"binomial/output_pkl/" + workdir + "/pooled_prob2/"
    path_AA = source + "ADM/" + workdir + "/output_pkl/AA_pval/"
    return path_model, path_base,path_base_p,path_AA,path_lambda1,path_lambda2,path_lambda3,path_lambda4,path_lambda5


def Generate_path_LAMBDA_selected(source,workdir,datatype,lamdbda1,lamdbda2):
    path_model = source + "SPP2-odd/" + workdir + "/output_pkl/model_prob1/"
    path_lambda1 = source + "SPP2-odd/" + workdir + "_lambda/output_pkl/lambda_"+str(lamdbda1)+"/"+str(datatype)+"/"
    path_model2 = source + "BEASTIE3/" + workdir + "/output_pkl/model_prob1/"
    path_lambda2 = source + "BEASTIE3/" + workdir + "_lambda/output_pkl/lambda_"+str(lamdbda2)+"/"+str(datatype)+"/"
    path_base = source +"binomial/output_pkl/" + workdir + "/prob3/"
    path_base2 = source +"binomial/output_pkl/" + workdir + "/prob4/"
    path_base_p = source +"binomial/output_pkl/" + workdir + "/pooled_prob1/"
    path_AA = source + "ADM/" + workdir + "/output_pkl/AA_pval/"
    return path_model, path_lambda1, path_model2,path_lambda2,path_base, path_base2,path_base_p,path_AA

def Generate_path_lambda(model,source,workdir,if_prob1):
    if if_prob1 == True:
        print("AUC is calculated using max posterior probability")
        path_model = source + str(model)+"/" + workdir + "/output_pkl/model_prob1/"
        path_lambda1 = source + str(model)+"/" + workdir + "_lambda/output_pkl/lambda_1.2/model_prob1/"
        path_lambda2 = source + str(model)+"/" + workdir + "_lambda/output_pkl/lambda_1.4/model_prob1/"
        path_lambda3 = source + str(model)+"/" + workdir + "_lambda/output_pkl/lambda_1.6/model_prob1/"
        path_lambda4 = source + str(model)+"/" + workdir + "_lambda/output_pkl/lambda_1.8/model_prob1/"
        path_lambda5 = source + str(model)+"/" + workdir + "_lambda/output_pkl/lambda_2/model_prob1/"
        path_base = source +"binomial/output_pkl/" + workdir + "/prob_new/"
        path_base_p = source +"binomial/output_pkl/" + workdir + "/pooled_prob_new/"
        path_AA = source + "ADM/output_pkl/" + workdir + "/AA_pval/"
    if if_prob1 == False:
        print("AUC is calculated log2 transformed max posterior probability")
        path_model = source + str(model)+"/" + workdir + "/output_pkl/model_prob2/"
        path_lambda1 = source + str(model)+"/" + workdir + "_lambda/output_pkl/lambda_1.2/model_prob2/"
        path_lambda2 = source + str(model)+"/" + workdir + "_lambda/output_pkl/lambda_1.4/model_prob2/"
        path_lambda3 = source + str(model)+"/" + workdir + "_lambda/output_pkl/lambda_1.6/model_prob2/"
        path_lambda4 = source + str(model)+"/" + workdir + "_lambda/output_pkl/lambda_1.8/model_prob2/"
        path_lambda5 = source + str(model)+"/" + workdir + "_lambda/output_pkl/lambda_2/model_prob2/"
        path_base = source +"binomial/output_pkl/" + workdir + "/prob_new/"
        path_base_p = source +"binomial/output_pkl/" + workdir + "/pooled_prob_new/"
        path_AA = source + "ADM/output_pkl/" + workdir + "/AA_pval/"
    if if_prob1 == None:
        print("AUC is calculated posterior median theta")
        path_model = source + str(model)+"/" + workdir + "/output_pkl/model_med/"
        path_lambda1 = source + str(model)+"/" + workdir + "_lambda/output_pkl/lambda_1.2/model_med/"
        path_lambda2 = source + str(model)+"/" + workdir + "_lambda/output_pkl/lambda_1.4/model_med/"
        path_lambda3 = source + str(model)+"/" + workdir + "_lambda/output_pkl/lambda_1.6/model_med/"
        path_lambda4 = source + str(model)+"/" + workdir + "_lambda/output_pkl/lambda_1.8/model_med/"
        path_lambda5 = source + str(model)+"/" + workdir + "_lambda/output_pkl/lambda_2/model_med/"
        path_base = source +"binomial/output_pkl/" + workdir + "/esti1/"
        path_base_p = source +"binomial/output_pkl/" + workdir + "/pooled_esti/"
        path_AA = source + "ADM/output_pkl/" + workdir + "/AA_esti/"
    #print(path_model)
    #print(path_lambda1)
    return path_model, path_lambda1,path_lambda2,path_lambda3,path_lambda4,path_lambda5,path_base, path_base_p,path_AA

def ROC_comparison_fix3_LAMBDA(source,model,workdir,datatype,lamdbda=None,theta_pos=None,theta_neg=None,gene=None,hets=None,depth=None,sigma=None,title=None,Num_para=3,Num_col=None):
    # judge whether  2 of the 4 are None, and 2 of the 4 are not None
    Make_judgement(Num_para=Num_para,theta_pos=theta_pos,theta_neg=theta_neg,gene=gene,hets=hets,depth=depth,sigma=sigma)
    Plot_ROC_fix3_LAMBDA(source,model=model,workdir=workdir,datatype=datatype,lamdbda=lamdbda,Num_col=Num_col,gene=gene, hets=hets, depth=depth, sigma=sigma,theta_pos=theta_pos,theta_neg=theta_neg,title=title)

def Plot_ROC_fix3_LAMBDA(source,model,workdir,datatype,lamdbda,Num_col,gene, hets, depth, sigma,theta_pos,theta_neg,title):
    path_model, path_base, path_base_p,path_AA,path_lambda1,path_lambda2,path_lambda3,path_lambda4,path_lambda5 = Generate_path_LAMBDA(model=model,source=source,workdir=workdir,datatype=datatype,lamdbda=lamdbda)
    d_group,var,var_map_np,fixed_var_np,var_fullname_map_np,variable_var_np,pos_pd,neg_pd = Prepare_data_fix_LAMBDA(gene, hets, depth, sigma,model=model,source=source,workdir=workdir,datatype=datatype,lamdbda=lamdbda,theta_pos=theta_pos,theta_neg=theta_neg,Num_para=3)
    if Num_col == None:
        Num_col = 3
    else:
        Num_col = int(Num_col)
    row = math.ceil(float(len(d_group))/Num_col)
    fig, axs = plt.subplots(row, Num_col, figsize = (20,5*row))
    if (row * Num_col > len(d_group)):
        for i in range(row * Num_col - len(d_group)):
            axs.flat[-1-i].set_axis_off()
    #print(d_group)
    #d_group=d_group[:-1]
    for i, each in enumerate(d_group):
        #print(i)
        #print(each)
        current_group_pos_list = pos_pd[pos_pd[var_map_np[np.array(var) == None][0]] == each].index
        current_group_neg_list = neg_pd[neg_pd[var_map_np[np.array(var) == None][0]] == each].index
        xlabels = "Fixed parameters "
        for idx in range(len(current_group_pos_list)):
            #g,h,d,s,reduced_file_pos,reduced_file_neg =  decompose(current_group_pos_list[idx])
            reduced_file_pos = current_group_pos_list[idx].rsplit("_",1)[0]+".pickle"
            reduced_file_neg = current_group_neg_list[idx].rsplit("_",1)[0]+".pickle"
            g=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[0].rsplit("-")[1]
            h=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[1].rsplit("-")[1]
            d=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[2].rsplit("-")[1]
            s=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[4].rsplit("-")[1]
            fpr_m,tpr_m,fpr_b,tpr_b,fpr_b_p,tpr_b_p,fpr_a,tpr_a,fpr_1,tpr_1,fpr_2,tpr_2,fpr_3,tpr_3,fpr_4,tpr_4,fpr_5,tpr_5 = Get_tpr_fpr_LAMBDA(path_model, path_base, path_base_p,path_AA,path_lambda1,path_lambda2,path_lambda3,path_lambda4,path_lambda5,reduced_file_pos,reduced_file_neg,current_group_pos_list[idx],current_group_neg_list[idx])
            var_dict = {"gene":g, "hets": h, "depth": d, "sigma": s}
            for each in fixed_var_np:
                xlabels += each+":"+var_dict[each]+' '
            axs.flat[i].set_xlabel("FPR")
            axs.flat[i].set_ylabel("TPR")
            axs.flat[i].set_title(var_fullname_map_np[np.array(var) == None][0]+":" + var_dict[var_fullname_map_np[np.array(var) == None][0]])
            axs.flat[i].plot(fpr_m, tpr_m, label =  str(model)+" :"+str(round(auc(fpr_m,tpr_m),3)),linewidth=3)
            if lamdbda != None:
                axs.flat[i].plot(fpr_1, tpr_1, label =  str(model)+" lambda 1.2 :"+str(round(auc(fpr_1,tpr_1),3)),linewidth=3)
                axs.flat[i].plot(fpr_2, tpr_2, label =  str(model)+" lambda 1.4: "+str(round(auc(fpr_2,tpr_2),3)),linewidth=3)
                axs.flat[i].plot(fpr_3, tpr_3, label =  str(model)+" lambda 1.6: "+str(round(auc(fpr_3,tpr_3),3)),linewidth=3)
                axs.flat[i].plot(fpr_4, tpr_4, label =  str(model)+" lambda 1.8: "+str(round(auc(fpr_4,tpr_4),3)),linewidth=3)
                axs.flat[i].plot(fpr_5, tpr_5, label =  str(model)+" lambda 2: "+str(round(auc(fpr_5,tpr_5),3)),linewidth=3)
            axs.flat[i].plot(fpr_b, tpr_b, label =  "Binomial 2-side: "+str(round(auc(fpr_b,tpr_b),3)),linewidth=3)
            axs.flat[i].plot(fpr_b_p, tpr_b_p, label =  "Binomial pool 2-side: "+str(round(auc(fpr_b_p,tpr_b_p),3)),linewidth=3)
            axs.flat[i].plot(fpr_a, tpr_a, label =  "ADM: "+str(round(auc(fpr_a,tpr_a),3)),linewidth=3)
            axs.flat[i].legend(fontsize=10,loc='lower right')
    plt.suptitle(xlabels+"theta pos: "+str(theta_pos)+" theta neg: "+str(theta_neg),fontsize=20)
    plt.show()

def ROC_comparison_fix3_LAMBDA_selected(source,workdir,datatype,lamdbda1,lamdbda2,theta_pos=None,theta_neg=None,gene=None,hets=None,depth=None,sigma=None,title=None,Num_para=3,Num_col=None):
    # judge whether  2 of the 4 are None, and 2 of the 4 are not None
    Make_judgement(Num_para=Num_para,theta_pos=theta_pos,theta_neg=theta_neg,gene=gene,hets=hets,depth=depth,sigma=sigma)
    Plot_ROC_fix3_LAMBDA_selected(source,workdir=workdir,datatype=datatype,lamdbda1=lamdbda1,lamdbda2=lamdbda2,Num_col=Num_col,gene=gene, hets=hets, depth=depth, sigma=sigma,theta_pos=theta_pos,theta_neg=theta_neg,title=title)


def Plot_ROC_fix3_LAMBDA_selected(source,workdir,datatype,lamdbda1,lamdbda2,Num_col,gene, hets, depth, sigma,theta_pos,theta_neg,title):
    path_model, path_lambda1, path_model2,path_lambda2,path_base,path_base2, path_base_p,path_AA = Generate_path_LAMBDA_selected(source=source,workdir=workdir,datatype=datatype,lamdbda1=lamdbda1,lamdbda2=lamdbda2)
    d_group,var,var_map_np,fixed_var_np,var_fullname_map_np,variable_var_np,pos_pd,neg_pd = Prepare_data_fix_LAMBDA(gene, hets, depth, sigma,model="BEASTIE3",source=source,workdir=workdir,datatype=datatype,lamdbda=lamdbda1,theta_pos=theta_pos,theta_neg=theta_neg,Num_para=3)
    if Num_col == None:
        Num_col = 3
    else:
        Num_col = int(Num_col)
    row = math.ceil(float(len(d_group))/Num_col)
    fig, axs = plt.subplots(row, Num_col, figsize = (20,5*row))
    if (row * Num_col > len(d_group)):
        for i in range(row * Num_col - len(d_group)):
            axs.flat[-1-i].set_axis_off()
    print(d_group)
    #d_group=d_group[:-1]
    xlabels = "Fixed parameters "
    for i, each in enumerate(d_group):
        #print(i)
        #print(each)
        current_group_pos_list = pos_pd[pos_pd[var_map_np[np.array(var) == None][0]] == each].index
        current_group_neg_list = neg_pd[neg_pd[var_map_np[np.array(var) == None][0]] == each].index
        xlabels = "Fixed parameters "
        for idx in range(len(current_group_pos_list)):
            #g,h,d,s,reduced_file_pos,reduced_file_neg =  decompose(current_group_pos_list[idx])
            reduced_file_pos = current_group_pos_list[idx].rsplit("_",1)[0]+".pickle"
            reduced_file_neg = current_group_neg_list[idx].rsplit("_",1)[0]+".pickle"
            g=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[0].rsplit("-")[1]
            h=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[1].rsplit("-")[1]
            d=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[2].rsplit("-")[1]
            s=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[4].rsplit("-")[1]
            fpr_m,tpr_m,fpr_m2,tpr_m2,fpr_b,tpr_b,fpr_b2,tpr_b2,fpr_b_p,tpr_b_p,fpr_a,tpr_a,fpr_1,tpr_1,fpr_2,tpr_2 = Get_tpr_fpr_LAMBDA_selected(path_model, path_lambda1, path_model2,path_lambda2,path_base, path_base2,path_base_p,path_AA,reduced_file_pos,reduced_file_neg,current_group_pos_list[idx],current_group_neg_list[idx])
            var_dict = {"gene":g, "hets": h, "depth": d, "sigma": s}
            for each in fixed_var_np:
                xlabels += each+":"+var_dict[each]+' '
            axs.flat[i].set_xlabel("FPR")
            axs.flat[i].set_ylabel("TPR")
            axs.flat[i].set_title(var_fullname_map_np[np.array(var) == None][0]+":" + var_dict[var_fullname_map_np[np.array(var) == None][0]])
            #axs.flat[i].plot(fpr_m, tpr_m, label =  "SPP2-odd :"+str(round(auc(fpr_m,tpr_m),3)),linewidth=3)
            axs.flat[i].plot(fpr_1, tpr_1, label =  "SPP2-odd lambda "+str(lamdbda1)+" :"+str(round(auc(fpr_1,tpr_1),3)),linewidth=3)
            #axs.flat[i].plot(fpr_m2, tpr_m2, label =  "BEASTIE3 :"+str(round(auc(fpr_m2,tpr_m2),3)),linewidth=3)
            axs.flat[i].plot(fpr_2, tpr_2, label =  "BEASTIE3 lambda"+str(lamdbda2)+" :"+str(round(auc(fpr_2,tpr_2),3)),linewidth=3)
            #axs.flat[i].plot(fpr_b, tpr_b, label =  "Binomial 2-side: "+str(round(auc(fpr_b,tpr_b),3)),linewidth=3)
            axs.flat[i].plot(fpr_b2, tpr_b2, label =  "Binomial (1st) 2-side: "+str(round(auc(fpr_b2,tpr_b2),3)),linewidth=3)
            axs.flat[i].plot(fpr_b_p, tpr_b_p, label =  "Binomial pool 2-side: "+str(round(auc(fpr_b_p,tpr_b_p),3)),linewidth=3)
            #axs.flat[i].plot(fpr_a, tpr_a, label =  "ADM: "+str(round(auc(fpr_a,tpr_a),3)),linewidth=3)
            axs.flat[i].legend(fontsize=10,loc='lower right')
    plt.suptitle(str(title)+"| "+str(xlabels)+" |theta "+str(theta_pos)+"vs theta"+str(theta_neg),fontsize=20)
    plt.show()

def Prepare_data_fix_LAMBDA(gene, hets, depth, sigma,model,source,workdir,datatype,lamdbda,theta_pos,theta_neg,Num_para):
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
    
    if lamdbda != None or lamdbda != False:
        _,_,_,_,_,_,_,_,path= Generate_path_LAMBDA(model=model,source=source,workdir=workdir,datatype=datatype,lamdbda=lamdbda)
    else:
        path,_,_,_,_,_,_,_,_= Generate_path_LAMBDA(model=model,source=source,workdir=workdir,datatype=datatype,lamdbda=lamdbda)
    all_file = sorted(os.listdir(path))
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

def Get_tpr_fpr_LAMBDA_selected(path_model, path_lambda1, path_model2,path_lambda2,path_base, path_base2,path_base_p,path_AA,reduced_file_pos,reduced_file_neg,full_file_pos,full_file_neg,if_prob=True):
    fpr_m=None
    tpr_m=None
    fpr_m2=None
    tpr_m2=None
    fpr_1=None
    tpr_1=None
    fpr_2=None
    tpr_2=None
    fpr_b=None
    tpr_b=None
    fpr_a=None
    tpr_a=None
    #fpr_m,tpr_m = get_ROC_AUC_V3(path_model,full_file_pos,full_file_neg,if_prob=if_prob)
    #fpr_m2,tpr_m2 = get_ROC_AUC_V3(path_model2,full_file_pos,full_file_neg,if_prob=if_prob)
    #fpr_1,tpr_1 = get_ROC_AUC_V3(path_lambda1,full_file_pos,full_file_neg,if_prob=if_prob)
    fpr_2,tpr_2 = ROC_common.get_ROC_AUC_V3(path_lambda2,full_file_pos,full_file_neg,if_prob=if_prob)
    #fpr_b,tpr_b = get_ROC_AUC_V3(path_base,reduced_file_pos,reduced_file_neg,if_prob=if_prob,if_baseline=True)
    fpr_b2,tpr_b2 = ROC_common.get_ROC_AUC_V3(path_base2,reduced_file_pos,reduced_file_neg,if_prob=if_prob,if_baseline=True)
    fpr_b_p,tpr_b_p = ROC_common.get_ROC_AUC_V3(path_base_p,reduced_file_pos,reduced_file_neg,if_prob=if_prob,if_baseline=True)
    #fpr_a,tpr_a = get_ROC_AUC_V3(path_AA,reduced_file_pos,reduced_file_neg,if_prob=if_prob,if_AA=True)
    return fpr_m,tpr_m,fpr_m2,tpr_m2,fpr_b,tpr_b,fpr_b2,tpr_b2,fpr_b_p,tpr_b_p,fpr_a,tpr_a,fpr_1,tpr_1,fpr_2,tpr_2

def Get_tpr_fpr_LAMBDA(path_model, path_base, path_base_p,path_AA,path_lambda1,path_lambda2,path_lambda3,path_lambda4,path_lambda5,reduced_file_pos,reduced_file_neg,full_file_pos,full_file_neg,if_prob=True):
    fpr_m=None
    tpr_m=None
    fpr_1=None  
    tpr_1=None
    fpr_2=None
    tpr_2=None
    fpr_3=None
    tpr_3=None
    fpr_4=None
    tpr_4=None
    fpr_5=None
    tpr_5=None
    if path_model != None:
        fpr_m,tpr_m = ROC_common.get_ROC_AUC_V3(path_model,full_file_pos,full_file_neg,if_prob=if_prob)
    if path_lambda1 != None:
        fpr_1,tpr_1 = ROC_common.get_ROC_AUC_V3(path_lambda1,full_file_pos,full_file_neg,if_prob=if_prob)
        fpr_2,tpr_2 = ROC_common.get_ROC_AUC_V3(path_lambda2,full_file_pos,full_file_neg,if_prob=if_prob)
        fpr_3,tpr_3 = ROC_common.get_ROC_AUC_V3(path_lambda3,full_file_pos,full_file_neg,if_prob=if_prob)
        fpr_4,tpr_4 = ROC_common.get_ROC_AUC_V3(path_lambda4,full_file_pos,full_file_neg,if_prob=if_prob)
        fpr_5,tpr_5 = ROC_common.get_ROC_AUC_V3(path_lambda5,full_file_pos,full_file_neg,if_prob=if_prob)
    fpr_b,tpr_b = ROC_common.get_ROC_AUC_V3(path_base,reduced_file_pos,reduced_file_neg,if_prob=if_prob,if_baseline=True)
    fpr_b_p,tpr_b_p = ROC_common.get_ROC_AUC_V3(path_base_p,reduced_file_pos,reduced_file_neg,if_prob=if_prob,if_baseline=True)
    fpr_a,tpr_a = ROC_common.get_ROC_AUC_V3(path_AA,reduced_file_pos,reduced_file_neg,if_prob=if_prob,if_AA=True)
    return fpr_m,tpr_m,fpr_b,tpr_b,fpr_b_p,tpr_b_p,fpr_a,tpr_a,fpr_1,tpr_1,fpr_2,tpr_2,fpr_3,tpr_3,fpr_4,tpr_4,fpr_5,tpr_5

def Find_cutoff_from_Null_allmodels(model, percentError,theta_alt,cutoff,model2=None,gene=None,hets=None,depth=None,sigma=None,num=3,if_AA=False,calculation="max_prob"):
    source="/data2/stan/"
    # judge whether  2 of the 4 are None, and 2 of the 4 are not None
    theta_pos=theta_alt
    theta_neg=1

    var=[gene,hets,depth,sigma]        
    if var.count(None)!=2:
        raise Exception('Two variables have to be set to None. The number of None from input was {}'.format(var.count(None)))
    # data directory is known 
    if if_AA:
        path_AA=f"{source}/ADM/{percentError}/AA_pval"
    else:   
        path_AA=None

    path_model,path_NS,path_MS,path_model2,path_beta1,path_beta2,path_beta3 = power.Generate_path_power(source=source,model=model,sigma=sigma,workdir=percentError,model2=model2,calculation=calculation)
    d_group,var,var_map_np,fixed_var_np,var_fullname_map_np,variable_var_np,pos_pd,neg_pd = power.Prepare_data_fix(gene, hets, depth, sigma,source,model,percentError,theta_pos,theta_neg=1,Num_para=2)

    ############################################################
    read_depth = pos_pd[var_map_np[np.array(var) == None][1]].unique()

    if num == None:
        num = 3
    else:
        num = int(num)
    row = math.ceil(float(len(d_group))/num)
    fig, axs = plt.subplots(row, num, figsize = (25,8*row))
    if (row * num > len(d_group)):
        for i in range(row * num - len(d_group)):
            axs.flat[-1-i].set_axis_off()

    xlabels = "Data with %s percent error, gene: %s , sigma: %s, cutoff set at %s-th percentile of NULL"%(str(percentError),gene,sigma,cutoff)

    labels = ""
    #df1 = pd.DataFrame({'Model':[],'Het':[],'d5':[], 'd10':[], 'd20':[], 'd30':[], 'd40':[], 'd50':[], 'd60':[], 'd70':[], 'd80':[], 'd90':[], 'd100':[]})
    #df2=df1
    #df3=df1
    #df4=df1
    #df5=df1
    #df6=df1
    #df7=df1
    #df8=df1
    for i, each in enumerate(d_group):
        current_group_pos_list = pos_pd[pos_pd[var_map_np[np.array(var) == None][0]] == each].index
        current_group_neg_list = neg_pd[neg_pd[var_map_np[np.array(var) == None][0]] == each].index
        power_model_list=[]
        power_model2_list=[]
        power_NS_list=[]
        power_MS_list=[]
        power_adam_list=[]
        power_beta11_list=[]
        power_beta1010_list=[]
        power_beta2020_list=[]
        #
        cutoff_model_list=[]
        cutoff_adam_list=[]
        cutoff_NS_list=[]
        cutoff_MS_list=[]
        cutoff_model2_list=[]
        cutoff_beta11_list=[]
        cutoff_beta1010_list=[]
        cutoff_beta2020_list=[]

        for idx in range(len(current_group_pos_list)):
            #print(current_group_pos_list[idx])
            reduced_file_pos = current_group_pos_list[idx].rsplit("_",1)[0]+".pickle"
            reduced_file_neg = current_group_neg_list[idx].rsplit("_",1)[0]+".pickle"
            cutoff1,power1=power.Calculate_cutoff(current_group_pos_list[idx],current_group_neg_list[idx],path_model,cutoff,calculation=calculation)
            cutoff2,power2=power.Calculate_cutoff(reduced_file_pos,reduced_file_neg,path_NS,cutoff,if_AA_baseline=True,calculation=calculation)
            cutoff3,power3=power.Calculate_cutoff(reduced_file_pos,reduced_file_neg,path_MS,cutoff,if_AA_baseline=True,calculation=calculation)
            if if_AA:
                cutoff4,power4=power.Calculate_cutoff(reduced_file_pos,reduced_file_neg,path_AA,cutoff,if_AA_baseline=True,calculation=calculation)
                cutoff_adam_list.append(cutoff4)
                power_adam_list.append(power4)
            cutoff5,power5=power.Calculate_cutoff(current_group_pos_list[idx],current_group_neg_list[idx],path_model2,cutoff,calculation=calculation)
            #print(">> beta(1,1)")
            if path_beta1 is not None:
                cutoff6,power6=power.Calculate_cutoff(reduced_file_pos,reduced_file_neg,path_beta1,cutoff,if_beta=True,calculation=calculation)
                #print(">> beta(10,10)")
                cutoff7,power7=power.Calculate_cutoff(reduced_file_pos,reduced_file_neg,path_beta2,cutoff,if_beta=True,calculation=calculation)
                #print(">> beta(20,20)")
                cutoff8,power8=power.Calculate_cutoff(reduced_file_pos,reduced_file_neg,path_beta3,cutoff,if_beta=True,calculation=calculation)
                cutoff_beta11_list.append(cutoff6)
                cutoff_beta1010_list.append(cutoff7)
                cutoff_beta2020_list.append(cutoff8)
                power_beta11_list.append(power6)
                power_beta1010_list.append(power7)
                power_beta2020_list.append(power8)
            # 
            cutoff_model_list.append(cutoff1)
            cutoff_NS_list.append(cutoff2)
            cutoff_MS_list.append(cutoff3)
            cutoff_model2_list.append(cutoff5)
            power_model_list.append(power1) 
            power_NS_list.append(power2)
            power_MS_list.append(power3)
            power_model2_list.append(power5) 


        # for each read depth, we plot    
        g=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[0].rsplit("-")[1]
        h=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[1].rsplit("-")[1]
        d=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[2].rsplit("-")[1]
        s=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[4].rsplit("-")[1]
        var_dict = {"gene":g, "hets": h, "depth": d, "sigma": s}
        for each in variable_var_np:
            if each != var_fullname_map_np[np.array(var) == None][0]:
                labels += each+":"+var_dict[each]+' '
        axs.flat[i].set_ylabel("Power",fontsize=20)
        axs.flat[i].set_xlabel("Read Depth per het site",fontsize=15)
        axs.flat[i].set_ylim(0,1.1)
        axs.flat[i].set_xlim(0,100)
        axs.flat[i].plot(read_depth, power_model_list,'--o',label=str(model))
        axs.flat[i].plot(read_depth, power_model2_list,'--o',label=str(model2))
        axs.flat[i].plot(read_depth, power_NS_list,'--o',label="Naive Sum")
        axs.flat[i].plot(read_depth, power_MS_list,'--o',label="Major Site")
        if path_beta1 is not None:
            axs.flat[i].plot(read_depth, power_beta11_list,'--o',label="betabinom (1,1)")
            axs.flat[i].plot(read_depth, power_beta1010_list,'--o',label="betabinom (10,10)")
            axs.flat[i].plot(read_depth, power_beta2020_list,'--o',label="betabinom (20,20)")
        if if_AA:
            axs.flat[i].plot(read_depth, power_adam_list,'--bo',label="ADAM")
        axs.flat[i].axhline(y=0.9,color='darkred',alpha=0.80,linestyle='--')
        axs.flat[i].axhline(y=0.8,color='red',alpha=0.80,linestyle='--')
        axs.flat[i].axhline(y=0.7,color='orangered',alpha=0.80,linestyle='--')
        axs.flat[i].axhline(y=0.6,color='lightsalmon',alpha=0.80,linestyle='--') 
        axs.flat[i].legend(loc='lower right',fontsize=15)
        axs.flat[i].set_title(var_fullname_map_np[np.array(var) == None][0]+":" + var_dict[var_fullname_map_np[np.array(var) == None][0]],fontsize=20)
        #
        #df1 = pd.concat([df1,pd.DataFrame([{'Model':str(model)+" "+str(percentError),'Het':var_dict[var_fullname_map_np[np.array(var) == None][0]],'d5':round(power_model_list[0],4), 'd10':round(power_model_list[1],4), 'd20':round(power_model_list[2],4), 'd30':round(power_model_list[3],4), 'd40':round(power_model_list[4],4), 'd50':round(power_model_list[5],4), 'd60':round(power_model_list[6],4), 'd70':round(power_model_list[7],4), 'd80':round(power_model_list[8],4), 'd90':round(power_model_list[9],4), 'd100':round(power_model_list[10],4)}])],ignore_index=True)
        #df3 = pd.concat([df3,pd.DataFrame([{'Model':"NS "+str(percentError),'Het':var_dict[var_fullname_map_np[np.array(var) == None][0]],'d5':round(power_NS_list[0],4), 'd10':round(power_NS_list[1],4), 'd20':round(power_NS_list[2],4), 'd30':round(power_NS_list[3],4), 'd40':round(power_NS_list[4],4), 'd50':round(power_NS_list[5],4), 'd60':round(power_NS_list[6],4), 'd70':round(power_NS_list[7],4), 'd80':round(power_NS_list[8],4), 'd90':round(power_NS_list[9],4), 'd100':round(power_NS_list[10],4)}])],ignore_index=True)
        #df4 = pd.concat([df4,pd.DataFrame([{'Model':"MS "+str(percentError),'Het':var_dict[var_fullname_map_np[np.array(var) == None][0]],'d5':round(power_MS_list[0],4), 'd10':round(power_MS_list[1],4), 'd20':round(power_MS_list[2],4), 'd30':round(power_MS_list[3],4), 'd40':round(power_MS_list[4],4), 'd50':round(power_MS_list[5],4), 'd60':round(power_MS_list[6],4), 'd70':round(power_MS_list[7],4), 'd80':round(power_MS_list[8],4), 'd90':round(power_MS_list[9],4), 'd100':round(power_MS_list[10],4)}])],ignore_index=True)
        #if if_AA:
        #    df2 = pd.concat([df2,pd.DataFrame([{'Model':"ADAM "+str(percentError),'Het':var_dict[var_fullname_map_np[np.array(var) == None][0]],'d5':round(power_adam_list[0],4), 'd10':round(power_adam_list[1],4), 'd20':round(power_adam_list[2],4), 'd30':round(power_adam_list[3],4), 'd40':round(power_adam_list[4],4), 'd50':round(power_adam_list[5],4), 'd60':round(power_adam_list[6],4), 'd70':round(power_adam_list[7],4), 'd80':round(power_adam_list[8],4), 'd90':round(power_adam_list[9],4), 'd100':round(power_adam_list[10],4)}])],ignore_index=True)
        #df5 = pd.concat([df5,pd.DataFrame([{'Model':str(model2)+" "+str(percentError),'Het':var_dict[var_fullname_map_np[np.array(var) == None][0]],'d5':round(power_model2_list[0],4), 'd10':round(power_model2_list[1],4), 'd20':round(power_model2_list[2],4), 'd30':round(power_model2_list[3],4), 'd40':round(power_model2_list[4],4), 'd50':round(power_model2_list[5],4), 'd60':round(power_model2_list[6],4), 'd70':round(power_model2_list[7],4), 'd80':round(power_model2_list[8],4), 'd90':round(power_model2_list[9],4), 'd100':round(power_model2_list[10],4)}])],ignore_index=True)
        #if path_beta1 is not None:
        #    df6 = pd.concat([df6,pd.DataFrame([{'Model':"beta(1,1) "+str(percentError),'Het':var_dict[var_fullname_map_np[np.array(var) == None][0]],'d5':round(power_beta11_list[0],4), 'd10':round(power_beta11_list[1],4), 'd20':round(power_beta11_list[2],4), 'd30':round(power_beta11_list[3],4), 'd40':round(power_beta11_list[4],4), 'd50':round(power_NS_list[5],4), 'd60':round(power_beta11_list[6],4), 'd70':round(power_beta11_list[7],4), 'd80':round(power_beta11_list[8],4), 'd90':round(power_beta11_list[9],4), 'd100':round(power_beta11_list[10],4)}])],ignore_index=True)
        #    df7 = pd.concat([df7,pd.DataFrame([{'Model':"beta(10,10) "+str(percentError),'Het':var_dict[var_fullname_map_np[np.array(var) == None][0]],'d5':round(power_beta1010_list[0],4), 'd10':round(power_beta1010_list[1],4), 'd20':round(power_beta1010_list[2],4), 'd30':round(power_beta1010_list[3],4), 'd40':round(power_beta1010_list[4],4), 'd50':round(power_beta1010_list[5],4), 'd60':round(power_beta1010_list[6],4), 'd70':round(power_beta1010_list[7],4), 'd80':round(power_beta1010_list[8],4), 'd90':round(power_beta1010_list[9],4), 'd100':round(power_beta1010_list[10],4)}])],ignore_index=True)
        #    df8 = pd.concat([df8,pd.DataFrame([{'Model':"beta(20,20) "+str(percentError),'Het':var_dict[var_fullname_map_np[np.array(var) == None][0]],'d5':round(power_beta2020_list[0],4), 'd10':round(power_beta2020_list[1],4), 'd20':round(power_beta2020_list[2],4), 'd30':round(power_beta2020_list[3],4), 'd40':round(power_beta2020_list[4],4), 'd50':round(power_beta2020_list[5],4), 'd60':round(power_beta2020_list[6],4), 'd70':round(power_beta2020_list[7],4), 'd80':round(power_beta2020_list[8],4), 'd90':round(power_beta2020_list[9],4), 'd100':round(power_beta2020_list[10],4)}])],ignore_index=True)

    plt.suptitle(xlabels,fontsize=25)
    plt.show()

    #df1.set_index(['Model', 'Het'])
    #df2.set_index(['Model', 'Het'])
    #df3.set_index(['Model', 'Het'])
    #df4.set_index(['Model', 'Het'])
    #df5.set_index(['Model', 'Het'])
    #df6.set_index(['Model', 'Het'])
    #df7.set_index(['Model', 'Het'])
    #df8.set_index(['Model', 'Het'])

    #final_df=pd.concat([df1,df5],axis=0,ignore_index=True)
    #final_df=pd.concat([final_df,df3],axis=0,ignore_index=True)
    #final_df=pd.concat([final_df,df4],axis=0,ignore_index=True)

    #return final_df,df1,df5,df3,df4