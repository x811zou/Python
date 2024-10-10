"""
Functions defined below are for model comparison using ROC score, no lambda results are shown
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
import sys
import matplotlib.pyplot as plt
from sklearn import metrics,preprocessing
from sklearn.metrics import roc_curve, auc,precision_recall_curve
import ROC_common
import read_data
import quickBeast

def ROC_comparison_fix3_beta(source,model,workdir,calculation,theta_pos=None,theta_neg=None,gene=None,hets=None,depth=None,sigma=None,title=None,Num_para=3,Num_col=None):
    # judge whether  2 of the 4 are None, and 2 of the 4 are not None
    ROC_common.Make_judgement(Num_para=Num_para,theta_pos=theta_pos,theta_neg=theta_neg,gene=gene,hets=hets,depth=depth,sigma=sigma)
    Plot_ROC_fix3_beta(source,model=model,workdir=workdir,calculation=calculation,Num_col=Num_col,gene=gene, hets=hets, depth=depth, sigma=sigma,theta_pos=theta_pos,theta_neg=theta_neg,title=title)

def ROC_comparison_fix3_sigma(source,model,workdir,calculation,theta_pos=None,theta_neg=None,gene=None,hets=None,depth=None,sigma=None,title=None,Num_para=3,Num_col=None):
    # judge whether  2 of the 4 are None, and 2 of the 4 are not None
    #ROC_common.Make_judgement(Num_para=Num_para,theta_pos=theta_pos,theta_neg=theta_neg,gene=gene,hets=hets,depth=depth,sigma=sigma)
    Plot_ROC_fix3_sigma(source,model=model,workdir=workdir,calculation=calculation,Num_col=Num_col,gene=gene, hets=hets, depth=depth, sigma=sigma,theta_pos=theta_pos,theta_neg=theta_neg,title=title)

def ROC_comparison_fix3_Lambda(source,model,workdir,calculation,lambda_model,chosen_lambda=None,theta_pos=None,theta_neg=None,gene=None,hets=None,depth=None,sigma=None,title=None,Num_para=3,Num_col=None):
    # judge whether  2 of the 4 are None, and 2 of the 4 are not None
    ROC_common.Make_judgement(Num_para=Num_para,theta_pos=theta_pos,theta_neg=theta_neg,gene=gene,hets=hets,depth=depth,sigma=sigma)
    Plot_ROC_fix3_Lambda(source,model=model,workdir=workdir,calculation=calculation,lambda_model=lambda_model,chosen_lambda=chosen_lambda,Num_col=Num_col,gene=gene, hets=hets, depth=depth,sigma=sigma, theta_pos=theta_pos,theta_neg=theta_neg,title=title)

def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))

def get_lambda_from_gam(
    model, hets, totalcount, expected_type1error, candidate_log_lambdas
):
    # prepare input
    data = [[hets, totalcount, lam] for lam in candidate_log_lambdas]

    # prediction
    prediction = inv_logit(model.predict(data))
    chosen_lambda = 3
    if min(prediction) <= expected_type1error:
        chosen_lambda = np.exp(
            data[np.where(prediction <= expected_type1error)[0][0]][2]
        )
    return chosen_lambda

def Plot_ROC_fix3_Lambda(source,model,workdir,calculation,lambda_model,chosen_lambda,Num_col,gene, hets, depth, sigma,theta_pos,theta_neg,title):
    gam_model_path="/home/scarlett/github/BEASTIE/BEASTIE/iBEASTIE4_s0.7_GAM/"
    gam_model = pickle.load(open(gam_model_path+str(lambda_model), "rb"))
    candidate_log_lambdas = np.log(np.linspace(1, 3, 3000))

    path_model,path_NS,path_MS = Generate_path(source,model,sigma,workdir)
    d_group,var,var_map_np,fixed_var_np,var_fullname_map_np,variable_var_np,pos_pd,neg_pd = Prepare_data_fix(gene, hets, depth, sigma,source,model,workdir,calculation,theta_pos,theta_neg=1,Num_para=3)
    if Num_col == None:
        Num_col = 3
    else:
        Num_col = int(Num_col)
    row = math.ceil(float(len(d_group))/Num_col)
    fig, axs = plt.subplots(row, Num_col, figsize = (15,5*row))
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
            reduced_file_pos = current_group_pos_list[idx].rsplit("_",1)[0]+".pickle"
            reduced_file_neg = current_group_neg_list[idx].rsplit("_",1)[0]+".pickle"                                  
            g=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[0].rsplit("-")[1]
            h=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[1].rsplit("-")[1]
            d=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[2].rsplit("-")[1]
            s=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[4].rsplit("-")[1]
            #
            if chosen_lambda is None:
                chosen_lambda=get_lambda_from_gam(gam_model,int(h),int(h)*int(d),0.05/int(g),candidate_log_lambdas)
            # calculate auc score
            fpr_m,tpr_m,s1 = get_ROC_AUC(path_model,current_group_pos_list[idx],current_group_neg_list[idx],calculation,lambdas=chosen_lambda)
            fpr_b,tpr_b,s2 = get_ROC_AUC(path_NS,reduced_file_pos,reduced_file_neg,calculation,if_baseline=True)
            fpr_b2,tpr_b2,s3 = get_ROC_AUC(path_MS,reduced_file_pos,reduced_file_neg,calculation,if_baseline=True)
            # 
            var_dict = {"gene":g, "hets": h, "depth": d, "sigma": s}
            for each in fixed_var_np:
                xlabels += each+":"+var_dict[each]+' '
            axs.flat[i].set_xlabel("FPR",fontsize=12)
            axs.flat[i].set_ylabel("TPR",fontsize=12)
            axs.flat[i].set_title(var_fullname_map_np[np.array(var) == None][0]+":" + var_dict[var_fullname_map_np[np.array(var) == None][0]],fontsize=12)
            axs.flat[i].plot(fpr_m, tpr_m, label =  str(model)+" :"+str(s1),alpha=0.8,linewidth=3)
            axs.flat[i].tick_params(axis='both', labelsize=10)
            axs.flat[i].plot(fpr_b, tpr_b, label =  "Naive Sum : "+str(s2),alpha=0.8,linewidth=3)
            axs.flat[i].plot(fpr_b2, tpr_b2, label =  "Major Site : "+str(s3),alpha=0.8,linewidth=3)
            axs.flat[i].legend(fontsize=15,loc='lower right')
    plt.suptitle(str(title)+"\n"+xlabels+"theta pos: "+str(theta_pos)+" theta neg: "+str(theta_neg),fontsize=20)
    plt.show()

def Plot_ROC_fix3_Lambda_GIAB(source,model,workdir,calculation,lambda_model,chosen_lambda,Num_col,gene, hets, depth, sigma,theta_pos,theta_neg,title):
    gam_model_path="/home/scarlett/github/BEASTIE/BEASTIE/"
    gam_model = pickle.load(open(gam_model_path+str(lambda_model), "rb"))
    candidate_log_lambdas = np.log(np.linspace(1, 3, 3000))

    path_model,path_GIAB, path_NS,path_MS = Generate_path_GIAB(source,model,sigma,workdir)
    d_group,var,var_map_np,fixed_var_np,var_fullname_map_np,variable_var_np,pos_pd,neg_pd = Prepare_data_fix(gene, hets, depth, sigma,source,model,workdir,calculation,theta_pos,theta_neg=1,Num_para=3)
    if Num_col == None:
        Num_col = 3
    else:
        Num_col = int(Num_col)
    row = math.ceil(float(len(d_group))/Num_col)
    fig, axs = plt.subplots(row, Num_col, figsize = (15,5*row))
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
        xlabels = title+", fixed parameters theta:"+str(theta_pos)+" "
        for idx in range(len(current_group_pos_list)):
            reduced_file_pos = current_group_pos_list[idx].rsplit("_",1)[0]+".pickle"
            reduced_file_neg = current_group_neg_list[idx].rsplit("_",1)[0]+".pickle"                                  
            g=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[0].rsplit("-")[1]
            h=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[1].rsplit("-")[1]
            d=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[2].rsplit("-")[1]
            s=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[4].rsplit("-")[1]
            #
            if chosen_lambda is None:
                chosen_lambda=get_lambda_from_gam(gam_model,int(h),int(h)*int(d),0.00001,candidate_log_lambdas)
            # calculate auc score
            fpr_m,tpr_m,s1 = get_ROC_AUC(path_model,current_group_pos_list[idx],current_group_neg_list[idx],calculation,lambdas=chosen_lambda)
            fpr_m_GIAB,tpr_m_GIAB,s2 = get_ROC_AUC(path_GIAB,current_group_pos_list[idx],current_group_neg_list[idx],calculation,lambdas=chosen_lambda)
            fpr_b,tpr_b,s3 = get_ROC_AUC(path_NS,reduced_file_pos,reduced_file_neg,calculation,if_baseline=True)
            fpr_b2,tpr_b2,s4 = get_ROC_AUC(path_MS,reduced_file_pos,reduced_file_neg,calculation,if_baseline=True)
            # 
            var_dict = {"gene":g, "hets": h, "depth": d, "sigma": s}
            for each in fixed_var_np:
                xlabels += each+":"+var_dict[each]+' '
            axs.flat[i].set_xlabel("FPR",fontsize=12)
            axs.flat[i].set_ylabel("TPR",fontsize=12)
            axs.flat[i].set_title(var_fullname_map_np[np.array(var) == None][0]+":" + var_dict[var_fullname_map_np[np.array(var) == None][0]],fontsize=12)
            axs.flat[i].plot(fpr_m, tpr_m, label =  "BEASTIE :"+str(s1),alpha=0.8,linewidth=3)
            axs.flat[i].tick_params(axis='both', labelsize=10)
            axs.flat[i].plot(fpr_b, tpr_b, label =  "Naive Sum : "+str(s3),alpha=0.8,linewidth=3)
            axs.flat[i].plot(fpr_b2, tpr_b2, label =  "Major Site : "+str(s4),alpha=0.8,linewidth=3)
            axs.flat[i].plot(fpr_m_GIAB, tpr_m_GIAB, label =  "iBEASTIE :"+str(s2),alpha=0.8,linewidth=3)
            axs.flat[i].legend(fontsize=15,loc='lower right')
    plt.suptitle(str(title)+"\n"+xlabels+"theta pos: "+str(theta_pos)+" theta neg: "+str(theta_neg),fontsize=20)
    plt.show()

def Plot_ROC_fix3_beta(source,model,workdir,calculation,Num_col,gene, hets, depth, sigma,theta_pos,theta_neg,title):
    path_model,path_model_beta,path_NS,path_MS = Generate_path_noLAMBDA_beta(source,model,sigma,workdir,calculation)
    d_group,var,var_map_np,fixed_var_np,var_fullname_map_np,variable_var_np,pos_pd,neg_pd = Prepare_data_fix(gene, hets, depth, sigma,source,model,workdir,calculation,theta_pos,theta_neg=1,Num_para=3)
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
            fpr_m,tpr_m,fpr_m2,tpr_m2,fpr_b,tpr_b,fpr_b2,tpr_b2,fpr_a,tpr_a = Get_tpr_fpr_noLAMBDA(path_model, path_model_beta,path_NS,path_MS,reduced_file_pos,reduced_file_neg,current_group_pos_list[idx],current_group_neg_list[idx],calculation=calculation)
            var_dict = {"gene":g, "hets": h, "depth": d, "sigma": s}
            for each in fixed_var_np:
                xlabels += each+":"+var_dict[each]+' '
            axs.flat[i].set_xlabel("FPR")
            axs.flat[i].set_ylabel("TPR")
            axs.flat[i].set_title(var_fullname_map_np[np.array(var) == None][0]+":" + var_dict[var_fullname_map_np[np.array(var) == None][0]])
            axs.flat[i].plot(fpr_m, tpr_m, label =  str(model)+" :"+str(round(auc(fpr_m,tpr_m),3)),alpha=0.8,linewidth=3)
            axs.flat[i].plot(fpr_m2, tpr_m2, label =  str(model)+"_beta0.05 :"+str(round(auc(fpr_m2,tpr_m2),3)),alpha=0.8,linewidth=3)
            # if lamdbda != None:
            #     axs.flat[i].plot(fpr_1, tpr_1, label =  str(model)+" lambda 1.2 :"+str(round(auc(fpr_1,tpr_1),3)),linewidth=3)
            #     axs.flat[i].plot(fpr_2, tpr_2, label =  str(model)+" lambda 1.4: "+str(round(auc(fpr_2,tpr_2),3)),linewidth=3)
            #     axs.flat[i].plot(fpr_3, tpr_3, label =  str(model)+" lambda 1.6: "+str(round(auc(fpr_3,tpr_3),3)),linewidth=3)
            #     axs.flat[i].plot(fpr_4, tpr_4, label =  str(model)+" lambda 1.8: "+str(round(auc(fpr_4,tpr_4),3)),linewidth=3)
            #     axs.flat[i].plot(fpr_5, tpr_5, label =  str(model)+" lambda 2: "+str(round(auc(fpr_5,tpr_5),3)),linewidth=3)
            axs.flat[i].plot(fpr_b, tpr_b, label =  "NS : "+str(round(auc(fpr_b,tpr_b),3)),alpha=0.8,linewidth=3)
            axs.flat[i].plot(fpr_b2, tpr_b2, label =  "MS : "+str(round(auc(fpr_b2,tpr_b2),3)),alpha=0.8,linewidth=3)
            axs.flat[i].plot(fpr_a, tpr_a, label =  "ADM : "+str(round(auc(fpr_a,tpr_a),3)),alpha=0.8,linewidth=3)
            axs.flat[i].legend(fontsize=10,loc='lower right')
    print(xlabels+"theta pos: "+str(theta_pos)+" theta neg: "+str(theta_neg))
    plt.suptitle(str(title),fontsize=20)
    plt.show()

def Plot_ROC_fix3_sigma(source,model,workdir,calculation,Num_col,gene, hets, depth,sigma,theta_pos,theta_neg,title):
    path_model1,path_model2,path_model3,path_NS,path_MS,path_ADM = Generate_path_noLAMBDA_sigma(source,model,workdir,calculation)
    d_group,var,var_map_np,fixed_var_np,var_fullname_map_np,variable_var_np,pos_pd,neg_pd = Prepare_data_fix(gene, hets, depth, sigma,source,model,workdir,calculation,theta_pos,theta_neg=1,Num_para=3)
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
            fpr_m,tpr_m,fpr_m2,tpr_m2,fpr_m3,tpr_m3,fpr_b,tpr_b,fpr_b2,tpr_b2 = Get_tpr_fpr_noLAMBDA_sigma(path_model1,path_model2,path_model3,path_NS,path_MS,path_ADM,reduced_file_pos,reduced_file_neg,current_group_pos_list[idx],current_group_neg_list[idx],calculation=calculation)
            var_dict = {"gene":g, "hets": h, "depth": d, "sigma": s}
            for each in fixed_var_np:
                xlabels += each+":"+var_dict[each]+' '
            axs.flat[i].set_xlabel("FPR")
            axs.flat[i].set_ylabel("TPR")
            axs.flat[i].set_title(var_fullname_map_np[np.array(var) == None][0]+":" + var_dict[var_fullname_map_np[np.array(var) == None][0]])
            axs.flat[i].plot(fpr_b, tpr_b, label =  "NS : "+str(round(auc(fpr_b,tpr_b),3)),alpha=0.8,linewidth=3)
            axs.flat[i].plot(fpr_b2, tpr_b2, label =  "MS : "+str(round(auc(fpr_b2,tpr_b2),3)),alpha=0.8,linewidth=3)
            axs.flat[i].plot(fpr_m, tpr_m, label =  str(model)+" sigma 0.5 :"+str(round(auc(fpr_m,tpr_m),3)),alpha=0.8,linewidth=3)
            axs.flat[i].plot(fpr_m2, tpr_m2, label =  str(model)+" sigma 0.7 :"+str(round(auc(fpr_m2,tpr_m2),3)),alpha=0.8,linewidth=3)
            axs.flat[i].plot(fpr_m3, tpr_m3, label =  str(model)+" sigma 1.0 :"+str(round(auc(fpr_m3,tpr_m3),3)),alpha=0.8,linewidth=3)
            # if lamdbda != None:
            #     axs.flat[i].plot(fpr_1, tpr_1, label =  str(model)+" lambda 1.2 :"+str(round(auc(fpr_1,tpr_1),3)),linewidth=3)
            #     axs.flat[i].plot(fpr_2, tpr_2, label =  str(model)+" lambda 1.4: "+str(round(auc(fpr_2,tpr_2),3)),linewidth=3)
            #     axs.flat[i].plot(fpr_3, tpr_3, label =  str(model)+" lambda 1.6: "+str(round(auc(fpr_3,tpr_3),3)),linewidth=3)
            #     axs.flat[i].plot(fpr_4, tpr_4, label =  str(model)+" lambda 1.8: "+str(round(auc(fpr_4,tpr_4),3)),linewidth=3)
            #     axs.flat[i].plot(fpr_5, tpr_5, label =  str(model)+" lambda 2: "+str(round(auc(fpr_5,tpr_5),3)),linewidth=3)
            #axs.flat[i].plot(fpr_a, tpr_a, label =  "ADM : "+str(round(auc(fpr_a,tpr_a),3)),alpha=0.8,linewidth=3)
            axs.flat[i].legend(fontsize=10,loc='lower right')
    print(xlabels+"theta pos: "+str(theta_pos)+" theta neg: "+str(theta_neg))
    plt.suptitle(str(title),fontsize=20)
    plt.show()

def Prepare_data_fix(gene, hets, depth, sigma,source,model,workdir,calculation,theta_pos,theta_neg,Num_para):
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
        
    path_model,_,_= Generate_path(source,model,sigma,workdir,calculation)
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

def Prepare_data_fix_semi(gene, hets, depth, sigma,source,model,workdir,calculation,theta_pos,theta_neg,Num_para):
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
        
    path_model,_,_= Generate_path(source,model,sigma,workdir,calculation)
    all_file = sorted(os.listdir(path_model))
    file_dict = {}
    for pkl in all_file:
        if (".pickle" in pkl) and ("tpi" not in pkl) and ("giab" not in pkl):
            #pkl=pkl.replace("CEU_","")
            name_parts = pkl.rsplit(".pickle")[0].rsplit("_")
            file_dict[pkl] = {}
            for part in name_parts:
                key_value = part.split("-")
                if len(key_value) == 2:  # Check if the part contains both key and value
                    file_dict[pkl][key_value[0]] = float(key_value[1])
                else:  # Handle parts that do not follow the key-value format
                    file_dict[pkl][part] = None  # or use a placeholder value
        else:
            continue
    file_dict_pd = pd.DataFrame(file_dict).transpose()
    file_dict_pd['file'] = file_dict_pd.index
    if Num_para == 3:
        pos_pd = file_dict_pd[(file_dict_pd[valid_var_np[0]] == valid_full_var_np[0])&(file_dict_pd[valid_var_np[1]] == valid_full_var_np[1])&(file_dict_pd[valid_var_np[2]] == valid_full_var_np[2])&(file_dict_pd['t'] == theta_pos)].sort_values(['d','h','g','s'])
        neg_pd = file_dict_pd[(file_dict_pd[valid_var_np[0]] == valid_full_var_np[0])&(file_dict_pd[valid_var_np[1]] == valid_full_var_np[1])&(file_dict_pd[valid_var_np[2]] == valid_full_var_np[2])&(file_dict_pd['t'] == theta_neg)].sort_values(['d','h','g','s'])
    elif Num_para == 2:
        pos_pd = file_dict_pd[(file_dict_pd[valid_var_np[0]] == valid_full_var_np[0])&(file_dict_pd[valid_var_np[1]] == valid_full_var_np[1])&(file_dict_pd['t'] == theta_pos)].sort_values(['d','h','g','s'])
        neg_pd = file_dict_pd[(file_dict_pd[valid_var_np[0]] == valid_full_var_np[0])&(file_dict_pd[valid_var_np[1]] == valid_full_var_np[1])&(file_dict_pd['t'] == theta_neg)].sort_values(['d','h','g','s'])
    d_group = neg_pd[var_map_np[np.array(var) == None][0]].unique()
    return d_group,var,var_map_np,fixed_var_np,var_fullname_map_np,variable_var_np,pos_pd,neg_pd


def get_ROC_AUC(path, file_pos, file_neg, calculation,lambdas=None,if_PRR=None,if_prob=None, if_baseline=None,if_AA=None,if_drop=True):
    # baseline estiamtes/prob
    if if_baseline == True and if_AA !=True:
        prob1 = read_data.read_one_pickle(path+"/"+file_pos)
        prob2 = read_data.read_one_pickle(path+"/"+file_neg)
        if calculation == "estimate":
            fpr,tpr, _ = roc_curve([0 for i in range(len(prob1))] + [1 for i in range(len(prob2))], prob1 + prob2, drop_intermediate=if_drop)
        else:
            fpr,tpr, _ = roc_curve([1 for i in range(len(prob1))] + [0 for i in range(len(prob2))], prob1 + prob2,pos_label=0,drop_intermediate=if_drop)
            precision,recall, _ = precision_recall_curve([1 for i in range(len(prob1))] + [0 for i in range(len(prob2))], prob1 + prob2,pos_label=0)

    # ADM estimates
    elif if_AA==True and if_baseline!=True:
        prob1 = read_data.read_one_pickle(path+"/"+file_pos)
        prob2 = read_data.read_one_pickle(path+"/"+file_neg)
        if calculation == "estimate":
            fpr, tpr, _ = roc_curve([1 for i in range(len(prob1))] + [0 for i in range(len(prob2))], prob1 + prob2,pos_label=1,drop_intermediate=if_drop)
            precision,recall, _ = precision_recall_curve([1 for i in range(len(prob1))] + [0 for i in range(len(prob2))], prob1 + prob2,pos_label=1)
        else:
            fpr, tpr, _ = roc_curve([0 for i in range(len(prob1))] + [1 for i in range(len(prob2))], prob1 + prob2,drop_intermediate=if_drop)

    # BEASTIE estimates/prob
    elif if_baseline !=True and if_AA != True:
        if calculation == "pval":
            NEG = file_neg.replace('.pickle', '.tsv')
            POS = file_pos.replace('.pickle', '.tsv')
            _,_,pos_p_st,neg_p_st,_,_ = quickBeast.get_tsv_p_values(POS,NEG,path)
            fpr, tpr, precision, recall = quickBeast.get_ROC_tsv(pos_p_st,neg_p_st)
        else:
            prob1 = read_data.read_one_pickle(path+"/"+file_pos)
            prob2 = read_data.read_one_pickle(path+"/"+file_neg)
            prob1=ROC_common.calculate_posterior_value(calculation,prob1,Lambda=lambdas)
            prob2=ROC_common.calculate_posterior_value(calculation,prob2,Lambda=lambdas)
            if calculation == "estimate": #estimates
                prob1_t = [abs(x - 1) for x in prob1]
                prob2_t = [abs(x - 1) for x in prob2]
                fpr, tpr, _ = roc_curve([0 for i in range(len(prob1_t))] + [1 for i in range(len(prob2_t))], prob1_t + prob2_t,drop_intermediate=if_drop)
            elif calculation == "max_prob": #max prob
                fpr, tpr, _ = roc_curve([1 for i in range(len(prob1))] + [0 for i in range(len(prob2))], prob1 + prob2,pos_label=1,drop_intermediate=if_drop)
                precision,recall, _ = precision_recall_curve([1 for i in range(len(prob1))] + [0 for i in range(len(prob2))], prob1 + prob2,pos_label=1)

    if if_PRR is not True:
        return(fpr, tpr, round(auc(fpr,tpr),3))
    else:
        return(recall,precision,round(auc(recall,precision),3))

#
# def Generate_path(source,model,sigma,workdir):
#     path_model = "/home/scarlett/DCC/software/cmdstan-2.27.0/examples/BEASTIE3/new_simulation/output_pkl/model_prob1/"
#     path_NS="/home/scarlett/DCC/output/simulation/binomial/output_pkl/new_simulation/NS_p/"
#     path_MS="/home/scarlett/DCC/output/simulation/binomial/output_pkl/new_simulation/MS_p/"
#     return path_model,path_NS,path_MS

def Generate_path_GIAB(source,model,sigma,workdir):
    path_model = f"/home/scarlett/DCC/simulation/semi_empirical/BEASTIE_empirical_data/g-1000/output_pkl/"
    path_GIAB = f"/home/scarlett/DCC/simulation/semi_empirical/iBEASTIE_empirical_data/g-1000/output_pkl/"
    path_NS=f"{source}/binomial/simulation/{workdir}/NS_p/" 
    path_MS=f"{source}/binomial/simulation/{workdir}/MS_p/" 
    return path_model,path_GIAB,path_NS,path_MS

def Generate_path(source,model,sigma,workdir,calculation="estimate"):
    path_model = f"{source}/{model}/sigma{sigma}/{workdir}/output_pkl/"
    if calculation == "pval":
        postfix="p"
    else:
        postfix="esti"
    path_NS=f"{source}/binomial/{workdir}/NS_"+postfix+"/" 
    path_MS=f"{source}/binomial/{workdir}/MS_"+postfix+"/" 

    return path_model,path_NS,path_MS

def Generate_path_beta(source,model,sigma,workdir,model2=None,calculation="max prob"):
    path_model = f"{source}/{model}/sigma{sigma}/{workdir}/output_pkl/"
    if model2 is not None:
        path_model2 = f"{source}/{model2}/sigma{sigma}/{workdir}/output_pkl/"
    else:
        path_model2=None
    if calculation == "max prob":
        postfix="p"
        path_beta1=f"{source}/betabinomial/{workdir}/betabinom_1_1_p/"
        path_beta10=f"{source}/betabinomial/{workdir}/betabinom_10_10_p/"
        path_beta20=f"{source}/betabinomial/{workdir}/betabinom_20_20_p/"
    else:
        postfix="esti"
        path_beta1=None
        path_beta10=None
        path_beta20=None
    path_NS=f"{source}/binomial/{workdir}/NS_"+postfix+"/" 
    path_MS=f"{source}/binomial/{workdir}/MS_"+postfix+"/" 
    return path_model,path_model2,path_NS,path_MS,path_beta1,path_beta10,path_beta20

# generate input data path
def Generate_path_noLAMBDA(source,model1,model2,model3,model4,sigma,workdir,calculation):
    path_model1 = f"{source}/{model1}/sigma{str(round(float(sigma),2))}/{workdir}/output_pkl/"
    path_model2 = f"{source}/{model2}/sigma{str(round(float(sigma),2))}/{workdir}/output_pkl/"
    path_model3 = f"{source}/{model3}/sigma{str(round(float(sigma),2))}/{workdir}/output_pkl/"
    path_model4 = f"{source}/{model4}/sigma{str(round(float(sigma),2))}/{workdir}/output_pkl/"
    if calculation=="max_prob":
        path_NS=f"{source}/binomial/{workdir}/NS_p/" 
        path_MS=f"{source}/binomial/{workdir}/MS_p/" 
        path_ADM=f"{source}/ADM/{workdir}/AA_p/"
    else:
        path_NS=f"{source}/binomial/{workdir}/NS_esti/" 
        path_MS=f"{source}/binomial/{workdir}/MS_esti/" 
        path_ADM=f"{source}/ADM/{workdir}/AA_esti/" 
    return path_model1,path_model2,path_model3,path_model4,path_NS,path_MS

def Generate_path_noLAMBDA_beta(source,model,sigma,workdir,calculation):
    path_model = f"{source}/{model}/sigma{str(round(float(sigma),2))}/{workdir}/output_pkl/"
    path_model_beta = f"{source}/{model}_beta0.05/sigma{str(round(float(sigma),2))}/{workdir}/output_pkl/"
    if calculation == "median":
        path_NS=f"{source}/binomial/simulation/{workdir}/NS_esti/" 
        path_MS=f"{source}/binomial/simulation/{workdir}/MS_esti/" 
        path_ADM=f"{source}/ADM/simulation/{workdir}/AA_esti/" 
    else:
        path_NS=f"{source}/binomial/simulation/{workdir}/NS_p/" 
        path_MS=f"{source}/binomial/simulation/{workdir}/MS_p/" 
        path_ADM=f"{source}/ADM/simulation/{workdir}/AA_p/" 
    return path_model,path_model_beta,path_NS,path_MS

def Generate_path_noLAMBDA_sigma(source,model,workdir,calculation):
    path_model1 = f"{source}/{model}/sigma0.5/{workdir}/output_pkl/"
    path_model2 = f"{source}/{model}/sigma0.7/{workdir}/output_pkl/"
    path_model3 = f"{source}/{model}/sigma1.0/{workdir}/output_pkl/"
    if calculation == "median":
        path_NS=f"{source}/binomial/{workdir}/NS_esti/" 
        path_MS=f"{source}/binomial/{workdir}/MS_esti/" 
        path_ADM=f"{source}/ADM/{workdir}/AA_esti/" 
    else:
        path_NS=f"{source}/binomial/{workdir}/NS_p/" 
        path_MS=f"{source}/binomial/{workdir}/MS_p/" 
        path_ADM=f"{source}/ADM/{workdir}/AA_p/" 
    return path_model1,path_model2,path_model3,path_NS,path_MS,path_ADM

# no lambda ROC comparison
def ROC_comparison_fix3_noLambda(source,model1,model2,workdir,calculation,model3=None,model4=None,theta_pos=None,theta_neg=None,gene=None,hets=None,depth=None,sigma=None,title=None,Num_para=3,Num_col=None):
    # judge whether  2 of the 4 are None, and 2 of the 4 are not None
    ROC_common.Make_judgement(Num_para=Num_para,theta_pos=theta_pos,theta_neg=theta_neg,gene=gene,hets=hets,depth=depth,sigma=sigma)
    Plot_ROC_fix3_noLambda(source,model1,model2,model3=model3,model4=model4,workdir=workdir,calculation=calculation,Num_col=Num_col,gene=gene, hets=hets, depth=depth,sigma=sigma, theta_pos=theta_pos,theta_neg=theta_neg,title=title)


def Plot_ROC_fix3_noLambda(source,model1,model2,workdir,calculation,Num_col,gene, hets, depth, sigma,theta_pos,theta_neg,title,model3=None,model4=None):
    path_model1,path_model2, path_model3,path_model4,path_MS,path_NS = Generate_path_noLAMBDA(source,model1=model1,model2=model2,model3=model3,model4=model4,sigma=sigma,workdir=workdir,calculation=calculation)
    d_group,var,var_map_np,fixed_var_np,var_fullname_map_np,variable_var_np,pos_pd,neg_pd = Prepare_data_fix(gene, hets, depth, sigma,source,model1,workdir,calculation,theta_pos,theta_neg=1,Num_para=3)
    if Num_col == None:
        Num_col = 3
    else:
        Num_col = int(Num_col)
    row = math.ceil(float(len(d_group))/Num_col)
    fig, axs = plt.subplots(row, Num_col, figsize = (15,5*row))
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
            reduced_file_pos = current_group_pos_list[idx].rsplit("_",1)[0]+".pickle"
            reduced_file_neg = current_group_neg_list[idx].rsplit("_",1)[0]+".pickle"                                    
            g=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[0].rsplit("-")[1]
            h=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[1].rsplit("-")[1]
            d=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[2].rsplit("-")[1]
            s=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[4].rsplit("-")[1]
            # calculate auc score
            fpr_m,tpr_m,s1 = get_ROC_AUC(path_model1,current_group_pos_list[idx],current_group_neg_list[idx],calculation,lambdas=None)
            fpr_m2,tpr_m2,s2 = get_ROC_AUC(path_model2,current_group_pos_list[idx],current_group_neg_list[idx],calculation,lambdas=None)
            if model3 is not None:
                fpr_m3,tpr_m3,s3 = get_ROC_AUC(path_model3,current_group_pos_list[idx],current_group_neg_list[idx],calculation,lambdas=None)
            if model4 is not None:
                fpr_m4,tpr_m4,s4 = get_ROC_AUC(path_model4,current_group_pos_list[idx],current_group_neg_list[idx],calculation,lambdas=None)
            fpr_b,tpr_b,s5 = get_ROC_AUC(path_NS,reduced_file_pos,reduced_file_neg,calculation,if_baseline=True)
            fpr_b2,tpr_b2,s6 = get_ROC_AUC(path_MS,reduced_file_pos,reduced_file_neg,calculation,if_baseline=True)
            # 
            var_dict = {"gene":g, "hets": h, "depth": d, "sigma": s}
            for each in fixed_var_np:
                xlabels += each+":"+var_dict[each]+' '
            axs.flat[i].set_xlabel("FPR",fontsize=12)
            axs.flat[i].set_ylabel("TPR",fontsize=12)
            axs.flat[i].set_title(var_fullname_map_np[np.array(var) == None][0]+":" + var_dict[var_fullname_map_np[np.array(var) == None][0]],fontsize=12)
            axs.flat[i].tick_params(axis='both', labelsize=10)
            axs.flat[i].plot(fpr_b, tpr_b, label =  "Major Site : "+str(s5),alpha=0.8,linewidth=3)
            axs.flat[i].plot(fpr_b2, tpr_b2, label =  "Naive Sum : "+str(s6),alpha=0.8,linewidth=3)
            axs.flat[i].plot(fpr_m, tpr_m, label =  str(model1)+" :"+str(s1),alpha=0.8,linewidth=3)
            axs.flat[i].plot(fpr_m2, tpr_m2, label =  str(model2)+" :"+str(s2),alpha=0.8,linewidth=3)
            if model3 is not None: 
                axs.flat[i].plot(fpr_m3, tpr_m3, label =  str(model3)+" :"+str(s3),alpha=0.8,linewidth=3)
            if model4 is not None:
                axs.flat[i].plot(fpr_m4, tpr_m4, label =  str(model4)+" :"+str(s4),alpha=0.8,linewidth=3)
            axs.flat[i].legend(fontsize=10,loc='lower right')
    plt.suptitle(str(title)+"\n"+xlabels+"theta pos: "+str(theta_pos)+" theta neg: "+str(theta_neg),fontsize=20)
    plt.show()

# no lambda ROC comparison for se_empirical data
def ROC_comparison_fix3_noLambda_semi(source,model1,model2,workdir,calculation,model3=None,model4=None,theta_pos=None,theta_neg=None,gene=None,hets=None,depth=None,sigma=None,title=None,Num_para=3,Num_col=None):
    # judge whether  2 of the 4 are None, and 2 of the 4 are not None
    ROC_common.Make_judgement(Num_para=Num_para,theta_pos=theta_pos,theta_neg=theta_neg,gene=gene,hets=hets,depth=depth,sigma=sigma)
    Plot_ROC_fix3_noLambda_semi(source,model1,model2,model3=model3,model4=model4,workdir=workdir,calculation=calculation,Num_col=Num_col,gene=gene, hets=hets, depth=depth,sigma=sigma, theta_pos=theta_pos,theta_neg=theta_neg,title=title)


def Plot_ROC_fix3_noLambda_semi(source,model1,model2,workdir,calculation,Num_col,gene, hets, depth, sigma,theta_pos,theta_neg,title,model3=None,model4=None):
    path_model1,path_model2, path_model3,path_model4,path_MS,path_NS = Generate_path_noLAMBDA(source,model1=model1,model2=model2,model3=model3,model4=model4,sigma=sigma,workdir=workdir,calculation=calculation)
    d_group,var,var_map_np,fixed_var_np,var_fullname_map_np,variable_var_np,pos_pd,neg_pd = Prepare_data_fix_semi(gene, hets, depth, sigma,source,model1,workdir,calculation,theta_pos,theta_neg=1,Num_para=3)
    if Num_col == None:
        Num_col = 3
    else:
        Num_col = int(Num_col)
    row = math.ceil(float(len(d_group))/Num_col)
    fig, axs = plt.subplots(row, Num_col, figsize = (15,5*row))
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
        print(current_group_neg_list)

        xlabels = "Fixed parameters "
        for idx in range(len(current_group_pos_list)):
            reduced_file_pos = current_group_pos_list[idx].rsplit("_",1)[0]+".pickle"
            reduced_file_neg = current_group_neg_list[idx].rsplit("_",1)[0]+".pickle"                                    
            g=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[0].rsplit("-")[1]
            h=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[1].rsplit("-")[1]
            d=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[2].rsplit("-")[1]
            s=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[5].rsplit("-")[1]
            # calculate auc score
            fpr_m,tpr_m,s1 = get_ROC_AUC(path_model1,current_group_pos_list[idx],current_group_neg_list[idx],calculation,lambdas=None)
            fpr_m2,tpr_m2,s2 = get_ROC_AUC(path_model2,current_group_pos_list[idx],current_group_neg_list[idx],calculation,lambdas=None)
            if model3 is not None:
                fpr_m3,tpr_m3,s3 = get_ROC_AUC(path_model3,current_group_pos_list[idx],current_group_neg_list[idx],calculation,lambdas=None)
            if model4 is not None:
                fpr_m4,tpr_m4,s4 = get_ROC_AUC(path_model4,current_group_pos_list[idx],current_group_neg_list[idx],calculation,lambdas=None)
            fpr_b,tpr_b,s5 = get_ROC_AUC(path_NS,reduced_file_pos,reduced_file_neg,calculation,if_baseline=True)
            fpr_b2,tpr_b2,s6 = get_ROC_AUC(path_MS,reduced_file_pos,reduced_file_neg,calculation,if_baseline=True)
            # 
            var_dict = {"gene":g, "hets": h, "depth": d, "sigma": s}
            for each in fixed_var_np:
                xlabels += each+":"+var_dict[each]+' '
            axs.flat[i].set_xlabel("FPR",fontsize=12)
            axs.flat[i].set_ylabel("TPR",fontsize=12)
            axs.flat[i].set_title(var_fullname_map_np[np.array(var) == None][0]+":" + var_dict[var_fullname_map_np[np.array(var) == None][0]],fontsize=12)
            axs.flat[i].tick_params(axis='both', labelsize=10)
            axs.flat[i].plot(fpr_b, tpr_b, label =  "Major Site : "+str(s5),alpha=0.8,linewidth=3)
            axs.flat[i].plot(fpr_b2, tpr_b2, label =  "Naive Sum : "+str(s6),alpha=0.8,linewidth=3)
            axs.flat[i].plot(fpr_m, tpr_m, label =  str(model1)+" :"+str(s1),alpha=0.8,linewidth=3)
            axs.flat[i].plot(fpr_m2, tpr_m2, label =  str(model2)+" :"+str(s2),alpha=0.8,linewidth=3)
            if model3 is not None: 
                axs.flat[i].plot(fpr_m3, tpr_m3, label =  str(model3)+" :"+str(s3),alpha=0.8,linewidth=3)
            if model4 is not None:
                axs.flat[i].plot(fpr_m4, tpr_m4, label =  str(model4)+" :"+str(s4),alpha=0.8,linewidth=3)
            axs.flat[i].legend(fontsize=10,loc='lower right')
    plt.suptitle(str(title)+"\n"+xlabels+"theta pos: "+str(theta_pos)+" theta neg: "+str(theta_neg),fontsize=20)
    plt.show()

def Plot_ROC_comparison_fix4_noLAMBDA(prefix,model,workdir,pi=None,beta=None,theta_pos=None,theta_neg=None,if_prob=None,gene=None,hets=None,depth=None,sigma=None,title=None,Num_para=4):
    source = prefix+"software/cmdstan/examples/"
    # judge whether  2 of the 4 are None, and 2 of the 4 are not None
    ROC_common.Make_judgement(Num_para=Num_para,theta_pos=theta_pos,theta_neg=theta_neg,gene=gene,hets=hets,depth=depth,sigma=sigma)
    path_model,path_pi,path_beta,path_base,path_base_p = Generate_path_noLAMBDA(source,model=model,workdir=workdir,pi=pi,beta=beta,if_prob=if_prob)
    reduced_file_pos,reduced_file_neg,full_file_pos,full_file_neg,xlabels = ROC_common.Get_file_name(gene,hets,depth,sigma,theta_pos,theta_neg)
    fpr_m,tpr_m,fpr_pi,tpr_pi,fpr_beta,tpr_beta,fpr_b,tpr_b,fpr_b_p,tpr_b_p = Get_tpr_fpr_noLAMBDA(path_model,path_pi,path_beta,path_base,path_base_p,reduced_file_pos,reduced_file_neg,full_file_pos,full_file_neg,if_prob=if_prob)
    Plot_ROC_fix4_noLAMBDA(fpr_m,tpr_m,fpr_pi,tpr_pi,fpr_beta,tpr_beta,fpr_b,tpr_b,fpr_b_p,tpr_b_p,xlabels,model=model,beta=beta,pi=pi,title=title)

def Plot_ROC_fix4_noLAMBDA(fpr_m,tpr_m,fpr_pi,tpr_pi,fpr_beta,tpr_beta,fpr_b,tpr_b,fpr_b_p,tpr_b_p,xlabels,model,beta,pi,title=None):
    plt.figure(figsize=(10,8))
    plt.plot(fpr_m,tpr_m, label = str(model)+": " + str(round(auc(fpr_m,tpr_m),3)),linewidth=3)
    if fpr_beta is not None:
        plt.plot(fpr_beta,tpr_beta, label = str(model)+"-beta"+str(beta)+": "+ str(round(auc(fpr_beta,tpr_beta),3)),linewidth=3)
    if fpr_pi is not None:
        plt.plot(fpr_pi,tpr_pi, label = str(model)+"-pi"+str(pi)+": "+ str(round(auc(fpr_pi,tpr_pi),3)),linewidth=3)
    plt.plot(fpr_b,tpr_b, label = "Binomial 1st site: "+ str(round(auc(fpr_b,tpr_b),3)),linewidth=3)
    plt.plot(fpr_b_p,tpr_b_p, label = "Binomial pooled p: "+ str(round(auc(fpr_b_p,tpr_b_p),3)),linewidth=3)
    #plt.plot(fpr_a,tpr_a, label = "ADM: " + str(round(auc(fpr_a,tpr_a),3)),linewidth=3)
    if title!=None:
        plt.title(str(title),fontsize=16)
    plt.ylabel("ROC",fontsize=25)
    plt.legend(fontsize=20)

def Get_tpr_fpr_noLAMBDA(path_model, path_model_beta,path_NS,path_MS,path_ADM,reduced_file_pos,reduced_file_neg,full_file_pos,full_file_neg,calculation):
    fpr_m,tpr_m,_ = get_ROC_AUC(path_model,full_file_pos,full_file_neg,calculation)
    fpr_m2,tpr_m2,_ = get_ROC_AUC(path_model_beta,full_file_pos,full_file_neg,calculation)
    fpr_b,tpr_b,_ = get_ROC_AUC(path_NS,reduced_file_pos,reduced_file_neg,calculation,if_baseline=True)
    fpr_b2,tpr_b2,_ = get_ROC_AUC(path_MS,reduced_file_pos,reduced_file_neg,calculation,if_baseline=True)
    fpr_a_p,tpr_a_p,_ = get_ROC_AUC(path_ADM,reduced_file_pos,reduced_file_neg,calculation,if_AA=True)
    return fpr_m,tpr_m,fpr_m2,tpr_m2,fpr_b,tpr_b,fpr_b2,tpr_b2,fpr_a_p,tpr_a_p

def Get_tpr_fpr_noLAMBDA_sigma(path_model, path_model2,path_model3,path_NS,path_MS,reduced_file_pos,reduced_file_neg,full_file_pos,full_file_neg,calculation):
    fpr_m,tpr_m,_ = get_ROC_AUC(path_model,full_file_pos,full_file_neg,calculation)
    fpr_m2,tpr_m2,_ = get_ROC_AUC(path_model2,full_file_pos.replace("-0.5.pickle", f"-0.7.pickle"),full_file_neg.replace("-0.5.pickle", f"-0.7.pickle"),calculation)
    fpr_m3,tpr_m3,_ = get_ROC_AUC(path_model3,full_file_pos.replace("-0.5.pickle", f"-1.0.pickle"),full_file_neg.replace("-0.5.pickle", f"-1.0.pickle"),calculation)
    fpr_b,tpr_b,_ = get_ROC_AUC(path_NS,reduced_file_pos,reduced_file_neg,calculation,if_baseline=True)
    fpr_b2,tpr_b2,_ = get_ROC_AUC(path_MS,reduced_file_pos,reduced_file_neg,calculation,if_baseline=True)
    #fpr_a_p,tpr_a_p,_ = get_ROC_AUC(path_ADM,reduced_file_pos,reduced_file_neg,calculation,if_AA=True)
    return fpr_m,tpr_m,fpr_m2,tpr_m2,fpr_m3,tpr_m3,fpr_b,tpr_b,fpr_b2,tpr_b2