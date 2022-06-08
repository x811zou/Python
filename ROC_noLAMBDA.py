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
import matplotlib.pyplot as plt
from sklearn import metrics,preprocessing
from sklearn.metrics import roc_curve, auc,precision_recall_curve
from ROC_common import get_ROC_AUC_V3,Make_judgement,Prepare_data_fix,Generate_path,Get_file_name

# prefix="/Users/scarlett/Desktop/HARDAC/scarlett/"

def get_ROC_PRR(path, file_pos, file_neg, if_PRR=None,if_prob=None, if_baseline=None,if_print=None,if_AA=None,if_drop=True):
    if if_print==True:
        print(file_pos)
    prob1 = pickle.load(open(path+file_pos,"rb"))
    prob2 = pickle.load(open(path+file_neg,"rb"))
    # baseline estiamtes/prob
    if if_baseline == True and if_AA !=True:
        if if_prob == True:
            fpr,tpr, _ = roc_curve([1 for i in range(len(prob1))] + [0 for i in range(len(prob2))], prob1 + prob2,pos_label=0,drop_intermediate=if_drop)
            precision,recall, _ = precision_recall_curve([1 for i in range(len(prob1))] + [0 for i in range(len(prob2))], prob1 + prob2,pos_label=0)
        else:
            fpr,tpr, _ = roc_curve([1 for i in range(len(prob1))] + [0 for i in range(len(prob2))], prob1 + prob2,drop_intermediate=if_drop)

    # ADM estimates
    elif if_AA==True and if_baseline!=True:
        if if_prob == True:
            fpr, tpr, _ = roc_curve([1 for i in range(len(prob1))] + [0 for i in range(len(prob2))], prob1 + prob2,pos_label=1,drop_intermediate=if_drop)
            precision,recall, _ = precision_recall_curve([1 for i in range(len(prob1))] + [0 for i in range(len(prob2))], prob1 + prob2,pos_label=1)
        else:
            fpr, tpr, _ = roc_curve([1 for i in range(len(prob1))] + [0 for i in range(len(prob2))], prob1 + prob2,drop_intermediate=if_drop)
    # model estimates/prob
    elif if_baseline !=True and if_AA != True:
        if if_prob != True: #estimates
            prob1_t = [abs(x - 1) for x in prob1]
            prob2_t = [abs(x - 1) for x in prob2]
            fpr, tpr, _ = roc_curve([1 for i in range(len(prob1_t))] + [0 for i in range(len(prob2_t))], prob1_t + prob2_t,drop_intermediate=if_drop)
        else:
            fpr, tpr, _ = roc_curve([1 for i in range(len(prob1))] + [0 for i in range(len(prob2))], prob1 + prob2,pos_label=1,drop_intermediate=if_drop)
            precision,recall, _ = precision_recall_curve([1 for i in range(len(prob1))] + [0 for i in range(len(prob2))], prob1 + prob2,pos_label=1)

    if if_PRR is not True:
        return(fpr, tpr, round(auc(fpr,tpr),3))
    else:
        return(recall,precision,round(auc(recall,precision),3))

def Generate_path_noLAMBDA(source,model,workdir,beta=None,pi=None,if_prob=None):
    if if_prob is True:
        path_model = source + model+"/"+workdir+"/output_pkl"+"/model_prob2/"
        path_base = source + "binomial"+"/output_pkl/"+workdir+"/prob4/"
        #path_base_p = source +"binomial/output_pkl/" + workdir + "/pooled_prob1/"
        path_base_p = source + "binomial"+"/output_pkl/"+workdir+"/pooled_prob1/"
        #path_AA = source + "ADM/" + workdir + "/output_pkl/AA_pval/"
        if beta != None:
            path_beta = source +str(model)+"-beta"+str(beta)+"/" + workdir + "/output_pkl/model_prob1/"
        if pi != None:
            path_pi = source +str(model)+"-pi"+str(pi)+"/" + workdir + "/output_pkl/model_prob1/"
    else:
        path_model = source +str(model)+"/" + workdir + "/output_pkl/model_med/"
        path_base = source +"binomial/output_pkl/" + workdir + "/esti1/"
        path_base_p = source +"binomial/output_pkl/" + workdir + "/pooled_esti1/"
        path_AA = source + "ADM/" + workdir + "/output_pkl/AA_esti/"
        if beta != None:
            path_beta = source +str(model)+"-beta"+str(beta)+"/" + workdir + "/output_pkl/model_med/"
        if pi != None:
            path_pi = source +str(model)+"-pi"+str(pi)+"/" + workdir + "/output_pkl/model_med/"
    if beta is None:
        path_beta=None
    if pi is None:
        path_pi=None

    return path_model, path_pi,path_beta,path_base,path_base_p

def ROC_comparison_fix3_noLAMBDA(prefix,model,workdir,beta=None,pi=None,theta_pos=None,theta_neg=None,if_prob=None,gene=None,hets=None,depth=None,sigma=None,title=None,Num_para=3,Num_col=None):
    source = prefix+"software/cmdstan/examples/"
    # judge whether  2 of the 4 are None, and 2 of the 4 are not None
    #Make_judgement(Num_para=Num_para,theta_pos=theta_pos,theta_neg=theta_neg,gene=gene,hets=hets,depth=depth,sigma=sigma)
    Plot_ROC_fix3_noLAMBDA(source=source,model=model,workdir=workdir,beta=beta,pi=pi,if_prob=if_prob,Num_col=Num_col,gene=gene, hets=hets, depth=depth, sigma=sigma,theta_pos=theta_pos,theta_neg=theta_neg,title=title)

def Plot_ROC_fix3_noLAMBDA(source,model,workdir,beta,pi,if_prob,Num_col,gene, hets, depth, sigma,theta_pos,theta_neg,title):
    path_model, path_pi,path_beta,path_base,path_base_p = Generate_path_noLAMBDA(source,model=model,workdir=workdir,beta=beta,pi=pi,if_prob=if_prob)
    d_group,var,var_map_np,fixed_var_np,var_fullname_map_np,variable_var_np,pos_pd,neg_pd = Prepare_data_fix(gene, hets, depth, sigma,source,workdir=workdir,if_prob=if_prob,theta_pos=theta_pos,theta_neg=theta_neg,Num_para=3)
    if Num_col == None:
        Num_col = 3
    else:
        Num_col = int(Num_col)

    row = math.ceil(float(len(d_group))/Num_col)
    fig, axs = plt.subplots(row, Num_col, figsize = (20,5*row))
    if (row * Num_col > len(d_group)):
        for i in range(row * Num_col - len(d_group)):
            axs.flat[-1-i].set_axis_off()
    if title!=None:
        plt.title(str(title),fontsize=16)
    for i, each in enumerate(d_group):
        #print(i)
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

            fpr_m,tpr_m,fpr_pi,tpr_pi,fpr_beta,tpr_beta,fpr_b,tpr_b,fpr_b_p,tpr_b_p = Get_tpr_fpr_noLAMBDA(path_model, path_pi,path_beta,path_base,path_base_p,reduced_file_pos,reduced_file_neg,current_group_pos_list[idx],current_group_neg_list[idx],if_prob=if_prob)
            #print(fpr_a)
            var_dict = {"gene":g, "hets": h, "depth": d, "sigma": s}
            for each in fixed_var_np:
                xlabels += each+":"+var_dict[each]+' '
            axs.flat[i].set_xlabel("FPR")
            axs.flat[i].set_ylabel("TPR")
            axs.flat[i].set_title(var_fullname_map_np[np.array(var) == None][0]+":" + var_dict[var_fullname_map_np[np.array(var) == None][0]])
            axs.flat[i].plot(fpr_m, tpr_m, label =  model+" : "+str(round(auc(fpr_m,tpr_m),3)),linewidth=3)
            if pi is not None:
                axs.flat[i].plot(fpr_pi, tpr_pi, label =  model+"-pi"+str(pi)+" : "+str(round(auc(fpr_pi,tpr_pi),3)),linewidth=3)
            if beta is not None:
                axs.flat[i].plot(fpr_beta, tpr_beta, label =  model+"-beta"+str(beta)+" : "+str(round(auc(fpr_beta,tpr_beta),3)),linewidth=3)         
            axs.flat[i].plot(fpr_b, tpr_b, label =  "Binomial 1st site: "+str(round(auc(fpr_b,tpr_b),3)),linewidth=3)
            axs.flat[i].plot(fpr_b_p, tpr_b_p, label =  "Binomial pooled: "+str(round(auc(fpr_b_p,tpr_b_p),3)),linewidth=3)
            #axs.flat[i].plot(fpr_a, tpr_a, label =  "ADM: "+str(round(auc(fpr_a,tpr_a),3)),linewidth=3)
            axs.flat[i].legend(fontsize=15,loc='lower right')
    plt.suptitle(str(title)+" >> "+xlabels,fontsize=20)
    plt.show()

def Plot_ROC_comparison_fix4_noLAMBDA(prefix,model,workdir,pi=None,beta=None,theta_pos=None,theta_neg=None,if_prob=None,gene=None,hets=None,depth=None,sigma=None,title=None,Num_para=4):
    source = prefix+"software/cmdstan/examples/"
    # judge whether  2 of the 4 are None, and 2 of the 4 are not None
    Make_judgement(Num_para=Num_para,theta_pos=theta_pos,theta_neg=theta_neg,gene=gene,hets=hets,depth=depth,sigma=sigma)
    path_model,path_pi,path_beta,path_base,path_base_p = Generate_path_noLAMBDA(source,model=model,workdir=workdir,pi=pi,beta=beta,if_prob=if_prob)
    reduced_file_pos,reduced_file_neg,full_file_pos,full_file_neg,xlabels = Get_file_name(gene,hets,depth,sigma,theta_pos,theta_neg)
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

def Get_tpr_fpr_noLAMBDA(path_model, path_pi,path_beta,path_base,path_base_p,reduced_file_pos,reduced_file_neg,full_file_pos,full_file_neg,if_prob):
    fpr_m,tpr_m,_ = get_ROC_PRR(path_model,full_file_pos,full_file_neg,if_prob=if_prob)
    #print(auc(fpr_m,tpr_m))
    if path_pi is not None:
        fpr_pi,tpr_pi= get_ROC_PRR(path_pi,full_file_pos,full_file_neg,if_prob=if_prob)
    else:
        fpr_pi = None
        tpr_pi = None
    if path_beta is not None:
        fpr_beta,tpr_beta= get_ROC_PRR(path_beta,full_file_pos,full_file_neg,if_prob=if_prob)
    else:
        fpr_beta = None
        tpr_beta = None
    fpr_b,tpr_b,_ = get_ROC_PRR(path_base,reduced_file_pos,reduced_file_neg,if_prob=if_prob,if_baseline=True)
    fpr_b_p,tpr_b_p,_ = get_ROC_PRR(path_base_p,reduced_file_pos,reduced_file_neg,if_prob=if_prob,if_baseline=True)
    #fpr_a,tpr_a,_ = get_ROC_PRR(path_AA,reduced_file_pos,reduced_file_neg,if_prob=if_prob,if_AA=True)
    return fpr_m,tpr_m,fpr_pi,tpr_pi,fpr_beta,tpr_beta,fpr_b,tpr_b,fpr_b_p,tpr_b_p


