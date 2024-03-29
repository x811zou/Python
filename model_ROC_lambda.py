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
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc

def ROC_comparison_fix3(source,model,workdir,datatype,lamdbda=None,theta_pos=None,theta_neg=None,gene=None,hets=None,depth=None,sigma=None,title=None,Num_para=3,Num_col=None):
    # judge whether  2 of the 4 are None, and 2 of the 4 are not None
    Make_judgement(Num_para=Num_para,theta_pos=theta_pos,theta_neg=theta_neg,gene=gene,hets=hets,depth=depth,sigma=sigma)
    Plot_ROC_fix3_lambda(source,model=model,workdir=workdir,datatype=datatype,lamdbda=lamdbda,Num_col=Num_col,gene=gene, hets=hets, depth=depth, sigma=sigma,theta_pos=theta_pos,theta_neg=theta_neg,title=title)

def Make_judgement(Num_para,theta_pos=None,theta_neg=None,gene=None,hets=None,depth=None,sigma=None):
    thetas=[theta_pos,theta_neg]
    if thetas.count(None) >0:
        raise Exception('Both thetas could not be None. The number of None from input was {}'.format(thetas.count(None)))
    var=[gene,hets,depth,sigma]
    N = 4-Num_para
    if var.count(None) > N:
        raise Exception('None of the variables could be set to None. The number of None from input was {}'.format(var.count(None)))

def Plot_ROC_fix3_lambda(source,model,workdir,datatype,lamdbda,Num_col,gene, hets, depth, sigma,theta_pos,theta_neg,title):
    path_model, path_base, path_base_p,path_AA,path_lambda1,path_lambda2,path_lambda3,path_lambda4,path_lambda5 = Generate_path_lambda(model=model,source=source,workdir=workdir,datatype=datatype,lamdbda=lamdbda)
    d_group,var,var_map_np,fixed_var_np,var_fullname_map_np,variable_var_np,pos_pd,neg_pd = Prepare_data_fix_lambda(gene, hets, depth, sigma,model=model,source=source,workdir=workdir,datatype=datatype,lamdbda=lamdbda,theta_pos=theta_pos,theta_neg=theta_neg,Num_para=3)
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
    d_group=d_group[:-1]
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
            fpr_m,tpr_m,fpr_b,tpr_b,fpr_b_p,tpr_b_p,fpr_a,tpr_a,fpr_1,tpr_1,fpr_2,tpr_2,fpr_3,tpr_3,fpr_4,tpr_4,fpr_5,tpr_5 = Get_tpr_fpr_lambda(path_model, path_base, path_base_p,path_AA,path_lambda1,path_lambda2,path_lambda3,path_lambda4,path_lambda5,reduced_file_pos,reduced_file_neg,current_group_pos_list[idx],current_group_neg_list[idx])
            var_dict = {"gene":g, "hets": h, "depth": d, "sigma": s}
            xlabels = "Fixed parameters "
            for each in fixed_var_np:
                xlabels += each+":"+var_dict[each]+' '
            axs.flat[i].set_xlabel("FPR")
            axs.flat[i].set_ylabel("TPR")
            axs.flat[i].set_title(var_fullname_map_np[np.array(var) == None][0]+":" + var_dict[var_fullname_map_np[np.array(var) == None][0]])
            if lamdbda != None:
                axs.flat[i].plot(fpr_1, tpr_1, label =  str(model)+" lambda 1.2 :"+str(round(auc(fpr_1,tpr_1),3)),linewidth=3)
                axs.flat[i].plot(fpr_2, tpr_2, label =  str(model)+" lambda 1.4: "+str(round(auc(fpr_2,tpr_2),3)),linewidth=3)
                axs.flat[i].plot(fpr_3, tpr_3, label =  str(model)+" lambda 1.6: "+str(round(auc(fpr_3,tpr_3),3)),linewidth=3)
                axs.flat[i].plot(fpr_4, tpr_4, label =  str(model)+" lambda 1.8: "+str(round(auc(fpr_4,tpr_4),3)),linewidth=3)
                axs.flat[i].plot(fpr_5, tpr_5, label =  str(model)+" lambda 2: "+str(round(auc(fpr_5,tpr_5),3)),linewidth=3)
            else:
                axs.flat[i].plot(fpr_m, tpr_m, label =  str(model)+" :"+str(round(auc(fpr_m,tpr_m),3)),linewidth=3)
            axs.flat[i].plot(fpr_b, tpr_b, label =  "Binomial: "+str(round(auc(fpr_b,tpr_b),3)),linewidth=3)
            axs.flat[i].plot(fpr_b_p, tpr_b_p, label =  "Binomial pooled: "+str(round(auc(fpr_b_p,tpr_b_p),3)),linewidth=3)
            axs.flat[i].plot(fpr_a, tpr_a, label =  "ADM: "+str(round(auc(fpr_a,tpr_a),3)),linewidth=3)
            axs.flat[i].legend(fontsize=10,loc='lower right')
    plt.suptitle(title+" "+xlabels,fontsize=20)
    plt.show()

def Prepare_data_fix_lambda(gene, hets, depth, sigma,model,source,workdir,datatype,lamdbda,theta_pos,theta_neg,Num_para):
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
    
    if lamdbda == True:
        _,_,_,_,_,_,_,_,path= Generate_path_lambda(model=model,source=source,workdir=workdir,datatype=datatype,lamdbda=lamdbda)
    else:
        path,_,_,_,_,_,_,_,_= Generate_path_lambda(model=model,source=source,workdir=workdir,datatype=datatype,lamdbda=lamdbda)
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

def Generate_path_lambda(model,source,workdir,datatype,lamdbda):
    if lamdbda == True:
        path_model = None
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
    path_base = source +"binomial/output_pkl/" + workdir + "/prob_new/"
    path_base_p = source +"binomial/output_pkl/" + workdir + "/pooled_prob_new/"
    path_AA = source + "ADM/" + workdir + "/output_pkl/AA_pval/"
    return path_model, path_base, path_base_p,path_AA,path_lambda1,path_lambda2,path_lambda3,path_lambda4,path_lambda5

def Get_tpr_fpr_lambda(path_model, path_base, path_base_p,path_AA,path_lambda1,path_lambda2,path_lambda3,path_lambda4,path_lambda5,reduced_file_pos,reduced_file_neg,full_file_pos,full_file_neg,if_prob=True):
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
        fpr_m,tpr_m = get_ROC_AUC_V3(path_model,full_file_pos,full_file_neg,if_prob=if_prob)
    if path_lambda1 != None:
        fpr_1,tpr_1 = get_ROC_AUC_V3(path_lambda1,full_file_pos,full_file_neg,if_prob=if_prob)
        fpr_2,tpr_2 = get_ROC_AUC_V3(path_lambda2,full_file_pos,full_file_neg,if_prob=if_prob)
        fpr_3,tpr_3 = get_ROC_AUC_V3(path_lambda3,full_file_pos,full_file_neg,if_prob=if_prob)
        fpr_4,tpr_4 = get_ROC_AUC_V3(path_lambda4,full_file_pos,full_file_neg,if_prob=if_prob)
        fpr_5,tpr_5 = get_ROC_AUC_V3(path_lambda5,full_file_pos,full_file_neg,if_prob=if_prob)
    fpr_b,tpr_b = get_ROC_AUC_V3(path_base,reduced_file_pos,reduced_file_neg,if_prob=if_prob,if_baseline=True)
    fpr_b_p,tpr_b_p = get_ROC_AUC_V3(path_base_p,reduced_file_pos,reduced_file_neg,if_prob=if_prob,if_baseline=True)
    fpr_a,tpr_a = get_ROC_AUC_V3(path_AA,reduced_file_pos,reduced_file_neg,if_prob=if_prob,if_AA=True)
    return fpr_m,tpr_m,fpr_b,tpr_b,fpr_b_p,tpr_b_p,fpr_a,tpr_a,fpr_1,tpr_1,fpr_2,tpr_2,fpr_3,tpr_3,fpr_4,tpr_4,fpr_5,tpr_5

def get_ROC_AUC_V3(path, file_pos, file_neg, if_prob=True, if_baseline=None,if_print=None,if_AA=None,if_drop=True):
    if if_print==True:
        print(file_pos)
    prob1 = pickle.load(open(path+file_pos,"rb"))
    prob2 = pickle.load(open(path+file_neg,"rb"))
    # baseline estiamtes/prob
    if if_baseline == True and if_AA !=True:
        if if_prob == True:
            fpr,tpr, _ = roc_curve([0 for i in range(len(prob1))] + [1 for i in range(len(prob2))], prob1 + prob2,drop_intermediate=if_drop)
        else:
            fpr,tpr, _ = roc_curve([1 for i in range(len(prob1))] + [0 for i in range(len(prob2))], prob1 + prob2,drop_intermediate=if_drop)

    # ADM estimates
    elif if_AA==True and if_baseline!=True:
        if if_prob == True:
            fpr, tpr, _ = roc_curve([0 for i in range(len(prob1))] + [1 for i in range(len(prob2))], prob1 + prob2,drop_intermediate=if_drop)
        else:
            fpr, tpr, _ = roc_curve([1 for i in range(len(prob1))] + [0 for i in range(len(prob2))], prob1 + prob2,drop_intermediate=if_drop)
    # model estimates/prob
    elif if_baseline !=True and if_AA != True:
        if if_prob != True: #estimates
            prob1_t = [abs(x - 1) for x in prob1]
            prob2_t = [abs(x - 1) for x in prob2]
            fpr, tpr, _ = roc_curve([1 for i in range(len(prob1_t))] + [0 for i in range(len(prob2_t))], prob1_t + prob2_t,drop_intermediate=if_drop)
        else:
            fpr, tpr, _ = roc_curve([1 for i in range(len(prob1))] + [0 for i in range(len(prob2))], prob1 + prob2,drop_intermediate=if_drop)
    return(fpr, tpr)
