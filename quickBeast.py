import re
import os
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from read_data import read_one_pickle
from sklearn.metrics import roc_curve, auc,precision_recall_curve
import matplotlib.pyplot as plt
import math
from read_data import read_one_pickle, get_NS_p_values
from ROC_common import Make_judgement, calculate_posterior_value
from ROC_noLAMBDA import get_ROC_AUC, Prepare_data_fix
import sys
sys.path.append('/home/scarlett/github/BEASTIE')
# from BEASTIE import predict_lambda_GAM
from math import log
from statsmodels.stats.multitest import multipletests
from prettytable import PrettyTable


def get_qb_output(qb_POS, qb_NEG, qb_path):
    qb_pos_file=pd.read_csv(f"{qb_path}/{qb_POS}",delimiter="\t",header=0)
    qb_neg_file=pd.read_csv(f"{qb_path}/{qb_NEG}",delimiter="\t",header=0)
    return qb_pos_file,qb_neg_file

def get_BEASTIE_pval(NEG,POS):
    NEG_pval = NEG["st_p_value"].values
    POS_pval = POS["st_p_value"].values
    return POS_pval,NEG_pval

def get_data(n_gene,n_hets,n_depth,alt,sigma=0.7):
    NEG=f"g-{n_gene}_h-{n_hets}_d-{n_depth}_t-1_s-{sigma}.pickle"
    POS=f"g-{n_gene}_h-{n_hets}_d-{n_depth}_t-{alt}_s-{sigma}.pickle"
    ns_NEG,ns_POS,qb_NEG_name,qb_POS_name = get_filename(POS,NEG)
    path_beastie = "/data2/stan/iBEASTIE4/sigma0.7/parametrized/ASE_0.05_error/output_pkl/"
    path_qb=f"/data2/stan/quickBEAST/a8.789625_b8.789625/lambda0.04545/parametrized/ASE_0.05_error/"
    NS_path="/data2/stan/binomial/parametrized/ASE_0.05_error/NS_p"
    # qb
    qb_POS, qb_NEG = get_qb_output(qb_POS_name, qb_NEG_name, path_qb)
    _,_,mode_qb_pos_p_st,mode_qb_neg_p_st = get_tsv_p_values(qb_POS_name, qb_NEG_name, path_qb)
    # b
    b_pos, b_neg = get_BEASTIE_tsv(NEG,POS,path_beastie)
    b_pos_p, b_neg_p = get_BEASTIE_pval(b_neg,b_pos)
    # ns
    ns_POS_p, ns_NEG_p = get_NS_p_values(ns_POS, ns_NEG, NS_path)
    return mode_qb_pos_p_st,mode_qb_neg_p_st, b_pos_p, b_neg_p,ns_POS_p, ns_NEG_p, qb_POS, qb_NEG, b_pos, b_neg

def get_qb_ns_data(n_gene,n_hets,n_depth,alt,sigma=0.7):
    NEG=f"g-{n_gene}_h-{n_hets}_d-{n_depth}_t-1_s-{sigma}.pickle"
    POS=f"g-{n_gene}_h-{n_hets}_d-{n_depth}_t-{alt}_s-{sigma}.pickle"
    ns_NEG,ns_POS,qb_NEG_name,qb_POS_name = get_filename(POS,NEG)
    path_qb=f"/data2/stan/quickBEAST/a8.789625_b8.789625/lambda0.04545/parametrized/ASE_0.05_error/"
    NS_path="/data2/stan/binomial/parametrized/ASE_0.05_error/NS_p"
    MS_path="/data2/stan/binomial/parametrized/ASE_0.05_error/MS_p"
    pseudo_path="/data2/stan/binomial/parametrized/ASE_0.05_error/pseudo_p"
    qb_POS, qb_NEG = get_qb_output(qb_POS_name, qb_NEG_name, path_qb)
    mean_qb_pos_p_st,mean_qb_neg_p_st,mode_qb_pos_p_st,mode_qb_neg_p_st = get_tsv_p_values(qb_POS_name, qb_NEG_name, path_qb)
    ns_POS_p, ns_NEG_p = get_NS_p_values(ns_POS, ns_NEG, NS_path)
    ms_POS_p, ms_NEG_p = get_NS_p_values(ns_POS, ns_NEG, MS_path)
    pseudo_POS_p, pseudo_NEG_p = get_NS_p_values(ns_POS, ns_NEG, pseudo_path)
    return mean_qb_pos_p_st,mean_qb_neg_p_st,mode_qb_pos_p_st, mode_qb_neg_p_st, ns_POS_p, ns_NEG_p, ms_POS_p, ms_NEG_p, pseudo_POS_p, pseudo_NEG_p,qb_POS, qb_NEG

def get_BEASTIE_tsv(NEG,POS,path_beastie):
    NEG = NEG.replace('.pickle', '.tsv')
    POS = POS.replace('.pickle', '.tsv')
    NEG = pd.read_csv(path_beastie+NEG,sep="\t")
    POS = pd.read_csv(path_beastie+POS,sep="\t")
    return POS, NEG

def calculate_AUC_qb(path, file_pos, file_neg):
    _, _, qb_NEG, qb_POS = get_filename(file_pos,file_neg)
    mean_qb_pos_p_st,mean_qb_neg_p_st,mode_qb_pos_p_st,mode_qb_neg_p_st = get_tsv_p_values(qb_POS, qb_NEG, path)
    #t_fpr, t_tpr, _, _, = get_ROC_tsv(qb_pos_p_t,qb_neg_p_t)
    mean_st_fpr, mean_st_tpr, _, _, = get_ROC_tsv(mean_qb_pos_p_st,mean_qb_neg_p_st)
    mode_st_fpr, mode_st_tpr, _, _, = get_ROC_tsv(mode_qb_pos_p_st,mode_qb_neg_p_st)
    #n_fpr, n_tpr, _, _, = get_ROC_tsv(qb_pos_p_n,qb_neg_p_n)
    return mean_st_fpr, mean_st_tpr,mode_st_fpr, mode_st_tpr

def get_tsv_p_values(qb_POS, qb_NEG, qb_path):
    qb_pos_file=pd.read_csv(f"{qb_path}/{qb_POS}",delimiter="\t",header=0)
    qb_neg_file=pd.read_csv(f"{qb_path}/{qb_NEG}",delimiter="\t",header=0)
    mean_qb_pos_p_st = qb_pos_file['mean_st_p_value'].tolist()
    mean_qb_neg_p_st = qb_neg_file['mean_st_p_value'].tolist()
    mode_qb_pos_p_st = qb_pos_file['mode_st_p_value'].tolist()
    mode_qb_neg_p_st = qb_neg_file['mode_st_p_value'].tolist()
    # mean_qb_pos_p_st = qb_pos_file['qb_mean'].tolist()
    # mean_qb_neg_p_st = qb_neg_file['qb_mean'].tolist()
    # mode_qb_pos_p_st = qb_pos_file['qb_mode'].tolist()
    # mode_qb_neg_p_st = qb_neg_file['qb_mode'].tolist()
    return mean_qb_pos_p_st,mean_qb_neg_p_st,mode_qb_pos_p_st,mode_qb_neg_p_st

def get_ROC_tsv(prob1,prob2):
    fpr, tpr, _ = roc_curve([1 for i in range(len(prob1))] + [0 for i in range(len(prob2))], prob1 + prob2,pos_label=0,drop_intermediate=True)
    precision,recall, _ = precision_recall_curve([1 for i in range(len(prob1))] + [0 for i in range(len(prob2))], prob1 + prob2,pos_label=1)
    return fpr, tpr, precision, recall

def Table_qb_NS_power(POS,NEG,qb_path,NS_path,MS_path,alpha_beta,type1error,lambdas = None):
    ns_NEG,ns_POS,qb_NEG,qb_POS = get_filename(POS,NEG)
    qb_POS_p, qb_NEG_p,_,_ = get_qb_p_values(qb_POS, qb_NEG, qb_path)
    ns_POS_p, ns_NEG_p = get_NS_p_values(ns_POS, ns_NEG, NS_path)
    ms_POS_p, ms_NEG_p = get_NS_p_values(ns_POS, ns_NEG, MS_path)
    #qb_power, qb_type1error = Calculate_bonferroni_power_type1error(qb_POS_p,qb_NEG_p,threshold=0.05)
    #ns_power, ns_type1error = Calculate_bonferroni_power_type1error(ns_POS_p,ns_NEG_p,threshold=0.05)
    fdr_qb_power, fdr_qb_type1error = Calculate_FDR_power_type1error(qb_POS_p,qb_NEG_p,threshold=0.05)
    fdr_ns_power, fdr_ns_type1error = Calculate_FDR_power_type1error(ns_POS_p,ns_NEG_p,threshold=0.05)
    fdr_ms_power, fdr_ms_type1error = Calculate_FDR_power_type1error(ms_POS_p,ms_NEG_p,threshold=0.05)
    # Plotting the histograms
    print((f"qB    FDR          power      = ALT (qB pval) <= 0.05"))
    print((f"qB    FDR          type1error = REF (qB pval) <= 0.05"))
    print((f"NS/MS FDR          power      = ALT (   pval) <= 0.05"))
    print((f"NS/MS FDR          type1error = REF (   pval) <= 0.05"))
    Table = PrettyTable(["sample Name", "qB (power)","NS (power)", "MS (power)","qb (type1er)","NS (type1er)","MS (type1er)"])
    Table.add_row([os.path.splitext(NEG)[0], fdr_qb_power,fdr_ns_power, fdr_ms_power,fdr_qb_type1error,fdr_ns_type1error,fdr_ms_type1error])
    print(Table)

def print_qb_NS_power_table(n_gene,n_hets,n_depth,alt,alpha_beta,lambdas,sigma,alpha):
    print(f">>>> expected type1error at {alpha}, alpha/beta parameter at {alpha_beta}")

    b_NEG=f"g-{n_gene}_h-{n_hets}_d-{n_depth}_t-1_s-{sigma}.pickle"
    b_POS=f"g-{n_gene}_h-{n_hets}_d-{n_depth}_t-{alt}_s-{sigma}.pickle"

    DCC_path="/data2/stan"
    # candidate_lambdas = np.linspace(1, 3, 2000)
    # gam_model= load(open("/home/scarlett/github/BEASTIE/BEASTIE/iBEASTIE4_s0.7_GAM/gam4_lambdamodel.pkl", "rb"))
    # predicted_lambda=predict_lambda_GAM.get_lambda_from_gam(gam_model, log(n_hets), log(n_depth), float(expected_type1error),candidate_lambdas)
    # print(f"predicted lambda is {predicted_lambda}")
    model="BEASTIE3-pi0.05"
    qb_path=f"/data2/stan/quickBEAST/a{alpha_beta}_b{alpha_beta}/lambda{lambdas}/parametrized/ASE_0.05_error"
    NS_path="/data2/stan/binomial/parametrized/ASE_0.05_error/NS_p"
    MS_path="/data2/stan/binomial/parametrized/ASE_0.05_error/MS_p"

    Table_qb_NS_power(b_POS,b_NEG,qb_path,NS_path,MS_path,alpha_beta,alpha)

def get_qb_data(filename,path_qb,parameter):
    base_filename = os.path.splitext(filename)[0] # remove extension
    parts = base_filename.split('_') # split by underscore
    qb_filename = '_'.join(parts[:-1])
    # read data
    #BEASTIE = read_one_pickle(path+"/"+filename)
    qb=pd.read_csv(f"{path_qb}/{qb_filename}.txt",delimiter="\t",header=None)
    qb.columns=['geneID','qb_posterior','qb_mean','qb_var','qb_zscore']
    # qb.columns=['geneID','qb_mean','qb_var','qb_zscore','normal_p_value','t_p_value','st_p_value']
    #qb['qb_lambda'] = pd.to_numeric(qb['qb_lambda'], errors='coerce')
    #qb['qb_posterior'] = pd.to_numeric(qb['qb_posterior'], errors='coerce')
    #qb['t_p_value'] = pd.to_numeric(qb['t_p_value'], errors='coerce')
    #qb['converted_qB_lambda_right'] = (0.5 + qb['qb_lambda']) / (1 - (0.5 + qb['qb_lambda']))
    #qb['converted_qB_lambda_left'] = (0.5 - qb['qb_lambda']) / (1 - (0.5 - qb['qb_lambda']))
    #qb['converted_qB_lambda'] = qb.apply(lambda row: max(row['converted_qB_lambda_left'], row['converted_qB_lambda_right']), axis=1)
    return qb

def calculate_qb_lambda_posterior(gene_dict,qb):
    value_list=[]
    for geneID in gene_dict:  # iterate over each key in the dictionary
        qb_lambda_value = qb.loc[qb['geneID'] == geneID, 'converted_qB_lambda'].values[0]
        thetas=gene_dict[geneID]
        log2_thetas = np.log2(np.array(thetas))
        n_total = len(log2_thetas)
        Lambda=qb_lambda_value
        min_l = 1 / Lambda
        max_l = Lambda
        min_l_log2 = math.log2(min_l)
        max_l_log2 = math.log2(max_l)
        n_less_log2 = np.count_nonzero(log2_thetas < min_l_log2)
        n_more_log2 = np.count_nonzero(log2_thetas > max_l_log2)
        max_log2_score = max(n_less_log2, n_more_log2) / n_total
        sum_log2_score = (n_less_log2 + n_more_log2) / n_total
        value_list.append(sum_log2_score)  # add the mean to the list of means
    return value_list

def Calculate_power_type1error_for_posterior(POS,NEG,path,parameter,type1error,lambdas=None):
    ALT,qb_ALT = get_data(POS,path,parameter)
    REF,qb_REF = get_data(NEG,path,parameter)
    # REF (B: B with gam lambda; B_qb: B with qb lambda)
    BEASTIE_REF = calculate_posterior_value(calculation="max_prob",prob=REF,Lambda=lambdas)
    BEATIE_qb_lambda_REF = calculate_qb_lambda_posterior(REF,qb_REF)
    qb_REF = qb_REF['qb_posterior'].tolist()
    # ALT
    BEASTIE_ALT=calculate_posterior_value(calculation="max_prob",prob=ALT,Lambda=lambdas)
    BEATIE_qb_lambda_ALT=calculate_qb_lambda_posterior(ALT,qb_ALT)
    qb_ALT=qb_ALT['qb_posterior'].tolist()
    # bonferroni power
    BEASTIE_power=len([i for i in BEASTIE_ALT if i>0.5])/1000
    BEATIE_qb_lambda_power=len([i for i in BEATIE_qb_lambda_ALT if i>0.5])/1000
    qb_power=len([i for i in qb_ALT if i>0.5])/1000
    # bonferroni type1error
    BEASTIE_type1error=len([i for i in BEASTIE_REF if i>0.5])/1000
    BEATIE_qb_lambda_type1rror=len([i for i in BEATIE_qb_lambda_REF if i>0.5])/1000
    qb_type1error=len([i for i in qb_REF if i>0.5])/1000
    # return output
    return format(BEASTIE_power,'.5f'),format(BEATIE_qb_lambda_power,'.5f'),format(qb_power,'.5f'),format(BEASTIE_type1error,'.5f'),format(BEATIE_qb_lambda_type1rror,'.5f'),format(qb_type1error,'.5f')


def get_filename(b_POS,b_NEG):  
    ns_NEG = re.sub(r'_s-\d+(\.\d+)?', '', b_NEG)
    ns_POS = re.sub(r'_s-\d+(\.\d+)?', '', b_POS)
    qb_NEG = re.sub(r'_s-\d+(\.\d+)?', '', b_NEG).replace('.pickle', '.txt')
    qb_POS = re.sub(r'_s-\d+(\.\d+)?', '', b_POS).replace('.pickle', '.txt')
    return ns_NEG,ns_POS,qb_NEG,qb_POS

def get_qb_p_values(qb_POS, qb_NEG, qb_path):
    qb_pos_file=pd.read_csv(f"{qb_path}/{qb_POS}",delimiter="\t",header=0)
    qb_neg_file=pd.read_csv(f"{qb_path}/{qb_NEG}",delimiter="\t",header=0)
    qb_pos_p_t = qb_pos_file['t_p_value'].tolist()
    qb_neg_p_t = qb_neg_file['t_p_value'].tolist()
    qb_pos_p_n = qb_pos_file['normal_p_value'].tolist()
    qb_neg_p_n = qb_neg_file['normal_p_value'].tolist()
    return qb_pos_p_t,qb_neg_p_t,qb_pos_p_n,qb_neg_p_n


def Calculate_power_type1error_pval(pvals_alt, pvals_null, threshold=0.05):
    n = len(pvals_null)
    ######### bonferroni correction
    alpha_corrected = threshold / n
        # Type I Error
    type1error = np.mean(np.array(pvals_null) < alpha_corrected)
        # Power
    power = np.mean(np.array(pvals_alt) < alpha_corrected)

    ######### FDR correction
    # Perform FDR correction using Benjamini-Hochberg method for POS
    _, corrected_POS, _, _ = multipletests(pvals_alt, method='fdr_bh')
    # Perform FDR correction using Benjamini-Hochberg method for NEG
    _, corrected_NEG, _, _ = multipletests(pvals_null, method='fdr_bh')
        # FDR
    # Number of declared positives and false positives based on corrected p-values
    R = np.sum(corrected_POS < threshold)
    V = np.sum(corrected_NEG < threshold)
    FDR = V / (V + R) if R > 0 else 0
        # TDR
    T = np.sum(np.array(pvals_alt) < threshold)  # Total true hypotheses
    S = np.sum(np.array(pvals_alt) < threshold)  # True hypotheses declared significant
    TDR = S / T if T > 0 else 0
        # type 1 error
    fdr_type1error = np.mean(np.array(corrected_NEG) < threshold)
        # power
    fdr_power = np.mean(np.array(corrected_POS) < threshold)
    return format(power,'.5f'), format(type1error,'.5f'),format(TDR,'.5f'), format(FDR,'.5f'),format(fdr_power,'.5f'), format(fdr_type1error,'.5f')

def Calculate_bonferroni_power_type1error(POS,NEG,threshold=0.05):
    cutoff = threshold/len(POS)
    power = len([i for i in POS if float(i) <= cutoff]) / len(POS)
    # false positive 
    type1error = len([i for i in NEG if float(i) <= cutoff]) / len(NEG)
    return format(power,'.5f'), format(type1error,'.5f')

def get_ROC_qb(prob1,prob2):
    fpr, tpr, _ = roc_curve([0 for i in range(len(prob1))] + [1 for i in range(len(prob2))], prob1 + prob2,pos_label=1,drop_intermediate=True)
    precision,recall, _ = precision_recall_curve([1 for i in range(len(prob1))] + [0 for i in range(len(prob2))], prob1 + prob2,pos_label=1)
    return fpr, tpr, precision, recall

def ROC_comparison_fix3_qb(source,model,workdir,calculation,lambda_model,chosen_lambda=None,theta_pos=None,theta_neg=None,gene=None,hets=None,depth=None,sigma=None,title=None,Num_para=None,Num_col=None):
    # judge whether  2 of the 4 are None, and 2 of the 4 are not None
    Make_judgement(Num_para=Num_para,theta_pos=theta_pos,theta_neg=theta_neg,gene=gene,hets=hets,depth=depth,sigma=sigma)
    Plot_ROC_fix3_qb(source,model=model,workdir=workdir,calculation=calculation,lambda_model=lambda_model,chosen_lambda=chosen_lambda,Num_col=Num_col,gene=gene, hets=hets, depth=depth,sigma=sigma, theta_pos=theta_pos,theta_neg=theta_neg,title=title)

def Generate_path_qb(source,model,sigma,workdir,calculation="max prob"):
    path_model = f"{source}/{model}/sigma{sigma}/{workdir}/output_pkl/"
    path_qb = f"{source}/quickBEAST/a8.789625_b8.789625/lambda0.04545/{workdir}/"
    if calculation == "max prob":
        postfix="p"
    else:
        postfix="esti"
    path_NS=f"{source}/binomial/{workdir}/NS_"+postfix+"/" 
    path_MS=f"{source}/binomial/{workdir}/MS_"+postfix+"/" 
    return path_model, path_qb, path_NS, path_MS

def Plot_ROC_fix3_qb(source,model,workdir,calculation,lambda_model,chosen_lambda,Num_col,gene, hets, depth, sigma,theta_pos,theta_neg,title):
    gam_model_path="/home/scarlett/github/BEASTIE/BEASTIE/iBEASTIE4_s0.7_GAM/"
    gam_model = pickle.load(open(gam_model_path+str(lambda_model), "rb"))

    path_model, path_qb, path_NS, path_MS = Generate_path_qb(source,model,sigma,workdir)
    d_group,var,var_map_np,fixed_var_np,var_fullname_map_np,variable_var_np,pos_pd,neg_pd = Prepare_data_fix(gene, hets, depth, sigma,source,model,workdir,calculation,theta_pos,theta_neg=1,Num_para=3)
    if Num_col == None:
        Num_col = 3
    else:
        Num_col = int(Num_col)
    row = math.ceil(float(len(d_group))/Num_col)
    fig, axs = plt.subplots(row, Num_col, figsize = (17,6*row))
    if (row * Num_col > len(d_group)):
        for i in range(row * Num_col - len(d_group)):
            axs.flat[-1-i].set_axis_off()

    for i, each in enumerate(d_group):
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
            hets=int(h)
            totalcount=int(h)*int(d)
            expected_type1error = 0.05/int(g)
            #
            if chosen_lambda is None:
                chosen_lambda = predict_BEASTIE_GAM_lambda(gam_model, hets, totalcount, expected_type1error)
            # calculate auc score
            fpr_m,tpr_m,s1 = get_ROC_AUC(path_model,current_group_pos_list[idx],current_group_neg_list[idx],calculation,lambdas=chosen_lambda)
            fpr_b,tpr_b,s2 = get_ROC_AUC(path_NS,reduced_file_pos,reduced_file_neg,calculation,if_baseline=True)
            fpr_b2,tpr_b2,s3 = get_ROC_AUC(path_MS,reduced_file_pos,reduced_file_neg,calculation,if_baseline=True)
            t_fpr, t_tpr, n_fpr, n_tpr = calculate_AUC_qb(path_qb, current_group_pos_list[idx], current_group_neg_list[idx])
            s4 = round(auc(t_fpr,t_tpr),3)
            # 
            var_dict = {"gene":g, "hets": h, "depth": d, "sigma": s}
            for each in fixed_var_np:
                xlabels += each+":"+var_dict[each]+' '
            axs.flat[i].set_xlabel("FPR",fontsize=12)
            axs.flat[i].set_ylabel("TPR",fontsize=12)
            axs.flat[i].set_title(var_fullname_map_np[np.array(var) == None][0]+":" + var_dict[var_fullname_map_np[np.array(var) == None][0]],fontsize=15)
            # 
            #axs.flat[i].plot(fpr_m, tpr_m, label =  f"{model} gam λ : {s1}",color = "dodgerblue", alpha=0.8,linewidth=3)
            axs.flat[i].plot(fpr_b, tpr_b, label =  "Naive Sum : "+str(s2), color="darkorange", alpha=0.8,linewidth=3)
            axs.flat[i].plot(fpr_b2, tpr_b2, label =  "Major Site : "+str(s3),color= "limegreen",alpha=0.8,linewidth=3)
            axs.flat[i].plot(t_fpr,t_tpr, label =  "qb fixed λ (t) : "+str(s4),color= "crimson", alpha=0.8,linewidth=3)
            # 
            axs.flat[i].tick_params(axis='both', labelsize=12)
            axs.flat[i].legend(fontsize=13,loc='lower right')
    plt.suptitle(str(title)+"\n"+xlabels+"theta pos: "+str(theta_pos)+" theta neg: "+str(theta_neg),fontsize=20)
    plt.show()

def predict_BEASTIE_GAM_lambda(gam_model, hets, totalcount, expected_type1error):
    lambda_pick = predict_lambda_GAM.get_lambda_from_gam(gam_model, log(hets), log(totalcount), expected_type1error, candidate_lambdas = np.linspace(1, 3, 2000))
    return lambda_pick