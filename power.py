import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels as sm
import scipy.stats as st
import matplotlib.pyplot as plt
import os
import sys
from . import ROC_common
from . import read_data
from math import log,log2,ceil
from scipy.special import expit
BEASTIE_path="/home/scarlett/github/BEASTIE"
sys.path.append(str(BEASTIE_path))
from BEASTIE import run_model_stan_wrapper,predict_lambda_GAM

# def Plot_power_curve(DCC_path,sigma,simulator,workdir,type1error,percentError,theta_alt,cutoff,lambda_model,gene=None,hets=None,depth=None,num=3):

#     gam_model_path="/home/scarlett/github/BEASTIE/BEASTIE/"
#     gam_model = pickle.load(open(gam_model_path+str(lambda_model), "rb"))
#     candidate_log_lambdas = np.log(np.linspace(1, 3, 3000))
#     source = DCC_path+"/"
#     # judge whether  2 of the 4 are None, and 2 of the 4 are not None
#     theta_pos=theta_alt
#     theta_neg=1
#     var=[gene,hets,depth,sigma]
#     if var.count(None)!=2:
#         raise Exception('Two variables have to be set to None. The number of None from input was {}'.format(var.count(None)))
#     ############################################################
#     # data directory is known 
#     path_imodel=source + "iBEASTIE3/sigma" +str(sigma)+"/"+str(simulator)+ "/"+str(workdir)+"/output_pkl" #g-1000_h-5_d-5_t-0.75_CEU_s-0.5.pickle
#     path_model=source + "BEASTIE3/sigma" +str(sigma)+"/"+str(simulator)+ "/"+str(workdir)+"/output_pkl"
#     #path_AA=source + "ADM/output_pkl/"+workdir+"/AA_pval"      
#     path_NS=source + "binomial/"+str(simulator)+ "/"+str(workdir)+"/NS_p"
#     path_MS=source + "binomial/"+str(simulator)+ "/"+str(workdir)+"/MS_p"
#     ############################################################
#     var_map_np = np.array(['g','h','d','s'])
#     var_fullname_map_np = np.array(['gene','hets','depth','sigma'])
#     full_var_map_np = np.array([gene, hets, depth, sigma])
#     valid_var_np = var_map_np[np.array(var) != None]
#     variable_var_np = var_fullname_map_np[np.array(var) == None]
#     fixed_var_np = var_fullname_map_np[np.array(var) != None]
#     valid_full_var_np = full_var_map_np[np.array(var) != None]

#     if hets is not None:
#         h=hets
#     if gene is not None:
#         g=gene
#     if depth is not None:
#         d=depth
#     if sigma is not None:
#         s=sigma

#     all_file = sorted(os.listdir(path_model))
#     file_dict = {}
#     if "CEU/g-1000" in workdir:
#         postfix="CEU_s-0.5.pickle"
#     elif "CEU_enrichedEr" in workdir:
#         postfix="CEU_enrichedEr_s-0.5.pickle"
#     else:
#         postfix=".pickle"

#     for pkl in all_file:
#         if postfix in pkl:
#             name=pkl.rsplit(".pickle")[0].rsplit("_")
#             if "CEU" in workdir:
#                 name.remove("CEU")
#                 if "CEU_enrichedEr" in workdir:
#                     name.remove("enrichedEr")
#             file_dict[pkl] = {}
#             for each_value in name:
#                 file_dict[pkl][each_value.split("-")[0]] = float(each_value.split("-")[1])
#         else:continue

#     file_dict_pd = pd.DataFrame(file_dict).transpose()
#     file_dict_pd['file'] = file_dict_pd.index

#     pos_pd = file_dict_pd[(file_dict_pd[valid_var_np[0]] == valid_full_var_np[0])&(file_dict_pd[valid_var_np[1]] == valid_full_var_np[1])&(file_dict_pd['t'] == theta_pos)].sort_values(['d','h','g','s'])
#     #neg_pd = file_dict_pd[(file_dict_pd[valid_var_np[0]] == valid_full_var_np[0])&(file_dict_pd[valid_var_np[1]] == valid_full_var_np[1])&(file_dict_pd['t'] == theta_neg)].sort_values(['d','h','g','s'])
#     ############################################################
#     ############################################################

#     gene_neg=gene
#     var_neg=[gene_neg,hets,depth,sigma]
#     var_map_np = np.array(['g','h','d','s'])
#     var_fullname_map_np = np.array(['gene','hets','depth','sigma'])
#     full_var_map_np = np.array([gene, hets, depth, sigma])
#     valid_var_np = var_map_np[np.array(var) != None]
#     variable_var_np = var_fullname_map_np[np.array(var) == None]
#     fixed_var_np = var_fullname_map_np[np.array(var) != None]
#     valid_full_var_np = full_var_map_np[np.array(var) != None]

#     if hets is not None:
#         h=hets
#     if sigma is not None:
#         s=sigma
#     if gene_neg is not None:
#         g=gene_neg
#     if depth is not None:
#         d=depth
#     all_file = sorted(os.listdir(path_model))
#     file_dict = {}

#     for pkl in all_file:
#         if postfix in pkl:
#             name=pkl.rsplit(".pickle")[0].rsplit("_")
#             if "CEU" in workdir:
#                 name.remove("CEU")
#                 if "CEU_enrichedEr" in workdir:
#                     name.remove("enrichedEr")
#             file_dict[pkl] = {}
#             for each_value in name:
#                 file_dict[pkl][each_value.split("-")[0]] = float(each_value.split("-")[1])
#         else:continue
#     file_dict_pd = pd.DataFrame(file_dict).transpose()
#     file_dict_pd['file'] = file_dict_pd.index
#     neg_pd = file_dict_pd[(file_dict_pd[valid_var_np[0]] == valid_full_var_np[0])&(file_dict_pd[valid_var_np[1]] == valid_full_var_np[1])&(file_dict_pd['t'] == theta_neg)].sort_values(['d','h','g','s'])
#     #############################################################
#     d_group = pos_pd[var_map_np[np.array(var) == None][0]].unique()
#     read_depth = pos_pd[var_map_np[np.array(var) == None][1]].unique()

#     if num == None:
#         num = 3
#     else:
#         num = int(num)
#     row = math.ceil(float(len(d_group))/num)
#     fig, axs = plt.subplots(row, num, figsize = (20,7*row))
#     if (row * num > len(d_group)):
#         for i in range(row * num - len(d_group)):
#             axs.flat[-1-i].set_axis_off()

#     xlabels = "Data with %s percent error, gene: %s , sigma: %s, cutoff set at %s-th percentile of NULL"%(str(percentError),gene_neg,sigma,cutoff)

#     labels = ""
#     df1 = pd.DataFrame({'Model':[],'Het':[],'d5':[], 'd10':[], 'd20':[], 'd30':[], 'd40':[], 'd50':[], 'd60':[], 'd70':[], 'd80':[], 'd90':[], 'd100':[]})
#     df2=df1
#     #df3=df1
#     df4=df1
#     df4=df1
#     df5=df1
#     #d_group=d_group[:-1]
#     #print(d_group)

#     for i, each in enumerate(d_group):
#         current_group_pos_list = pos_pd[pos_pd[var_map_np[np.array(var) == None][0]] == each].index
#         current_group_neg_list = neg_pd[neg_pd[var_map_np[np.array(var) == None][0]] == each].index
#         power_imodel_list=[]
#         power_model_list=[]
#         power_NS_list=[]
#         power_MS_list=[]
        
#         cutoff_imodel_list=[]
#         cutoff_model_list=[]
#         cutoff_NS_list=[]
#         cutoff_MS_list=[]
#         for idx in range(len(current_group_pos_list)):
#             if "CEU_" in workdir:
#                 name = current_group_pos_list[idx].replace("_CEU", "")
#             if "CEU_enrichedEr" in workdir:
#                 name = current_group_pos_list[idx].replace("_CEU_enrichedEr", "")
#             g=name.rsplit(".pickle")[0].rsplit("_")[0].rsplit("-")[1]
#             h=name.rsplit(".pickle")[0].rsplit("_")[1].rsplit("-")[1]
#             d=name.rsplit(".pickle")[0].rsplit("_")[2].rsplit("-")[1]
#             t=name.rsplit(".pickle")[0].rsplit("_")[3].rsplit("-")[1]
#             s=name.rsplit(".pickle")[0].rsplit("_")[4].rsplit("-")[1]
#             if "CEU/g-1000" in workdir:
#                 new_POS = f"g-{g}_h-{h}_d-{d}_t-{theta_alt}_CEU_s-{s}.pickle"
#                 new_NEG = f"g-{g}_h-{h}_d-{d}_t-{theta_neg}_CEU_s-{s}.pickle"
#                 reduced_POS=f"g-{g}_h-{h}_d-{d}_t-{theta_alt}_CEU.pickle"
#                 reduced_NEG=f"g-{g}_h-{h}_d-{d}_t-{theta_neg}_CEU.pickle"
#             elif "CEU_enrichedEr" in workdir:
#                 new_POS = f"g-{g}_h-{h}_d-{d}_t-{theta_alt}_CEU_enrichedEr_s-{s}.pickle"
#                 new_NEG = f"g-{g}_h-{h}_d-{d}_t-{theta_neg}_CEU_enrichedEr_s-{s}.pickle"
#                 reduced_POS=f"g-{g}_h-{h}_d-{d}_t-{theta_alt}_CEU_enrichedEr.pickle"
#                 reduced_NEG=f"g-{g}_h-{h}_d-{d}_t-{theta_neg}_CEU_enrichedEr.pickle"
#             else:
#                 new_POS = f"g-{g}_h-{h}_d-{d}_t-{theta_alt}_s-{s}.pickle"
#                 new_NEG = f"g-{g}_h-{h}_d-{d}_t-{theta_neg}_s-{s}.pickle" 
#                 reduced_POS=f"g-{g}_h-{h}_d-{d}_t-{theta_alt}.pickle"
#                 reduced_NEG=f"g-{g}_h-{h}_d-{d}_t-{theta_neg}.pickle"

#             chosen_lambda=get_lambda_from_gam(gam_model,int(h),int(h)*int(d),type1error/int(g),candidate_log_lambdas)
#             #print(current_group_pos_list[idx])
#             if simulator=="semi_empirical":
#                 cutoff1,power1=Calculate_cutoff(new_POS,new_NEG,path_imodel,cutoff,lambdas=chosen_lambda)
#                 power_imodel_list.append(power1)
#                 cutoff_imodel_list.append(cutoff1)
#             cutoff2,power2=Calculate_cutoff(new_POS,new_NEG,path_model,cutoff,lambdas=chosen_lambda)
#             #cutoff3,power3=Calculate_cutoff(current_group_pos_list[idx],current_group_neg_list[idx],path_AA,cutoff,if_AA_baseline=True)
#             cutoff4,power4=Calculate_cutoff(reduced_POS,reduced_NEG,path_NS,cutoff,if_AA_baseline=True)
#             cutoff5,power5=Calculate_cutoff(reduced_POS,reduced_NEG,path_MS,cutoff,if_AA_baseline=True)
            
#             cutoff_model_list.append(cutoff2)
#             #cutoff_adam_list.append(cutoff3)
#             cutoff_NS_list.append(cutoff4)
#             cutoff_MS_list.append(cutoff5)
            
#             power_model_list.append(power2)
#             #power_adam_list.append(power3)
#             power_NS_list.append(power4)
#             power_MS_list.append(power5)
#         # for each read depth, we plot    
#         g=name.rsplit(".pickle")[0].rsplit("_")[0].rsplit("-")[1]
#         h=name.rsplit(".pickle")[0].rsplit("_")[1].rsplit("-")[1]
#         d=name.rsplit(".pickle")[0].rsplit("_")[2].rsplit("-")[1]
#         s=name.rsplit(".pickle")[0].rsplit("_")[4].rsplit("-")[1]
#         var_dict = {"gene":g, "hets": h, "depth": d, "sigma": s}
#         for each in variable_var_np:
#             if each != var_fullname_map_np[np.array(var) == None][0]:
#                 labels += each+":"+var_dict[each]+' '
#         axs.flat[i].set_ylabel("Power",fontsize=20)
#         axs.flat[i].set_xlabel("Read Depth per het site",fontsize=15)
#         axs.flat[i].set_ylim(0,1.1)
#         axs.flat[i].set_xlim(0,100)
#         if simulator=="semi_empirical":
#             axs.flat[i].plot(read_depth, power_imodel_list,'--ro',label="iBEASTIE")
#         axs.flat[i].plot(read_depth, power_model_list,'--mo',label="BEASTIE")
#         #axs.flat[i].plot(read_depth, power_adam_list,'--bo',label="ADAM")
#         axs.flat[i].plot(read_depth, power_NS_list,'--yo',label="NAIVE SUM")
#         axs.flat[i].plot(read_depth, power_MS_list,'--go',label="MAJOR SITE")
#         axs.flat[i].axhline(y=0.95,color='darkred',alpha=0.95,linestyle='--',label="power=95%")
#         axs.flat[i].axhline(y=0.85,color='red',alpha=0.95,linestyle='--',label="power=85%")
#         axs.flat[i].axhline(y=0.75,color='orangered',alpha=0.95,linestyle='--',label="power=75%")
#         axs.flat[i].axhline(y=0.5,color='lightsalmon',alpha=0.95,linestyle='--',label="power=50%") 
#         axs.flat[i].legend(loc='lower right',fontsize=14)
#         axs.flat[i].set_title(var_fullname_map_np[np.array(var) == None][0]+":" + var_dict[var_fullname_map_np[np.array(var) == None][0]],fontsize=20)
#         #
#         if simulator=="semi_empirical":
#             df1 = df1.append({'Model':"iBEASTIE "+str(percentError)+"%error",'Het':var_dict[var_fullname_map_np[np.array(var) == None][0]],'d5':round(power_imodel_list[0],4), 'd10':round(power_imodel_list[1],4), 'd20':round(power_imodel_list[2],4), 'd30':round(power_imodel_list[3],4), 'd40':round(power_model_list[4],4), 'd50':round(power_imodel_list[5],4), 'd60':round(power_imodel_list[6],4), 'd70':round(power_imodel_list[7],4), 'd80':round(power_imodel_list[8],4), 'd90':round(power_imodel_list[9],4), 'd100':round(power_imodel_list[10],4)}, ignore_index=True)
#         df2 = df2.append({'Model':"BEASTIE "+str(percentError)+"%error",'Het':var_dict[var_fullname_map_np[np.array(var) == None][0]],'d5':round(power_model_list[0],4), 'd10':round(power_model_list[1],4), 'd20':round(power_model_list[2],4), 'd30':round(power_model_list[3],4), 'd40':round(power_model_list[4],4), 'd50':round(power_model_list[5],4), 'd60':round(power_model_list[6],4), 'd70':round(power_model_list[7],4), 'd80':round(power_model_list[8],4), 'd90':round(power_model_list[9],4), 'd100':round(power_model_list[10],4)}, ignore_index=True)
#         #df3 = df3.append({'Model':"ADAM "+str(percentError)+"%error",'Het':var_dict[var_fullname_map_np[np.array(var) == None][0]],'d5':round(cutoff_adam_list[0],4), 'd10':round(cutoff_adam_list[1],4), 'd20':round(cutoff_adam_list[2],4), 'd30':round(cutoff_adam_list[3],4), 'd40':round(cutoff_adam_list[4],4), 'd50':round(cutoff_adam_list[5],4), 'd60':round(cutoff_adam_list[6],4), 'd70':round(cutoff_adam_list[7],4), 'd80':round(cutoff_adam_list[8],4), 'd90':round(cutoff_adam_list[9],4), 'd100':round(cutoff_adam_list[10],4)}, ignore_index=True)
#         df4 = df4.append({'Model':"NAIVE SUM "+str(percentError)+"%error",'Het':var_dict[var_fullname_map_np[np.array(var) == None][0]],'d5':round(power_NS_list[0],4), 'd10':round(power_NS_list[1],4), 'd20':round(power_NS_list[2],4), 'd30':round(power_NS_list[3],4), 'd40':round(power_NS_list[4],4), 'd50':round(power_NS_list[5],4), 'd60':round(power_NS_list[6],4), 'd70':round(power_NS_list[7],4), 'd80':round(power_NS_list[8],4), 'd90':round(power_NS_list[9],4), 'd100':round(power_NS_list[10],4)}, ignore_index=True)
#         df5 = df5.append({'Model':"MAJOR SITE "+str(percentError)+"%error",'Het':var_dict[var_fullname_map_np[np.array(var) == None][0]],'d5':round(power_MS_list[0],4), 'd10':round(power_MS_list[1],4), 'd20':round(power_MS_list[2],4), 'd30':round(power_MS_list[3],4), 'd40':round(power_MS_list[4],4), 'd50':round(power_MS_list[5],4), 'd60':round(power_MS_list[6],4), 'd70':round(power_MS_list[7],4), 'd80':round(power_MS_list[8],4), 'd90':round(power_MS_list[9],4), 'd100':round(power_MS_list[10],4)}, ignore_index=True)
        
#     plt.suptitle(xlabels,fontsize=25)
#     plt.show()
#     df2.set_index(['Model', 'Het'])
#     #df3.set_index(['Model', 'Het'])
#     df4.set_index(['Model', 'Het'])
#     df5.set_index(['Model', 'Het'])
#     df2=pd.DataFrame(df2)
#     df4=pd.DataFrame(df4)
#     df5=pd.DataFrame(df5)
#     if simulator=="semi_empirical":
#         df1.set_index(['Model', 'Het'])
#         df1=pd.DataFrame(df1)
#         combined_df = pd.concat([df1, df2, df4,df5])
#     else:
#         combined_df = pd.concat([df2, df4,df5])
#     return combined_df

def calculate_beastie_score(dict,geneID,Lambda):
    thetas=dict.get(geneID)
    log2_thetas = np.log2(np.array(thetas))
    _,sum_log2_score=run_model_stan_wrapper.computeBeastieScoreLog2(log2_thetas, Lambda)
    return sum_log2_score

def predict_lambda_from_realdata(selected_df,expected_type1error,gam1_model,gam2_model,gam3_model,gam4_model,gam5_model,gam6_model,plk):
    # prepare model input
    candidate_lambdas = np.linspace(1, 3, 2000)
    selected_df["log_hets"]=np.log(selected_df["number.of.hets"])
    selected_df["log_totalcount"]=np.log(selected_df["totalCount"])

    selected_df["gam1_lambda"] = selected_df.apply(
        lambda x: predict_lambda_GAM.get_lambda_from_gam(
            gam1_model, x["log_hets"], x["log_totalcount"], expected_type1error,candidate_lambdas
        ),
        axis=1,
    )
    selected_df["gam2_lambda"] = selected_df.apply(
        lambda x: predict_lambda_GAM.get_lambda_from_gam(
            gam2_model, x["log_hets"], x["log_totalcount"], expected_type1error,candidate_lambdas
        ),
        axis=1,
    )
    selected_df["gam3_lambda"] = selected_df.apply(
        lambda x: predict_lambda_GAM.get_lambda_from_gam(
            gam3_model, x["log_hets"], x["log_totalcount"], expected_type1error,candidate_lambdas
        ),
        axis=1,
    )
    selected_df["gam4_lambda"] = selected_df.apply(
        lambda x: predict_lambda_GAM.get_lambda_from_gam(
            gam4_model, x["log_hets"], x["log_totalcount"], expected_type1error,candidate_lambdas
        ),
        axis=1,
    )
    selected_df["gam5_lambda"] = selected_df.apply(
        lambda x: predict_lambda_GAM.get_lambda_from_gam(
            gam5_model, x["log_hets"], x["log_totalcount"], expected_type1error,candidate_lambdas
        ),
        axis=1,
    )
    selected_df["gam6_lambda"] = selected_df.apply(
        lambda x: predict_lambda_GAM.get_lambda_from_gam(
            gam6_model, x["log_hets"], x["log_totalcount"], expected_type1error,candidate_lambdas
        ),
        axis=1,
    )
    selected_df["posterior_mass_support_ALT_gam1"] = selected_df.apply(
        lambda x: calculate_beastie_score(
            plk, x["geneID"], x["gam1_lambda"]
        ),
        axis=1,
    )
    selected_df["posterior_mass_support_ALT_gam2"] = selected_df.apply(
        lambda x: calculate_beastie_score(
            plk, x["geneID"], x["gam2_lambda"]
        ),
        axis=1,
    )
    selected_df["posterior_mass_support_ALT_gam3"] = selected_df.apply(
        lambda x: calculate_beastie_score(
            plk, x["geneID"], x["gam3_lambda"]
        ),
        axis=1,
    )
    selected_df["posterior_mass_support_ALT_gam4"] = selected_df.apply(
        lambda x: calculate_beastie_score(
            plk, x["geneID"], x["gam4_lambda"]
        ),
        axis=1,
    )
    selected_df["posterior_mass_support_ALT_gam5"] = selected_df.apply(
        lambda x: calculate_beastie_score(
            plk, x["geneID"], x["gam5_lambda"]
        ),
        axis=1,
    )
    selected_df["posterior_mass_support_ALT_gam6"] = selected_df.apply(
        lambda x: calculate_beastie_score(
            plk, x["geneID"], x["gam6_lambda"]
        ),
        axis=1,
    )
    selected_df["posterior_mass_support_ALT_BEASTIE"] = selected_df.apply(
        lambda x: calculate_beastie_score(
            plk, x["geneID"], x["gam_lambda"]
        ),
        axis=1,
    )
    return selected_df


def getBatabinomial_pooled(fields):
    if len(fields) >= 4:
        Mreps = int(fields[1])
        pooled_A = 0
        pooled_R = 0
        pooled_min = 0
        for rep in range(Mreps):
            A = float(fields[2 + rep * 2])
            R = float(fields[3 + rep * 2])
            pooled_A = pooled_A + A
            pooled_R = pooled_R + R
            pooled_min = pooled_min + min(A, R)
        sum_AR = pooled_A + pooled_R

        p_value_11 = get_2sided_pval(pooled_A, sum_AR, 1, 1)
        p_value_1010 = get_2sided_pval(pooled_A, sum_AR, 10, 10)
        p_value_2020 = get_2sided_pval(pooled_A, sum_AR, 20, 20)
        p_value_5050 = get_2sided_pval(pooled_A, sum_AR, 50, 50)
        p_value_100100 = get_2sided_pval(pooled_A, sum_AR, 100, 100)

        return (
            round(p_value_11, 10),
            round(p_value_1010, 10),
            round(p_value_2020, 10),
            round(p_value_5050, 10),
            round(p_value_100100, 10),
        )
    else:
        return (None, None, None,None,None)

def Calculate_cutoff(POS,NEG,path,null_cutoff,calculation,expected_type1error,lambdas=None,if_AA_baseline=False,if_beta=False):
    ALT = read_data.read_one_pickle(path+"/"+POS)
    REF = read_data.read_one_pickle(path+"/"+NEG)
    if if_AA_baseline == True or if_beta is True:
        #cutoff=np.percentile(REF, 100-null_cutoff)
        power=len([i for i in ALT if i<=expected_type1error])/1000
        type1error=len([i for i in REF if i<=expected_type1error])/1000
    else:
        ALT=ROC_common.calculate_posterior_value(calculation,ALT,Lambda=lambdas)
        REF=ROC_common.calculate_posterior_value(calculation,REF,Lambda=lambdas)
        #if calculation == "max_prob":
            #cutoff=np.percentile(REF, null_cutoff)
        power=len([i for i in ALT if i>0.5])/1000
        type1error=len([i for i in REF if i>0.5])/1000
            #print('%s: critical p-values at %d-th percentile:%8.4f; The power is %8.4f' % (POS,null_cutoff,cutoff,power))
        #elif calculation == "median":
            #null_cutoff = 100-null_cutoff
            #cutoff=np.percentile(REF, null_cutoff)
            #power=len([i for i in ALT if i<cutoff])/1000
            #print('%s: psterior estimates at %d-th percentile:%8.4f; The power is %8.4f' % (POS,null_cutoff,cutoff,power))
    return format(power,'.4f'),format(type1error,'.4f'),ALT,REF

def Calculate_type1error(NEG,path,expected_type1error,lambdas=None,if_AA_baseline=False,if_beta=False):
    REF = read_data.read_one_pickle(path+"/"+NEG)
    if if_AA_baseline == True or if_beta == True:
        #cutoff=np.percentile(REF, 100-null_cutoff)
        type1error=len([i for i in REF if i<=expected_type1error])/1000
    else:
        REF=ROC_common.calculate_posterior_value("max_prob",REF,lambdas)
        type1error=len([i for i in REF if i>0.5])/1000
        #print('%s: critical p-values at %d-th percentile:%8.4f; The power is %8.4f' % (POS,null_cutoff,cutoff,power))
    return type1error

def Plot_cutoff(POS,NEG,path,null_cutoff,expected_type1error,lambdas=None,if_AA_baseline=False,if_beta=False,calculation="max_prob",title=""):
    power,type1er,ALT,REF=Calculate_cutoff(POS,NEG,path,null_cutoff,lambdas=lambdas,if_AA_baseline=if_AA_baseline,if_beta=if_beta,calculation=calculation,expected_type1error=expected_type1error)
    # Plotting the histograms
    plt.hist(REF, alpha=0.5, label=NEG)
    plt.hist(ALT, alpha=0.5, label=POS)
    if if_AA_baseline is True or if_beta is True:
        plt.title(f"{title} power: {power} (ALT (BEASTIE score w/ GAM lambda) > 0.5)\ntype1error: {type1er} (REF (BEASTIE score w/ GAM lambda) > 0.5)")
        plt.axvline(x=expected_type1error, color='r', linestyle='--',label=f"expected type1error: {expected_type1error}")
    else:
        print(f"predicted lambda: {lambdas} for type1error {expected_type1error}")
        if calculation == "max_prob":
            plt.title(f"{title} power: {power} (ALT (BEASTIE score w/ GAM lambda) > 0.5)\ntype1error: {type1er} (REF (BEASTIE score w/ GAM lambda) > 0.5)")
        else:
            null_cutoff = 100-null_cutoff
            plt.title(f"{title} power: {power} (ALT < null {null_cutoff}-th perncetile cutoff)")
        plt.axvline(x=0.5, color='r', linestyle='--',label=f"ASE cutoff: 0.5")
    if calculation == "max_prob":
        plt.xlabel('posterior mass support ALT with lambda')
    else:
        plt.xlabel('posterior estimates')
    plt.ylabel('counts')
    # Adding legend
    plt.legend()
    # Show the plot
    plt.show()

def Generate_path_power(source,model,sigma,workdir,model2,calculation="max_prob"):
    path_model = f"{source}/{model}/sigma{sigma}/{workdir}/output_pkl"
    path_model2 = f"{source}/{model2}/sigma{sigma}/{workdir}/output_pkl"
    path_beta1=None
    path_beta2=None
    path_beta3=None
    if calculation == "max_prob":
        path_NS=f"{source}/binomial/{workdir}/NS_p"
        path_MS=f"{source}/binomial/{workdir}/MS_p"
        path_beta1=f"{source}/betabinomial/{workdir}/betabinom_1_1_p"
        path_beta2=f"{source}/betabinomial/{workdir}/betabinom_50_50_p"
        path_beta3=f"{source}/betabinomial/{workdir}/betabinom_100_100_p"
    else:
        path_NS=f"{source}/binomial/{workdir}/NS_esti"
        path_MS=f"{source}/binomial/{workdir}/MS_esti"
    return path_model,path_NS,path_MS,path_model2,path_beta1,path_beta2,path_beta3

def Prepare_data_fix(gene, hets, depth, sigma,source,model,workdir,theta_pos,theta_neg,Num_para,model2=None):
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
        
    path_model,_,_,_,_,_,_= Generate_path_power(source,model,sigma,workdir,model2)
    all_file = sorted(os.listdir(path_model))
    file_dict = {}

    if "CEU/g-1000" in workdir:
        postfix="CEU_s-0.5.pickle"
    elif "CEU_enrichedEr" in workdir:
        postfix="CEU_enrichedEr_s-0.5.pickle"
    else:
        postfix=".pickle"

    for pkl in all_file:
        if postfix in pkl:
            name=pkl.rsplit(".pickle")[0].rsplit("_")
            if "CEU" in workdir:
                name.remove("CEU")
                if "CEU_enrichedEr" in workdir:
                    name.remove("enrichedEr")
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

def Plot_trend(df0_1,df0_2,df0_3,df0_4,model1,model2,percentError):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(16, 12))
    x=[5,10,20,30,40,50,60,70,80,90,100]
    for i in range(len(df0_1)):
        ax1.plot(x,np.array(df0_1.iloc[i][2:]),label="Het "+df0_1.iloc[i][1])
        ax1.legend(loc="lower right",fontsize=10)
        ax1.set_title(str(model)+':'+str(percentError)+'% error',fontsize=20)
        ax1.set_xlabel('Read depth')
    for i in range(len(df0_2)):
        ax2.plot(x,np.array(df0_2.iloc[i][2:]),label="Het "+df0_2.iloc[i][1])
        ax2.legend(loc="upper right",fontsize=10)
        ax2.set_title(str(model2)+':'+str(percentError)+'% error',fontsize=20)
        ax2.set_xlabel("Read depth")
    for i in range(len(df0_3)):
        ax3.plot(x,np.array(df0_3.iloc[i][2:]),label="Het "+df0_3.iloc[i][1])
        ax3.legend(loc="upper right",fontsize=10)
        ax3.set_title('NS:'+str(percentError)+'% error',fontsize=20) 
    for i in range(len(df0_4)):
        ax4.plot(x,np.array(df0_4.iloc[i][2:]),label="Het "+df0_4.iloc[i][1])
        ax4.legend(loc="upper right",fontsize=10)
        ax4.set_title('MS:'+str(percentError)+'% error',fontsize=20)


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

    path_model,path_NS,path_MS,path_model2,path_beta1,path_beta2,path_beta3 = Generate_path_power(source=source,model=model,sigma=sigma,workdir=percentError,model2=model2,calculation=calculation)
    d_group,var,var_map_np,fixed_var_np,var_fullname_map_np,variable_var_np,pos_pd,neg_pd = Prepare_data_fix(gene, hets, depth, sigma,source,model,percentError,theta_pos,theta_neg=1,Num_para=2)

    ############################################################
    read_depth = pos_pd[var_map_np[np.array(var) == None][1]].unique()

    if num == None:
        num = 3
    else:
        num = int(num)
    row = ceil(float(len(d_group))/num)
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
            cutoff1,power1=Calculate_cutoff(current_group_pos_list[idx],current_group_neg_list[idx],path_model,cutoff,calculation=calculation)
            cutoff2,power2=Calculate_cutoff(reduced_file_pos,reduced_file_neg,path_NS,cutoff,if_AA_baseline=True,calculation=calculation)
            cutoff3,power3=Calculate_cutoff(reduced_file_pos,reduced_file_neg,path_MS,cutoff,if_AA_baseline=True,calculation=calculation)
            if if_AA:
                cutoff4,power4=Calculate_cutoff(reduced_file_pos,reduced_file_neg,path_AA,cutoff,if_AA_baseline=True,calculation=calculation)
                cutoff_adam_list.append(cutoff4)
                power_adam_list.append(power4)
            cutoff5,power5=Calculate_cutoff(current_group_pos_list[idx],current_group_neg_list[idx],path_model2,cutoff,calculation=calculation)
            #print(">> beta(1,1)")
            if path_beta1 is not None:
                cutoff6,power6=Calculate_cutoff(reduced_file_pos,reduced_file_neg,path_beta1,cutoff,if_beta=True,calculation=calculation)
                #print(">> beta(10,10)")
                cutoff7,power7=Calculate_cutoff(reduced_file_pos,reduced_file_neg,path_beta2,cutoff,if_beta=True,calculation=calculation)
                #print(">> beta(20,20)")
                cutoff8,power8=Calculate_cutoff(reduced_file_pos,reduced_file_neg,path_beta3,cutoff,if_beta=True,calculation=calculation)
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


