#!/usr/bin/env python

import os
import math
import glob
import pickle
import fnmatch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc

prefix="/Users/scarlett/Desktop/HARDAC/"

def Plot_theta_group_distri_lambda(prefix,workdir,lambdaN,datatype,theta,gene=None,hets=None,depth=None,sigma=None):
    source = prefix+"scarlett/software/cmdstan/examples/"
    var=[gene,hets,depth,sigma]
    if var.count(None) !=1:
        raise Exception('One variables have to be set to None. The number of None from input was {}'.format(var.count(None)))
    path_model = source + "BEASTIE3" + "/" + workdir+"_lambda/output_pkl/lambda_"+str(lambdaN)+"/"+str(datatype)+"/"
    path_base = source + "SPP2-odd" + "/" + workdir+"_lambda/output_pkl/lambda_"+str(lambdaN)+"/"+str(datatype)+"/"
    var_map_np = np.array(['g','h','d','s'])
    var_fullname_map_np = np.array(['gene','hets','depth','sigma'])
    full_var_map_np = np.array([gene, hets, depth, sigma])
    valid_var_np = var_map_np[np.array(var) != None]
    variable_var_np = var_fullname_map_np[np.array(var) == None]
    fixed_var_np = var_fullname_map_np[np.array(var) != None]
    valid_full_var_np = full_var_map_np[np.array(var) != None]

    h=hets
    s=sigma
    g=gene
    d=depth
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
    pos_pd = file_dict_pd[(file_dict_pd[valid_var_np[0]] == valid_full_var_np[0])&(file_dict_pd[valid_var_np[1]] == valid_full_var_np[1])&(file_dict_pd[valid_var_np[2]] == valid_full_var_np[2])&(file_dict_pd['t'] == theta)].sort_values(['d','h','g','s'])

    current_group_pos_list = pos_pd.index

    labels= " "
    var_dict = {"gene":g, "hets": h, "depth": d, "sigma": s}
    for each in variable_var_np:
        if each != var_fullname_map_np[np.array(var) == None][0]:
            labels = each  
    d_group = pos_pd[var_map_np[np.array(var) == None][0]].unique()
 
    theta_dist=[]
    x_var = []
    
    for idx in range(len(current_group_pos_list)):
        model_rmse = get_theta_group(theta, current_group_pos_list[idx], path_model, datatype=datatype,d=depth,h=hets,prob=None, base=None, hue = "BEASTIE lambda")
        theta_dist.extend(model_rmse)
        base_rmse = get_theta_group(theta, current_group_pos_list[idx], path_base, datatype=datatype,d=depth,h=hets,prob=None, base=None, hue = "SPP2-odd lambda")
        theta_dist.extend(base_rmse)
        #x_var.append(file_dict[current_group_pos_list[idx]][var_map_np[np.array(var) == None][0]])
    theta_dist_pd = pd.DataFrame(theta_dist)

    error=workdir.rsplit("_",2)[1]
    if error == "simulation":
        error_term="No error"
    else:
        error_term=error+" %error"
    ####### Plot
    plt.figure(figsize=(12,9))
    color_map = {"BEASTIE lambda":"lightskyblue", "SPP2-odd lambda":"lightsalmon"}
    if h is None:
        sns.violinplot(x="h", y="theta", hue="hue", data=theta_dist_pd, palette=color_map, split=True)
    else:
        sns.violinplot(x="d", y="theta", hue="hue", data=theta_dist_pd, palette=color_map, split=True)        
    plt.axhline(y= theta, color='r', linestyle='-')
    plt.ylabel("probability",fontsize=20)
    plt.legend(title = "Category",fontsize=14)
    plt.title(" Distribution of probability: True " + r"$\theta$ = "+ theta.__str__()+" @"+error_term+" "+r"$\lambda$ = "+lambdaN.__str__())



def Plot_theta_group_distri(prefix,workdir,datatype,theta,gene=None,hets=None,depth=None,sigma=None):
    source = prefix+"scarlett/software/cmdstan/examples/"
    var=[gene,hets,depth,sigma]
    if var.count(None) !=1:
        raise Exception('One variables have to be set to None. The number of None from input was {}'.format(var.count(None)))
    path_model = source + "BEASTIE3" + "/" + workdir+"/output_pkl/"+str(datatype)+"/"
    path_base = source + "SPP2-odd" + "/" + workdir+"/output_pkl/"+str(datatype)+"/"
    var_map_np = np.array(['g','h','d','s'])
    var_fullname_map_np = np.array(['gene','hets','depth','sigma'])
    full_var_map_np = np.array([gene, hets, depth, sigma])
    valid_var_np = var_map_np[np.array(var) != None]
    variable_var_np = var_fullname_map_np[np.array(var) == None]
    fixed_var_np = var_fullname_map_np[np.array(var) != None]
    valid_full_var_np = full_var_map_np[np.array(var) != None]

    h=hets
    s=sigma
    g=gene
    d=depth
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
    pos_pd = file_dict_pd[(file_dict_pd[valid_var_np[0]] == valid_full_var_np[0])&(file_dict_pd[valid_var_np[1]] == valid_full_var_np[1])&(file_dict_pd[valid_var_np[2]] == valid_full_var_np[2])&(file_dict_pd['t'] == theta)].sort_values(['d','h','g','s'])

    current_group_pos_list = pos_pd.index

    labels= " "
    var_dict = {"gene":g, "hets": h, "depth": d, "sigma": s}
    for each in variable_var_np:
        if each != var_fullname_map_np[np.array(var) == None][0]:
            labels = each  
    d_group = pos_pd[var_map_np[np.array(var) == None][0]].unique()
 
    theta_dist=[]
    x_var = []
    
    for idx in range(len(current_group_pos_list)):
        model_rmse = get_theta_group(theta, current_group_pos_list[idx], path_model, datatype=datatype,d=depth,h=hets,prob=None, base=None, hue = "BEASTIE")
        theta_dist.extend(model_rmse)
        base_rmse = get_theta_group(theta, current_group_pos_list[idx], path_base, datatype=datatype,d=depth,h=hets,prob=None, base=None, hue = "SPP2-odd")
        theta_dist.extend(base_rmse)
        #x_var.append(file_dict[current_group_pos_list[idx]][var_map_np[np.array(var) == None][0]])
    theta_dist_pd = pd.DataFrame(theta_dist)

    error=workdir.rsplit("_",2)[1]
    if error == "simulation":
        error_term="No error"
    else:
        error_term=error+" %error"
    ####### Plot
    plt.figure(figsize=(12,9))
    color_map = {"BEASTIE":"lightskyblue", "SPP2-odd":"lightsalmon"}
    if h is None:
        sns.violinplot(x="h", y="theta", hue="hue", data=theta_dist_pd, palette=color_map, split=True)
    else:
        sns.violinplot(x="d", y="theta", hue="hue", data=theta_dist_pd, palette=color_map, split=True)        
    plt.axhline(y= (math.log2(theta+1e-18) if datatype == "model_prob2" else theta), color='r', linestyle='-')
    plt.ylabel(r'$\log_2 \theta$' if datatype == "model_prob2" else r'$\theta$',fontsize=20)
    plt.legend(title = "Category",fontsize=14)
    plt.title(" Distribution of "+datatype+": True "+ r"$\theta$ = "+ theta.__str__()+" @"+error_term)

def get_theta_group(true_theta, file_name,path, datatype,d=None,h=None,prob=None, base=None, hue = None):
    theta_hat = pickle.load(open(path+file_name,'rb'))
    ret_list=[]
    if d is None and h is not None:
        d = int(file_name.split("_")[2].split("-")[1])      
        for each in theta_hat:
            ret_list.append({"d":d, "hue": hue, "theta": math.log2(each+1e-18) if datatype == "model_prob2" else each})
    elif h is None and d is not None:
        h = int(file_name.split("_")[1].split("-")[1])
        for each in theta_hat:
            ret_list.append({"h":h, "hue": hue, "theta": math.log2(each+1e-18) if datatype == "model_prob2" else each})
    return (ret_list) 

def get_theta_group_binomial(true_theta, file_name,path,d=None,h=None,prob=None, base=None, hue = None):
    #file_name = file_name.rsplit("_",1)[0]+".pickle"
    theta_hat = pickle.load(open(path+file_name,'rb'))
    ret_list=[]
    if d is None and h is not None:
        d = int(file_name.split("_")[2].split("-")[1])      
        for each in theta_hat:
            ret_list.append({"d":d, "hue": hue, "theta": each})
    elif h is None and d is not None:
        h = int(file_name.split("_")[1].split("-")[1])
        for each in theta_hat:
            ret_list.append({"h":h, "hue": hue, "theta": each})
    return (ret_list) 

def Plot_theta_group_distri_binomial(prefix,workdir,theta,if_prob=None,gene=None,hets=None,depth=None):
    source = prefix+"scarlett/software/cmdstan/examples/"
    var=[gene,hets,depth]
    if var.count(None) !=1:
        raise Exception('One variables have to be set to None. The number of None from input was {}'.format(var.count(None)))
    if if_prob is None:
        path_model = source + "binomial" + "/" +"output_pkl/"+workdir+"/pooled_esti/"
        path_base = source + "ADM" + "/" + workdir+"/output_pkl/AA_esti/"
    else:
        path_model = source + "binomial" + "/" +"output_pkl/"+workdir+"/pooled_prob_new/"
        path_base = source + "ADM" + "/" + workdir+"/output_pkl/AA_pval/"      
    var_map_np = np.array(['g','h','d'])
    var_fullname_map_np = np.array(['gene','hets','depth'])
    full_var_map_np = np.array([gene, hets, depth])
    valid_var_np = var_map_np[np.array(var) != None]
    variable_var_np = var_fullname_map_np[np.array(var) == None]
    fixed_var_np = var_fullname_map_np[np.array(var) != None]
    valid_full_var_np = full_var_map_np[np.array(var) != None]

    h=hets
    g=gene
    d=depth
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
    pos_pd = file_dict_pd[(file_dict_pd[valid_var_np[0]] == valid_full_var_np[0])&(file_dict_pd[valid_var_np[1]] == valid_full_var_np[1])&(file_dict_pd['t'] == theta)].sort_values(['d','h','g'])

    current_group_pos_list = pos_pd.index

    labels= " "
    var_dict = {"gene":g, "hets": h, "depth": d}
    for each in variable_var_np:
        if each != var_fullname_map_np[np.array(var) == None][0]:
            labels = each  
    d_group = pos_pd[var_map_np[np.array(var) == None][0]].unique()
 
    theta_dist=[]
    x_var = []
    
    for idx in range(len(current_group_pos_list)):
        model_rmse = get_theta_group_binomial(theta, current_group_pos_list[idx], path_model, d=depth,h=hets,prob=None, base=None, hue = "Binomial")
        theta_dist.extend(model_rmse)
        base_rmse = get_theta_group_binomial(theta, current_group_pos_list[idx], path_base, d=depth,h=hets,prob=None, base=None, hue = "ADM")
        theta_dist.extend(base_rmse)
        #x_var.append(file_dict[current_group_pos_list[idx]][var_map_np[np.array(var) == None][0]])
    theta_dist_pd = pd.DataFrame(theta_dist)

    error=workdir.rsplit("_",2)[1]
    if error == "simulation":
        error_term="No error"
    else:
        error_term=error+" %error"
    ####### Plot
    plt.figure(figsize=(12,9))
    color_map = {"Binomial":"lightskyblue", "ADM":"lightsalmon"}
    if h is None:
        sns.violinplot(x="h", y="theta", hue="hue", data=theta_dist_pd, palette=color_map, split=True)
    else:
        sns.violinplot(x="d", y="theta", hue="hue", data=theta_dist_pd, palette=color_map, split=True)        
    plt.axhline(y= 0, color='r', linestyle='-')
    plt.ylabel("probability" if if_prob is True else r'$\theta$',fontsize=20)
    plt.legend(title = "Category",fontsize=14)
    if if_prob is True:
        plt.title(" Distribution of probability: True " + r"$\theta$ = "+ theta.__str__()+" @"+error_term)
    else:
        plt.title(" Distribution of estimates: True "+ r"$\theta$ = "+ theta.__str__()+" @"+error_term)



def Plot_theta_binomial_distri(prefix,workdir,data1,data2,theta,if_prob=None,gene=None,hets=None,depth=None):
    source = prefix+"scarlett/software/cmdstan/examples/"
    var=[gene,hets,depth]
    if var.count(None) !=1:
        raise Exception('One variables have to be set to None. The number of None from input was {}'.format(var.count(None)))
    path_model = source + "binomial" + "/" +"output_pkl/"+workdir+"/"+data1+"/"
    path_base = source + "binomial" + "/" +"output_pkl/"+workdir+"/"+data2+"/"
    var_map_np = np.array(['g','h','d'])
    var_fullname_map_np = np.array(['gene','hets','depth'])
    full_var_map_np = np.array([gene, hets, depth])
    valid_var_np = var_map_np[np.array(var) != None]
    variable_var_np = var_fullname_map_np[np.array(var) == None]
    fixed_var_np = var_fullname_map_np[np.array(var) != None]
    valid_full_var_np = full_var_map_np[np.array(var) != None]

    h=hets
    g=gene
    d=depth
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
    pos_pd = file_dict_pd[(file_dict_pd[valid_var_np[0]] == valid_full_var_np[0])&(file_dict_pd[valid_var_np[1]] == valid_full_var_np[1])&(file_dict_pd['t'] == theta)].sort_values(['d','h','g'])

    current_group_pos_list = pos_pd.index

    labels= " "
    var_dict = {"gene":g, "hets": h, "depth": d}
    for each in variable_var_np:
        if each != var_fullname_map_np[np.array(var) == None][0]:
            labels = each  
    d_group = pos_pd[var_map_np[np.array(var) == None][0]].unique()
 
    theta_dist=[]
    x_var = []
    
    for idx in range(len(current_group_pos_list)):
        model_rmse = get_theta_group_binomial(theta, current_group_pos_list[idx], path_model, d=depth,h=hets,prob=None, base=None, hue = data1)
        theta_dist.extend(model_rmse)
        base_rmse = get_theta_group_binomial(theta, current_group_pos_list[idx], path_base, d=depth,h=hets,prob=None, base=None, hue = data2)
        theta_dist.extend(base_rmse)
        #x_var.append(file_dict[current_group_pos_list[idx]][var_map_np[np.array(var) == None][0]])
    theta_dist_pd = pd.DataFrame(theta_dist)

    error=workdir.rsplit("_",2)[1]
    if error == "simulation":
        error_term="No error"
    else:
        error_term=error+" %error"
    ####### Plot
    plt.figure(figsize=(12,9))
    color_map = {data1:"lightskyblue", data2:"lightsalmon"}
    if h is None:
        sns.violinplot(x="h", y="theta", hue="hue",data=theta_dist_pd,palette=color_map, split=True)
    else:
        sns.violinplot(x="d", y="theta", hue="hue",data=theta_dist_pd, palette=color_map, split=True)        
    plt.axhline(y= 0, color='r', linestyle='-')
    plt.ylabel("probability" if if_prob is True else r'$\theta$',fontsize=20)
    plt.legend(title = "Category",fontsize=14)
   
    plt.title(" Distribution of binomial"+data1+" & "+data2+": True " + r"$\theta$ = "+ theta.__str__()+" @"+error_term)


def Plot_theta_single(prefix,model,workdir,datatype,g,h,d,theta,s=0.5):
    source = prefix+"scarlett/software/cmdstan/examples/"
    if "binomial" in model:
        file_name="g-"+str(g)+"_h-"+str(h)+"_d-"+str(d)+"_t-"+str(theta)+".pickle"
        path = source + model + "/" +"output_pkl/"+workdir+"/"+str(datatype)+"/"
    else:
        file_name="g-"+str(g)+"_h-"+str(h)+"_d-"+str(d)+"_t-"+str(theta)+"_s-"+str(s)+".pickle"
        path = source + model + "/" + workdir+"/output_pkl/"+str(datatype)+"/"
        
    theta_hat = pickle.load(open(path+file_name,'rb'))
    sns.distplot(theta_hat, hist=True)
    plt.ylabel(datatype,fontsize=20)
    plt.title(" Distribution of binomial: "+datatype+"; True " + r"$\theta$ = "+ theta.__str__()+" @"+workdir)
    plt.show()

def sample_inverseCDF(data,n,plot=False,title=None):
    # An empirical distribution function provides a way to model and sample cumulative probabilities for a data sample that does not fit a standard probability distribution.
    if title!=None:
        print("The distribution of %s:"%title)
    print("\tMax:",max(data),", Min:",min(data),", Median:",np.median(data),", Mean:",round(np.mean(data),2))

    x_cloned = EmpiricalDistribution(data).rvs(n)
    if plot != False:
        sns.set(font_scale=2)  
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(ncols=1, figsize=(8,6))
        sns.distplot(x_cloned, hist = False, kde = True,kde_kws = {'linewidth': 3},label = 'inverse cdf')
        sns.distplot(data, hist = False, kde = True,kde_kws = {'linewidth': 3},label = 'real data')
        plt.legend(loc="upper right",fontsize=17)
        if title != None:
            plt.title(title,fontsize=26) 
    if n == 1:
        x_cloned = np.ndarray.tolist(x_cloned)[0]
    #return x_cloned