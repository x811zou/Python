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

def autolabel(rects,labels):
    """
    Attach a text label above each bar displaying its height
    """
    i=0
    for rect,label in zip(rects,labels):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 1.001*height,round(label,2),ha='center', va='bottom',fontsize=8)
        i+=1

def get_StanOutput_summary(workdir):
    dir = "/data/reddylab/scarlett/1000G/software/cmdstan/examples/" + workdir
    all_file = sorted(os.listdir(dir))
    file_dict = {}
    for pkl in all_file:
        if ".pickle" in pkl: 
            name=pkl.rsplit(".pickle")[0].rsplit("_")
            file_dict[pkl] = {}
            for each_value in name:
                file_dict[pkl][each_value.split("-")[0]] = float(each_value.split("-")[1])
            #print(name)
        else:continue
    file_dict_pd = pd.DataFrame(file_dict).transpose()
    for each in file_dict_pd.columns:
        print (each+'-unique',sorted(file_dict_pd[each].unique()))


def get_ROC_AUC(path, file_pos, file_neg, if_prob=None, if_baseline=None):
    if if_prob == True and if_baseline == True:
        #print ("TrueTrue")
        file_pos = '_'.join(file_pos.rsplit(".pickle")[0].rsplit("_")[0:4])+'.pickle'
        file_neg = '_'.join(file_neg.rsplit(".pickle")[0].rsplit("_")[0:4])+'.pickle'
        prob1 = pickle.load(open(path+file_pos,"rb"))
        prob2 = pickle.load(open(path+file_neg,"rb"))
        fpr, tpr, _ = roc_curve([0 for i in range(len(prob1))] + [1 for i in range(len(prob2))], prob1 + prob2)
    elif if_prob == True and if_baseline != True: 
        #print ("TrueFalse")
        prob1 = pickle.load(open(path+file_pos,"rb"))
        prob2 = pickle.load(open(path+file_neg,"rb"))
        fpr, tpr, _ = roc_curve([1 for i in range(len(prob1))] + [0 for i in range(len(prob2))], prob1 + prob2)
    else: #if_prob != True and if_baseline != True: 
        if if_baseline == True:
            #print ("here")
            file_pos = '_'.join(file_pos.rsplit(".pickle")[0].rsplit("_")[0:4])+'.pickle'
            file_neg = '_'.join(file_neg.rsplit(".pickle")[0].rsplit("_")[0:4])+'.pickle'
        prob1 = pickle.load(open(path+file_pos,"rb"))
        prob2 = pickle.load(open(path+file_neg,"rb"))
        prob1_t = [abs(x - 1) for x in prob1]
        prob2_t = [abs(x - 1) for x in prob2]
        fpr, tpr, _ = roc_curve([1 for i in range(len(prob1_t))] + [0 for i in range(len(prob2_t))], prob1_t + prob2_t)
    return(fpr, tpr, round(auc(fpr,tpr),3))

def CompareAUC_model_baseline(label,error,theta_pos,theta_neg,if_prob=None,gene=None,hets=None,depth=None,sigma=None):
    source = "/data/reddylab/scarlett/1000G/software/cmdstan/examples/"
    #workdir =  "ase/new_simulation_05_error/"
    #workdir =  "ase/new_simulation_01_error/"
    #workdir =  "ase/new_simulation/"
    thetas=[theta_pos,theta_neg]
    if thetas.count(None) >0:
        raise Exception('Both thetas could not be None. The number of None from input was {}'.format(thetas.count(None)))    
    var=[gene,hets,depth,sigma]
    if var.count(None) !=1:
        raise Exception('One variables have to be set to None. The number of None from input was {}'.format(var.count(None)))
    if if_prob != True and if_prob !=None:
        raise Exception('if_prob == True - means using probability, if_prob == None - means using theta') 
    if if_prob == True:
        path_model = source + label + "/"+  error +"/output_pkl/model_prob/"
        path_base = source + label + "/"+  error  + "/output_pkl/baseline_prob/"
    else:
        path_model = source + label + "/"+  error  + "/output_pkl/model_med/"
        path_base = source + label + "/"+  error  + "/output_pkl/baseline_theta/"     

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
    pos_pd = file_dict_pd[(file_dict_pd[valid_var_np[0]] == valid_full_var_np[0])&(file_dict_pd[valid_var_np[1]] == valid_full_var_np[1])&(file_dict_pd[valid_var_np[2]] == valid_full_var_np[2])&(file_dict_pd['t'] == theta_pos)].sort_values(['d','h','g','s'])
    neg_pd = file_dict_pd[(file_dict_pd[valid_var_np[0]] == valid_full_var_np[0])&(file_dict_pd[valid_var_np[1]] == valid_full_var_np[1])&(file_dict_pd[valid_var_np[2]] == valid_full_var_np[2])&(file_dict_pd['t'] == theta_neg)].sort_values(['d','h','g','s'])
    
    current_group_pos_list = pos_pd.index
    current_group_neg_list = neg_pd.index

    labels= " "
    var_dict = {"gene":g, "hets": h, "depth": d, "sigma": s}
    for each in variable_var_np:
        if each != var_fullname_map_np[np.array(var) == None][0]:
            labels = each  
    d_group = pos_pd[var_map_np[np.array(var) == None][0]].unique()

    AUC=[]
    x_var = []
    AUC_base=[]
    x_var_base = []
    
    for idx in range(len(current_group_pos_list)):
        # model
        _,_,auc12 =  get_ROC_AUC(path_model, current_group_pos_list[idx],current_group_neg_list[idx],if_prob=if_prob)
        AUC.append(auc12)
        x_var.append(file_dict[current_group_pos_list[idx]][var_map_np[np.array(var) == None][0]])
        # baseline
        if if_prob == True:
            current_file_pos = current_group_pos_list[idx].rsplit("_",1)[0]+".pickle"
            current_file_neg = current_group_neg_list[idx].rsplit("_",1)[0]+".pickle"
        else: 
            current_file_pos = current_group_pos_list[idx]
            current_file_neg = current_group_neg_list[idx]
        _,_,auc34 =  get_ROC_AUC(path_base, current_file_pos,current_file_neg,if_prob=if_prob,if_baseline=True)
        AUC_base.append(auc34)
        x_var_base.append(file_dict[current_group_pos_list[idx]][var_map_np[np.array(var) == None][0]])

    ####### Plot
    plt.figure(figsize=(12,8))
    barWidth = 0.3
    bars1 = AUC
    bars2 = AUC_base
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    # Make the plot
    if "01" in path_model:
        rects1=plt.bar(r1, bars1, color='teal', width=barWidth, edgecolor='white', label='%s Model with error 0.1'%(label))
        rects2=plt.bar(r2, bars2, color='salmon', width=barWidth, edgecolor='white', label='%s Baseline with error 0.1'%(label)) 
    elif "05" in path_model:      
        rects1=plt.bar(r1, bars1, color='darkblue', width=barWidth, edgecolor='white', label='%s Model with error 0.5'%(label))
        rects2=plt.bar(r2, bars2, color='darksalmon', width=barWidth, edgecolor='white', label='%s Baseline with error 0.5'%(label))
    else:
        rects1=plt.bar(r1, bars1, color='lightskyblue', width=barWidth, edgecolor='white', label='%s Model with no error'%(label))
        rects2=plt.bar(r2, bars2, color='lightsalmon', width=barWidth, edgecolor='white', label='%s Baseline with no error'%(label))        
    # Add xticks on the middle of the group bars
    plt.xlabel(var_fullname_map_np[np.array(var) == None][0], fontweight='bold',fontsize=12)
    plt.xticks([r + barWidth for r in range(len(bars1))], x_var)
    plt.ylabel("AUC")
    # Create legend & Show graphic
    plt.axhline(y=1, linewidth=1, linestyle='--',color='r')
    autolabel(rects1,AUC)
    autolabel(rects2,AUC_base)
    plt.ylim(0.5,1.2)
    plt.legend(loc=2,fontsize=14)
    plt.show()


def CompareAUC_3model(label,theta_pos,theta_neg,if_prob=None,gene=None,hets=None,depth=None,sigma=None):
    source = "/data/reddylab/scarlett/1000G/software/cmdstan/examples/"
    thetas=[theta_pos,theta_neg]
    if thetas.count(None) >0:
        raise Exception('Both thetas could not be None. The number of None from input was {}'.format(thetas.count(None)))    
    var=[gene,hets,depth,sigma]
    if var.count(None) !=1:
        raise Exception('One variables have to be set to None. The number of None from input was {}'.format(var.count(None)))
    if if_prob != True and if_prob !=None:
        raise Exception('if_prob == True - means using probability, if_prob == None - means using theta') 
    if if_prob == True:
        model_path = source + label + "/"+ "new_simulation/output_pkl/model_prob/"
        model_path_e1 = source + label + "/"+ "new_simulation_01_error/output_pkl/model_prob/"
        model_path_e5 = source + label + "/"+ "new_simulation_05_error/output_pkl/model_prob/"
    else:
        model_path = source + label + "/"+ "new_simulation/output_pkl/model_med/"
        model_path_e1 = source + label + "/"+ "new_simulation_01_error/output_pkl/model_med/"
        model_path_e5 = source + label + "/"+ "new_simulation_05_error/output_pkl/model_med/"
        
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
    all_file = sorted(os.listdir(model_path))
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
    pos_pd = file_dict_pd[(file_dict_pd[valid_var_np[0]] == valid_full_var_np[0])&(file_dict_pd[valid_var_np[1]] == valid_full_var_np[1])&(file_dict_pd[valid_var_np[2]] == valid_full_var_np[2])&(file_dict_pd['t'] == theta_pos)].sort_values(['d','h','g','s'])
    neg_pd = file_dict_pd[(file_dict_pd[valid_var_np[0]] == valid_full_var_np[0])&(file_dict_pd[valid_var_np[1]] == valid_full_var_np[1])&(file_dict_pd[valid_var_np[2]] == valid_full_var_np[2])&(file_dict_pd['t'] == theta_neg)].sort_values(['d','h','g','s'])
    
    current_group_pos_list = pos_pd.index
    current_group_neg_list = neg_pd.index

    labels= " "
    var_dict = {"gene":g, "hets": h, "depth": d, "sigma": s}
    for each in variable_var_np:
        if each != var_fullname_map_np[np.array(var) == None][0]:
            labels = each  
    d_group = pos_pd[var_map_np[np.array(var) == None][0]].unique()

    AUC=[]
    x_var = []
    AUC_e1=[]
    AUC_e5=[]
    x_var_e1 = []
    x_var_e5 = []
    
    for idx in range(len(current_group_pos_list)):
        current_file_pos = current_group_pos_list[idx]
        current_file_neg = current_group_neg_list[idx]
        # model without error, baseline without error
        _,_,auc12 =  get_ROC_AUC(model_path, current_group_pos_list[idx],current_group_neg_list[idx],if_prob=if_prob)
        AUC.append(auc12)
        x_var.append(file_dict[current_group_pos_list[idx]][var_map_np[np.array(var) == None][0]])
        # model with 10% error, baseline with 10% error
        _,_,auc34 =  get_ROC_AUC(model_path_e1, current_file_pos,current_file_neg,if_prob=if_prob)
        AUC_e1.append(auc34)
        #x_var_e1.append(file_dict[current_group_pos_list[idx]][var_map_np[np.array(var) == None][0]])
        # model with 50% error, baseline with 50% error
        _,_,auc56 =  get_ROC_AUC(model_path_e5, current_file_pos,current_file_neg,if_prob=if_prob)
        AUC_e5.append(auc56)
        #x_var_e5.append(file_dict[current_group_pos_list[idx]][var_map_np[np.array(var) == None][0]])

    ####### Plot
    plt.figure(figsize=(12,8))
    barWidth = 0.3
    bars1 = AUC
    bars2 = AUC_e1
    bars3 = AUC_e5
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    # Make the plot
    rects1=plt.bar(r1, bars1, color='lightskyblue', width=barWidth, edgecolor='white', label='No error')
    rects2=plt.bar(r2, bars2, color='teal', width=barWidth, edgecolor='white', label='10% error')
    rects3=plt.bar(r3, bars3, color='darkblue', width=barWidth, edgecolor='white', label='50% error')
    # Add xticks on the middle of the group bars
    plt.xlabel(var_fullname_map_np[np.array(var) == None][0], fontweight='bold',fontsize=12)
    plt.xticks([r + barWidth for r in range(len(bars1))], x_var)
    plt.ylabel("AUC")
    # Create legend & Show graphic
    autolabel(rects1,AUC)
    autolabel(rects2,AUC_e1)
    autolabel(rects3,AUC_e5)
    plt.ylim(0.5,1.2)
    plt.axhline(y=1, linewidth=1, linestyle='--',color='r')
    plt.legend(loc=2,fontsize=20)
    plt.title("The distribution of Model AUC",fontsize=25)
    plt.show()

def CompareAUC_3baseline(label,theta_pos,theta_neg, data=None,gene=None,hets=None,depth=None):
    source = "/data/reddylab/scarlett/1000G/software/cmdstan/examples/"
    thetas = [theta_pos, theta_neg]
    if thetas.count(None) >0:
        raise Exception('Both thetas could not be None. The number of None from input was {}'.format(thetas.count(None)))    
    var=np.array([gene,hets,depth])
    name = np.array(['gene','hets','depth'])

    model_path = source + label + "/"+ "new_simulation/output_pkl/baseline_" + data +"/"
    model_path_e1 = source + label + "/"+ "new_simulation_01_error/output_pkl/baseline_"+ data + "/"
    model_path_e5 = source + label + "/"+ "new_simulation_05_error/output_pkl/baseline_"+ data + "/"
    if_prob = None
    if "prob" in data:
        if_prob = True
    file_pos = "g-%s_h-%s_d-%s_t-%s.pickle" %(gene, hets,depth, theta_pos)
    file_neg = "g-%s_h-%s_d-%s_t-%s.pickle" %(gene, hets,depth, theta_neg)
    
    if "None" in file_pos:
        #print ("correct")
        listing = glob.glob(model_path+file_pos.replace("None","*"))
        #print (model_path+file_pos.replace("None","*"))
        AUC = []
        AUC_e1 = []
        AUC_e5 = []
        x_var = []
        #print (listing)
        for each_file in listing:
            cur_pos = each_file.rsplit("/",1)[1]
            cur_neg = cur_pos.replace(str(theta_pos), str(theta_neg))
            _,_,auc12 =  get_ROC_AUC(model_path, cur_pos,cur_neg,if_prob=if_prob,if_baseline=True)
            AUC.append(auc12)
            _,_,auc34 =  get_ROC_AUC(model_path_e1, cur_pos,cur_neg,if_prob=if_prob,if_baseline=True)
            AUC_e1.append(auc34)
            _,_,auc56 =  get_ROC_AUC(model_path_e5, cur_pos,cur_neg,if_prob=if_prob,if_baseline=True)
            AUC_e5.append(auc56)
            x_var.append(float(np.array(cur_pos.split("_")[:3])[var==None][0].split("-")[1]))
        plt.figure(figsize=(12,8))
        barWidth = 0.3
        bars1 = AUC
        bars2 = AUC_e1
        bars3 = AUC_e5
        r1 = np.arange(len(bars1))
        r2 = [x + barWidth for x in r1]
        r3 = [x + barWidth for x in r2]
        # Make the plot
        rects1=plt.bar(r1, bars1, color='lightskyblue', width=barWidth, edgecolor='white', label='No error')
        rects2=plt.bar(r2, bars2, color='teal', width=barWidth, edgecolor='white', label='10% error')
        rects3=plt.bar(r3, bars3, color='darkblue', width=barWidth, edgecolor='white', label='50% error')
        # Add xticks on the middle of the group bars
        plt.xlabel(name[var==None][0])
        plt.xticks([r + barWidth for r in range(len(bars1))], sorted(x_var))
        plt.ylabel("AUC")
        # Create legend & Show graphic
        autolabel(rects1,AUC)
        autolabel(rects2,AUC_e1)
        autolabel(rects3,AUC_e5)
        plt.ylim(0.5,1.2)
        plt.axhline(y=1, linewidth=1, linestyle='--',color='r')
        plt.legend(loc=2,fontsize=14)
        plt.title("The distribution of Baseline AUC",fontsize=14)

        plt.show()        
    else:
        AUC = []
        AUC_e1 = []
        AUC_e5 = []
        _,_,auc12 =  get_ROC_AUC(model_path, cur_pos,cur_neg,if_prob=if_prob,if_baseline=True)
        AUC.append(auc12)
        _,_,auc34 =  get_ROC_AUC(model_path_e1, cur_pos,cur_neg,if_prob=if_prob,if_baseline=True)
        AUC_e1.append(auc34)
        _,_,auc56 =  get_ROC_AUC(model_path_e5, cur_pos,cur_neg,if_prob=if_prob,if_baseline=True)
        AUC_e5.append(auc56)
        plt.figure(figsize=(15,8))
        barWidth = 0.35
        bars1 = AUC
        bars2 = AUC_e1
        bars3 = AUC_e5
        r1 = np.arange(len(bars1))
        r2 = [x + barWidth for x in r1]
        r3 = [x + barWidth for x in r2]
        # Make the plot
        rects1=plt.bar(r1, bars1, color='lightskyblue', width=barWidth, edgecolor='white', label='No error')
        rects2=plt.bar(r2, bars2, color='teal', width=barWidth, edgecolor='white', label='10% error')
        rects3=plt.bar(r3, bars3, color='darkblue', width=barWidth, edgecolor='white', label='50% error')
        # Add xticks on the middle of the group bars
        #plt.xlabel(var_fullname_map_np[np.array(var) == None][0], fontweight='bold',fontsize=12)
        #plt.xticks([r + barWidth for r in range(len(bars1))], x_var)
        plt.ylabel("AUC")
        # Create legend & Show graphic
        autolabel(rects1,AUC)
        autolabel(rects2,AUC_e1)
        autolabel(rects3,AUC_e5)
        plt.ylim(0.5,1.2)
        plt.axhline(y=1, linewidth=1, linestyle='--',color='r')
        plt.legend(loc=2,fontsize=14)
        plt.title("The distribution of Baseline AUC",fontsize=14)
        plt.show()

def Plot_ROC_comparison_fix2(label,workdir,theta_pos,theta_neg,if_prob=None,prob_t=None,gene=None,hets=None,depth=None,sigma=None, num = None):
    source = "/data/reddylab/scarlett/1000G/software/cmdstan/examples/"
    # judge whether  2 of the 4 are None, and 2 of the 4 are not None
    thetas=[theta_pos,theta_neg]
    if thetas.count(None) >0:
        raise Exception('Both thetas could not be None. The number of None from input was {}'.format(thetas.count(None)))    
    var=[gene,hets,depth,sigma]
    if var.count(None)!=2:
        raise Exception('Two variables have to be set to None. The number of None from input was {}'.format(var.count(None)))
    # data directory is known 
    if if_prob == None:
        path=source + label + "/" + workdir+"/output_pkl/model_med/"
    else:
        if prob_t == None:
            path=source + label + "/" + workdir+"/output_pkl/model_prob/"
        #else:
        #    path=source + label + "/" + workdir+"/output_pkl/model_prob2/"
    
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
    pos_pd = file_dict_pd[(file_dict_pd[valid_var_np[0]] == valid_full_var_np[0])&(file_dict_pd[valid_var_np[1]] == valid_full_var_np[1])&(file_dict_pd['t'] == theta_pos)].sort_values(['d','h','g','s'])
    neg_pd = file_dict_pd[(file_dict_pd[valid_var_np[0]] == valid_full_var_np[0])&(file_dict_pd[valid_var_np[1]] == valid_full_var_np[1])&(file_dict_pd['t'] == theta_neg)].sort_values(['d','h','g','s'])
    
    d_group = pos_pd[var_map_np[np.array(var) == None][0]].unique()
    #print (d_group)
    if num == None:
        num = 3
    else:
        num = int(num)
    row = math.ceil(float(len(d_group))/num)
    fig, axs = plt.subplots(row, num, figsize = (20,5*row))
    if (row * num > len(d_group)):
        for i in range(row * num - len(d_group)):
            axs.flat[-1-i].set_axis_off()
            
    xlabels = "Fixed parameters "
    for i, each in enumerate(d_group):
        current_group_pos_list = pos_pd[pos_pd[var_map_np[np.array(var) == None][0]] == each].index
        current_group_neg_list = neg_pd[neg_pd[var_map_np[np.array(var) == None][0]] == each].index
        xlabels = "Fixed parameters "
        for idx in range(len(current_group_pos_list)):
            fpr, tpr, roc_auc = get_ROC_AUC(path, current_group_pos_list[idx], current_group_neg_list[idx],if_prob=if_prob)
            g=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[0].rsplit("-")[1]
            h=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[1].rsplit("-")[1]
            d=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[2].rsplit("-")[1]
            s=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[4].rsplit("-")[1]
            #labels=" sigma:"+str(s)
            labels= " AUC: "+str(roc_auc)+" "
            var_dict = {"gene":g, "hets": h, "depth": d, "sigma": s}
            for each in variable_var_np:
                if each != var_fullname_map_np[np.array(var) == None][0]:
                    labels += each+":"+var_dict[each]+' '
            xlabels = "Fixed parameters "
            for each in fixed_var_np:
                xlabels += each+":"+var_dict[each]+' '
            #axs[i].suptitle(xlabels)
            axs.flat[i].set_xlabel("FPR")
            axs.flat[i].set_ylabel("TPR")
            axs.flat[i].set_title(var_fullname_map_np[np.array(var) == None][0]+":" + var_dict[var_fullname_map_np[np.array(var) == None][0]])
            axs.flat[i].plot(fpr, tpr, label =  labels)
            axs.flat[i].legend()
    plt.suptitle(xlabels,fontsize=20)
    plt.show()



def Plot_theta_group_distri(label,workdir,theta,gene=None,hets=None,depth=None,sigma=None, if_log = None):
    source = "/data/reddylab/scarlett/1000G/software/cmdstan/examples/"
    var=[gene,hets,depth,sigma]
    if var.count(None) !=1:
        raise Exception('One variables have to be set to None. The number of None from input was {}'.format(var.count(None)))
    path_model = source + label + "/" + workdir+"/output_pkl/model_med/"
    path_base = source + label + "/" +  workdir+"/output_pkl/baseline_theta_pool/"
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
        model_rmse = get_theta_group(theta, current_group_pos_list[idx], path_model, prob=None, base=None, hue = "model" ,if_log= if_log)
        theta_dist.extend(model_rmse)
        base_rmse = get_theta_group(theta, current_group_pos_list[idx], path_base, prob=None, base=None, hue = "baseline",if_log= if_log)
        theta_dist.extend(base_rmse)
        #x_var.append(file_dict[current_group_pos_list[idx]][var_map_np[np.array(var) == None][0]])
    theta_dist_pd = pd.DataFrame(theta_dist)
    ####### Plot
    plt.figure(figsize=(12,9))
    color_map = {"model":"lightskyblue", "baseline":"lightsalmon"}
    sns.violinplot(x="h", y="theta", hue="hue", data=theta_dist_pd, palette=color_map, split=True)
    plt.axhline(y= (math.log2(theta+1e-18) if if_log == True else theta), color='r', linestyle='-')
    plt.ylabel(r'$\log_2 \theta$' if if_log == True else r'$\theta$',fontsize=20)
    plt.legend(title = "Category",fontsize=14)
    plt.title(r"True $\theta$ = "+ theta.__str__())


def get_theta_group(true_theta, file_name, path, prob=None, base=None, hue = None, if_log = None):
    if "baseline" in path:
        file_name = file_name.rsplit("_",1)[0]+".pickle"
    theta_hat = pickle.load(open(path+file_name,'rb'))
    h = int(file_name.split("_")[1].split("-")[1])
    ret_list = []
    for each in theta_hat:
        ret_list.append({"h":h, "hue": hue, "theta": math.log2(each+1e-18) if if_log == True else each})
    return (ret_list) 



def Plot_prob_group_distri(workdir,theta,gene=None,hets=None,depth=None,sigma=None):
    source = "/data/reddylab/scarlett/1000G/software/cmdstan/examples/ase/"
    var=[gene,hets,depth,sigma]
    if var.count(None) !=1:
        raise Exception('One variables have to be set to None. The number of None from input was {}'.format(var.count(None)))
    path_model = source + workdir+"/output_pkl/model_prob/"
    path_base = source + workdir+"/output_pkl/baseline_prob/"    
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
        model_rmse = get_theta_group(theta, current_group_pos_list[idx], path_model, prob=None, base=None, hue = "model")
        theta_dist.extend(model_rmse)
        base_rmse = get_theta_group(theta, current_group_pos_list[idx], path_base, prob=None, base=None, hue = "baseline")
        theta_dist.extend(base_rmse)
    theta_dist_pd = pd.DataFrame(theta_dist)
    ####### Plot
    plt.figure(figsize=(12,9))
    color_map = {"model":"lightskyblue", "baseline":"lightsalmon"}
    sns.violinplot(x="h", y="theta", hue="hue", data=theta_dist_pd, palette=color_map, split=True)
    plt.ylabel("Probability",fontsize=20)
    plt.legend(title = "Category",fontsize=14,loc=1)
    plt.title(r"True $\theta$ = "+ theta.__str__())


def Plot_prob_group_pos_neg_hist(workdir,theta_pos,theta_neg,kind,gene=None,hets=None,depth=None,sigma=None, if_log = None):
    source = "/data/reddylab/scarlett/1000G/software/cmdstan/examples/ase/"
    var=[gene,hets,depth,sigma]
    path_model = source+ workdir+"/output_pkl/model_prob/"
    path_base = source + workdir+"/output_pkl/baseline_prob/"
    used_path = ""
    used_path = (path_model if kind == 'model' else  path_base)
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
    pos_pd = file_dict_pd[(file_dict_pd[valid_var_np[0]] == valid_full_var_np[0])&(file_dict_pd[valid_var_np[1]] == valid_full_var_np[1])&(file_dict_pd[valid_var_np[2]] == valid_full_var_np[2])&(file_dict_pd['t'] == theta_neg)].sort_values(['d','h','g','s'])
    pos_pd = pos_pd[(pos_pd['h'] == hets) &(pos_pd['s'] == sigma)]
    current_group_pos_list = pos_pd.index
 
    theta_dist=[]
    model = get_theta_group_prob(theta_pos, current_group_pos_list[0], used_path, prob=None, base=None, hue = theta_pos,if_log= if_log)
    theta_dist.extend(model)
    base = get_theta_group_prob(theta_neg, current_group_pos_list[0], used_path, prob=None, base=None, hue = theta_neg,if_log= if_log)
    theta_dist.extend(base)
    #x_var.append(file_dict[current_group_pos_list[idx]][var_map_np[np.array(var) == None][0]])
    theta_dist_pd = pd.DataFrame(theta_dist)
    ####### Plot
    plt.figure(figsize=(12,9))
    #sns.violinplot(x="h", y="theta", hue="hue", data=theta_dist_pd, palette="muted", split=True)
    theta_dist_pd[theta_dist_pd['hue'] == theta_pos]['theta'].hist(bins = 20, label=str(theta_pos), alpha = 0.35)
    theta_dist_pd[theta_dist_pd['hue'] == theta_neg]['theta'].hist(bins = 20,label = str(theta_neg), alpha = 0.35)
    #print (theta_dist_pd['theta'].describe())
    #plt.axhline(y= (math.log2(theta+1e-18) if if_log == True else theta), color='r', linestyle='-')
    #plt.ylabel(r'$\log_2 \theta$' if if_log == True else r'Probability',fontsize=20)
    plt.legend(title = "theta",fontsize=14)
    plt.xlabel(r"The probability of positive")


def get_theta_group_prob(true_theta, file_name, path, prob=None, base=None, hue = None, if_log = None):
    if "baseline_prob" in path:
        file_name = file_name.rsplit("_",1)[0]+".pickle"
    if true_theta != 1:
        file_name = file_name.replace("t-1","t-"+str(true_theta))
    #print (file_name)
    theta_hat = pickle.load(open(path+file_name,'rb'))
    h = int(file_name.split("_")[1].split("-")[1])
    ret_list = []
    for each in theta_hat:
        ret_list.append({"h":h, "hue": hue, "theta": math.log2(each+1e-18) if if_log == True else each})
    return (ret_list)



def Plot_prob_group_pos_neg_distri(workdir,theta,theta_neg,kind,gene=None,hets=None,depth=None,sigma=None, if_log = None):
    source = "/data/reddylab/scarlett/1000G/software/cmdstan/examples/ase/"
    var=[gene,hets,depth,sigma]
    if var.count(None) !=1:
        raise Exception('One variables have to be set to None. The number of None from input was {}'.format(var.count(None)))
    path_model = source + workdir + "/output_pkl/model_prob/"
    path_base = source + workdir + "/output_pkl/baseline_prob/"
    used_path = ""
    used_path = (path_model if kind == 'model' else  path_base)
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
        model_rmse = get_theta_group_prob(theta, current_group_pos_list[idx], used_path, prob=None, base=None, hue = theta,if_log= if_log)
        theta_dist.extend(model_rmse)
        base_rmse = get_theta_group_prob(theta_neg, current_group_pos_list[idx], used_path, prob=None, base=None, hue = theta_neg,if_log= if_log)
        theta_dist.extend(base_rmse)
        #x_var.append(file_dict[current_group_pos_list[idx]][var_map_np[np.array(var) == None][0]])
    theta_dist_pd = pd.DataFrame(theta_dist)
    ####### Plot
    plt.figure(figsize=(12,9))
    #color_map = {"model":"lightskyblue", "baseline":"lightsalmon"}
    sns.violinplot(x="h", y="theta", hue="hue", data=theta_dist_pd, palette="muted", split=True)
    #print (theta_dist_pd['theta'].describe())
    #plt.axhline(y= (math.log2(theta+1e-18) if if_log == True else theta), color='r', linestyle='-')
    plt.ylabel(r'$\log_2 \theta$' if if_log == True else r'Probability',fontsize=20)
    plt.legend(title = "theta",fontsize=14)
    plt.title(r"The probability of positive")

  


def Plot_prob_group_distri(workdir,theta,gene=None,hets=None,depth=None,sigma=None):
    source = "/data/reddylab/scarlett/1000G/software/cmdstan/examples/ase/"
    var=[gene,hets,depth,sigma]
    if var.count(None) !=1:
        raise Exception('One variables have to be set to None. The number of None from input was {}'.format(var.count(None)))
    path_model = source + workdir+"/output_pkl/model_prob/"
    path_base = source + workdir+"/output_pkl/baseline_prob/"    
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
        model_rmse = get_theta_group(theta, current_group_pos_list[idx], path_model, prob=None, base=None, hue = "model")
        theta_dist.extend(model_rmse)
        base_rmse = get_theta_group(theta, current_group_pos_list[idx], path_base, prob=None, base=None, hue = "baseline")
        theta_dist.extend(base_rmse)
        #x_var.append(file_dict[current_group_pos_list[idx]][var_map_np[np.array(var) == None][0]])
    theta_dist_pd = pd.DataFrame(theta_dist)
    ####### Plot
    plt.figure(figsize=(12,9))
    color_map = {"model":"lightskyblue", "baseline":"lightsalmon"}
    sns.violinplot(x="h", y="theta", hue="hue", data=theta_dist_pd, palette=color_map, split=True)
    #plt.axhline(y= (math.log2(theta+1e-18) if if_log == True else theta), color='r', linestyle='-')
    plt.ylabel("Probability",fontsize=20)
    plt.legend(title = "Category",fontsize=14)
    plt.title(r"True $\theta$ = "+ theta.__str__())



def get_ROC_AUC(path, file_pos, file_neg, if_prob=None, if_baseline=None):
    #print (path, file_pos, file_neg)
    if if_prob == True and if_baseline == True:        
        file_pos = '_'.join(file_pos.rsplit(".pickle")[0].rsplit("_")[0:4])+'.pickle'
        file_neg = '_'.join(file_neg.rsplit(".pickle")[0].rsplit("_")[0:4])+'.pickle'
        prob1 = pickle.load(open(path+file_pos,"rb"))
        prob2 = pickle.load(open(path+file_neg,"rb"))
        fpr, tpr, _ = roc_curve([0 for i in range(len(prob1))] + [1 for i in range(len(prob2))], prob1 + prob2)
    elif if_prob == True and if_baseline != True: 
        prob1 = pickle.load(open(path+file_pos,"rb"))
        prob2 = pickle.load(open(path+file_neg,"rb"))
        fpr, tpr, _ = roc_curve([1 for i in range(len(prob1))] + [0 for i in range(len(prob2))], prob1 + prob2)
    else: #if_prob != True and if_baseline != True: 
        if if_baseline == True:
            file_pos = '_'.join(file_pos.rsplit(".pickle")[0].rsplit("_")[0:4])+'.pickle'
            file_neg = '_'.join(file_neg.rsplit(".pickle")[0].rsplit("_")[0:4])+'.pickle'
        prob1 = pickle.load(open(path+file_pos,"rb"))
        prob2 = pickle.load(open(path+file_neg,"rb"))
        prob1_t = [abs(x - 1) for x in prob1]
        prob2_t = [abs(x - 1) for x in prob2]
        fpr, tpr, _ = roc_curve([1 for i in range(len(prob1_t))] + [0 for i in range(len(prob2_t))], prob1_t + prob2_t)
    return(fpr, tpr, round(auc(fpr,tpr),3))



def Plot_ROC_comparison_fix4(workdir,theta_pos=None,theta_neg=None,if_prob=None,gene=None,hets=None,depth=None,sigma=None):
    source = "/data/allenlab/scarlett/software/cmdstan/examples/ase/"
    # judge whether  2 of the 4 are None, and 2 of the 4 are not None
    thetas=[theta_pos,theta_neg]
    if thetas.count(None) >0:
        raise Exception('Both thetas could not be None. The number of None from input was {}'.format(thetas.count(None)))    
    var=[gene,hets,depth,sigma]
    if var.count(None)>0:
        raise Exception('None of the variables could be set to None. The number of None from input was {}'.format(var.count(None)))
    # data directory is known 
    h=hets
    s=sigma
    g=gene
    d=depth
    if if_prob == True:
        path_model = source + workdir+"/output_pkl/model_prob/"
        path_base = source + workdir+"/output_pkl/baseline_prob/"
        base_file_pos = "g-%s_h-%s_d-%s_t-%g.pickle" % (g, h, d, float(theta_pos))
        base_file_neg = "g-%s_h-%s_d-%s_t-%s.pickle" % (g, h, d, int(theta_neg))
    else: 
        path_model=workdir+"model_med/"
        path_base=workdir+"baseline_theta/"
        base_file_pos = "g-%s_h-%s_d-%s_t-%g_s-%s.pickle" % (g, h, d, float(theta_pos),s)
        base_file_neg = "g-%s_h-%s_d-%s_t-%s_s-%s.pickle" % (g, h, d, int(theta_neg),s)
        
    model_file_pos = "g-%s_h-%s_d-%s_t-%g_s-%s.pickle" % (g, h, d, float(theta_pos), s)
    model_file_neg = "g-%s_h-%s_d-%s_t-%s_s-%s.pickle" % (g, h, d, int(theta_neg), s)
    
    plt.figure(figsize=(12,8))
    p1,p2,roc12 = get_ROC_AUC(path_model, model_file_pos, model_file_neg,if_prob=if_prob)
    p3,p4,roc34 = get_ROC_AUC(path_base, base_file_pos, base_file_neg,if_prob=if_prob,if_baseline=True)
    plt.plot(p1,p2, label = "model" +" AUC: "+ str(roc12),color='lightskyblue',linewidth=3)
    plt.plot(p3,p4, label = "baseline" +" AUC: "+ str(roc34),color='lightsalmon',linewidth=3)
    plt.ylabel("ROC",fontsize=14)
    plt.legend(fontsize=12)
    

def Plot_ROC_comparison_binomial_fix4(prefix,workdir,theta_pos=None,theta_neg=None,if_prob=None,gene=None,hets=None,depth=None):
    source = prefix+"scarlett/software/cmdstan/examples/binomial/output_pkl/"
    # judge whether  2 of the 4 are None, and 2 of the 4 are not None
    thetas=[theta_pos,theta_neg]
    if thetas.count(None) >0:
        raise Exception('Both thetas could not be None. The number of None from input was {}'.format(thetas.count(None)))    
    var=[gene,hets,depth]
    if var.count(None)>0:
        raise Exception('None of the variables could be set to None. The number of None from input was {}'.format(var.count(None)))
    # data directory is known 
    h=hets
    g=gene
    d=depth

    base_file_pos = "g-%s_h-%s_d-%s_t-%g.pickle" % (g, h, d, float(theta_pos))
    base_file_neg = "g-%s_h-%s_d-%s_t-%s.pickle" % (g, h, d, int(theta_neg))

    if if_prob != True:
        path1 = source + workdir+"/esti1/"
        path2 = source + workdir+"/esti2/"
        path3 = source + workdir+"/esti3/"
        path4 = source + workdir+"/esti4/"
        path5 = source + workdir+"/esti5/"
        path6 = source + workdir+"/esti6/"
        path7 = source + workdir+"/pooled_esti1/"
    else: 
        path1 = source + workdir+"/prob1/"
        path2 = source + workdir+"/prob2/"
        path3 = source + workdir+"/prob3/"
        path4 = source + workdir+"/pooled_prob1/"
        path5 = None
        path6 = None
        path7 = None
        if_prob=True

    base1="esti"
    base2="prob"
    
    p11,p12,roc1 = get_ROC_AUC_binomial(path1, base_file_pos, base_file_neg,if_prob=if_prob)
    p21,p22,roc2 = get_ROC_AUC_binomial(path2, base_file_pos, base_file_neg,if_prob=if_prob)
    p31,p32,roc3 = get_ROC_AUC_binomial(path3, base_file_pos, base_file_neg,if_prob=if_prob)
    p41,p42,roc4 = get_ROC_AUC_binomial(path4, base_file_pos, base_file_neg,if_prob=if_prob)

    plt.figure(figsize=(12,8))

    if "prob" in path1:
        plt.plot(p11,p12, label = base2 +"1 AUC: "+ str(roc1),linewidth=3)
    else:
        plt.plot(p11,p12, label = base1 +"1 AUC: "+ str(roc1),linewidth=3)     
    if "prob" in path2:
        plt.plot(p21,p22, label = base2 +"2 AUC: "+ str(roc2),linewidth=3)
    else:
        plt.plot(p21,p22, label = base1 +"2 AUC: "+ str(roc2),linewidth=3)     
    if "prob" in path3:
        plt.plot(p31,p32, label = base2 +"3 AUC: "+ str(roc3),linewidth=3)
    else:
        plt.plot(p31,p32, label = base1 +"3 AUC: "+ str(roc3),linewidth=3)   
    if "prob" in path4:
        plt.plot(p41,p42, label = "pooled prob1 AUC: "+ str(roc4),linewidth=3)
    else:
        plt.plot(p41,p42, label = base1 +"4 AUC: "+ str(roc4),linewidth=3)   
    if path5 != None:
        p51,p52,roc5 = get_ROC_AUC_binomial(path5, base_file_pos, base_file_neg)
        plt.plot(p51,p52, label = base1 +"5 AUC: "+ str(roc5),linewidth=3)  
    if path6 != None:
        p61,p62,roc6 = get_ROC_AUC_binomial(path6, base_file_pos, base_file_neg)
        plt.plot(p61,p62, label = base1 +"6 AUC: "+ str(roc6),linewidth=3)  
    if path7 != None:
        p71,p72,roc7 = get_ROC_AUC_binomial(path7, base_file_pos, base_file_neg)
        plt.plot(p71,p72, label = "pooled esti1 AUC: "+ str(roc7),linewidth=3)

    plt.ylabel("ROC",fontsize=14)
    plt.legend(fontsize=12)

def get_ROC_AUC_binomial(path, file_pos, file_neg, if_prob=None):
    prob1 = pickle.load(open(path+file_pos,"rb"))
    prob2 = pickle.load(open(path+file_neg,"rb"))
    if if_prob==True:
        fpr,tpr, _ = roc_curve([0 for i in range(len(prob1))] + [1 for i in range(len(prob2))], prob1 + prob2)
    else:
        fpr, tpr, _ = roc_curve([1 for i in range(len(prob1))] + [0 for i in range(len(prob2))], prob1 + prob2)
    return(fpr, tpr, round(auc(fpr,tpr),3)) 


def Plot_AUC_group_bar(workdir,theta_pos,theta_neg,if_prob=None,gene=None,hets=None,depth=None,sigma=None):
    source = "/data/reddylab/scarlett/1000G/software/cmdstan/examples/ase/"
    thetas=[theta_pos,theta_neg]
    if thetas.count(None) >0:
        raise Exception('Both thetas could not be None. The number of None from input was {}'.format(thetas.count(None)))    
    var=[gene,hets,depth,sigma]
    if var.count(None) !=1:
        raise Exception('One variables have to be set to None. The number of None from input was {}'.format(var.count(None)))
    if if_prob != True and if_prob !=None:
        raise Exception('if_prob == True - means using probability, if_prob == None - means using theta') 
    if if_prob == True:
        path_model = source + workdir+"/output_pkl/model_prob/"
        path_base = source + workdir+"/output_pkl/baseline_prob/"
    else:
        path_model = source + workdir+"/output_pkl/model_med/"
        path_base = source + workdir+"/output_pkl/baseline_theta/"    # data directory is known 

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
    pos_pd = file_dict_pd[(file_dict_pd[valid_var_np[0]] == valid_full_var_np[0])&(file_dict_pd[valid_var_np[1]] == valid_full_var_np[1])&(file_dict_pd[valid_var_np[2]] == valid_full_var_np[2])&(file_dict_pd['t'] == theta_pos)].sort_values(['d','h','g','s'])
    neg_pd = file_dict_pd[(file_dict_pd[valid_var_np[0]] == valid_full_var_np[0])&(file_dict_pd[valid_var_np[1]] == valid_full_var_np[1])&(file_dict_pd[valid_var_np[2]] == valid_full_var_np[2])&(file_dict_pd['t'] == theta_neg)].sort_values(['d','h','g','s'])
    
    current_group_pos_list = pos_pd.index
    current_group_neg_list = neg_pd.index

    labels= " "
    var_dict = {"gene":g, "hets": h, "depth": d, "sigma": s}
    for each in variable_var_np:
        if each != var_fullname_map_np[np.array(var) == None][0]:
            labels = each  
    d_group = pos_pd[var_map_np[np.array(var) == None][0]].unique()

    AUC=[]
    x_var = []
    AUC_base=[]
    x_var_base = []
    
    for idx in range(len(current_group_pos_list)):
        # model
        _,_,auc12 =  get_ROC_AUC(path_model, current_group_pos_list[idx],current_group_neg_list[idx],if_prob=if_prob)
        AUC.append(auc12)
        x_var.append(file_dict[current_group_pos_list[idx]][var_map_np[np.array(var) == None][0]])
        # baseline
        if if_prob == True:
            current_file_pos = current_group_pos_list[idx].rsplit("_",1)[0]+".pickle"
            current_file_neg = current_group_neg_list[idx].rsplit("_",1)[0]+".pickle"
        else: 
            current_file_pos = current_group_pos_list[idx]
            current_file_neg = current_group_neg_list[idx]
        _,_,auc34 =  get_ROC_AUC(path_base, current_file_pos,current_file_neg,if_prob=if_prob,if_baseline=True)
        AUC_base.append(auc34)
        x_var_base.append(file_dict[current_group_pos_list[idx]][var_map_np[np.array(var) == None][0]])

    ####### Plot
    plt.figure(figsize=(12,8))
    barWidth = 0.3
    bars1 = AUC
    bars2 = AUC_base
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    # Make the plot
    rects1=plt.bar(r1, bars1, color='lightskyblue', width=barWidth, edgecolor='white', label='Model')
    rects2=plt.bar(r2, bars2, color='lightsalmon', width=barWidth, edgecolor='white', label='Baseline')
    # Add xticks on the middle of the group bars
    plt.xlabel(var_fullname_map_np[np.array(var) == None][0], fontweight='bold',fontsize=12)
    plt.xticks([r + barWidth for r in range(len(bars1))], x_var)
    plt.ylabel("AUC")
    # Create legend & Show graphic
    autolabel(rects1,AUC)
    autolabel(rects2,AUC_base)
    #plt.ylim(0.5,1.2)
    plt.legend(loc=2,fontsize=14)
    plt.show()

def Plot_AUC_single_bar(workdir,theta_pos,theta_neg,if_base=None,if_prob=None,gene=None,hets=None,depth=None,sigma=None):
    source = "/data/reddylab/scarlett/1000G/software/cmdstan/examples/ase/"
    thetas=[theta_pos,theta_neg]
    if thetas.count(None) >0:
        raise Exception('Both thetas could not be None. The number of None from input was {}'.format(thetas.count(None)))    
    var=[gene,hets,depth,sigma]
    if var.count(None) !=1:
        raise Exception('One variables have to be set to None. The number of None from input was {}'.format(var.count(None)))
    if if_prob != True and if_prob !=None:
        raise Exception('if_prob == True - means using probability, if_prob == None - means using theta') 
    if if_prob == True:
        path = source + workdir+"/output_pkl/model_prob/"
    else:
        path = source + workdir+"/output_pkl/model_med/"
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
    pos_pd = file_dict_pd[(file_dict_pd[valid_var_np[0]] == valid_full_var_np[0])&(file_dict_pd[valid_var_np[1]] == valid_full_var_np[1])&(file_dict_pd[valid_var_np[2]] == valid_full_var_np[2])&(file_dict_pd['t'] == theta_pos)].sort_values(['d','h','g','s'])
    neg_pd = file_dict_pd[(file_dict_pd[valid_var_np[0]] == valid_full_var_np[0])&(file_dict_pd[valid_var_np[1]] == valid_full_var_np[1])&(file_dict_pd[valid_var_np[2]] == valid_full_var_np[2])&(file_dict_pd['t'] == theta_neg)].sort_values(['d','h','g','s'])
    
    current_group_pos_list = pos_pd.index
    current_group_neg_list = neg_pd.index

    labels= " "
    var_dict = {"gene":g, "hets": h, "depth": d, "sigma": s}
    for each in variable_var_np:
        if each != var_fullname_map_np[np.array(var) == None][0]:
            labels = each
            
    d_group = pos_pd[var_map_np[np.array(var) == None][0]].unique()
    AUC=[]
    x_var = []
    
    if theta_pos > theta_neg:
        rev = True
    else:
        rev = False
        
    for idx in range(len(current_group_pos_list)):
        _,_,auc =  get_ROC_AUC(path, current_group_pos_list[idx],current_group_neg_list[idx],if_prob=if_prob,if_baseline=if_base)
        AUC.append(auc)
        x_var.append(file_dict[current_group_pos_list[idx]][var_map_np[np.array(var) == None][0]])
    ####### Plotting
    bars1 = AUC
    barWidth = 0.3
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]

    
    plt.figure(figsize=(12,8))
    rects1=plt.bar(r1, bars1, color='lightskyblue', width=barWidth, edgecolor='white', label='Model')
    plt.xlabel(var_fullname_map_np[np.array(var) == None][0], fontweight='bold')
    plt.xticks(range(len(bars1)), x_var)
    plt.ylim(0.5,1)
    plt.ylabel("AUC")
    autolabel(rects1,AUC)
    
    plt.legend(loc=2)
    plt.show()


def Plot_RMSE_group_bar(workdir,theta,gene=None,hets=None,depth=None,sigma=None):
    source = "/data/reddylab/scarlett/1000G/software/cmdstan/examples/ase/"
    var=[gene,hets,depth,sigma]
    if var.count(None) !=1:
        raise Exception('One variables have to be set to None. The number of None from input was {}'.format(var.count(None)))
    path_model = source+ workdir+"/output_pkl/model_med/"
    path_base = source + workdir+"/output_pkl/baseline_theta/"
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
 
    RMSE=[]
    RMSE_base=[]
    x_var = []
    for idx in range(len(current_group_pos_list)):
        model_rmse = get_Rmse_from_theta(theta,current_group_pos_list[idx], path_model)
        RMSE.append(model_rmse)
        base_rmse = get_Rmse_from_theta(theta,current_group_pos_list[idx], path_base)
        RMSE_base.append(base_rmse)
        x_var.append(file_dict[current_group_pos_list[idx]][var_map_np[np.array(var) == None][0]])

    ####### Plot
    plt.figure(figsize=(12,8))
    barWidth = 0.3
    bars1 = RMSE
    bars2 = RMSE_base
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    # Make the plot
    #print (r1, bars2)
    rects1=plt.bar(r1, bars1, color='lightskyblue', width=barWidth, edgecolor='white', label='Model')
    rects2=plt.bar(r2, bars2, color='lightsalmon', width=barWidth, edgecolor='white', label='Baseline')
    # Add xticks on the middle of the group bars
    plt.xlabel(var_fullname_map_np[np.array(var) == None][0], fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bars1))], x_var)
    plt.ylabel("RMSE")
    # Create legend & Show graphic
    autolabel(rects1,RMSE)
    autolabel(rects2,RMSE_base)
    plt.legend(loc=1,fontsize = 'large')
    plt.show()

def get_Rmse_from_theta(theta,file_name, path):
    if "baseline" in path:
        file_name = file_name.rsplit("_",1)[0]+".pickle"
    theta_hat = pickle.load(open(path+file_name,'rb'))
    true_theta=theta
    #print (true_theta, theta_hat)
    RMSE = round(math.sqrt(np.mean( [(x - float(true_theta))**2 for x in theta_hat])),3)
    return(RMSE) 




def Plot_AUC_group_line(workdir,theta_pos,theta_neg,if_single=None,if_prob=None,gene=None,hets=None,depth=None,sigma=None):
    source = "/data/reddylab/scarlett/1000G/software/cmdstan/examples/ase/"    
    thetas=[theta_pos,theta_neg]
    if thetas.count(None) >0:
        raise Exception('Both thetas could not be None. The number of None from input was {}'.format(thetas.count(None)))    
    var=[gene,hets,depth,sigma]
    if var.count(None) !=1:
        raise Exception('One variables have to be set to None. The number of None from input was {}'.format(var.count(None)))
    if if_prob != True and if_prob !=None:
        raise Exception('if_prob == True - means using probability, if_prob == None - means using theta') 
    if if_prob == True:
        path_model = source + workdir+"/output_pkl/model_prob/"
        path_base = source + workdir+"/output_pkl/baseline_prob/"
    else:
        path_model = source + workdir+"/output_pkl/model_med/"
        path_base = source + workdir+"/output_pkl/baseline_theta/"    # data directory is known 

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
    pos_pd = file_dict_pd[(file_dict_pd[valid_var_np[0]] == valid_full_var_np[0])&(file_dict_pd[valid_var_np[1]] == valid_full_var_np[1])&(file_dict_pd[valid_var_np[2]] == valid_full_var_np[2])&(file_dict_pd['t'] == theta_pos)].sort_values(['d','h','g','s'])
    neg_pd = file_dict_pd[(file_dict_pd[valid_var_np[0]] == valid_full_var_np[0])&(file_dict_pd[valid_var_np[1]] == valid_full_var_np[1])&(file_dict_pd[valid_var_np[2]] == valid_full_var_np[2])&(file_dict_pd['t'] == theta_neg)].sort_values(['d','h','g','s'])
    
    current_group_pos_list = pos_pd.index
    current_group_neg_list = neg_pd.index

    labels= " "
    var_dict = {"gene":g, "hets": h, "depth": d, "sigma": s}
    for each in variable_var_np:
        if each != var_fullname_map_np[np.array(var) == None][0]:
            labels = each  
    d_group = pos_pd[var_map_np[np.array(var) == None][0]].unique()
    AUC=[]
    x_var = []
    for idx in range(len(current_group_pos_list)):
        _,_,auc =  get_ROC_AUC(path_model, current_group_pos_list[idx],current_group_neg_list[idx],if_prob=if_prob)
        AUC.append(auc)
        x_var.append(file_dict[current_group_pos_list[idx]][var_map_np[np.array(var) == None][0]])
    AUC_base=[]
    x_var_base = []
    for idx in range(len(current_group_pos_list)):
        _,_,auc =  get_ROC_AUC(path_base, current_group_pos_list[idx],current_group_neg_list[idx],if_prob=if_prob,if_baseline=True)
        AUC_base.append(auc)
        x_var_base.append(file_dict[current_group_pos_list[idx]][var_map_np[np.array(var) == None][0]])
    plt.figure(figsize=(6,5))
    plt.xlabel(var_fullname_map_np[np.array(var) == None][0])
    plt.plot(x_var, AUC, marker = 'o', c = 'r', label = 'model')
    plt.plot(x_var_base, AUC_base, marker = 'o', label = 'baseline')
    plt.legend()


def Plot_AUC_single_line(workdir,theta_pos,theta_neg,if_base=None,if_prob=None,gene=None,hets=None,depth=None,sigma=None):
    source = "/data/reddylab/scarlett/1000G/software/cmdstan/examples/ase/"
    thetas=[theta_pos,theta_neg]
    if thetas.count(None) >0:
        raise Exception('Both thetas could not be None. The number of None from input was {}'.format(thetas.count(None)))    
    var=[gene,hets,depth,sigma]
    if var.count(None) !=1:
        raise Exception('One variables have to be set to None. The number of None from input was {}'.format(var.count(None)))
    if if_prob != True and if_prob !=None:
        raise Exception('if_prob == True - means using probability, if_prob == None - means using theta') 
    if if_prob == True:
        path = source + workdir+"/output_pkl/model_prob/"
    else:
        path = source + workdir+"/output_pkl/model_med/"
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
    pos_pd = file_dict_pd[(file_dict_pd[valid_var_np[0]] == valid_full_var_np[0])&(file_dict_pd[valid_var_np[1]] == valid_full_var_np[1])&(file_dict_pd[valid_var_np[2]] == valid_full_var_np[2])&(file_dict_pd['t'] == theta_pos)].sort_values(['d','h','g','s'])
    neg_pd = file_dict_pd[(file_dict_pd[valid_var_np[0]] == valid_full_var_np[0])&(file_dict_pd[valid_var_np[1]] == valid_full_var_np[1])&(file_dict_pd[valid_var_np[2]] == valid_full_var_np[2])&(file_dict_pd['t'] == theta_neg)].sort_values(['d','h','g','s'])
    
    current_group_pos_list = pos_pd.index
    current_group_neg_list = neg_pd.index

    labels= " "
    var_dict = {"gene":g, "hets": h, "depth": d, "sigma": s}
    for each in variable_var_np:
        if each != var_fullname_map_np[np.array(var) == None][0]:
            labels = each
            
    d_group = pos_pd[var_map_np[np.array(var) == None][0]].unique()
    AUC=[]
    x_var = []
    for idx in range(len(current_group_pos_list)):        
        _,_,auc =  get_ROC_AUC(path, current_group_pos_list[idx],current_group_neg_list[idx],if_baseline=if_base,if_prob=if_prob)
        AUC.append(auc)
        x_var.append(file_dict[current_group_pos_list[idx]][var_map_np[np.array(var) == None][0]])
    plt.figure(figsize=(6,5))
    plt.xlabel(var_fullname_map_np[np.array(var) == None][0])
    plt.ylabel("AUC")
    plt.title("AUC vs #het sites",fontsize=15)
    #plt.ylim(0.5,1.1)
    plt.plot(x_var, AUC, marker = 'o', c = 'r')
    for idx in range(len(AUC)):
        plt.text(x_var[idx],AUC[idx],round(AUC[idx],2).__str__(),fontsize=12)

def draw_roc_prob(file_pos, file_neg, leg):
    prob1 = plk.load(open(file_pos,"rb"))
    prob2 = plk.load(open(file_neg,"rb"))
    p1, p2, _ = roc_curve([1 for i in range(len(prob1))] + [0 for i in range(len(prob2))], prob1 + prob2)
    plt.plot(p1,p2, label = leg +"AUC: "+ str(auc(p1,p2)))
    print (file_pos,auc(p1,p2))


def print_GSD_theta(sample,model):
    sigma=(0.0001,0.001,0.01,0.1,0.5,1)
    sources = ("star","tophat")
    for source in sources:
        #print("%s: sigma - theta probability" %(source))
        print(">>>>> %s" %(source))
        print("\t\tProbability ; Theta median")
        for s in sigma:
            theta=pickle.load(open("/data/reddylab/scarlett/1000G/software/cmdstan/examples/SPP2-odd-unphased/output_pkl/GSD/"+str(sample)+"/"+str(source)+"/theta_s-"+str(s)+".pickle","rb"))
            theta_prob=pickle.load(open("/data/reddylab/scarlett/1000G/software/cmdstan/examples/SPP2-odd-unphased/output_pkl/GSD/"+str(sample)+"/"+str(source)+"/theta_prob_s-"+str(s)+".pickle","rb"))
            theta_med=pickle.load(open("/data/reddylab/scarlett/1000G/software/cmdstan/examples/SPP2-odd-unphased/output_pkl/GSD/"+str(sample)+"/"+str(source)+"/theta_med_s-"+str(s)+".pickle","rb"))
            #print("       %s - %s" %(s,round(theta_prob,4)))
            print("sigma %f:  %f \t%f"%(s,theta_prob[0],theta_med[0])) 

def plot_GSD_theta(sample,model,source):
    sigma=(0.1,0.5,1)
    print(">>>>> %s" %(source))
    print("\t\tProbability ; Theta median")
    fig, ax = plt.subplots(ncols=1, figsize=(10,6))
    for s in sigma:
        theta=pickle.load(open("/data/reddylab/scarlett/1000G/software/cmdstan/examples/"+str(model)+"/output_pkl/GSD/"+str(sample)+"/"+str(source)+"/theta_s-"+str(s)+".pickle","rb"))
        theta_prob=pickle.load(open("/data/reddylab/scarlett/1000G/software/cmdstan/examples/"+str(model)+"/output_pkl/GSD/"+str(sample)+"/"+str(source)+"/theta_prob_s-"+str(s)+".pickle","rb"))
        theta_med=pickle.load(open("/data/reddylab/scarlett/1000G/software/cmdstan/examples/"+str(model)+"/output_pkl/GSD/"+str(sample)+"/"+str(source)+"/theta_med_s-"+str(s)+".pickle","rb"))
        sns.distplot(theta, hist =True, kde = True,kde_kws = {'linewidth': 3},label = 'sigma:'+str(s))
    plt.legend(fontsize=20)
    plt.title("Posterior theta distribution (Unphased no Error model)",fontsize=20) 

def plot_GSD_theta_med(sample,model):
    theta=pickle.load(open("/data/reddylab/scarlett/1000G/software/cmdstan/examples/"+str(model)+"/output_pkl/GSD/122687/star/theta_s-0.01.pickle","rb"))
    theta_med=pickle.load(open("/data/reddylab/scarlett/1000G/software/cmdstan/examples/"+str(model)+"/output_pkl/GSD/122687/star/theta_med_s-0.01.pickle","rb"))
    fig, ax = plt.subplots(ncols=1, figsize=(10,6))
    sns.distplot(theta, hist =True, kde = True,kde_kws = {'linewidth': 3},label = str(model))
    plt.legend(fontsize=12)
    plt.title("Posterior theta distribution",fontsize=15) 
    plt.axvline(x=theta_med,color='r', linestyle='--')


def Plot_FDR_comparison_fix2(label,workdir,theta_pos,theta_neg,fdr_cutoff,if_prob=None,prob_t=None,gene=None,hets=None,depth=None,sigma=None, num = None):
    source = "/data/reddylab/scarlett/1000G/software/cmdstan/examples/"
    # judge whether  2 of the 4 are None, and 2 of the 4 are not None
    thetas=[theta_pos,theta_neg]
    if thetas.count(None) >0:
        raise Exception('Both thetas could not be None. The number of None from input was {}'.format(thetas.count(None)))
    var=[gene,hets,depth,sigma]
    if var.count(None)!=2:
        raise Exception('Two variables have to be set to None. The number of None from input was {}'.format(var.count(None)))
    # data directory is known 
    if if_prob == None:
        path=source + label + "/" + workdir+"/output_pkl/model_med/"
    else:
        if prob_t == None:
            path=source + label + "/" + workdir+"/output_pkl/model_prob/"
        else:
            path=source + label + "/" + workdir+"/output_pkl/model_prob2/"

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
    pos_pd = file_dict_pd[(file_dict_pd[valid_var_np[0]] == valid_full_var_np[0])&(file_dict_pd[valid_var_np[1]] == valid_full_var_np[1])&(file_dict_pd['t'] == theta_pos)].sort_values(['d','h','g','s'])
    neg_pd = file_dict_pd[(file_dict_pd[valid_var_np[0]] == valid_full_var_np[0])&(file_dict_pd[valid_var_np[1]] == valid_full_var_np[1])&(file_dict_pd['t'] == theta_neg)].sort_values(['d','h','g','s'])

    d_group = pos_pd[var_map_np[np.array(var) == None][0]].unique()
    #print (d_group)
    if num == None:
        num = 3
    else:
        num = int(num)
    row = math.ceil(float(len(d_group))/num)
    fig, axs = plt.subplots(row, num, figsize = (20,8*row))
    if (row * num > len(d_group)):
        for i in range(row * num - len(d_group)):
            axs.flat[-1-i].set_axis_off()

    xlabels = "Fixed parameters "
    for i, each in enumerate(d_group):
        current_group_pos_list = pos_pd[pos_pd[var_map_np[np.array(var) == None][0]] == each].index
        current_group_neg_list = neg_pd[neg_pd[var_map_np[np.array(var) == None][0]] == each].index
        xlabels = "Fixed parameters "
        for idx in range(len(current_group_pos_list)):
            fpr, tpr, roc_auc = get_ROC_AUC(path, current_group_pos_list[idx], current_group_neg_list[idx],if_prob=if_prob)
            #for j in range(len(fpr)):
            #    if fpr[j] == 0:
            #        fpr[j]=0.00001
            #fdr = fpr/(fpr+tpr)
            g=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[0].rsplit("-")[1]
            h=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[1].rsplit("-")[1]
            d=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[2].rsplit("-")[1]
            s=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[4].rsplit("-")[1]
            #labels=" sigma:"+str(s)
            labels= " AUC: "+str(roc_auc)+" "
            var_dict = {"gene":g, "hets": h, "depth": d, "sigma": s}
            for each in variable_var_np:
                if each != var_fullname_map_np[np.array(var) == None][0]:
                    labels += each+":"+var_dict[each]+' '
            xlabels = "Fixed parameters "
            for each in fixed_var_np:
                xlabels += each+":"+var_dict[each]+' '
            #axs[i].suptitle(xlabels)
            axs.flat[i].set_xlabel("FDR = FPR/(FPR+TPR)")
            axs.flat[i].set_ylabel("TPR")
            axs.flat[i].set_title(var_fullname_map_np[np.array(var) == None][0]+":" + var_dict[var_fullname_map_np[np.array(var) == None][0]])
            axs.flat[i].plot(fpr/(fpr+tpr), tpr, label =  labels)
            axs.flat[i].legend(loc='lower right')
            axs.flat[i].axvline(x=fdr_cutoff,linewidth=2,color='r',linestyle="dashed")
    plt.suptitle(xlabels,fontsize=20)
    plt.show()



def Plot_ROC_curve_fix4(workdir,model,theta_pos=None,theta_neg=None,if_prob=None,gene=None,hets=None,depth=None,sigma=None):
    source = "/data/allenlab/scarlett/software/cmdstan/examples/"
    # judge whether  2 of the 4 are None, and 2 of the 4 are not None
    thetas=[theta_pos,theta_neg]
    if thetas.count(None) >0:
        raise Exception('Both thetas could not be None. The number of None from input was {}'.format(thetas.count(None)))
    var=[gene,hets,depth,sigma]
    if var.count(None)>0:
        raise Exception('None of the variables could be set to None. The number of None from input was {}'.format(var.count(None)))
    # data directory is known 
    h=hets
    s=sigma
    g=gene
    d=depth
    if if_prob == True:
        path_model = source + model+workdir+"/output_pkl/model_prob/"
        path_base = source + "ase/" + workdir+"/output_pkl/model_prob/"
        path_AA = source + "ADM/"+"output_pkl/"+workdir+"/AA_pval/"
    else:
        path_model = source + model+workdir+"/output_pkl/model_med/"
        path_base = source + "ase/" + workdir+"/output_pkl/model_med/"
        path_AA = source + "ADM/"+"output_pkl/"+workdir+"/AA_esti/"
        #base_file_pos = "g-%s_h-%s_d-%s_t-%g_s-%s.pickle" % (g, h, d, float(theta_pos),s)
        #base_file_neg = "g-%s_h-%s_d-%s_t-%s_s-%s.pickle" % (g, h, d, int(theta_neg),s)
    # AA's estimates
    AA_file_pos = "g-%s_h-%s_d-%s_t-%g.pickle" % (g, h, d, float(theta_pos))
    AA_file_neg = "g-%s_h-%s_d-%s_t-%s.pickle" % (g, h, d, int(theta_neg))
    #AA_file_pos = '_'.join(AA_file_pos.rsplit(".pickle")[0].rsplit("_")[0:4])+'.pickle'
    #AA_file_neg = '_'.join(AA_file_neg.rsplit(".pickle")[0].rsplit("_")[0:4])+'.pickle'
    model_file_pos = "g-%s_h-%s_d-%s_t-%g_s-%s.pickle" % (g, h, d, float(theta_pos), s)
    model_file_neg = "g-%s_h-%s_d-%s_t-%s_s-%s.pickle" % (g, h, d, int(theta_neg), s)

    # Plot
    plt.figure(figsize=(12,8))
    p1,p2,roc12 = get_ROC_AUC_V2(path_model, model_file_pos, model_file_neg,if_prob=if_prob)
    p3,p4,roc34 = get_ROC_AUC_V2(path_base, model_file_pos, model_file_neg,if_prob=if_prob)
    p5,p6,roc56 = get_ROC_AUC_V2(path_AA,AA_file_pos, AA_file_neg,if_AA=True,if_prob=if_prob)
    
    plt.plot(p1,p2, label = "BEASTIE" +" AUC: "+ str(roc12),color='lightskyblue',linewidth=3)# SPP2-odd
    plt.plot(p3,p4, label = "Binomial" +" AUC: "+ str(roc34),color='lightsalmon',linewidth=3)# ASE
    plt.plot(p5,p6, label = "ADM" +" AUC: "+ str(roc56),color='mediumseagreen',linewidth=3)
    plt.ylabel("ROC",fontsize=14)
    plt.legend(fontsize=12)




def get_ROC_AUC_V2(path, file_pos, file_neg, if_prob=None, if_baseline=None,if_AA=None,if_new=None):
    prob1 = pickle.load(open(path+file_pos,"rb"))
    prob2 = pickle.load(open(path+file_neg,"rb"))
    if (if_AA!=True and if_prob == True and if_baseline!=True) or (if_AA==True and if_prob!=True):
        fpr, tpr, _ = roc_curve([1 for i in range(len(prob1))] + [0 for i in range(len(prob2))], prob1 + prob2)
    elif (if_AA==True and if_prob==True) or if_baseline==True:
        fpr,tpr, _ = roc_curve([0 for i in range(len(prob1))] + [1 for i in range(len(prob2))], prob1 + prob2)
    elif if_prob!=True and if_baseline !=True:
        prob1_t = [abs(x - 1) for x in prob1]
        prob2_t = [abs(x - 1) for x in prob2]
        fpr, tpr, _ = roc_curve([1 for i in range(len(prob1_t))] + [0 for i in range(len(prob2_t))], prob1_t + prob2_t)
    return(fpr, tpr, round(auc(fpr,tpr),3))                          


def Check_tpr_w_fdr(fpr_list,tpr_list,fdr_cutoff,if_print=None,title="model"):
    for i in range(len(fpr_list)):
        if fpr_list[i]==0:
            fpr_list[i]=0.0000000001
    FDR = fpr_list/(fpr_list+tpr_list)
    FDR=FDR.tolist()
    #idx,fdr_cloest=min(enumerate(FDR), key=lambda x: abs(fdr_cutoff - x[1]))
    if min(FDR) > fdr_cutoff:
        fdr_closest=min(FDR)
    elif max(FDR) < fdr_cutoff:
        fdr_closest=max(FDR)
    else:
        fdr_closest=max(i for i in FDR if i <= fdr_cutoff) 
    idx=FDR.index(fdr_closest)
    if if_print==True:
        print(">> %s: With FDR cutoff of %s, we find the closest FPR: %s - TPR: %s"%(title,fdr_cutoff,round(fdr_closest,4),round(tpr_list[idx],4)))
    return(tpr_list[idx])


def Generate_path(source,model,workdir,if_prob):
    if if_prob == True:
        path_model = source + model+workdir+"/output_pkl/model_prob/"
        path_base = source + "ase/" + workdir+"/output_pkl/model_prob/"
        path_AA = source + "ADM/"+"output_pkl/"+"AA_pval/"
    else:
        path_model = source + model+workdir+"/output_pkl/model_med/"
        path_base = source + "ase/" + workdir+"/output_pkl/model_med/"
        path_AA = source + "ADM/"+"output_pkl/"+"AA_esti/"
    return path_model, path_base, path_AA


def Lookup_files_plot(thetas,gene,hets,depth,sigma,path_model,path_base,path_AA,fdr_cutoff,if_prob,if_print):
    theta_pos = thetas[0]
    theta_neg = thetas[1]
    var=[gene,hets,depth,sigma]
    var_map_np = np.array(['g','h','d','s']) # array(['g', 'h', 'd', 's'], dtype='<U1')
    var_fullname_map_np = np.array(['gene','hets','depth','sigma']) #array(['gene', 'hets', 'depth', 'sigma'], dtype='<U5')
    full_var_map_np = np.array([gene, hets, depth, sigma]) #array([1000, 3, None, 0.5], dtype=object)
    valid_var_np = var_map_np[np.array(var) != None] #array(['g', 'h', 's'], dtype='<U1')
    variable_var_np = var_fullname_map_np[np.array(var) == None] #array(['depth'], dtype='<U5')
    fixed_var_np = var_fullname_map_np[np.array(var) != None] #array(['gene', 'hets', 'sigma'], dtype='<U5')
    valid_full_var_np = full_var_map_np[np.array(var) != None] #array(['gene', 'hets', 'sigma'], dtype='<U5')

    if hets is not None:
        h=hets
    if sigma is not None:
        s=sigma
    if gene is not None:
        g=gene
    if depth is not None:
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
    pos_pd = file_dict_pd[(file_dict_pd[valid_var_np[0]] == valid_full_var_np[0])&(file_dict_pd[valid_var_np[1]] == valid_full_var_np[1])&(file_dict_pd[valid_var_np[2]] == valid_full_var_np[2])&(file_dict_pd['t'] == theta_pos)].sort_values(['d','h','g','s'])
    neg_pd = file_dict_pd[(file_dict_pd[valid_var_np[0]] == valid_full_var_np[0])&(file_dict_pd[valid_var_np[1]] == valid_full_var_np[1])&(file_dict_pd[valid_var_np[2]] == valid_full_var_np[2])&(file_dict_pd['t'] == theta_neg)].sort_values(['d','h','g','s'])
    d_group = pos_pd[var_map_np[np.array(var) == None][0]].unique()
    loop_var=d_group
    
#     if num_per_row == None:
#         num_per_row = 3
#     else:
#         num_per_row = int(num_per_row)
#     row = math.ceil(float(len(loop_var))/num_per_row)
#     fig, axs = plt.subplots(row, num_per_row, figsize = (20,5*row))
#     if (row * num_per_row > len(loop_var)):
#         for i in range(row * num_per_row - len(loop_var)):
#             axs.flat[-1-i].set_axis_off()

    TPR_list_m=[]
    Var_list_m=[]
    TPR_list_b=[]
    Var_list_b=[]
    #FDR_list_a=()

    for i, each in enumerate(loop_var):
        current_group_pos_list = pos_pd[pos_pd[var_map_np[np.array(var) == None][0]] == each].index
        current_group_neg_list = neg_pd[neg_pd[var_map_np[np.array(var) == None][0]] == each].index

        for idx in range(len(current_group_pos_list)):
            print(current_group_pos_list[0])
            # old estimate for model
            fpr_m, tpr_m, roc_auc_m = get_ROC_AUC(path_model, current_group_pos_list[idx], current_group_neg_list[idx],if_prob=if_prob)
            #print(fpr_m)
            TPR_m = Check_tpr_w_fdr(fpr_m, tpr_m,fdr_cutoff=fdr_cutoff,if_print=if_print)
            TPR_list_m.append(TPR_m)
            Var_m = math.sqrt(TPR_m*(1-TPR_m)/1000)
            Var_list_m.append(Var_m)

            # baseline
            fpr_b, tpr_b, roc_auc_b = get_ROC_AUC(path_base, current_group_pos_list[idx], current_group_neg_list[idx],if_prob=if_prob)
            TPR_b = Check_tpr_w_fdr(fpr_b, tpr_b,fdr_cutoff=fdr_cutoff,if_print=if_print)
            TPR_list_b.append(TPR_b)
            Var_b = math.sqrt(TPR_b*(1-TPR_b)/1000)
            Var_list_b.append(Var_b)
            # AA
            reduced_file_pos = current_group_pos_list[idx].rsplit("_",1)[0]+".pickle"
            reduced_file_neg = current_group_neg_list[idx].rsplit("_",1)[0]+".pickle"
            #fpr_a, tpr_a, roc_auc_a = get_ROC_AUC(path_AA,reduced_file_pos, reduced_file_neg,if_prob=if_prob)
            #TPR_a = Check_tpr_w_fdr(fpr_a, tpr_a,fdr_cutoff=fdr_cutoff,if_print=True)

            g=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[0].rsplit("-")[1]
            h=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[1].rsplit("-")[1]
            d=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[2].rsplit("-")[1]
            s=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[4].rsplit("-")[1]
            #labels=" sigma:"+str(s)
            labels= " model: "+str(roc_auc_m)+" "
            var_dict = {"gene":g, "hets": h, "depth": d, "sigma": s}
            for each in variable_var_np:
                if each != var_fullname_map_np[np.array(var) == None][0]:
                    labels += each+":"+var_dict[each]+' '
            xlabels = "Fixed parameters "
            for each in fixed_var_np:
                xlabels += each+":"+var_dict[each]+','

    plt.figure(figsize=(12,10))
    plt.errorbar(d_group,TPR_list_m,Var_list_m,uplims=True, lolims=True, label = "BEASTIE" ,color='mediumseagreen',linewidth=4)
    plt.errorbar(d_group,TPR_list_b, Var_list_b,uplims=True, lolims=True,label = "BINOMIAL" ,color='royalblue',linewidth=4)
    plt.ylabel("Power / TPR / Sensitivity",fontsize=14)
    plt.xlabel("Read depth",fontsize=14)
    plt.title("FDR cutoff at %s"%(fdr_cutoff),fontsize=20)
    plt.axhline(y=0.9,color='darkred',alpha=0.95,linestyle='--',label="sensitivity=90%")
    plt.axhline(y=0.8,color='red',alpha=0.95,linestyle='--',label="sensitivity=80%")
    plt.axhline(y=0.75,color='orangered',alpha=0.95,linestyle='--',label="sensitivity=75%")
    plt.axhline(y=0.5,color='lightsalmon',alpha=0.95,linestyle='--',label="sensitivity=50%")
    plt.legend(fontsize=12)


def FDR_curve_fix3(workdir,theta_pos,theta_neg,fdr_cutoff=0.05,if_print=None,if_prob=None,gene=None,hets=None,depth=None,sigma=None):
    source = "/data/reddylab/scarlett/1000G/software/cmdstan/examples/"
    model="SPP2-odd/"
    # judge whether 1 of the 4 are None, and 3 of the 4 are not None
    thetas=[theta_pos,theta_neg]
    if thetas.count(None) >0:
        raise Exception('Both thetas could not be None. The number of None from input was {}'.format(thetas.count(None)))
    var=[gene,hets,depth,sigma]
    if var.count(None)!=1:
        raise Exception('One variables have to be set to None. The number of None from input was {}'.format(var.count(None)))
    path_model, path_base, path_AA = Generate_path(source,model,workdir=workdir,if_prob=if_prob)
    Lookup_files_plot(thetas,gene,hets,depth,sigma,path_model,path_base,path_AA,fdr_cutoff=fdr_cutoff,if_prob=if_prob,if_print=if_print)
    #Plot(3,pos_pd, neg_pd,loop_var,var_map_np, variable_var_np,var,path_model,path_base,path_AA,fdr_cutoff=fdr_cutoff)


def Plot_ROC_curve_fix3(workdir,theta_pos,theta_neg,fdr_cutoff,if_FDR=None,if_prob=None,gene=None,hets=None,depth=None,sigma=None, num = None):
    source = "/data/reddylab/scarlett/1000G/software/cmdstan/examples/"
    model="SPP2-odd"
    # judge whether 1 of the 4 are None, and 3 of the 4 are not None
    thetas=[theta_pos,theta_neg]
    if thetas.count(None) >0:
        raise Exception('Both thetas could not be None. The number of None from input was {}'.format(thetas.count(None)))
    var=[gene,hets,depth,sigma]
    if var.count(None)!=1:
        raise Exception('One variables have to be set to None. The number of None from input was {}'.format(var.count(None)))
    # data directory is known 
    if if_prob == None:
        model_path=source + model + "/" + workdir+"/output_pkl/model_med/"
        baseline_path = source +"ase/" + workdir+"/output_pkl/model_med/"
        #aa_path = source +"ADM/output_pkl/"+workdir+"/AA_esti/"
    else:
        model_path=source + model + "/" + workdir+"/output_pkl/model_prob/"
        model_path2=source + model + "/" + workdir+"/output_pkl/model_prob2/"
        baseline_path = source +"ase/" + workdir+"/output_pkl/model_prob/"
        baseline_path2 = source +"ase/" + workdir+"/output_pkl/model_prob2/"
        #aa_path = source +"ADM/output_pkl/"+workdir+"/AA_pval/"
    
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
    all_file = sorted(os.listdir(model_path))
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
    pos_pd = file_dict_pd[(file_dict_pd[valid_var_np[0]] == valid_full_var_np[0])&(file_dict_pd[valid_var_np[1]] == valid_full_var_np[1])&(file_dict_pd[valid_var_np[2]] == valid_full_var_np[2])&(file_dict_pd['t'] == theta_pos)].sort_values(['d','h','g','s'])
    neg_pd = file_dict_pd[(file_dict_pd[valid_var_np[0]] == valid_full_var_np[0])&(file_dict_pd[valid_var_np[1]] == valid_full_var_np[1])&(file_dict_pd[valid_var_np[2]] == valid_full_var_np[2])&(file_dict_pd['t'] == theta_neg)].sort_values(['d','h','g','s'])

    d_group = pos_pd[var_map_np[np.array(var) == None][0]].unique()
    #print (d_group)
    if num == None:
        num = 3
    else:
        num = int(num)
    row = math.ceil(float(len(d_group))/num)
    fig, axs = plt.subplots(row, num, figsize = (20,5*row))
    if (row * num > len(d_group)):
        for i in range(row * num - len(d_group)):
            axs.flat[-1-i].set_axis_off()

    xlabels = "Fixed parameters "
    
    if if_prob == True:
        #print("need probability")
        for i, each in enumerate(d_group):
            current_group_pos_list = pos_pd[pos_pd[var_map_np[np.array(var) == None][0]] == each].index
            current_group_neg_list = neg_pd[neg_pd[var_map_np[np.array(var) == None][0]] == each].index
            xlabels = "Fixed parameters "
            for idx in range(len(current_group_pos_list)):
                # old estimate for model
                fpr_m, tpr_m, roc_auc_m = get_ROC_AUC(model_path, current_group_pos_list[idx], current_group_neg_list[idx],if_prob=if_prob)
                # old estimate for baseline
                fpr_b, tpr_b, roc_auc_b = get_ROC_AUC(baseline_path, current_group_pos_list[idx], current_group_neg_list[idx],if_prob=if_prob)  #if_baseline=True
                ##### change the reading file method a little bit because ADM results do not have parameter sigma
                reduced_file_pos = current_group_pos_list[idx].rsplit("_",1)[0]+".pickle"
                reduced_file_neg = current_group_neg_list[idx].rsplit("_",1)[0]+".pickle"
                # ADM estimates pval
                #fpr_a, tpr_a, roc_auc_a = get_ROC_AUC_V2(aa_path, reduced_file_pos, reduced_file_neg,if_prob=if_prob,if_baseline=True,if_AA=True)
                g=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[0].rsplit("-")[1]
                h=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[1].rsplit("-")[1]
                d=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[2].rsplit("-")[1]
                s=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[4].rsplit("-")[1]
                #labels=" sigma:"+str(s)
                labels= " model: "+str(roc_auc_m)+" "
                var_dict = {"gene":g, "hets": h, "depth": d, "sigma": s}
                for each in variable_var_np:
                    if each != var_fullname_map_np[np.array(var) == None][0]:
                        labels += each+":"+var_dict[each]+' '
                xlabels = "Fixed parameters "
                for each in fixed_var_np:
                    xlabels += each+":"+var_dict[each]+' '
                #axs[i].suptitle(xlabels)
                axs.flat[i].set_ylabel("TPR")

                if if_FDR!=None:
                    axs.flat[i].set_xlabel("FDP = FPR/(FPR+TPR)")
                    axs.flat[i].plot(fpr_m/(fpr_m+tpr_m), tpr_m, label =  "BEASTIE:"+str(roc_auc_m))
                    axs.flat[i].plot(fpr_b/(fpr_b+tpr_b), tpr_b, label =  "Binomial:"+str(roc_auc_b))
                    #axs.flat[i].plot(fpr_a/(fpr_a+tpr_a), tpr_a, label =  "AA:"+str(roc_auc_a))
                    axs.flat[i].axvline(x=fdr_cutoff,linewidth=2,color='r',linestyle="dashed")
                else:
                    axs.flat[i].set_xlabel("FPR")
                    axs.flat[i].plot(fpr_m, tpr_m, label =  "BEASTIE:"+str(roc_auc_m))
                    axs.flat[i].plot(fpr_b, tpr_b, label =  "Binomial:"+str(roc_auc_b))
                    #axs.flat[i].plot(fpr_a, tpr_a, label =  "ADM:"+str(roc_auc_a))
                #axs.flat[i].set_title(var_fullname_map_np[np.array(var) == None][0]+":" + var_dict[var_fullname_map_np[np.array(var) == None][0]])
                axs.flat[i].legend(loc='lower right')
    else:
        for i, each in enumerate(d_group):
            current_group_pos_list = pos_pd[pos_pd[var_map_np[np.array(var) == None][0]] == each].index
            current_group_neg_list = neg_pd[neg_pd[var_map_np[np.array(var) == None][0]] == each].index
            xlabels = "Fixed parameters "
            for idx in range(len(current_group_pos_list)):
                #print(current_group_pos_list[idx])
                # old estimate for model
                fpr_m, tpr_m, roc_auc_m = get_ROC_AUC(model_path, current_group_pos_list[idx], current_group_neg_list[idx],if_prob=if_prob)
                # old estimate for baseline
                fpr_b, tpr_b, roc_auc_b = get_ROC_AUC(baseline_path, current_group_pos_list[idx], current_group_neg_list[idx],if_prob=if_prob)#if_baseline=True
                reduced_file_pos = current_group_pos_list[idx].rsplit("_",1)[0]+".pickle"
                reduced_file_neg = current_group_neg_list[idx].rsplit("_",1)[0]+".pickle"
                # aa
                #fpr_a, tpr_a, roc_auc_a = get_ROC_AUC_V2(aa_path, reduced_file_pos, reduced_file_neg,if_prob=if_prob,if_AA=True)
                g=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[0].rsplit("-")[1]
                h=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[1].rsplit("-")[1]
                d=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[2].rsplit("-")[1]
                s=current_group_pos_list[idx].rsplit(".pickle")[0].rsplit("_")[4].rsplit("-")[1]
                #labels=" sigma:"+str(s)
                labels= " model: "+str(roc_auc_m)+" "
                var_dict = {"gene":g, "hets": h, "depth": d, "sigma": s}
                for each in variable_var_np:
                    if each != var_fullname_map_np[np.array(var) == None][0]:
                        labels += each+":"+var_dict[each]+' '
                xlabels = "Fixed parameters "
                for each in fixed_var_np:
                    xlabels += each+":"+var_dict[each]+' '
                #axs[i].suptitle(xlabels)
                axs.flat[i].set_title(var_fullname_map_np[np.array(var) == None][0]+":" + var_dict[var_fullname_map_np[np.array(var) == None][0]])
                axs.flat[i].set_ylabel("TPR")
                if if_FDR!=None:
                    axs.flat[i].set_xlabel("FDR = FPR/(FPR+TPR)")
                    axs.flat[i].plot(fpr_m/(fpr_m+tpr_m), tpr_m, label =  "BEASTIE:"+str(roc_auc_m))
                    axs.flat[i].plot(fpr_b/(fpr_b+tpr_b), tpr_b, label =  "Binomial:"+str(roc_auc_b))
                    #axs.flat[i].plot(fpr_a/(fpr_a+tpr_a), tpr_a, label =  "AA:"+str(roc_auc_a))
                    axs.flat[i].axvline(x=fdr_cutoff,linewidth=2,color='r',linestyle="dashed")
                else:
                    axs.flat[i].set_xlabel("FPR")
                    axs.flat[i].plot(fpr_m, tpr_m, label =  "BEASTIE:"+str(roc_auc_m))
                    axs.flat[i].plot(fpr_b, tpr_b, label =  "Binomial:"+str(roc_auc_b))
                    #axs.flat[i].plot(fpr_a, tpr_a, label =  "ADM:"+str(roc_auc_a))
                axs.flat[i].legend(loc='lower right')
    plt.suptitle(xlabels,fontsize=20)
    plt.show()                      
