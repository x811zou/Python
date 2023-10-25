#!/bin/python

import os
import pandas as pd
import glob  
import time
import multiprocessing
import numpy as np
import pickle
from . import calculateCDF

def get_directories(path):
    subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return subdirs

def get_sample_df(DCC_path,sample_dict):
    all_sample_df = pd.DataFrame()
    counter=0
    for sample in sample_dict:
        df_path = "/output/RNAseq/1000Genome/"+str(sample)+"/beastie_202301/beastie_SNPs_even_100/beastie_shapeit2/chr1-22_alignBiasp0.05_ase0.5_s0.5_a0.05_sinCov0_totCov1_W1000K1000/result"
        path_df = DCC_path + df_path + "/" + str(sample) +"_ASE_sub.tsv"
        file = pd.read_csv(path_df,sep="\t")
        file["sample"]=sample
        file["ancestry"]=sample_dict[sample]
        file["abs_log2"] = np.abs(np.log2(file["posterior_mean"]))
        file['simplified_geneID'] = file.geneID.apply(lambda x: x.split(".")[0])
        all_sample_df=pd.concat([all_sample_df,file],axis=0)
#         counter+=1
#         if counter==10
#             break
    return all_sample_df

def get_success_sample(DCC_path):
    path = DCC_path + "/output/RNAseq/1000Genome/"
    sample_list = glob.glob(path + "*/", recursive=True)
    success_dict = {}
    counter=0
    for sample_path in sample_list:
        sample = os.path.basename(os.path.dirname(sample_path))
        success_file = path + str(sample) + "/success"
        ancestry_file = path +str(sample)+ "/ancestry"
        counter+=1
        if os.path.isfile(success_file):
            with open(ancestry_file) as f:
                ancestry = f.read().strip()
            #close()
            #tring = str(ancestry).replace('\n','')
            #print(ancestry)
            success_dict[sample] = ancestry
    return success_dict

def get_success_sample(path):
    sample_list = get_directories(path)
    success_list = []
    for sample in sample_list:
        success_file = path + str(sample) + "/success"
        if os.path.isfile(success_file):
            success_list.append(sample)
    return success_list

def get_success_model(path, model, sigma,sample_list):
    theta_path = (
        "/beastie/runModel_phased_even100/chr1-22_alignBiasp0.05_s"
        + str(round(float(sigma),2))
        + "_a0.05_sinCov0_totCov1_W1000K1000/"
        + str(model)
        + "/output_pkl/iBEASTIE/theta/"
    )
    #sample_list = get_success_sample(path)
    success_list = []
    for sample in sample_list:
        success_file = path + str(sample) + theta_path + "stan.pickle"
        if os.path.isfile(success_file):
            success_list.append(sample)
    return success_list


def clean_data(df):
    filtered_df = df.drop(columns=['Unnamed: 0', 'Pseudo_pval','CI_left','CI_right','beta_1_1_pval','beta_10_10_pval','beta_20_20_pval','beta_50_50_pval','beta_100_100_pval'])
    return filtered_df

def get_gene_counts_folderpath(path,sigma):
    subdirectories = get_success_sample(path)
    all_data_list=[]
    for sample in subdirectories:
        new_data = get_gene_counts_file(path,sample,sigma)
        all_data_list.append(new_data)
    all_data = pd.concat(all_data_list, axis=0)
    return all_data

def get_gene_counts_file(path,sample,sigma):
    file_path=path+"/"+str(sample)+"/beastie/runModel_phased_even100/chr1-22_alignBiasp0.05_s"+str(sigma)+"_a0.05_sinCov0_totCov1_W1000K1000/tmp/"
    filename=str(sample)+"_real_alignBiasafterFilter.phasedByshapeit2.cleaned.forlambda.tsv"
    data=pd.read_csv(file_path+filename,sep="\t",index_col=False)
    data["sample"]=str(sample)
    return data

def get_SNP_counts_folderpath(path,sigma):
    subdirectories = get_success_sample(path)
    all_data_list=[]
    for sample in subdirectories:
        new_data = get_SNP_counts_file(path,sample,sigma)
        all_data_list.append(new_data)
    all_data = pd.concat(all_data_list, axis=0)
    return all_data

def get_SNP_counts_file(path,sample,sigma):
    file_path=path+"/"+str(sample)+"/beastie/runModel_phased_even100/chr1-22_alignBiasp0.05_s"+str(sigma)+"_a0.05_sinCov0_totCov1_W1000K1000/tmp/"
    filename=str(sample)+"_real_alignBiasafterFilter.phasedByshapeit2.cleaned.tsv"
    data=pd.read_csv(file_path+filename,sep="\t",index_col=False)
    data=data[["chr","pos","rsid","geneID","patCount","matCount","totalCount"]]
    data["sample"]=str(sample)
    return data

def get_ASE_output_folderpath(path,sigma,subdirectories):
    #subdirectories = get_success_sample(path)
    all_data_list=[]
    for sample in subdirectories:
        file_path=path+"/"+str(sample)+"/beastie/runModel_phased_even100/chr1-22_alignBiasp0.05_s"+str(sigma)+"_a0.05_sinCov0_totCov1_W1000K1000/iBEASTIE3/"
        filename=str(sample)+"_ASE_all.tsv"
        if os.path.isfile(file_path+filename):
            new_data = get_ASEoutput_file(file_path+filename,path,sample,sigma)
            all_data_list.append(new_data)
    all_data = pd.concat(all_data_list, axis=0)
    cleaned_all_data=clean_data(all_data)
    return cleaned_all_data

def get_ASEoutput_file(filename,path,sample,sigma):
    data=pd.read_csv(filename,sep="\t",index_col=False)
    data["sample"]=str(sample)
    #filtered_total_reads=get_filtered_total_reads(path,sample)
    #filtered_primary_reads=get_filtered_primary_reads(path,sample)
    #sex=get_sex(path,sample)
    #data["filtered_total_reads"]=filtered_total_reads
    #data["filtered_primary_reads"]=filtered_primary_reads
    #data["sex"]=sex
    return data

def get_filtered_total_reads(path,sample):
    file_path=path+"/"+str(sample)+"/filtered_total_seqDepth"
    with open(file_path, 'r') as file:
        data = file.read()
    if data.strip():
        return int(data.strip())
    else:
        print(f"{sample} does not have filtered_total_seqDepth info")
        return "NA"

def get_filtered_primary_reads(path,sample):
    file_path=path+"/"+str(sample)+"/filtered_mapped_seqDepth"
    with open(file_path, 'r') as file:
        data = file.read()
    if data.strip():
        return int(data.strip())
    else:
        print(f"{sample} does not have filtered_mapped_seqDepth info")
        return "NA"

def get_sex(path,sample):
    file_path=path+"/"+str(sample)+"/sex"
    with open(file_path, 'r') as file:
        data = file.read()
    if data.strip():
        return data.strip()
    else:
        print(f"{sample} does not have sex info")
        return "NA"
        
def get_diff_list(list1,list2):
    diff = [item for item in list1 if item not in list2]
    return diff

def get_all_gene_dict(DCC_path, theta_path,sample_list, cutoff_list, debug=False):
    debug_print = (lambda x: print(x)) if debug else (lambda x: None)
    abslog2_cutoff = [abs(np.log2(x)) for x in cutoff_list]
    # debugging
    debug_print(f">>>>>> thetas")
    debug_print("un-transformed cutoff: ")
    debug_print(cutoff_list)
    debug_print("abs log2 transformed cutoff: ")
    debug_print(abslog2_cutoff)
    start_ns = time.time_ns()
    # output initialization
    all_sample_gene_dict = {}
    # start looping through each sample
    for i, sample in enumerate(sample_list):
        debug_print(f">>>>>> Sample: {sample}")
        sample_gene_dict = processSampleDict(DCC_path,sample,abslog2_cutoff,debug=debug)
        updateGeneToSampleDict(all_sample_gene_dict, sample_gene_dict)
        # debugging
        debug_print(f"completed sample {sample}")
        if (i + 1) % 10 == 0:
            now_ns = time.time_ns()
            print(
                f"finished {i+1}/{len(sample_list)} samples {(now_ns - start_ns) / 1e9}s"
            )
            start_ns = now_ns
    return all_sample_gene_dict

def get_all_gene_dict_parallel(path, model,sigma,sample_list, cutoff_list):
    theta_path = (
        "/beastie/runModel_phased_even100/chr1-22_alignBiasp0.05_s"
        + str(round(float(sigma),2))
        + "_a0.05_sinCov0_totCov1_W1000K1000/"
        + str(model)
        + "/output_pkl/iBEASTIE/theta/"
    )
    abslog2_cutoff = [abs(np.log2(x)) for x in cutoff_list]
    sample_tuples = [(x, path, theta_path,abslog2_cutoff) for x in sample_list]
    # output initialization
    all_sample_gene_dict = {}
    # start looping through each sample
    with multiprocessing.Pool(processes=16) as pool:
        i = 0
        start_ns = time.time_ns()
        for result_dict in pool.imap_unordered(processSampleDictWrapper, sample_tuples):
            updateGeneToSampleDict(all_sample_gene_dict, result_dict)

            i += 1
            if i % 10 == 0:
                now_ns = time.time_ns()
                print(
                    f"finished {i}/{len(sample_list)} samples {(now_ns - start_ns) / 1e9}s"
                )
                start_ns = now_ns
    return all_sample_gene_dict

def processSampleDictWrapper(t):
    return processSampleDict(*t)

def processSampleDict(sample, DCC_path, theta_path,abslog2_cutoff):
    gene_dict = {}
    gene_thetas_dict = read_one_posteriors(DCC_path, theta_path,sample)
    for gene, thetas in gene_thetas_dict.items():
        simplified_gene = gene.split(".", 1)[0]

        if simplified_gene not in gene_dict.keys():
            gene_dict[simplified_gene] = {}

        prob_tuple = calculateCDF.calculate_below_prob_fromThetas(thetas, abslog2_cutoff)
        gene_dict[simplified_gene][sample] = prob_tuple
    return gene_dict

def read_one_posteriors(path, theta_path, sample):
    path = path + str(sample) + theta_path
    file = open(path + "stan.pickle", "rb")
    object_file = pickle.load(file)
    file.close()
    return object_file

def read_one_pickle(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)

def updateGeneToSampleDict(a, b):
  for k, v in b.items():
    if k in a:
      a[k].update(v)
    else:
      a[k] = v

def get_all_gene_dict_MCMC_parallel(path, model,sigma,sample_list):
    theta_path = (
        "/beastie/runModel_phased_even100/chr1-22_alignBiasp0.05_s"
        + str(round(float(sigma),2))
        + "_a0.05_sinCov0_totCov1_W1000K1000/"
        + str(model)
        + "/output_pkl/iBEASTIE/theta/"
    )
    #abslog2_cutoff = [abs(np.log2(x)) for x in cutoff_list]
    sample_tuples = [(x, path, theta_path) for x in sample_list]
    # output initialization
    all_sample_gene_dict = {}
    # start looping through each sample
    with multiprocessing.Pool(processes=16) as pool:
        i = 0
        start_ns = time.time_ns()
        for result_dict in pool.imap_unordered(processSampleDictWrapper_MCMC,sample_tuples):
            updateGeneToSampleDict(all_sample_gene_dict, result_dict)

            i += 1
            if i % 10 == 0:
                now_ns = time.time_ns()
                print(
                    f"finished {i}/{len(sample_list)} samples {(now_ns - start_ns) / 1e9}s"
                )
                start_ns = now_ns
    return all_sample_gene_dict

def processSampleDictWrapper_MCMC(t):
    return processSampleDict_MCMC(*t)

def updateGeneToSampleDict(a, b):
  for k, v in b.items():
    if k in a:
      a[k].update(v)
    else:
      a[k] = v

def processSampleDict_MCMC(sample, DCC_path, theta_path):
    gene_dict = {}
    gene_thetas_dict = read_one_posteriors(DCC_path, theta_path,sample)
    for gene, thetas in gene_thetas_dict.items():
        simplified_gene = gene.split(".", 1)[0]

        if simplified_gene not in gene_dict.keys():
            gene_dict[simplified_gene] = {}

        #prob_tuple = calculateCDF.calculate_below_prob_fromThetas(thetas, abslog2_cutoff)
        gene_dict[simplified_gene][sample] = gene_thetas_dict[simplified_gene]

    return gene_dict