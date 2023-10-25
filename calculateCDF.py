#!/bin/python

import pickle
import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import time
import multiprocessing
import statistics

def get_percentile_figure(mylist, val=None, geneset=None, title=None):
    if geneset is not None:
        if title is not None:
            print("{} geneset include : {} genes".format(title, len(geneset)))

    mylist_log2 = np.log2(mylist)
    mylist_abslog2 = [abs(x) for x in mylist_log2]

    mylist_df = pd.DataFrame(mylist)
    mylist_log2_df = pd.DataFrame(mylist_log2)
    mylist_abslog2_df = pd.DataFrame(mylist_abslog2)

    mylist_df.columns = ["thetas"]
    mylist_log2_df.columns = ["log2(thetas)"]
    mylist_abslog2_df.columns = ["abs(log2(thetas))"]

    all = pd.concat([mylist_df, mylist_log2_df, mylist_abslog2_df], axis=1)
    print(all.describe())

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    # un-transformed
    ax1.hist(mylist)
    if val is None:
        ax1.axvline(
            x=0.5,
            color="r",
            linestyle="--",
            label="theta 0.5 at percentile "
            + str(round(get_prob_from_dist(mylist, 0.5), 2))
            + "%",
        )
        ax1.axvline(
            x=0.75,
            color="g",
            linestyle="--",
            label="theta 0.75 at percentile "
            + str(round(get_prob_from_dist(mylist, 0.75), 2))
            + "%",
        )
        ax1.axvline(
            x=1,
            color="b",
            linestyle="--",
            label="theta 1 at percentile "
            + str(round(get_prob_from_dist(mylist, 1), 2))
            + "%",
        )
    else:
        percentile = get_prob_from_dist(mylist, val)
        ax1.axvline(
            x=val,
            color="r",
            linestyle="--",
            label=str(val) + " at percentile " + str(round(percentile, 2)) + "%",
        )
    # ax1.set_ylim([0.5, 1.1])
    # ax1.set_title(str(cv) + " fold CV scores on test")
    ax1.set_xlabel("un-transformed thetas", fontsize=15)
    # ax1.set_ylabel("N")
    ax1.legend(loc="best", fontsize=12)

    # log2
    ax2.hist(mylist_log2)
    if val is None:
        ax2.axvline(
            x=np.log2(0.5),
            color="r",
            linestyle="--",
            label=str(round(np.log2(0.5), 2))
            + " at percentile "
            + str(round(get_prob_from_dist(mylist_log2, np.log2(0.5)), 2))
            + "%",
        )
        ax2.axvline(
            x=np.log2(0.75),
            color="g",
            linestyle="--",
            label=str(round(np.log2(0.75), 2))
            + " at percentile "
            + str(round(get_prob_from_dist(mylist_log2, np.log2(0.75)), 2))
            + "%",
        )
        ax2.axvline(
            x=np.log2(1),
            color="b",
            linestyle="--",
            label=str(round(np.log2(1), 2))
            + " at percentile "
            + str(round(get_prob_from_dist(mylist_log2, np.log2(1)), 2))
            + "%",
        )
    else:
        val_log2 = np.log2(val)
        percentile_log2 = get_prob_from_dist(mylist_log2, val_log2)
        ax2.axvline(
            x=val_log2,
            color="r",
            linestyle="--",
            label=str(val_log2)
            + " at percentile "
            + str(round(percentile_log2, 2))
            + "%",
        )
    ax2.set_xlabel("log2 transformed thetas", fontsize=15)
    ax2.legend(loc="best", fontsize=12)

    # abs log2
    if val is None:
        ax3.axvline(
            x=abs(np.log2(0.5)),
            color="r",
            linestyle="--",
            label=str(round(abs(np.log2(0.5)), 2))
            + " at percentile "
            + str(round(get_prob_from_dist(mylist_abslog2, abs(np.log2(0.5))), 2))
            + "%",
        )
        ax3.axvline(
            x=abs(np.log2(0.75)),
            color="g",
            linestyle="--",
            label=str(round(abs(np.log2(0.75)), 2))
            + " at percentile "
            + str(round(get_prob_from_dist(mylist_abslog2, abs(np.log2(0.75))), 2))
            + "%",
        )
        ax3.axvline(
            x=abs(np.log2(1)),
            color="b",
            linestyle="--",
            label=str(round(abs(np.log2(1)), 2))
            + " at percentile "
            + str(round(get_prob_from_dist(mylist_abslog2, abs(np.log2(1))), 2))
            + "%",
        )
    else:
        val_abslog2 = abs(val_log2)
        percentile_abslog2 = get_prob_from_dist(mylist_abslog2, val_abslog2)
        ax3.axvline(
            x=val_abslog2,
            color="r",
            linestyle="--",
            label=str(val_abslog2)
            + " at percentile "
            + str(round(percentile_abslog2, 2))
            + "%",
        )
    ax3.hist(mylist_abslog2)
    ax3.set_xlabel("abs log2 transformed thetas", fontsize=15)
    ax3.legend(loc="best", fontsize=12)

    plt.tight_layout()
    plt.show()


def get_prob_from_dist(list_posterior, val):
    myList = sorted(list_posterior)
    # Find indices where elements should be inserted to maintain order.
    return np.searchsorted(myList, val) / len(myList) * 100


def read_one_posteriors(DCC_path, theta_path, sample):
    path = DCC_path + "/output/RNAseq/1000Genome/" + str(sample) + theta_path
    file = open(path + "stan.pickle", "rb")
    object_file = pickle.load(file)
    file.close()
    return object_file


def get_prob_tuple(DCC_path, sample_list, cutoff_list, gene):
    print(">>>>>> Gene: {}".format(gene))
    abslog2_cutoff = [abs(np.log2(x)) for x in cutoff_list]
    print("un-transformed cutoff: ")
    print(cutoff_list)
    print("abs log2 transformed cutoff: ")
    print(abslog2_cutoff)
    sample_p = []
    for sample in sample_list:
        dictionary = read_one_posteriors(DCC_path, sample)
        prob_tuple = calculate_below_prob(dictionary, gene, abslog2_cutoff)
        if len(prob_tuple) == 0:
            continue
        sample_p.append(prob_tuple)
    print("print out tuple list for each ind")
    print(sample_p)
    dataframe = get_df(cutoff_list, sample_p)
    return dataframe


def get_gene_dict(DCC_path, sample_list, gene_list, cutoff_list, debug=False):
    debug_print = (lambda x: print(x)) if debug else (lambda x: None)
    abslog2_cutoff = [abs(np.log2(x)) for x in cutoff_list]

    debug_print(f">>>>>> thetas")
    debug_print("un-transformed cutoff: ")
    debug_print(cutoff_list)
    debug_print("abs log2 transformed cutoff: ")
    debug_print(abslog2_cutoff)

    gene_dict = {}
    for gene in gene_list:
        gene_dict[gene] = {}

    start_ns = time.time_ns()
    for i, sample in enumerate(sample_list):
        debug_print(f">>>>>> Sample: {sample}")
        gene_thetas_dict = read_one_posteriors(DCC_path, sample)
        for gene in gene_list:
            prob_tuple = calculate_below_prob(
                sample, gene_thetas_dict, gene, abslog2_cutoff
            )
            if len(prob_tuple) == 0:
                debug_print(f"sample {sample} does not have gene {gene}")
                continue
            gene_dict[gene][sample] = prob_tuple
        debug_print(f"completed sample {sample}")
        if (i + 1) % 10 == 0:
            now_ns = time.time_ns()
            print(
                f"finished {i+1}/{len(sample_list)} samples {(now_ns - start_ns) / 1e9}s"
            )
            start_ns = now_ns
    return gene_dict


def get_all_gene_dict(DCC_path, theta_path, sample_list, cutoff_list, debug=False):
    debug_print = (lambda x: print(x)) if debug else (lambda x: None)
    abslog2_cutoff = [abs(np.log2(x)) for x in cutoff_list]

    debug_print(f">>>>>> thetas")
    debug_print("un-transformed cutoff: ")
    debug_print(cutoff_list)
    debug_print("abs log2 transformed cutoff: ")
    debug_print(abslog2_cutoff)

    gene_dict = {}

    start_ns = time.time_ns()
    for i, sample in enumerate(sample_list):
        debug_print(f">>>>>> Sample: {sample}")
        gene_thetas_dict = read_one_posteriors(DCC_path, theta_path, sample)
        for gene, thetas in gene_thetas_dict.items():
            simplified_gene = gene.split(".", 1)[0]
            if simplified_gene not in gene_dict.keys():
                gene_dict[simplified_gene] = {}
            prob_tuple = calculate_below_prob_fromThetas(thetas, abslog2_cutoff)
            if len(prob_tuple) == 0:
                debug_print(f"sample {sample} does not have gene {gene}")
                continue
            gene_dict[simplified_gene][sample] = prob_tuple
        debug_print(f"completed sample {sample}")
        if (i + 1) % 10 == 0:
            now_ns = time.time_ns()
            print(
                f"finished {i+1}/{len(sample_list)} samples {(now_ns - start_ns) / 1e9}s"
            )
            start_ns = now_ns
    return gene_dict


def calculate_below_prob_fromThetas(thetas, cutoff_list):
    prob = ()
    thetas_len = len(thetas)
    thetas_abs_log2 = np.abs(np.log2(thetas))
    np.ndarray.sort(thetas_abs_log2)
    for cutoff in cutoff_list:
        p_i = np.searchsorted(thetas_abs_log2, cutoff) / thetas_len * 100
        prob = prob + (p_i / 100,)
    return prob


def calculate_below_prob(sample, gene_thetas_dict, gene, cutoff_list):
    prob = ()
    for k in gene_thetas_dict:
        if gene == k:
            thetas = gene_thetas_dict.get(k)
            if thetas == None:
                size = 0
            else:
                size = len(thetas)
                # print("{} - {} - {} - {}".format(sample,gene,k,size))
                prob = ()
                for cutoff in cutoff_list:
                    mylist_log2 = np.log2(thetas)
                    mylist_abslog2 = [abs(x) for x in mylist_log2]
                    p_i = get_prob_from_dist(mylist_abslog2, cutoff)
                    prob = prob + (p_i / 100,)
    return prob


def get_onegene_prob(sample_dict, cutoff_list, gene):
    print(">>>>>> Gene: {}".format(gene))
    abslog2_cutoff = [abs(np.log2(x)) for x in cutoff_list]
    print("un-transformed cutoff: ")
    print(cutoff_list)
    print("abs log2 transformed cutoff: ")
    print(abslog2_cutoff)
    sample_p = []
    sample_size = 0
    for sample in sample_dict:
        gene_thetas_dict = sample_dict[sample]
        prob_tuple = calculate_below_prob(
            sample, gene_thetas_dict, gene, abslog2_cutoff
        )
        if len(prob_tuple) == 0:
            continue
        else:
            sample_size += 1
        sample_p.append(prob_tuple)
    print(">>>>>> {} individuals have gene {}".format(sample_size, gene))
    dataframe = get_df(cutoff_list, sample_p)
    print(">>>>>> print out tuple list for each ind")
    print(sample_p)
    return dataframe

def get_probTuple_list_from_samples(all_gene_dict, geneID):
    sample_prob_dict = all_gene_dict[geneID]
    sample_size = 0
    sample_p = []
    for sample in sample_prob_dict:
        prob_tuple = sample_prob_dict[sample]
        if len(prob_tuple) == 0:
            continue
        else:
            sample_size += 1
        sample_p.append(prob_tuple)
    return sample_p, sample_size


def get_gene_annotation(path):
    conversion_annotation = pd.read_csv(path + "ensemble_HGNC.txt", sep="\t", header=0)
    conversion_annotation.columns = ["geneID", "Gene.name"]
    conversion_annotation2 = pd.read_csv(
        path + "manual_annotation_geneID.txt", sep=" ", header=0
    )
    conversion_annotation2["geneID"] = (
        conversion_annotation2["geneID"].str.split(".").str[0]
    )
    annotation = pd.concat([conversion_annotation, conversion_annotation2])
    annotation_uniq = annotation.dropna()
    annotation_uniq = annotation_uniq.groupby("geneID").first().reset_index()
    return annotation_uniq


def get_geneID_from_geneName(annotation, gene):
    if gene in annotation["Gene.name"].values:
        geneID = annotation[annotation["Gene.name"] == gene]["geneID"].values[0]
        return geneID
    else:
        return None


def find_avalGenes_from_list(gene_list, annotation,debug=False):
    debug_print = (lambda x: print(x)) if debug else (lambda x: None)
    geneName_dict = {}
    geneID_list = []
    for geneName in gene_list:
        geneID = get_geneID_from_geneName(annotation, geneName)
        if geneID is None:
            geneID = "NA"
        else:
            geneID_list.append(geneID)
        geneName_dict[geneName] = geneID
    df = pd.DataFrame(geneName_dict.items(), columns=["geneName", "geneID"])
    debug_print(df)
    return geneID_list


def get_geneset_prob(sample_dict, cutoff_list, gene_list):
    df = pd.DataFrame()
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    abslog2_cutoff = [abs(np.log2(x)) for x in cutoff_list]
    for gene in gene_list:
        sample_p = []
        sample_size = 0
        for sample in sample_dict:
            gene_thetas_dict = sample_dict[sample]
            prob_tuple = calculate_below_prob(
                sample, gene_thetas_dict, gene, abslog2_cutoff
            )
            if len(prob_tuple) == 0:
                continue
            else:
                sample_size += 1
            sample_p.append(prob_tuple)
        print(">>>>>> {} individuals have gene {}".format(sample_size, gene))
        if sample_size > 0:
            nobody_prob, atleastone_prob = get_prob_list(cutoff_list, sample_p)
            df1[gene] = nobody_prob
            df2[gene] = atleastone_prob
    df1["avg_nobody"] = df1.mean(axis=1)
    df2["avg_atleast_one"] = df2.mean(axis=1)
    df1["thetas"] = cutoff_list
    df2["thetas"] = cutoff_list
    df["thetas"] = cutoff_list
    df["abslog2_theta"] = abslog2_cutoff
    df["avg_nobody"] = df1["avg_nobody"]
    # df["avg_nobody_cu"]=df["avg_nobody"].cumsum()
    df["avg_atleast_one"] = df2["avg_atleast_one"]
    # df["avg_atleast_one_cu"]=df["avg_atleast_one"].cumsum()
    print(df1)
    print(df2)
    return df

def get_df(cutoff_list, prob_list):
    cutoff_prob = {}
    for i in range(len(cutoff_list)):
        below_prob_list = []
        for prob in prob_list:
            below_prob_list.append(prob[i])
        cutoff_prob[cutoff_list[i]] = np.prod(below_prob_list)
    df = pd.DataFrame(cutoff_prob.items(), columns=["thetas", "P_nobody_achieves"])
    df["P_atleast_one"] = 1 - df["P_nobody_achieves"]
    df["abslog2_thetas"] = abs(np.log2(df["thetas"]))
    df = df[["thetas", "abslog2_thetas", "P_atleast_one", "P_nobody_achieves"]]
    return df


def get_onegene_prob_genedict(
    GeneName, all_gene_dict, cutoff_list, annotation, prior,n_sample=445,geneID=None, debug=True
):
    if geneID is None:
        debug_print = (lambda x: print(x)) if debug else (lambda x: None)
        abslog2_cutoff = [abs(np.log2(x)) for x in cutoff_list]
        geneID = get_geneID_from_geneName(annotation, GeneName)
    if geneID is None:
        raise Exception(f"No annotation found for this gene name: {GeneName}")
    if geneID in all_gene_dict:
        sample_p, sample_size = get_probTuple_list_from_samples(all_gene_dict, geneID)
        debug_print(f">>>>>> {sample_size} individuals have gene {GeneName}")
        if sample_size > 0:
            nobody_prob, atleastone_prob = get_prob_list(cutoff_list, sample_p,prior, n_sample)
            df = pd.DataFrame(cutoff_list, columns=["thetas"])
            df["abslog2_thetas"] = abs(np.log2(df["thetas"]))
            df["P_nobody_achieves"] = nobody_prob
            df["P_atleast_one"] = 1 - df["P_nobody_achieves"]
            return df
    else:
        debug_print(f"this gene : {GeneName} cannot be found in input dict")

def get_geneset_prob_genedict(
    geneset_df, annotation, cutoff_list, all_gene_dict, prior, n_sample,debug=False, plot=False, title=None,
    allgenetable=None,lognormal=None
):
    debug_print = (lambda x: print(x)) if debug else (lambda x: None)
    abslog2_cutoff = [abs(np.log2(x)) for x in cutoff_list]
    if isinstance(geneset_df,list):
        if "ENSG" in geneset_df[0]:
            avalgene_list = geneset_df
        else:
            geneName_list = geneset_df
            avalgene_list = find_avalGenes_from_list(geneName_list, annotation, debug=debug)
    else:
        geneName_list = geneset_df["Gene.name"].tolist()
        avalgene_list = find_avalGenes_from_list(geneName_list, annotation, debug=debug)
    if len(avalgene_list) == 0:
        raise Exception(f"No gene in this geneset has annotation")
    if lognormal is not None:
        prior_nobody_prob, prior_atleastone_prob = get_prior_prob_list(abslog2_cutoff,lognormal)
    df = pd.DataFrame()
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()

    gene_n = {}

    if plot:
        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(13,6))
    meanprob_genedict={}
    for idx, geneID in enumerate(avalgene_list):
        for k in all_gene_dict.keys():
            if geneID in k:
                sample_p, sample_size = get_probTuple_list_from_samples(
                    all_gene_dict, k
                )
                debug_print(
                    f">>>>>> {sample_size} individuals have gene {geneID}"
                )
                if sample_size > 0:
                    nobody_prob, atleastone_prob = get_prob_list(cutoff_list, sample_p,prior, n_sample)
                    gene_n[geneID] = (sample_size,statistics.mean(nobody_prob),nobody_prob)
                    df1[geneID] = nobody_prob
                    df2[geneID] = atleastone_prob
                    if plot:
                        if len(avalgene_list)>5:
                            label=""
                        else:
                            label=geneID
                        ax1.plot(abslog2_cutoff,df1[geneID],"-o",fillstyle='none',alpha=0.5,label=label)
                        ax2.plot(abslog2_cutoff,df2[geneID],"-o",fillstyle='none',alpha=0.5,label=label)
            # else:
            #     debug_print(
            #         f"this gene : {geneName_list[idx]} cannot be found in input dict"
            #     )
    df1["avg_nobody"] = df1.mean(axis=1)
    df2["avg_atleast_one"] = df2.mean(axis=1)
    df["avg_nobody"] = df1["avg_nobody"]
    df["avg_atleast_one"] = df2["avg_atleast_one"]
    if lognormal is not None:
        df["prior_nobody"] = prior_nobody_prob
        df["prior_atleast_one"] = prior_atleastone_prob
    df1["thetas"] = cutoff_list
    df2["thetas"] = cutoff_list
    df["thetas"] = cutoff_list
    df["abslog2_theta"] = abslog2_cutoff
    if plot:
        if allgenetable is not None:
            ax1.plot(allgenetable["abslog2_theta"],allgenetable["avg_nobody"],"--ro",label="avg genome prob",linewidth=3)
            ax2.plot(allgenetable["abslog2_theta"],allgenetable["avg_atleast_one"],"--ro",label="avg genome prob",linewidth=3)
        ax1.plot(df["abslog2_theta"],df["avg_nobody"],"--bo",label="avg geneset prob",alpha=0.8,linewidth=3)
        ax2.plot(df["abslog2_theta"],df["avg_atleast_one"],"--bo",label="avg geneset prob",alpha=0.8,linewidth=3)
        if lognormal is not None:
            ax1.plot(df["abslog2_theta"],df["prior_nobody"],"black",label="prior",alpha=0.8,linewidth=3)
            ax1.scatter(df["abslog2_theta"],df["prior_nobody"],marker='o',color="black")
            ax2.plot(df["abslog2_theta"],df["prior_atleast_one"],"black",label="prior",alpha=0.8,linewidth=3)
            ax2.scatter(df["abslog2_theta"],df["prior_atleast_one"],marker='o',color="black")
        ax1.set_ylabel("CDF",fontsize=12)
        ax2.set_ylabel("CDF",fontsize=12)
        ax1.set_title("Nobody achieves certain ASE level",fontsize=15)
        ax2.set_title("At least one achieves certain ASE level",fontsize=15)
        ax1.legend(fontsize=11)
        ax1.set_xlim(-0.25,3.5)
        ax1.set_ylim(-0.05,1.05)
        ax2.set_xlim(-0.25,3.5)
        ax2.set_ylim(-0.05,1.05)
        fig.supxlabel("ASE level abs(log2(Θ))",fontsize=12)
        # 
        if title:
            plt.suptitle(title,fontsize=20)
    else:
        debug_print(df1)
        debug_print(df2)
    # convert dictionary to dataframe
    # Create empty lists for each column of the dataframe
    gene_ids = []
    n_samples = []
    means = []
    val_cols = [[] for i in range(10)]

    # Loop through the dictionary and extract the values for each column
    for gene, values in gene_n.items():
        gene_ids.append(gene)
        n_samples.append(values[0])
        means.append(values[1])
        for i, val in enumerate(values[2]):
            val_cols[i].append(val)

    # Create the dataframe
    gene_n_df = pd.DataFrame({"geneID": gene_ids,
                    "n_sample": n_samples,
                    "mean": means,
                    "prob1": val_cols[0],
                    "prob2": val_cols[1],
                    "prob3": val_cols[2],
                    "prob4": val_cols[3],
                    "prob5": val_cols[4],
                    "prob6": val_cols[5],
                    "prob7": val_cols[6],
                    "prob8": val_cols[7],
                    "prob9": val_cols[8],
                    "prob10": val_cols[9]})

    sorted_gene_n_df = gene_n_df.sort_values(by='mean', ascending=False)
    # display the resulting dataframe
    return df, sorted_gene_n_df

def get_prob_list(cutoff_list, prob_list, prior_list, n_sample):
    nobody_prob = []
    atleastone_prob = []
    for i in range(len(cutoff_list)):
        below_prob_list = []
        for prob in prob_list:
            below_prob_list.append(prob[i])
        nodata_prior = [prior_list[i]] * (n_sample - len(prob_list))
        below_prob_list.extend(nodata_prior)
        nobody_prob.append(np.prod(below_prob_list))
        atleastone_prob.append(1 - np.prod(below_prob_list))
    return nobody_prob, atleastone_prob


def plot_one_gene(iBEASTIE3_dict,iBEASTIE3_1M_dict,geneID,cutoff_list,annotation):
    iBEASTIE3 = get_onegene_prob_genedict(geneID,iBEASTIE3_dict,cutoff_list,annotation,debug=True)
    iBEASTIE3_1M = get_onegene_prob_genedict(geneID,iBEASTIE3_1M_dict,cutoff_list,annotation,debug=True)
    #iBEASTIE3_improper = get_onegene_prob_genedict(geneID,iBEASTIE3_improper_dict,cutoff_list,annotation,debug=True)
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(13,6))
    ax1.plot(iBEASTIE3["abslog2_thetas"],iBEASTIE3["P_nobody_achieves"],label="iBEASTIE3 sigma=0.5",color='blue')
    ax1.scatter(iBEASTIE3["abslog2_thetas"],iBEASTIE3["P_nobody_achieves"],marker='o',color='blue')
    ax1.plot(iBEASTIE3_1M["abslog2_thetas"],iBEASTIE3_1M["P_nobody_achieves"],label="iBEASTIE3 sigma=1million",color='green')
    ax1.scatter(iBEASTIE3_1M["abslog2_thetas"],iBEASTIE3_1M["P_nobody_achieves"],marker='o',color='green')
    #ax1.plot(iBEASTIE3_improper["abslog2_thetas"],iBEASTIE3_improper["P_nobody_achieves"],label="iBEASTIE3-improper")
    ax1.set_xlabel("abslog2_theta")
    ax1.set_ylabel("CDF (nobody achieves)",fontsize=12)
    ax1.legend(loc="upper left",fontsize=11)
    ax1.set_xlim(-0.25,3.5)
    ax1.set_ylim(-0.05,1.05)

    ax2.plot(iBEASTIE3["abslog2_thetas"],iBEASTIE3["P_atleast_one"],label="iBEASTIE3 sigma=0.5",color='blue')
    ax2.scatter(iBEASTIE3["abslog2_thetas"],iBEASTIE3["P_atleast_one"],marker='o',color='blue')
    ax2.plot(iBEASTIE3_1M["abslog2_thetas"],iBEASTIE3_1M["P_atleast_one"],label="iBEASTIE3 sigma=1million",color='green')
    ax2.scatter(iBEASTIE3_1M["abslog2_thetas"],iBEASTIE3_1M["P_atleast_one"],marker='o',color='green')
    #ax2.plot(iBEASTIE3_improper["abslog2_thetas"],iBEASTIE3_improper["P_atleast_one"],label="iBEASTIE3-improper")
    ax2.set_xlabel("abslog2_theta")
    ax2.set_ylabel("CDF (at least one achieves)",fontsize=12)
    ax2.legend(loc="lower left",fontsize=11)
    ax2.set_xlim(-0.25,3.5)
    ax2.set_ylim(-0.05,1.05)
    fig.suptitle(geneID,fontsize=20)
    fig.tight_layout()
    plt.show()


def get_prior_prob_list(cutoff_list,prob_list,n=445):

    below_prob = calculate_below_prob_fromThetas(prob_list, cutoff_list)
    nobody_prob = [i**n for i in below_prob]
    atleastone_prob = [1-i for i in nobody_prob]
    return nobody_prob, atleastone_prob

def get_prior_prob_genedict(
    lognormal, cutoff_list,n=445, title=None,allgenetable=None
):
    abslog2_cutoff = [abs(np.log2(x)) for x in cutoff_list]

    df = pd.DataFrame()
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(13,6))


    abslog2_cutoff = [abs(np.log2(x)) for x in cutoff_list]
    
    nobody_prob, atleastone_prob = get_prior_prob_list(abslog2_cutoff,lognormal,n)
    df["nobody"] = nobody_prob
    df["atleast_one"] = atleastone_prob
    df["thetas"] = cutoff_list
    df["abslog2_theta"] = abslog2_cutoff

    if allgenetable is not None:
        ax1.plot(allgenetable["abslog2_theta"],allgenetable["avg_nobody"],"--ro",label="avg genome prob",linewidth=3)
        ax2.plot(allgenetable["abslog2_theta"],allgenetable["avg_atleast_one"],"--ro",label="avg genome prob",linewidth=3)
    ax1.plot(df["abslog2_theta"],df["nobody"],"black",label="prior",alpha=0.8,linewidth=3)
    ax1.scatter(df["abslog2_theta"],df["nobody"],marker='o',color="black")
    ax2.plot(df["abslog2_theta"],df["atleast_one"],"black",label="prior",alpha=0.8,linewidth=3)
    ax2.scatter(df["abslog2_theta"],df["atleast_one"],marker='o',color="black")
    ax1.set_ylabel("CDF",fontsize=12)
    ax2.set_ylabel("CDF",fontsize=12)
    ax1.set_title("Nobody achieves certain ASE level",fontsize=15)
    ax2.set_title("At least one achieves certain ASE level",fontsize=15)
    ax1.legend(fontsize=11)
    ax1.set_xlim(-0.25,3.5)
    ax1.set_ylim(-0.05,1.05)
    ax2.set_xlim(-0.25,3.5)
    ax2.set_ylim(-0.05,1.05)
    fig.supxlabel("ASE level abs(log2(Θ))",fontsize=12)
    # 
    if title:
        plt.suptitle(title,fontsize=20)

    print(df)
    return df

