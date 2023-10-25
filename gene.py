#!/bin/python

import pandas as pd
import subprocess
import sys
import os
from pathlib import Path
from scipy.stats import hypergeom
import numpy as np
from scipy.stats import fisher_exact


def collect_info_one_gene(data_dir, save_dir, geneID):
    subfolders = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    counter = 0
    df0 = pd.DataFrame()
    for sub_dir in subfolders:
        sample = os.path.basename(sub_dir)
        filename1 = (
            sub_dir
            + "/beastie/beastie_SNPs_even_100/beastie_shapeit2/chr1-22_alignBiasp0.05_ase0.5_s0.5_a0.05_sinCov0_totCov1_W1000K1000/tmp/"
            + sample
            + "_real_alignBiasafterFilter.phasedByshapeit2.cleaned.tsv"
        )
        filename2 = (
            sub_dir
            + "/beastie/beastie_SNPs_even_100/beastie_shapeit2/chr1-22_alignBiasp0.05_ase0.5_s0.5_a0.05_sinCov0_totCov1_W1000K1000/result/"
            + sample
            + "_ASE_sub.tsv"
        )
        if os.path.isfile(filename1) and os.path.isfile(filename2):
            df1 = pd.read_csv(filename1, sep="\t", header=0)
            df2 = pd.read_csv(filename2, sep="\t", header=0)
            df1_filtered = df1[df1["geneID"] == geneID]
            df2_filtered = df2[df2["geneID"] == geneID]
            if df1_filtered.shape[0] > 0 and df2_filtered.shape[0] > 0:
                beastie_score = df2_filtered["posterior_mass_support_ALT"].iloc[0]
                df1_filtered["sample"] = sample
                if beastie_score > 0.5:
                    ASE_gene = "ASE"
                else:
                    ASE_gene = "no ASE"
                df1_filtered["ASE gene"] = ASE_gene
                df1_filtered["compare_2allele"] = np.where(
                    df1_filtered["refCount"] > df1_filtered["altCount"], "REF", "ALT"
                )
                df1_filtered["compare_2allele"] = df1_filtered[
                    "compare_2allele"
                ].astype("category")
                if counter == 0:
                    df0 = df1_filtered
                else:
                    df1_filtered.reset_index(drop=True, inplace=True)
                    df0.reset_index(drop=True, inplace=True)
                    df0 = pd.concat([df0, df1_filtered])
                counter += 1
    Path(save_dir + "/" + geneID).mkdir(parents=True, exist_ok=True)
    df0.to_csv(save_dir + "/" + geneID + "/allinfo.tsv", sep="\t")
    return df0


def collect_info_SNPs(save_dir, data):
    data_snp_list = data["pos"].value_counts().index.tolist()
    data_snp = data["pos"].value_counts()
    geneID = data["geneID"].iloc[0]
    data_snp.to_csv(save_dir + "/" + geneID + "/SNPs_frequency_table.tsv", sep="\t")
    return data_snp, data_snp_list


def calculate_2sidedfisherexact_variant(variant, data):
    data_filtered = data[data["pos"] == variant]
    data_filtered.groupby(["ASE gene"])["compare_2allele"].value_counts().reindex(
        data_filtered.compare_2allele.unique(), fill_value=0
    )
    # ASE gene  compare_2allele
    # ASE       ALT                14
    #          REF                13
    # no ASE    ALT                12
    #          REF                 8
    ASE_REF = (
        data_filtered[data_filtered["ASE gene"] == "ASE"]["compare_2allele"] == "REF"
    ).sum()
    ASE_ALT = (
        data_filtered[data_filtered["ASE gene"] == "ASE"]["compare_2allele"] == "ALT"
    ).sum()
    noASE_REF = (
        data_filtered[data_filtered["ASE gene"] == "no ASE"]["compare_2allele"] == "REF"
    ).sum()
    noASE_ALT = (
        data_filtered[data_filtered["ASE gene"] == "no ASE"]["compare_2allele"] == "ALT"
    ).sum()
    #           ASE   no ASE\n",
    # REF>ALT    13      8\n",
    # REF<=ALT   14      12\n",
    table = np.array([[ASE_REF, noASE_REF], [ASE_ALT, noASE_ALT]])
    oddsr, p = fisher_exact(table, alternative="two-sided")
    return p, ASE_REF, noASE_REF, ASE_ALT, noASE_ALT


# The probability that we would observe this or an even more imbalanced ratio by chance is about 3.5%. A commonly used significance level is 5%–if we adopt that, we can therefore conclude that our observed imbalance is statistically significant; whales prefer the Atlantic while sharks prefer the Indian ocean."


# main
data_dir = "/home/scarlett/data/1000Genome"
save_dir = "/home/scarlett/data/genes"
cutoff = 10
genelist = pd.read_csv("/home/scarlett/data/genes/geneID_list.tsv", sep="\t", header=0)
gene_list = genelist["geneID"].tolist()  # ["ENSG00000164308.12"]
gene_counter = 0
sig_counter = 0
snp_counter = 0
output = pd.DataFrame()
outputFilename = save_dir + "/significant_fisherexact.tsv"
finished_genes = save_dir + "/finished_genelist.txt"
if os.path.isfile(finished_genes):
    finished_genelist = pd.read_csv(finished_genes, sep="\t", header=0)
    finished_gene_list = finished_genelist["geneID"].tolist()
    finished_genes = open(finished_genes, "a")
    out_stream = open(outputFilename, "a")
else:
    finished_genes = open(finished_genes, "w")
    finished_genes.write("geneID\n")
    finished_genelist = []
    out_stream = open(outputFilename, "w")
    out_stream.write(
        "geneID\tchr\tpos\tn_individual\tpval\tASE_REF\tnoASE_REF\tASE_ALT\tnoASE_ALT\n"
    )

for gene_candidate in gene_list:
    if gene_candidate in finished_gene_list:
        continue
    gene_counter += 1
    data = collect_info_one_gene(data_dir, save_dir, gene_candidate)
    if data.empty:
        continue
    data_snp, data_snp_list = collect_info_SNPs(save_dir, data)
    for pos in data_snp_list:
        if data_snp.to_dict()[pos] >= cutoff:
            count = data_snp.to_dict()[pos]
            chrN = data["chr"].iloc[0]
            (
                p,
                ASE_REF,
                noASE_REF,
                ASE_ALT,
                noASE_ALT,
            ) = calculate_2sidedfisherexact_variant(pos, data)
            if p <= 0.05:
                info = [
                    [
                        gene_candidate,
                        chrN,
                        pos,
                        count,
                        p,
                        ASE_REF,
                        noASE_REF,
                        ASE_ALT,
                        noASE_ALT,
                    ]
                ]
                for r in info:
                    out_stream.write("\t".join(map(str, r)))
                    out_stream.write("\n")
    finished_genes.write(gene_candidate)
    finished_genes.write("\n")
out_stream.close()
finished_genes.close()
