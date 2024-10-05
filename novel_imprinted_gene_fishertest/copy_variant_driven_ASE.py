from logging import raiseExceptions
import pandas as pd
import subprocess
import sys
import os
from pathlib import Path
from scipy.stats import hypergeom
import numpy as np
from scipy.stats import fisher_exact


def SNP_composition(data_dir, save_dir, geneID):
    if os.path.isfile(save_dir + "/" + geneID + "/SNP_composition.tsv"):
        df0 = pd.read_csv(
            save_dir + "/" + geneID + "/SNP_composition.tsv", sep="\t", header=0
        )
    else:
        print("No pre-existed SNP_composition information")
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
                df2_filtered = df2_filtered[df2_filtered["totalCount"] >= 30]
                if not df1_filtered.empty and not df2_filtered.empty:
                    chr = df1_filtered["chr"].iloc[0]
                    SNP_list = df1_filtered["pos"].tolist()
                    combined_SNPs = "_".join(map(str, SNP_list))
                else:
                    continue
                df2_filtered["combined_SNPs"] = combined_SNPs
                beastie_score = df2_filtered["posterior_mass_support_ALT"].iloc[0]
                df2_filtered["sample"] = sample
                if beastie_score > 0.5:
                    ASE_gene = "ASE"
                else:
                    ASE_gene = "no_ASE"
                df2_filtered["ASE gene"] = ASE_gene
                df2_filtered["chr"] = chr
                if counter == 0:
                    df0 = df2_filtered
                else:
                    df2_filtered.reset_index(drop=True, inplace=True)
                    df0.reset_index(drop=True, inplace=True)
                    df0 = pd.concat([df0, df2_filtered])
                counter += 1
        if counter > 0:
            Path(save_dir + "/" + geneID).mkdir(parents=True, exist_ok=True)
            df0.to_csv(save_dir + "/" + geneID + "/SNP_composition.tsv", sep="\t")
    if df0.shape[0] != 0:
        df0 = df0[
            [
                "geneID",
                "chr",
                "number.of.hets",
                "totalCount",
                "combined_SNPs",
                "sample",
                "ASE gene",
            ]
        ]
    else:
        print("No individual have this gene")
        sys.exit()
    return df0


def get_variant_combination(data, data_snp, data_snp_list, cutoff_nind, n_total_ind):
    print(
        "geneID\tchr\tSNPs_combination\ttotal_n\tn_SNP\tn_noSNP\tpval\tASE_SNP\tnoASE_SNP\tASE_noSNP\tnoASE_noSNP\n"
    )
    for pos in data_snp_list:
        if data_snp.to_dict()[pos] >= cutoff_nind:
            data_w_SNP = data[data["combined_SNPs"] == pos]
            data_wo_SNP = data[data["combined_SNPs"] != pos]
            n_have_SNP = len(pd.unique(data_w_SNP["sample"]))
            n_nothave_SNP = len(pd.unique(data_wo_SNP["sample"]))
            count = data_snp.to_dict()[pos]
            chrN = data["chr"].iloc[0]
            (
                p,
                ASE_have,
                noASE_have,
                ASE_nothave,
                noASE_nothave,
            ) = calculate_2sidedfisherexact_variant(data_w_SNP, data_wo_SNP)
            if p <= 0.1:
                info = [
                    [
                        gene_candidate,
                        chrN,
                        pos,
                        n_total_ind,
                        n_have_SNP,
                        n_nothave_SNP,
                        round(p, 2),
                        ASE_have,
                        noASE_have,
                        ASE_nothave,
                        noASE_nothave,
                    ]
                ]
                for r in info:
                    print("\t".join(map(str, r)))


def get_variant(data, data_snp, data_snp_list, cutoff_nind, n_total):
    print(
        "geneID\tchr\tSNP\tvariant\ttotal_n\tn_SNP\tn_noSNP\tpval\tASE_SNP\tnoASE_SNP\tASE_noSNP\tnoASE_noSNP\n"
    )

    for pos in data_snp_list:
        if data_snp.to_dict()[pos] >= cutoff_nind:
            ind_w_SNP = data[data["pos"] == pos]["sample"].tolist()
            data_w_SNP = data[data["sample"].isin(ind_w_SNP)]
            data_wo_SNP = data[~data["sample"].isin(ind_w_SNP)]
            n_have_SNP = len(pd.unique(data_w_SNP["sample"]))
            n_nothave_SNP = len(pd.unique(data_wo_SNP["sample"]))
            count = data_snp.to_dict()[pos]
            rsid = data_w_SNP["rsid"].iloc[0]
            chrN = data_w_SNP["chr"].iloc[0]
            (
                p,
                ASE_have,
                ASE_nothave,
                noASE_have,
                noASE_nothave,
            ) = calculate_2sidedfisherexact_variant(data_w_SNP, data_wo_SNP)
            if p <= 0.1:
                info = [
                    [
                        gene_candidate,
                        chrN,
                        pos,
                        rsid,
                        n_total,
                        n_have_SNP,
                        n_nothave_SNP,
                        round(p, 2),
                        ASE_have,
                        ASE_nothave,
                        noASE_have,
                        noASE_nothave,
                    ]
                ]
                for r in info:
                    print("\t".join(map(str, r)))


def collect_info_one_gene(data_dir, save_dir, geneID):
    if os.path.isfile(save_dir + "/" + geneID + "/allinfo.tsv"):
        df0 = pd.read_csv(save_dir + "/" + geneID + "/allinfo.tsv", sep="\t", header=0)
    else:
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
                df2_filtered = df2_filtered[df2_filtered["totalCount"] >= 30]
                if df1_filtered.shape[0] > 0 and df2_filtered.shape[0] > 0:
                    chr = df1_filtered["chr"].iloc[0]
                    beastie_score = df2_filtered["posterior_mass_support_ALT"].iloc[0]
                    df1_filtered["sample"] = sample
                    if beastie_score > 0.5:
                        ASE_gene = "ASE"
                    else:
                        ASE_gene = "no_ASE"
                    df1_filtered["ASE gene"] = ASE_gene
                    df1_filtered["compare_2allele"] = np.where(
                        df1_filtered["refCount"] > df1_filtered["altCount"],
                        "REF",
                        "ALT",
                    )
                    df1_filtered["chr"] = chr
                    if counter == 0:
                        df0 = df1_filtered
                    else:
                        df1_filtered.reset_index(drop=True, inplace=True)
                        df0.reset_index(drop=True, inplace=True)
                        df0 = pd.concat([df0, df1_filtered])
                    counter += 1
        Path(save_dir + "/" + geneID).mkdir(parents=True, exist_ok=True)
        df0.to_csv(save_dir + "/" + geneID + "/allinfo.tsv", sep="\t")

    df0 = df0[
        [
            "geneID",
            "chr",
            "pos",
            "rsid",
            "ref",
            "refCount",
            "alt",
            "altCount",
            "totalCount",
            "sample",
            "ASE gene",
            "compare_2allele",
        ]
    ]
    return df0


def collect_info_SNPs(save_dir, data, cutoff_totalcount):
    data = data[data["totalCount"] >= cutoff_totalcount]
    data_snp_list = data["pos"].value_counts().index.tolist()
    data_snp = data["pos"].value_counts()
    geneID = data["geneID"].iloc[0]
    data_snp.to_csv(save_dir + "/" + geneID + "/frequency_snps_table.tsv", sep="\t")
    return data_snp, data_snp_list


def collect_info_SNPs_combination(save_dir, data, cutoff_totalcount):
    data = data[data["totalCount"] >= cutoff_totalcount]
    data_snp_list = data["combined_SNPs"].value_counts().index.tolist()
    data_snp = data["combined_SNPs"].value_counts()
    geneID = data["geneID"].iloc[0]
    data_snp.to_csv(
        save_dir + "/" + geneID + "/frequency_snpscombination_table.tsv", sep="\t"
    )
    return data_snp, data_snp_list


def calculate_2sidedfisherexact_variant(data_w_SNP, data_wo_SNP):
    data_w_SNP = data_w_SNP[["sample", "ASE gene"]]
    data_w_SNP = data_w_SNP.drop_duplicates()
    data_wo_SNP = data_wo_SNP[["sample", "ASE gene"]]
    data_wo_SNP = data_wo_SNP.drop_duplicates()
    ASE_SNP = (data_w_SNP["ASE gene"] == "ASE").sum()
    ASE_noSNP = (data_wo_SNP["ASE gene"] == "ASE").sum()
    noASE_SNP = (data_w_SNP["ASE gene"] == "no_ASE").sum()
    noASE_noSNP = (data_wo_SNP["ASE gene"] == "no_ASE").sum()
    #                ASE   no ASE
    # have SNP       188     0
    # not have SNP    7     70
    table = np.array([[ASE_SNP, noASE_SNP], [ASE_noSNP, noASE_noSNP]])
    oddsr, p = fisher_exact(table, alternative="two-sided")
    return (
        p,
        ASE_SNP,
        noASE_SNP,
        ASE_noSNP,
        noASE_noSNP,
    )


# The probability that we would observe this or an even more imbalanced ratio by chance is about 3.5%. A commonly used significance level is 5%–if we adopt that, we can therefore conclude that our observed imbalance is statistically significant; whales prefer the Atlantic while sharks prefer the Indian ocean."
# python variant_driven_ASE.py ENSG00000164308.12

# main
data_dir = "/home/scarlett/data/1000Genome"
save_dir = "/home/scarlett/data/genes"
cutoff_nind = 1
cutoff_totalcount = 1
# genelist = pd.read_csv("/home/scarlett/data/genes/geneID_list.tsv", sep="\t", header=0)
# gene_list = genelist["geneID"].tolist()  # ["ENSG00000164308.12"]
gene_counter = 0
sig_counter = 0
snp_counter = 0
# output = pd.DataFrame()
# outputFilename = save_dir + "/significant_fisherexact.tsv"
# finished_genes = save_dir + "/finished_genelist.txt"
# if os.path.isfile(finished_genes):
#    finished_genelist = pd.read_csv(finished_genes, sep="\t", header=0)
#    finished_gene_list = finished_genelist["geneID"].tolist()
#    finished_genes = open(finished_genes, "a")
#    out_stream = open(outputFilename, "a")
# else:
#    finished_genes = open(finished_genes, "w")
#    finished_genes.write("geneID\n")
#    finished_genelist = []
#    out_stream = open(outputFilename, "w")

# python variant_driven_ASE.py ENSG00000177879.10
gene_candidate = sys.argv[1]

####################################### preparation work
data_option1 = SNP_composition(data_dir, save_dir, gene_candidate)
data_snp_option1, data_snp_list_option1 = collect_info_SNPs_combination(
    save_dir, data_option1, cutoff_totalcount
)
n_total_ind_option1 = len(pd.unique(data_option1["sample"]))
data_option2 = collect_info_one_gene(data_dir, save_dir, gene_candidate)
data_snp_option2, data_snp_list_option2 = collect_info_SNPs(
    save_dir, data_option2, cutoff_totalcount
)
n_total_ind_option2 = len(pd.unique(data_option2["sample"]))

if n_total_ind_option1 != n_total_ind_option2:
    raiseExceptions
n_total = n_total_ind_option1

print(">>>>>>>>>>")
print(
    f">>>>>>>>>> we only keep SNP(s) with >= {cutoff_totalcount} total counts, and shared in at least {cutoff_nind} individuals"
)
print(">>>>>>>>>> variant combination")
#######################################
####################################### option1: variant_combination
#######################################
get_variant_combination(
    data_option1,
    data_snp_option1,
    data_snp_list_option1,
    cutoff_nind,
    n_total,
)
print(">>>>>>>>>> single variant")
#######################################
####################################### option2: variant single
#######################################
get_variant(
    data_option2,
    data_snp_option2,
    data_snp_list_option2,
    cutoff_nind,
    n_total,
)
