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
from scipy.stats import percentileofscore

def extract_pLI(df, gene_symbol):
    if gene_symbol not in df['gene'].values:
        print("This gene could not be found in the database.")
    else:
        # extract the pLI value for a specific gene symbol (e.g., 'BRCA1')
        pLI_value = df.loc[df['gene'] == gene_symbol, 'pLI'].values[0]
        if not pd.isna(pLI_value):
            # calculate the percentile of the pLI value in the DataFrame
            percentile = percentileofscore(df['pLI'].dropna(), pLI_value)
            # print the pLI value
            print(f"The pLI value for {gene_symbol} is {pLI_value} at the {percentile:.2f}th percentile")
        else:
            print(f"The pLI value for {gene_symbol} is missing in the database.")
            
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


def get_geneset_genes(geneset, new_dict):
    gene_ensemble = geneset["geneID"].values
    gene_display = geneset["Gene.name"].values
    thetas = []
    for gene in gene_ensemble:
        if gene in new_dict:
            thetas.extend(new_dict.get(gene))
    return thetas, gene_display


def get_imprinted_geneset(path, annotation):
    innerjoin = annotation[
        (annotation["Gene.name"] == "SNRPN") | (annotation["Gene.name"] == "PEG10")
    ]
    print(
        "imprinted geneset has {} genes, {} with proper annotation".format(
            0, len(innerjoin)
        )
    )
    return innerjoin


def get_RVIS_haploinsufficient_geneset(geneset_path, annotation):
    file_path = geneset_path + "/OMIM_Haploinsufficiency.txt"
    df = pd.read_csv(file_path, sep="\t")
    df.columns = ["Gene.name"]
    df_uniq = df.groupby("Gene.name").first().reset_index()
    innerjoin = pd.merge(df_uniq, annotation, on="Gene.name")
    innerjoin_uniq = innerjoin.groupby("Gene.name").first().reset_index()
    print(
        "OMIM haploinsufficient geneset has {} genes, {} with proper annotation".format(
            len(df_uniq), len(innerjoin_uniq)
        )
    )
    return innerjoin_uniq


def get_lof_tolerant_geneset(geneset_path, annotation):
    file_path = geneset_path + "/homozygous_lof_tolerant_twohit.tsv"
    df = pd.read_csv(file_path, sep="\t")
    df.columns = ["Gene.name"]
    df_uniq = df.groupby("Gene.name").first().reset_index()
    innerjoin = pd.merge(df_uniq, annotation, on="Gene.name")
    innerjoin_uniq = innerjoin.groupby("Gene.name").first().reset_index()
    print(
        "lof geneset has {} genes, {} with proper annotation".format(
            len(df_uniq), len(innerjoin_uniq)
        )
    )
    return innerjoin_uniq


def get_mgi_geneset(geneset_path, annotation):
    file_path = geneset_path + "/mgi_essential.tsv"
    df = pd.read_csv(file_path, sep="\t")
    df.columns = ["Gene.name"]
    df_uniq = df.groupby("Gene.name").first().reset_index()
    innerjoin = pd.merge(df_uniq, annotation, on="Gene.name")
    innerjoin_uniq = innerjoin.groupby("Gene.name").first().reset_index()
    print(
        "mgi essential geneset has {} genes, {} with proper annotation".format(
            len(df_uniq), len(innerjoin_uniq)
        )
    )
    return innerjoin_uniq


def get_RVIS_recessive_geneset(geneset_path, annotation):
    file_path = geneset_path + "/OMIM_recessive.txt"
    df = pd.read_csv(file_path, sep="\t")
    df.columns = ["Gene.name"]
    df_uniq = df.groupby("Gene.name").first().reset_index()
    innerjoin = pd.merge(df_uniq, annotation, on="Gene.name")
    innerjoin_uniq = innerjoin.groupby("Gene.name").first().reset_index()
    print(
        "OMIM recessive geneset has {} genes, {} with proper annotation".format(
            len(df_uniq), len(innerjoin_uniq)
        )
    )
    return innerjoin_uniq


def get_still_birth_geneset(geneset_path, annotation):
    file_path = geneset_path + "/stillbirth_candidates.txt"
    df = pd.read_csv(file_path, sep="\t")
    df.columns = ["Gene.name"]
    df_uniq = df.groupby("Gene.name").first().reset_index()
    innerjoin = pd.merge(df_uniq, annotation, on="Gene.name")
    innerjoin_uniq = innerjoin.groupby("Gene.name").first().reset_index()
    print(
        "still birth geneset has {} genes, {} with proper annotation".format(
            len(df_uniq), len(innerjoin_uniq)
        )
    )
    return innerjoin_uniq
