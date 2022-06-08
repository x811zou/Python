#!/usr/bin/env python
#=======================================================================================
# Run code as : python
#=======================================================================================
from __future__ import (absolute_import, division, print_function,
                        unicode_literals, generators, nested_scopes, with_statement)
from builtins import (bytes, dict, int, list, object, range, str, ascii,
                      chr, hex, input, next, oct, open, pow, round, super, filter, map, zip)
# The above imports should allow this program to run in both Python 2 and
# Python 3.  You might need to update your version of module "future".
#from GffTranscriptReader import GffTranscriptReader
import matplotlib.pyplot as plt
from itertools import count
from scipy import stats
#from Gene import Gene
#from Pipe import Pipe
import pandas as pd
import numpy as np
#import ProgramName
import subprocess
import argparse
import calendar
import tempfile
import pickle
import sys
import tqdm
import time
import gzip
import os
import pysam
import random
import statistics
import collections
import vcf
from cyvcf2 import VCF
import tqdm
from HARDAC_hetsMeta_for_mpileup_GSD import isHeterozygous
import seaborn as sns
import os.path
from os import path

#def FindKey(element,data):
#    for each in data:
#        if element in data[each]:
#            return each

def FindKey(chr,pos,data,if_list=False):
    element=str(chr)+"_"+str(pos)
    if if_list == True:
        listOfKeys = list()
        for k,v in data.items():
            if element in v:
                listOfKeys.append(k)
        return listOfKeys
    else:
        for each in data:
            if element in data[each]:
                return each

def getKeysByValue(dictOfElements, valueToFind):
    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item  in listOfItems:
        if item[1] == valueToFind:
            listOfKeys.append(item[0])
    return  listOfKeys

def if_alt(read,ref,allele):
    if read.lower() in allele.get(ref) or read.upper() in allele.get(ref):
        return True
    else:
        return False

def ref_alt_dict(REF,ALT):
    allele={}
    allele[REF] = allele.get(REF, set())
    for i in ALT:
        allele[REF].add(str(i))
    return(allele)


def count_raw_depth(reads):
    qualified_reads=[]
    i=0
    for x in reads:
        if (x != ">" and x != "<"):
            i=i+1
            qualified_reads.append(x)
    return(i)
def save_parsed_mpileup_allChr(sample,prefix,gz_dir,pileup_dir,min_cov=0,if_write = True,if_1KGP=False):
    # sample ="HG00096"
    #prefix="/data/allenlab/scarlett"
    #gz_dir="/data/VCF/1KGP/VCF"
    #pileup_dir=prefix+"/data/PileupOutput/1KGP/S6_M10"

    inputPileup = None
    inputGZ = None

    for file_name in os.listdir(prefix+gz_dir):
        if if_1KGP == True:
            if str(sample)+".vcf.gz" == file_name:
                inputGZ = prefix+gz_dir+"/"+file_name
        else:
            if sample+".vcf.recode.vcf.gz" == file_name:
                inputGZ = gz_dir+"/"+file_name
    for file_name in os.listdir(pileup_dir+"/"+str(sample)):
        if sample in file_name:
            if ".pileup" in file_name:
                inputPileup = pileup_dir+"/"+str(sample)+"/"+file_name
  
    #raise ValueError("inputPileup Not found")

    pileup_out=inputPileup
    output     = inputPileup.replace(".pileup", ".altratio")
    if str(path.exists(output)):
        print("Output altratio file exists!")
        #raise ValueError("Output altratio file exists!")
    #out_path = "/data/reddylab/scarlett/1000G/result/Pileup_counts_for_model/GSD/"+str(sample)+"/"
    stream_in = open(pileup_out, "r")
    if if_write != True:
        print("contig,position,variantID,refAllele,refCount,altAllele,altCount,totalCount,altRatio,if_Indel,if_SV,if_SNP,if_biallelic,lowMAPQDepth,lowBaseQDepth,rawDepth,otherCount\n")
    else:
        out_stream = open(output, "w")
        out_stream.write("contig\tposition\tvariantID\trefAllele\trefCount\taltAllele\taltCount\ttotalCount\taltRatio\tif_Indel\tif_SV\tif_SNP\tif_biallelic\tlowMAPQDepth\tlowBaseQDepth\trawDepth\totherCount\n")
    pipeup_dict = {}

    with open(pileup_out, "r") as stream_in:
        for i, line in enumerate(stream_in):             # start reading in my pileup results line by line
            chr_info = line.split("\t")[0].strip("chr")
            start_info = line.split("\t")[1]
            key_info = chr_info+"-"+start_info
            #print("we look at this variant: %s"%key_info) # we look at this variant: 16-30761285
            pipeup_dict[key_info] = line                  # save key: [1] value: line
        start_timestamp = calendar.timegm(time.gmtime())
        counter = 0
        hets = 0
        snp = 0
        indels=0
        bi = 0
        vcf = VCF(inputGZ)
        vcf.set_samples([sample])
        for record in vcf:
            counter += 1
            if counter % 1000000 == 0:
                print("%d processed." % counter)
            columns = str(record).strip("\n").split("\t") # parse it because cyvcf2 does not support extracting the genotype]
            vcf_key = str(record.CHROM)+"-"+str(record.start+1)
            pipeup_line = pipeup_dict.get(vcf_key)
            if not pipeup_line:
                continue
            #print(pipeup_line) #chr16	30761285	c	25	.,.,-1a.-1A,.,,-1a...-1A.-1A..,-1a,-1a.-1A,-1a.-1A.-1A,,,-1a,-1a	?????????????????????????	]]]]]]]]]]]]]]]]]]]]]]]]]
            if not isHeterozygous(columns[-1]):
                continue
            chrNum,start,rsid,ref,ref_count,alt,alt_count,total_count,ratio,if_Indel,if_SV,if_SNP,if_biallelic,low_mapg,low_baseq,raw_depth,other_count,hets,indels,snp,counter,bi,i = GATK_ParseMpileup(record,hets,indels,snp,counter,bi,i,pipeup_dict,min_cov)
            #print(",".join([chrNum,start,rsid,ref,ref_count,alt,alt_count,total_count,ratio,if_Indel,if_SV,if_SNP,if_biallelic,low_mapg,low_baseq,raw_depth,other_count,"\n"]))
            if if_write == True:
                out_stream.write("\t".join([chrNum,start,rsid,ref,ref_count,alt,alt_count,total_count,ratio,if_Indel,if_SV,if_SNP,if_biallelic,low_mapg,low_baseq,raw_depth,other_count,"\n"]))
            else:
                print(",".join([chrNum,start,rsid,ref,ref_count,alt,alt_count,total_count,ratio,if_Indel,low_mapg,low_baseq,raw_depth,other_count,"\n"]))
    if if_write == True:
        out_stream.close()
        stream_in.close()

    snp_ratio=snp/hets
    no_indel_ratio=1-indels/hets
    bi_ratio = bi/hets
    stop_timestamp = calendar.timegm(time.gmtime())
    print(str(sample))
    print("Chr" + str(chrNum)+":")
    print("\tPileup dimension: " + str(len(pipeup_dict)))
    print("\tVCF dimension: " + str(counter))
    print("\tHet sites found with total counts above " + str(min_cov) + ": "+ str(hets))
    print("\t\t"+str(hets-indels) +" without Indels "+"("+str(round(no_indel_ratio*100,2))+"%"+")")
    print("\t\t"+str(bi) +" bi-allelic sites "+"("+str(round(bi_ratio*100,2))+"%"+")")
    print("\t\t"+str(snp) +" only bi-allelic SNP "+"("+str(round(snp_ratio*100,2))+"%"+")")
    print("\tTotal time to complete: %d seconds"%(stop_timestamp-start_timestamp))
    df=pd.read_csv(output,sep="\t",header=0,index_col=False)
    df = df.iloc[:,0:9]
    if sample == "122687_merged":
        sample1 = "122687"
    else:
        sample1 = sample
    #df['gene']=df.apply(lambda x: FindKey(['position'],anno_dict), axis = 1)

    #anno_dict =report_hetsRatio(prefix,sample1,if_Num=str(chrNum),if_trans=False,if_print=False,if_1KGP=if_1KGP)
    #df['gene']=df.apply(lambda x: getKeysByValue(['position'],anno_dict), axis = 1)
    print("Parsed pileup reads file saved to %s"%(output))
    return df



def save_parsed_mpileup_byChr(sample,chrNum,prefix,gz_dir,pileup_dir,min_cov=0,if_write = True,if_1KGP=False):
    #sample = "125249"
    #prefix="/Users/scarlett/Desktop/HARDAC/scarlett"
    #gz_dir=prefix+"/data/VCF/GSD/DNA_vcf"
    #pileup_dir=prefix+"/data/PileupOutput/GSD/given_aligned_bam/target_gene_hets"

    # sample ="HG00096"
    #prefix="/data/allenlab/scarlett"
    #gz_dir="/data/common/1000_genomes/VCF/20130502/bgzip"
    #pileup_dir=prefix+"/data/PileupOutput/1KGP/S6_M10/bychr"

    inputPileup = None
    inputGZ = None

    for file_name in os.listdir(gz_dir):
        if if_1KGP == True:
            if "ALL.chr"+str(chrNum)+".phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz" == file_name:
                inputGZ = gz_dir+"/"+file_name
        else:
            if sample+".vcf.recode.vcf.gz" == file_name:
                inputGZ = gz_dir+"/"+file_name
    for file_name in os.listdir(pileup_dir):
        if sample in file_name:
            if ".pileup" in file_name:
                inputPileup = pileup_dir+"/"+file_name
  
    #raise ValueError("inputPileup Not found")

    pileup_out=inputPileup
    output     = inputPileup.replace(".pileup", ".altratio")
    if str(path.exists(output)):
        print("Output altratio file exists!")
        #raise ValueError("Output altratio file exists!")
    #out_path = "/data/reddylab/scarlett/1000G/result/Pileup_counts_for_model/GSD/"+str(sample)+"/"
    stream_in = open(pileup_out, "r")
    if if_write != True:
        print("contig,position,variantID,refAllele,refCount,altAllele,altCount,totalCount,altRatio,if_Indel,if_SV,if_SNP,if_biallelic,lowMAPQDepth,lowBaseQDepth,rawDepth,otherCount\n")
    else:
        out_stream = open(output, "w")
        out_stream.write("contig\tposition\tvariantID\trefAllele\trefCount\taltAllele\taltCount\ttotalCount\taltRatio\tif_Indel\tif_SV\tif_SNP\tif_biallelic\tlowMAPQDepth\tlowBaseQDepth\trawDepth\totherCount\n")
    pipeup_dict = {}

    with open(pileup_out, "r") as stream_in:
        for i, line in enumerate(stream_in):             # start reading in my pileup results line by line
            chr_info = line.split("\t")[0].strip("chr")
            start_info = line.split("\t")[1]
            key_info = chr_info+"-"+start_info
            #print("we look at this variant: %s"%key_info) # we look at this variant: 16-30761285
            pipeup_dict[key_info] = line                  # save key: [1] value: line
        start_timestamp = calendar.timegm(time.gmtime())
        counter = 0
        hets = 0
        snp = 0
        indels=0
        bi = 0
        vcf = VCF(inputGZ)
        vcf.set_samples([sample])
        for record in vcf:
            counter += 1
            if counter % 1000000 == 0:
                print("%d processed." % counter)
            columns = str(record).strip("\n").split("\t") # parse it because cyvcf2 does not support extracting the genotype]
            vcf_key = str(record.CHROM)+"-"+str(record.start+1)
            pipeup_line = pipeup_dict.get(vcf_key)
            if not pipeup_line:
                continue
            #print(pipeup_line) #chr16	30761285	c	25	.,.,-1a.-1A,.,,-1a...-1A.-1A..,-1a,-1a.-1A,-1a.-1A.-1A,,,-1a,-1a	?????????????????????????	]]]]]]]]]]]]]]]]]]]]]]]]]
            if not isHeterozygous(columns[-1]):
                continue
            chrNum,start,rsid,ref,ref_count,alt,alt_count,total_count,ratio,if_Indel,if_SV,if_SNP,if_biallelic,low_mapg,low_baseq,raw_depth,other_count,hets,indels,snp,counter,bi,i = GATK_ParseMpileup(record,hets,indels,snp,counter,bi,i,pipeup_dict,min_cov)
            print(",".join([chrNum,start,rsid,ref,ref_count,alt,alt_count,total_count,ratio,if_Indel,if_SV,if_SNP,if_biallelic,low_mapg,low_baseq,raw_depth,other_count,"\n"]))
            if if_write == True:
                out_stream.write("\t".join([chrNum,start,rsid,ref,ref_count,alt,alt_count,total_count,ratio,if_Indel,if_SV,if_SNP,if_biallelic,low_mapg,low_baseq,raw_depth,other_count,"\n"]))
            else:
                print(",".join([chrNum,start,rsid,ref,ref_count,alt,alt_count,total_count,ratio,if_Indel,low_mapg,low_baseq,raw_depth,other_count,"\n"]))
    if if_write == True:
        out_stream.close()
        stream_in.close()

    snp_ratio=snp/hets
    no_indel_ratio=1-indels/hets
    bi_ratio = bi/hets
    stop_timestamp = calendar.timegm(time.gmtime())
    print(str(sample))
    print("Chr" + str(chrNum)+":")
    print("\tPileup dimension: " + str(len(pipeup_dict)))
    print("\tVCF dimension: " + str(counter))
    print("\tHet sites found with total counts above " + str(min_cov) + ": "+ str(hets))
    print("\t\t"+str(hets-indels) +" without Indels "+"("+str(round(no_indel_ratio*100,2))+"%"+")")
    print("\t\t"+str(bi) +" bi-allelic sites "+"("+str(round(bi_ratio*100,2))+"%"+")")
    print("\t\t"+str(snp) +" only bi-allelic SNP "+"("+str(round(snp_ratio*100,2))+"%"+")")
    print("\tTotal time to complete: %d seconds"%(stop_timestamp-start_timestamp))
    df=pd.read_csv(output,sep="\t",header=0,index_col=False)
    df = df.iloc[:,0:9]
    if sample == "122687_merged":
        sample1 = "122687"
    else:
        sample1 = sample
    #df['gene']=df.apply(lambda x: FindKey(['position'],anno_dict), axis = 1)

    #anno_dict =report_hetsRatio(prefix,sample1,if_Num=str(chrNum),if_trans=False,if_print=False,if_1KGP=if_1KGP)
    #df['gene']=df.apply(lambda x: getKeysByValue(['position'],anno_dict), axis = 1)
    print("Parsed pileup reads file saved to %s"%(output))
    return df

def report_hetsRatio(prefix,label,if_Num=None,if_trans=False,if_print=False,if_1KGP=False):
    if if_1KGP == True: 
        if if_trans==True:
            filedir_t=prefix+'/result/hetsDict_0715/trans/'+str(label)+'/'
        else:
            filedir_t=prefix+'/result/hetsDict_0715/genom/'+str(label)+'/'       
    else:
        if if_trans==True:
            filedir_t=prefix+'/result/hetsDict_GSD/trans_all/'+str(label)+'/'
        else:
            filedir_t=prefix+'/result/hetsDict_GSD/genom_all/'+str(label)+'/'
    if if_Num != None:
        with open(filedir_t +'chr'+str(if_Num)+'.pickle', 'rb') as handle:
                translen_t = pickle.load(handle)
    else: 
        for Num in list(range(1,23)):#+['X','Y']:
            with open(filedir_t +'chr'+str(if_Num)+'.pickle', 'rb') as handle:
                translen_t = pickle.load(handle)
                if Num ==1:
                    translen0_t = translen_t
                else:
                    translen_t.update(translen0_t)
                    translen0_t=translen_t

    genes_w_hets = len(longest_trans_length(translen_t,region=None,include_zero=False))
    all_genes = len(longest_trans_length(translen_t,region=None,include_zero=True))
    ratio = genes_w_hets / all_genes
    if if_print==True:
        print("%d%% of genes with at least one het site (%d out of %d genes in chr1-22 from %s"%(round(ratio*100,2),genes_w_hets,all_genes,label))
    return(translen_t)


def output_modelCount(gene_df,out=None):
    #print(output_modelCount(df,out=out_path+str(sample)+"_targetHets.txt"))
    unique_gene = gene_df['gene'].unique()
    rst = "" 
    for each in unique_gene:
        idx = each
        gene_lst = [idx, sum(gene_df["gene"] == each)] + list(np.ravel(gene_df[gene_df["gene"] == each][["altCount","refCount"]].values))
        rst += "\t".join([str(x) for x in gene_lst]) +"\n"
    if out is not None: 
        file1 = open(out,"w")
        file1.write(rst) 
        file1.close() 
    return rst

def output_modelCount_V2(gene_df,target_gene,out=None):
    # force the gene ID to be the target Gene
    gene_df['gene']=str(target_gene)
    unique_gene = gene_df['gene'].unique()
    rst = ""
    for each in unique_gene:
        idx = each
        gene_lst = [idx, sum(gene_df["gene"] == each)] + list(np.ravel(gene_df[gene_df["gene"] == each][["altCount","refCount"]].values))
        rst += "\t".join([str(x) for x in gene_lst]) +"\n"
    if out is not None:
        file1 = open(out,"w")
        file1.write(rst)
        file1.close()
    return rst

def GATK_ParseMpileup(record,hets,indels,snp,counter,bi,i,pipeup_dict,min_cov):
    ################################ updated on 07/05/2020
    columns = str(record).strip("\n").split("\t") 
    # parse it because cyvcf2 does not support extracting the genotype]
    ref = ""
    alt = ""
    rsid = "."  
    # read in the dict of pileup result because it is relatively small file 
    vcf_key = str(record.CHROM)+"-"+str(record.start+1)
    pipeup_line = pipeup_dict.get(vcf_key).strip("\n").split("\t")
    alleles = ref_alt_dict(record.REF,record.ALT)
    ref = ", ".join([key for key in alleles.keys()])
    alt = ", ".join([", ".join(value) for value in alleles.values()])
    length_key = len(alleles[str(ref)])
    rsid=columns[2]
    if ref != "" and alt != "":
        #print(list(pipeup_line[4]))
        reads = list(pipeup_line[4])                  # read bases
        reads = [x for x in reads if (x != "$" and x != "^" and x != "~" and x != "\"" and x != "!")] 
        # remove this non-sense symbols
        block = 0
        out_reads = []
        for read in reads:
            if block == -1:
                i=i+1
                if ord(read) >= 48 and ord(read) <= 57: # up to 9 insertion/deletion
                    block_str += read
                else:
                    if block_str !="":
                        block = int(block_str) - 1
            elif read == "+" or read == "-":
                block = -1
                block_str = ""
            elif block > 0:
                i=i+1
                block -= 1
            elif block == 0:
                out_reads.append(read)
        reads = out_reads
        baseqs = list(pipeup_line[5])          # base quality
        mapqs = list(pipeup_line[6].strip())   # mapping quality
        low_baseq = 0
        low_mapq = 0
        ref_count = 0
        alt_count = 0
        other_count = 0
        raw_depth = count_raw_depth(reads) # number of qualified reads 
        
        #print("read: "+str(out_reads))
        for read,baseq,mapq in zip(reads,baseqs,mapqs):
            if read != "<" and read != ">":
                # D-68; E-69,F-70; H-72; J-74
                basequal = ord(baseq)-33
                mapqual = ord(mapq)-33 
                if basequal >= 0 and mapqual >= 0: #min_mapq=93
                    # count this base
                    if read == "." or read == ",":
                        #print("ref: "+str(read))
                        ref_count += 1
                    elif if_alt(read,record.REF,alleles):
                        #print("alt: "+str(read))
                        alt_count += 1
                    else:
                        other_count += 1
                        #print("other: "+str(read))
                if basequal < 0:
                    low_baseq += 1
                if mapqual < 0:
                    low_mapq += 1
                    
        totalCount = ref_count+alt_count
        if totalCount>0:
            ratio=alt_count/totalCount
        else:
            ratio=0
        if totalCount >= min_cov:
            hets+=1
            if length_key == 1:
                if_biallelic="Y"
                bi+=1
            else:
                if_biallelic="N"
            if record.is_indel:
                if_Indel="Y"
                indels+=1
            else:
                if_Indel="N"
            if record.is_sv:
                if_sv="Y"
            else:
                if_sv="N"
            if record.is_snp:
                if_snp="Y"
                snp+=1
            else:
                if_snp="N"
        return(columns[0],columns[1],rsid,str(ref),str(ref_count),str(alt),str(alt_count),str(totalCount),str(ratio),if_Indel,if_sv,if_snp,if_biallelic,str(low_mapq),str(low_baseq),str(raw_depth),str(other_count),hets,indels,snp,counter,bi,i)
    raise ValueError("Empty content")



def output_modelCount_V2(gene_df,target_gene,out=None):
    # force the gene ID to be the target Gene
    gene_df['gene']=str(target_gene)
    unique_gene = gene_df['gene'].unique()
    rst = ""
    for each in unique_gene:
        idx = each
        gene_lst = [idx, sum(gene_df["gene"] == each)] + list(np.ravel(gene_df[gene_df["gene"] == each][["altCount","refCount"]].values))
        rst += "\t".join([str(x) for x in gene_lst]) +"\n"
    if out is not None:
        file1 = open(out,"w")
        file1.write(rst)
        file1.close()
    return rst