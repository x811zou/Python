#=======================================================================================
# Run code as : python hetsMeta_for_mpileup_GSD.py 125249
# updated and saved in hardac on 07/05

##### make sure VCF file is zipped and has index: 
#module load vcftools/0.1.15-gcb01
#sample="383581"
#sample2="123667"
#outDir=/data/allenlab/scarlett/data/VCF/GSD/DNA_vcf
#vcftools --gzvcf /data/reddylab/GSD/kishnani_uwcmg_gsd_1.HF.final.vcf.gz --indv ${sample} --out $outDir/${sample2}.vcf --recode-INFO-all --recode
#bgzip -c ${sample2}.vcf.recode.vcf > ${sample2}.vcf.recode.vcf.gz
#tabix -p vcf ${sample2}.vcf.recode.vcf.gz


####### hetsMeta_GSD saves information for all hets by chr (Allchr_hets_all_transcript.tsv) for mpileup
####### hetsDict_GSD saves pickle file storing the gene:(pos set) for annotation (in altratio files)
#=======================================================================================

from __future__ import (absolute_import, division, print_function,
  unicode_literals, generators, nested_scopes, with_statement)
from builtins import (bytes, dict, int, list, object, range, str, ascii,
  chr, hex, input, next, oct, open, pow, round, super, filter, map, zip)
# The above imports should allow this program to run in both Python 2 and
# Python 3.  You might need to update your version of module "future".
#import ProgramName
#from GffTranscriptReader import GffTranscriptReader
#from Pipe import Pipe
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from scipy import stats
#from Gene import Gene
import sys
import time
import gzip
import os

""" Check if a genotype is heterozygous by testing whether them match with the 6 types of homozygous options
"""
def isHeterozygous(genotype):
    Homo = ["0|0","1|1","2|2","3|3","4|4","5|5","6|6","7|7","0/0","1/1","2/2"]
    if genotype in Homo:return False
    else:return True


""" Starting the loop for chr1-22 to obtain the dict {geneID:(het sites)}
    Before computing distances, first map the genomic coordinate of each variant to the transcript.  
    That way, introns won't be counted in the distances.
    Example code to subset the gff file: awk '{print > $1}' coding-and-noncoding.gff
"""


def count_longest_het_sites(sample,savedir,savedir2,filedir2,filedir3,prefix):
    print(">>> sample %s"%(sample))
    for file_name in os.listdir(filedir2):
        if sample+".vcf.recode.vcf.gz" == file_name:
            vcfFilename = filedir2+file_name
    print(vcfFilename)
    outputdir=prefix+"/result/hetsMeta_GSD/"+str(sample)
    ######
    #if str(sample) == "122687_2":
    #    vcfFilename="/data/reddylab/scarlett/1000G/data/VCF/GSD/"+str(sample)+".buffycoat.rnaseq.untrt.rep1.uniq.unmapped.variant_filtered.vcf.gz"
    #else:
    #    vcfFilename="/data/reddylab/scarlett/1000G/data/VCF/GSD/"+str(sample)+".buffycoat.rnaseq.untrt.rep1.unmapped.variant_filtered.vcf.gz"
    ######
    outputFilename=outputdir+"/Allchr_hets_all_transcript.tsv"
    out_stream = open(outputFilename, "w")
    out_stream.write("chr\tgeneID\tlongest_transID\ttransCoord_pos\tgenomicCoord_start\tgenomicCoord_end\tgenomicCoord_pos\tSNP_id\tinfo\tgenotype\n")
    print("- longest transcripts:")
    for Num in range(1,23):
        reader=GffTranscriptReader()
        print("we are working with chr" + str(Num))
        geneList=reader.loadGenes(filedir3+"chr"+str(Num))
        print(len(geneList),"Genes loaded")  #439  genes for chr22
        byGene={}
        byGene_trans={}
        total_biSNP = 0
        for gene in geneList:
            transcript=gene.longestTranscript()
            transID = transcript.getTranscriptId()# column 6
            geneID=transcript.getGeneId()      # column 5
            byGene[geneID]=byGene.get(geneID, set())
            byGene_trans[geneID]=byGene_trans.get(geneID, set())
            chrom=transcript.getSubstrate()      # column 1
            chromN=chrom.strip("chr")
            rawExons=transcript.getRawExons()
            for exon in rawExons:
                begin=exon.begin    # column 7
                end=exon.end      # column 8
                cmd = "tabix " + vcfFilename + " "+chromN+":"+str(begin)+"-"+str(end)#         
                #tabix /data/common/1000_genomes/VCF/20130502/bgzip/ALL.chr1.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz 1:10042358-10045556
                output=Pipe.run(cmd)
                #print(cmd)
                #print(output)
                if(not len(output)==0):
                    lines=output.split("\n")
                    for line in lines:
                        fields=line.split("\t")
                        if(fields[6]!="PASS"): continue
                        pos=fields[1]                         # column 9
                        #if(fields[2]=="."): continue
                        rs = fields[2]                        # column 10
                        genotype = fields[9].split(':')[0] # for HG00097
                        transcriptCoord=transcript.mapToTranscript(int(pos))#    
                        total_biSNP += 1 ###########################
                        #if if_HomoAlt == True:
                        #    if(not isHeterozygous_HomoAlt(genotype)): continue # go back to the begining of the loop
                        #else:
                        if(not isHeterozygous(genotype)): continue
                        byGene[geneID].add(pos)
                        byGene_trans[geneID].add(str(transcriptCoord))
                        out_stream.write("\t".join([str(chrom),str(geneID),str(transID),str(transcriptCoord),str(begin),str(end),str(pos),str(rs),str(genotype),"\n"]))
        # loop finishes here
        #  Done
        print("we finished with chr" + str(Num))
    #out_stream.close()
        with open(savedir+'chr'+str(Num)+'.pickle', 'wb') as handle:
            pickle.dump(byGene, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(savedir2+'chr'+str(Num)+'.pickle', 'wb') as handle:
            pickle.dump(byGene_trans, handle, protocol=pickle.HIGHEST_PROTOCOL)
    out_stream.close()

def count_all_het_sites(sample,savedir,savedir2,filedir2,filedir3,prefix):
    print(">>> sample %s"%(sample))
    for file_name in os.listdir(filedir2):
        if sample+".vcf.recode.vcf.gz" == file_name:
            vcfFilename = filedir2+file_name
    print(vcfFilename)

    outputdir=prefix+"/result/hetsMeta_GSD/"+str(sample)
    ######
    #if str(sample) == "122687_2":
    #    vcfFilename="/data/reddylab/scarlett/1000G/data/VCF/GSD/"+str(sample)+".buffycoat.rnaseq.untrt.rep1.uniq.unmapped.variant_filtered.vcf.gz"
    #else:
    #    vcfFilename="/data/reddylab/scarlett/1000G/data/VCF/GSD/"+str(sample)+".buffycoat.rnaseq.untrt.rep1.unmapped.variant_filtered.vcf.gz"
    ######
    outputFilename=outputdir+"/Allchr_hets_all_transcript.tsv"
    out_stream = open(outputFilename, "w")
    out_stream.write("chr\tgeneID\tlongest_transID\ttransCoord_pos\tgenomicCoord_start\tgenomicCoord_end\tgenomicCoord_pos\tSNP_id\tinfo\tgenotype\n")
    #
    #output2Filename="/data/reddylab/scarlett/1000G/result/chrGeneHet_GSD/all_chr.tsv"
    #out_stream2 = open(output2Filename, "w")
    #out_stream2.write("Chr\tTotal_gene\tGene_w_hets\tTotal_biSNP\tBi_hets\n")
  # iterate chr1-22
    print("- all transcripts:")
    for Num in range(1,23):
        reader=GffTranscriptReader()
        print("we are working with chr" + str(Num))
        geneList=reader.loadGenes(filedir3+"chr"+str(Num))
        print(len(geneList),"Genes loaded")  #439  genes for chr22
        byGene={}
        byGene_trans={}
        total_biSNP = 0
        for gene in geneList:
            transcript=gene.longestTranscript()
            transID = transcript.getTranscriptId()# column 6
            geneID=transcript.getGeneId()      # column 5
            byGene[geneID]=byGene.get(geneID, set())
            byGene_trans[geneID]=byGene_trans.get(geneID, set())
            chrom=transcript.getSubstrate()      # column 1
            chromN=chrom.strip("chr")
            rawExons=transcript.getRawExons()
            for exon in rawExons:
                # exon=rawExons[0]  ## for debugging only
                begin=exon.begin    # column 7
                end=exon.end      # column 8
                cmd = "tabix " + vcfFilename + " "+chromN+":"+str(begin)+"-"+str(end)#         
                #tabix /data/allenlab/scarlett/data/VCF/GSD/DNA_vcf/125249.vcf.recode.vcf.gz 1:10042358-10045556
                output=Pipe.run(cmd)
                if(not len(output)==0):
                    lines=output.split("\n")
                    for line in lines:
                        fields=line.split("\t")
                        if(fields[6]!="PASS"): continue
                        pos=fields[1]                         # column 9
                        if(fields[2]=="."): continue
                        rs = fields[2]                        # column 10
                        genotype = fields[9].split(':')[0] # for HG00097
                        transcriptCoord=transcript.mapToTranscript(int(pos))#    
                        total_biSNP += 1 ###########################
                        byGene[geneID].add(pos)
                        byGene_trans[geneID].add(str(transcriptCoord))
                        if(not isHeterozygous(genotype)): continue # go back to the begining of the loop
                        out_stream.write("\t".join([str(chromN),str(geneID),str(transID),str(transcriptCoord),str(begin),str(end),str(pos),str(rs),str(genotype),"\n"]))
        time.sleep(1/len(geneList))
        # write up the basic information 
        all_set_list = []
        non_empty_count = 0
        non_empty_count_len = 0
        for each_key in byGene:
            all_set_list.extend(list(byGene[each_key]))
            if len(byGene[each_key])!= 0:
                non_empty_count += 1
                non_empty_count_len += len(byGene[each_key])
    #    out_stream2.write("\t".join([str(Num),str(len(byGene)),str(non_empty_count),str(total_biSNP),str(len(all_set_list)),"\n"]))
    # loop finishes here
    #out_stream.close()
    #out_stream2.close()
    #  Done
        print("we finished with chr" + str(Num))
        with open(savedir+'chr'+str(Num)+'.pickle', 'wb') as handle:
            pickle.dump(byGene, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(savedir2+'chr'+str(Num)+'.pickle', 'wb') as handle:
            pickle.dump(byGene_trans, handle, protocol=pickle.HIGHEST_PROTOCOL)
    out_stream.close()

if __name__ == "__main__":
    sample = sys.argv[1]
    # There is only one VCF file for GSD case
    prefix="/data/allenlab/scarlett"
    #savedir='/data/allenlab/scarlett/result/hetsDict_GSD/genom_all/'+str(sample)+'/'
    #savedir2='/data/allenlab/scarlett/result/hetsDict_GSD/trans_all/'+str(sample)+'/'
    #filedir2='/data/allenlab/scarlett/data/VCF/GSD/'
    #filedir3='/data/allenlab/scarlett/data/coding-noncoding/'

    genom_dir = prefix+'/result/hetsDict_GSD/genom_all/'+str(sample)+'/'
    trans_dir = prefix+'/result/hetsDict_GSD/trans_all/'+str(sample)+'/'
    vcf_dir = prefix + '/data/VCF/GSD/DNA_vcf/'
    file_dir = prefix + '/data/coding-noncoding/'
    count_all_het_sites(sample,genom_dir,trans_dir,vcf_dir,file_dir,prefix)

    genom_dir = prefix+'/result/hetsDict_GSD/genom_longest/'+str(sample)+'/'
    trans_dir = prefix+'/result/hetsDict_GSD/trans_longest/'+str(sample)+'/'
    #count_longest_het_sites(sample,genom_dir,trans_dir,vcf_dir,file_dir,prefix)
