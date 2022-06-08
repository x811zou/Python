#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function,
  unicode_literals, generators, nested_scopes, with_statement)
from builtins import (bytes, dict, int, list, object, range, str, ascii,
  chr, hex, input, next, oct, open, pow, round, super, filter, map, zip)
# The above imports should allow this program to run in both Python 2 and
# Python 3.  You might need to update your version of module "future".
#import ProgramName
from GffTranscriptReader import GffTranscriptReader
from Pipe import Pipe
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from scipy import stats
from Gene import Gene
import sys
#import tqdm
import time
#from cyvcf2 import VCF
import gzip
from parse_gff import isHeterozygous


def count_AF(sample,patient):
  filedir3='/data/reddylab/scarlett/1000G/data/coding-noncoding/'


  with open(outputFilename, "w") as out_stream:
    out_stream.write("chr\tref_geneID\tquery_geneID\ttnovel_exon_start\tnovel_exon_end\tgenomicCoord_pos\tSNP_id\tinfo\tgenotype\tAF\n")
    for Num in range(1,23):
        reader=GffTranscriptReader()
        #print("we are working with chr" + str(Num))
        geneList=reader.loadGenes(filedir3+"chr"+str(Num))
        #print(len(geneList),"Genes loaded")  #439  genes for chr22
        byGene={}
        byGene_trans={}
        #for gene in tqdm.tqdm(geneList):

        vcfFilename=filedir2+"ALL.chr"+str(Num)+".phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz"

        total_row = int(Pipe.run("zcat "+vcfFilename+" |grep -E \"^[^#]\" | wc -l"))

        for gene in geneList:
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
                if(not len(output)==0):
                    lines=output.split("\n")
                    for line in lines:

                        total_var += 1
                        fields=line.split("\t")
                        #print (len(fields), line)
                        if(fields[6]!="PASS"): continue
                        pos=fields[1]                         # column 9
                        #begin_trans = transcript.mapToTranscript(begin)  # column 2
                        #end_trans = transcript.mapToTranscript(end)     # column 3
                        #pos_trans= transcript.mapToTranscript(int(pos))  # column 4
                        rs = fields[2]                        # column 10
                        #genotype = fields[9]  # FOR HG00096; CHANGE THIS LATER  # column
                        genotype = fields[patient] # for HG00097
                        transcriptCoord=transcript.mapToTranscript(int(pos))#    
                        info=fields[7].split('VT=')[1].split(';')[0]
                        if(info!='SNP'): continue
                        total_SNP += 1
                        if(not isHeterozygous(genotype)): continue # go back to the begining of the loop
                        #variantsInGene=byGene.get(geneID,None)
                        AF=fields[7].split('AF=')[1].split(';')[0]
                        byGene[geneID].add(pos)
                        byGene_trans[geneID].add(transcriptCoord)
                        out_stream.write("\t".join([str(chrom),str(geneID),str(transID),str(transcriptCoord),str(begin),str(end),str(pos),str(rs),str(info),str(genotype),str(AF),"\n"]))
        all_set_list = []
        non_empty_count = 0
        non_empty_count_len = 0
        for each_key in byGene:
            all_set_list.extend(list(byGene[each_key]))
            if len(byGene[each_key])!= 0:
                non_empty_count += 1
                non_empty_count_len += len(byGene[each_key])
        print("\t".join([str(chrom),str(len(byGene)),str(non_empty_count),str(total_var),str(total_SNP),str(len(all_set_list)),"\n"]))
    out_stream.close()
    #return byGene,byGene_trans
    with open(savedir+'wgs.pickle', 'wb') as handle:
        pickle.dump(byGene, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(savedir2+'wgs.pickle', 'wb') as handle:
        pickle.dump(byGene_trans, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
  samples=['HG00403','HG00404','HG00551','HG00553','HG02307','HG02308','HG03237','HG03238','HG00096','HG00097']
  for sample in samples:
  #sample=sys.argv[1]
  #i=sys.argv[2]
    header = pickle.load(open("/data/reddylab/scarlett/1000G/data/ASEreadCounter/vcf/vcf_header_list.plk","rb"))
    patient=header.index(str(sample))
    print("================================")
    print(">>>>>>>> Individual %s"%(sample))
   # check the transcript length for each gene for a specific chromosome
    count_AF(sample,patient)





cat transcripts.gtf | awk '{ if ($1=="chr16" && (($4 >= 30759591 && $4 <= 30772490)||($5 >= 30759591 && $5 <= 30772490))) { print } }' 