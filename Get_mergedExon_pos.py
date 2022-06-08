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

def get_mergedExon_pos():
    """ Starting the loop for chr1-22 to obtain the dict {geneID:(het sites)}
    Before computing distances, first map the genomic coordinate of each variant to the transcript.  
    That way, introns won't be counted in the distances.
    Example code to subset the gff file: awk '{print > $1}' coding-and-noncoding.gff
    """
    savedir='/data/allenlab/scarlett/result/exon_pos/'
    vcfdir='/data/common/1000_genomes/VCF/20130502/bgzip/'
    refdir='/data/allenlab/scarlett/data/coding-noncoding/'
    outputFilename=str(savedir)+"mergedExon_pos.tsv"

    with open(outputFilename, "w") as out_stream:
        out_stream.write("chr\tgeneID\texon_start\texon_end\n")
        for Num in range(1,23):
            reader=GffTranscriptReader()
            print("we are working with chr" + str(Num))
            geneList=reader.loadGenes(refdir+"chr"+str(Num))
            print(len(geneList),"Genes loaded")  #439  genes for chr22
            for gene in geneList:
                transcript=gene.longestTranscript()
                geneID=transcript.getGeneId()
                chrom=transcript.getSubstrate()
                mergedExons=gene.getMergedExons()
                for exon in mergedExons:
                    begin=exon.begin
                    end=exon.end 
                    out_stream.write("\t".join([str(chrom),str(geneID),str(begin),str(end),"\n"]))
        out_stream.close()

if __name__ == "__main__":
    get_mergedExon_pos()
