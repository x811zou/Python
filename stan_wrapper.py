#!/usr/bin/env python
# Script was updated on July 9th by Scarlett

from __future__ import (absolute_import, division, print_function,
   unicode_literals, generators, nested_scopes, with_statement)
from builtins import (bytes, dict, int, list, object, range, str, ascii,
   chr, hex, input, next, oct, open, pow, round, super, filter, map, zip)
import os
import tempfile
import sys
import math
import ProgramName
import TempFilename
import getopt
#from __future__ import print_function
from Pipe import Pipe
import pickle
from StanParser import StanParser
import numpy as np
import statistics
import os.path
# class RunStan:
#    def __init__(self, simulated_data_file):
#        self.simulated_data_file = simulated_data_file

#    def __enter__(self):


#    def __exit__(self ,type, value, traceback):


def writeInitializationFile(filename):
    OUT=open(filename,"wt")
    print("theta <- 1",file=OUT)
    OUT.close()

def writeReadCounts(fields,start,numReps,varName,OUT):
    print(varName,"<- c(",file=OUT,end="")
    for rep in range(numReps):
        print(fields[start+rep*2],file=OUT,end="")
        if(rep+1<numReps): print(",",file=OUT,end="")
    print(")",file=OUT)

def writeInputsFile(fields,filename,sigma):
    Mreps=int(fields[1])
    #filename='output.txt'
    OUT=open(filename,"wt")
    print("M <-",Mreps,file=OUT)
    writeReadCounts(fields,2,Mreps,"A",OUT) # alt
    writeReadCounts(fields,3,Mreps,"R",OUT) # ref
    print("sigma <-",sigma,file=OUT)
    OUT.close()

def getBaseline(fields):
    if(len(fields)>=5):
        base_thetas=[]
        Mreps=int(fields[1])
        for rep in range(Mreps):
            A = float(fields[2+rep*2])
            R = float(fields[3+(rep)*2])
            base = (A+1)/(R+1)
            #abs_base = abs(base-1)
            base_thetas.append(base)
        med_base_theta=statistics.median(base_thetas)
    true_theta = fields[-1]
    return med_base_theta, true_theta

def getFieldIndex(label,fields):
    numFields=len(fields)
    index=None
    for i in range(7,numFields):
        if(fields[i]==label): index=i
    return index


def getMaxProb(thetas):
    p_less = len([i for i in thetas if i <= 1])/len(thetas)
    p_more = 1 - p_less
    max_prob = max(p_less,p_more)
    #diff = [(x - 1)**2 for x in thetas]
    #rmse = np.sqrt(np.mean(diff))
    return max_prob

def runVariant(model,fields,input_file,tmp_output_file,stan_output_file,init_file,sigma):
    if(len(fields)>=5):
        writeInputsFile(fields,tmp_output_file,sigma)
        #writeInputsFile(fields,tmp_output_file)
        writeInitializationFile(init_file)
        cmd = "%s sample data file=%s init=%s output file=%s" % (model,tmp_output_file,init_file,stan_output_file) #/data/reddylab/scarlett/1000G/software/cmdstan/examples/ase/ase sample data file=/data/reddylab/scarlett/1000G/software/cmdstan/examples/ase/tmp_output.txt init=/data/reddylab/scarlett/1000G/software/cmdstan/examples/ase/initialization_stan.txt output file=/data/reddylab/scarlett/1000G/software/cmdstan/examples/ase/output_theta.txt      
        #cmd = "%s sample data file=%s output file=%s" % (model,tmp_output_file,stan_output_file)
        #print (cmd)
        os.system(cmd)# Parse MCMC output
        parser=StanParser(stan_output_file)
        thetas=parser.getVariable("theta")
        med,_,_,_,_ = parser.getSummary("theta")
        max_prob = getMaxProb(thetas)
        return thetas, med, max_prob
    else:
        return (None,None,None)


def getMedian(thetas):
    # Precondition: thetas is already sorted
    n=len(thetas)
    mid=int(n/2)
    if(n%2==0): return (thetas[mid-1]+thetas[mid])/2.0
    return thetas[mid]

def getCredibleInterval(thetas,alpha):
    halfAlpha=alpha/2.0
    n=len(thetas)
    leftIndex=int(halfAlpha*n)
    rightIndex=n-leftIndex
    left=thetas[leftIndex+1]
    right=thetas[rightIndex-1]
    return (left,right)

  
def wrappingProcess(source,File,inFile,sigma,para):
    in_path="/data/reddylab/scarlett/1000G/software/cmdstan/examples/ase" + "/" + File + "/simulated_data/"
    out_path="/data/reddylab/scarlett/1000G/software/cmdstan/examples/" + source + "/"+ File + "/output_pkl/"
    outfix=inFile.rsplit(".txt")[0]
    # check whether expected outfiles are already existed! 
    out1 = out_path+"model_theta/"+str(outfix)+"_s-"+str(sigma)+".pickle"
    out2 = out_path+"model_med/"+str(outfix)+"_s-"+str(sigma)+".pickle"
    out3 = out_path+"model_prob/"+str(outfix)+"_s-"+str(sigma)+".pickle"
    if (os.path.isfile(out1)) and (os.path.isfile(out2)) and (os.path.isfile(out3)):
        print ("Already Processed"+str(outfix)+"_s-"+str(sigma))
        os._exit(0)
    # model
    model = "/data/reddylab/scarlett/1000G/software/cmdstan/examples/" + source + "/" + source
    tmpFile= "tmp_output."+str(para)+".txt"
    initFile = "initialization_stan."+str(para)+".txt"
    outFile= "stan_output."+str(para)+".txt"
    input_file=in_path+inFile
    tmp_output_file=out_path+tmpFile
    init_file=out_path+initFile
    stan_output_file=out_path+outFile
    # parameters and create lists
        # ALPHA=0.05
    model_theta_list = []  # 150,000
    model_theta_med = []   # 150
    model_med_prob = []    # 150
    # start processing 
    with open(input_file,"rt") as IN:
        for line in IN:
            fields=line.rstrip().split()
            ID=fields[0]
            thetas,med,prob=runVariant(model,fields,input_file,tmp_output_file,stan_output_file,init_file,sigma)
            if thetas is not None:
                model_theta_list.extend(thetas)
                model_theta_med.append(med)
                model_med_prob.append(prob)
    # output 
    pickle.dump(model_theta_list,open(out_path+"model_theta/"+str(outfix)+"_s-"+str(sigma)+".pickle",'wb'))
    pickle.dump(model_theta_med,open(out_path+"model_med/"+str(outfix)+"_s-"+str(sigma)+".pickle",'wb'))
    pickle.dump(model_med_prob,open(out_path+"model_prob/"+str(outfix)+"_s-"+str(sigma)+".pickle",'wb'))
    print(str(outfix)+"_s-"+str(sigma) + " Done!")


if __name__ == "__main__":
    model = sys.argv[1]    # model="SPP2"
    File = sys.argv[2]     # File="new_simulation"
    inFile = sys.argv[3]   # inFile ="g-1000_h-5_d-5_t-1.txt"
    sigma = sys.argv[4]    # sigma =0.5
    para = sys.argv[5]     # para=0

    # call function 
    wrappingProcess(model,File,inFile,sigma,para)
