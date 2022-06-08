import pickle
import statistics
import numpy as np
from math import log
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from astroML.density_estimation import EmpiricalDistribution
from scipy.stats import percentileofscore

def longest_trans_length(dictionary,region,include_zero=False,if_print=False):
    """ The length of the longest transcript by summing the length of raw exons
    
    Args:

    Return: 
    """
    len_trans=[]
    for k in dictionary.keys():
        if include_zero != True:
            if (len(dictionary[k]) >0) :
                len_trans.append(int(len(dictionary[k])))
        else:
            if (len(dictionary[k]) >=0) :
                len_trans.append(int(len(dictionary[k])))   
    data=len_trans
    #if isinstance(region, int): # e.g. if restrcit to 1kb region
    #    len_trans.sort()
    #    ind = bisect.bisect(len_trans, region)
    #    len_trans_1kb  = len_trans[0:ind]
    #    data=len_trans_1kb
    if if_print == True:
        print("Total genes %s"%len(hets))
        if include_zero==True:
            print(">> Include genes with 0 hets")
        else:
            print(">> Did not include genes with 0 hets")
            
        print("The distribution of number of hets per gene:")

        print("\tMax:",max(data),", Min:",min(data),", Median:",np.median(data),", Mean:",round(np.mean(data),2))
    return data

def sample_inverseCDF(data,n,plot=False,title=None,print=True):
    #if title != None:
        #print("The distribution of %s:"%title)
    #if print == True:
    #    print("\tMax:",max(data),", Min:",min(data),", Median:",np.median(data),", Mean:",round(np.mean(data),2),", Variance:",round(np.var(data),2))

    x_cloned = EmpiricalDistribution(data).rvs(n)
    if plot != False:
        sns.set(font_scale=2)  
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(ncols=1, figsize=(8,6))
        sns.distplot(x_cloned, hist = False, kde = True,kde_kws = {'linewidth': 3},label = 'inverse cdf')
        sns.distplot(data, hist = False, kde = True,kde_kws = {'linewidth': 3},label = 'real data')
        plt.legend(loc="upper right",fontsize=17)
        if title != None:
            plt.title(title,fontsize=26) 
    if n == 1:
        x_cloned = np.ndarray.tolist(x_cloned)[0]
    return x_cloned

def guessN_processData_96(prefix,patient_number,min_totalcounts=None,min_bothcounts=None,stats=False,percentile=None,if_1KGP=True):
    inputDir=prefix+"/data/PileupOutput/1KGP/S6_M10"
    altRatio=[]
    tCount=[]
    rCount=[]
    aCount=[]
    medAltRatio=[]
    #for Num in range(1,23):
    df=pd.read_csv(inputDir+"/"+str(patient_number)+"/HG00096_wgs_all.altratio",sep="\t",header=0,index_col=False)
    if len(df): 
        if isinstance(min_bothcounts, int) and isinstance(min_totalcounts, int):
            df2 = df[df['refCount'] >= int(min_bothcounts)]
            df3 = df2[df2['altCount'] >= int(min_bothcounts)]
            df4 = df3[df3['totalCount'] >= int(min_totalcounts)]
            if percentile is not None:
                pct = np.quantile(df4['totalCount'],percentile)
                df4 = df4[df4['totalCount'] <= pct]
            df5 = df4[df4['if_SNP'] == "Y"]
            df6 = df5[df5['if_biallelic'] == "Y"]
            freq = df6['altCount']/df6['totalCount']
            data = report_hetsRatio(prefix,str(patient_number),if_Num=None,if_trans=False,if_print=False,if_1KGP=if_1KGP)
            df6['gene']=df6.apply(lambda x: FindKey(str(x['position']),data,if_list=False), axis = 1)
            df6_med = df6.groupby(['gene']).agg({"altRatio":'median'}).reset_index().rename(columns={'altRatio': 'median altRatio'})
            med_atr = np.ndarray.tolist(df6_med['median altRatio'].values)

    altRatio=altRatio+list(freq.values)
    tCount=tCount+list(df6['totalCount'].values)
    rCount=rCount+list(df6['refCount'].values)
    aCount=aCount+list(df6['altCount'].values)
    medAltRatio=medAltRatio+med_atr
    return(medAltRatio,altRatio,tCount,rCount,aCount)

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
            with open(filedir_t +'chr'+str(Num)+'.pickle', 'rb') as handle:
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


def FindKey(element,data,if_list=False):
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

def print_simulation_1KGP(prefix,Num_gene,sample,min_totalcounts,min_bothcounts,if_1KGP=True):
    print("We are now simulating "+str(Num_gene)+" genes using distribution extracted from HG00096 with filtering threshold at each allele with min counts :"+str(min_bothcounts) +" and total counts :" +str(min_totalcounts))
    N= Num_gene # number of gene
    data_96 = report_hetsRatio(prefix,"HG00096",if_Num=None,if_trans=False,if_print=False,if_1KGP=True)
    hets=longest_trans_length(data_96,region=None,include_zero=False,if_print=False)
    medAltRatio,altRatio,tCount,rCount,aCount=guessN_processData_96(prefix,str(sample),min_totalcounts=min_totalcounts,min_bothcounts=min_bothcounts,stats=False,percentile=None,if_1KGP=True)
    for k in range(N):
        # sample a number of hets for one gene
        M = round(sample_inverseCDF(hets,1,print=False))
        print('Number of hets: {:-2}'.format(M))
        # sample theta for one gene
        p = sample_inverseCDF(medAltRatio,1,print=False)
        print('Sampled Theta: {:-2}'.format(M))
        AR=[]
        for i in range(M):
            # sample a read depth for a het
            D = round(sample_inverseCDF(tCount,1,print=False))  
            print('Sampled Read depth: {:-2}'.format(D))
            A=np.random.binomial(D, p, size=1)
            if i ==0:
                AR_0 = "%d\t%d" % (A, D-A)
            else:
                AR = "%d\t%d" % (A, D-A)
                AR_0 = '%s\t%s'%(AR_0,AR)
        line = '%d\t%d\t%s\t%s\n' % (k+1,M, AR_0, p/(1-p))
        print(line)

def write_simulation(prefix,Num_gene,sample,min_totalcounts,min_bothcounts,out1,out2,if_1KGP=True):
    print("We are now simulating "+str(Num_gene)+" genes using distribution extracted from "+str(sample)+" with filtering threshold at each allele with min counts :"+str(min_bothcounts) +" and total counts :" +str(min_totalcounts))
    N= Num_gene # number of gene
    data_96 = report_hetsRatio(prefix,sample,if_Num=None,if_trans=False,if_print=False,if_1KGP=True)
    hets=longest_trans_length(data_96,region=None,include_zero=False,if_print=False)
    medAltRatio,altRatio,tCount,rCount,aCount=guessN_processData_96(prefix,str(sample),min_totalcounts=min_totalcounts,min_bothcounts=min_bothcounts,stats=False,percentile=None,if_1KGP=True)
    line1 = ""
    line2 = ""
    p0 = 0.5
    for k in range(N):
        # sample a number of hets for one gene
        M = round(sample_inverseCDF(hets,1,print=False))
        #print('Number of hets: {:-2}'.format(M))
        # sample theta for one gene
        p = sample_inverseCDF(medAltRatio,1,print=False)
        #print('Sampled Theta: {:-2}'.format(M))
        AR=[]
        for i in range(M):
            # sample a read depth for a het
            D = round(sample_inverseCDF(tCount,1,print=False))  
            #print('Sampled Read depth: {:-2}'.format(D))
            A=np.random.binomial(D, p, size=1)
            A0=np.random.binomial(D, p0, size=1)
            if i ==0:
                AR_1st = "%d\t%d" % (A, D-A)
                AR0_1st = "%d\t%d" % (A0, D-A0)
            else:
                AR = "%d\t%d" % (A, D-A)
                AR0 = "%d\t%d" % (A0, D-A0)
                AR_1st = '%s\t%s'%(AR_1st,AR)
                AR0_1st = '%s\t%s'%(AR0_1st,AR0)
        gene="gene"+ str(k+1)
        line1 += '%s\t%d\t%s\t%s\n' % (gene, M, AR_1st, p/(1-p))
        line2 += '%s\t%d\t%s\t%s\n' % (gene, M, AR0_1st, p0/(1-p0))
    if out1 is not None:
        file1 = open(out1,"w")
        file1.write(line1)
        file1.close()
    if out2 is not None:
        file2 = open(out2,"w")
        file2.write(line2)
        file2.close()
    return line1,line2

def print_simulation_GSD(prefix,Num_gene,sample,min_totalcounts,min_bothcounts,if_1KGP=False):
    print("We are now simulating "+str(Num_gene)+" genes using distribution extracted from "+str(sample)+" with filtering threshold at each allele with min counts :"+str(min_bothcounts) +" and total counts :" +str(min_totalcounts))
    N= Num_gene # number of gene
    data_GSD = report_hetsRatio(prefix,str(sample),if_Num=None,if_trans=False,if_print=False,if_1KGP=if_1KGP)
    hets=longest_trans_length(data_GSD,region=None,include_zero=False,if_print=False)
    medAltRatio,altRatio,tCount,rCount,aCount=guessN_processData_96(prefix,str(sample),min_totalcounts=min_totalcounts,min_bothcounts=min_bothcounts,stats=False,percentile=None,if_1KGP=True)
    for k in range(N):
        # sample a number of hets for one gene
        M = round(sample_inverseCDF(hets,1,print=False))
        print('Number of hets: {:-2}'.format(M))
        # sample theta for one gene
        p = sample_inverseCDF(medAltRatio,1,print=False)
        print('Sampled Theta: {:-2}'.format(M))
        AR=[]
        for i in range(M):
            # sample a read depth for a het
            D = round(sample_inverseCDF(tCount,1,print=False))  
            print('Sampled Read depth: {:-2}'.format(D))
            A=np.random.binomial(D, p, size=1)
            if i ==0:
                AR_0 = "%d\t%d" % (A, D-A)
            else:
                AR = "%d\t%d" % (A, D-A)
                AR_0 = '%s\t%s'%(AR_0,AR)
        line = '%d\t%d\t%s\t%s\n' % (k+1,M, AR_0, p/(1-p))
        print(line)
