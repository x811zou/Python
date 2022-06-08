#!/usr/bin/env python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from scipy import stats
import bisect
import pickle
import statistics 


def report_hetsRatio(hetsDir,sample,if_Num=None,if_trans=False,if_print=False,if_1KGP=False):
    #if if_trans==True:
    #    filedir_t=hetsDir+'/'+sample+'/hetsDict/trans_all/'
    #else:
    #    filedir_t=hetsDir+'/'+sample+'/hetsDict/genom_all/'       
    filedir_t=hetsDir+'/NA12878/hetsDict/genom_all/'
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

# def report_hetsRatio(label,if_trans=False,if_print=False):
#     if if_trans==True:
#         filedir_t='/data/reddylab/scarlett/1000G/result/hetsDict_0715/trans/' + label +'/'
#     else:
#         filedir_t='/data/reddylab/scarlett/1000G/result/hetsDict_0715/genom/' + label +'/'
#     #for Num in range(22):
#     for Num in list(range(1,23))+['X','Y']:
#         with open(filedir_t +'chr'+str(Num)+'.pickle', 'rb') as handle:
#             translen_t = pickle.load(handle)
#             if Num ==1:
#                 translen0_t = translen_t
#             else:
#                 translen_t.update(translen0_t)
#                 translen0_t=translen_t
            
#     genes_w_hets = len(longest_trans_length(translen_t,region=None,include_zero=False))
#     all_genes = len(longest_trans_length(translen_t,region=None,include_zero=True))
#     ratio = genes_w_hets / all_genes
#     if if_print==True:
#         print("%d%% of genes with at least one het site (%d out of %d genes in chr1-22 from %s"%(round(ratio*100,2),genes_w_hets,all_genes,label))
#     return(translen_t)

def summaryOf_TransLength(data):
    print("Top 10 longest transcript length:")
    for i in range(10):
        print("\t Top %d : %d "%(i+1,longest_trans_length(data,region=None,include_zero=True).count(i)))

def summary_output(patient):
    path="/data/reddylab/scarlett/1000G/result/chrGeneHet_0715/"+patient+"/chr"
    over_all_df = []
    for i in list(range(1,23)):#+['X','Y']:
        #print(i)
        over_all_df.append(pd.read_csv(path + str(i) +".tsv",sep="\t",index_col= False))
        #over_all_df.append(pd.read_csv(path + str(i) +".tsv",sep="\t"))

    final_df = pd.concat(over_all_df)
    final_df = final_df.reset_index().drop(["index"], axis = 1)
    
    print("For patient: "+patient+"\n")
    print("Total {:-2} genes (in exon) in all 1-22 chromosomes, {:2.2%} of them have at least one heterozygous site".format(final_df.sum()[1],final_df.sum()[2]/final_df.sum()[1]))
    print("Total {:-2} bi-allelic SNPs (in exon) in all 1-22 chromosomes, {:2.2%} of them are heterozygous in this individual\n".format(final_df.sum()[4],final_df.sum()[5]/final_df.sum()[4]))


def sort_distance_het(dictionary,region=None, sort=True, include_zero=True):
    """ Calculate the distance between consecutive het sites within gene
    Args:

    Return: 
    """
    diff=[]
    for k in dictionary.keys():
        if include_zero != True:
                if sort==True:
                    diff.extend(np.ndarray.tolist(np.diff(np.array([int(x) for x in sorted(list(map(int, dictionary[k])))]))))
                else:
                    diff.extend(np.ndarray.tolist(abs(np.diff(np.array([int(x) for x in sorted(list(map(int, dictionary[k])))])))))
        else:
            if len(dictionary[k]) == 1:
                diff.append(0)
            if len(dictionary[k]) > 1:
                if sort==True:
                    diff.extend(np.ndarray.tolist(np.diff(np.array([int(x) for x in sorted(list(map(int, dictionary[k])))]))))
                else:
                    diff.extend(np.ndarray.tolist(abs(np.diff(np.array([int(x) for x in sorted(list(map(int, dictionary[k])))])))))
                
    data=diff
    if isinstance(region, int): # e.g. if restrcit to 1kb region
        diff.sort()
        ind = bisect.bisect(diff, region)
        diff_1kb  = diff[0:ind]
        data=diff_1kb
    else:
        print("no region limit")
    print("Max:",max(data),"Min:",min(data),"Median:",statistics.median(data),"Mean:",statistics.mean(data))
    return data



def longest_trans_length(dictionary,region,include_zero,if_print=False):
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
        print("The distribution of number of hets per gene:")
        print("\tMax:",max(data),", Min:",min(data),", Median:",np.median(data),", Mean:",round(np.mean(data),2))
    return data



def plot_histogram_cumulative(data,title="#het sites (only bi-allelic SNPs",
                              xlabel="# Het sites",cumu_label="HG00096 Chr1-22",save=False,plotTitle=False):
    """ Plot the histogram and cumulative distribution 
    Args:
    Return: 
    """ 
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(16, 8))
    graphs = (ax1, ax2)
    
    # histogram plot 
    graphs[0].grid(True)
    graphs[0].hist(data,bins=100)
    graphs[0].set_title("Distribution of "+title)

    # cumulative plot 
    n, bins, patches = graphs[1].hist(data, bins=50, density=True, histtype='step',cumulative=True, label=cumu_label)
    graphs[1].legend(loc='right')
    graphs[1].set_ylabel("Percentage")
    graphs[1].set_title("Cumulative step histograms for "+title)
    
    for i in range(2):
        graphs[i].grid(True)
        graphs[i].set_xlabel(xlabel)
        #graphs[i].set_xlim(left=0)
        
    if save == True:
        plt.savefig(savedir+plotTitle+"dist_btw_hetsite_hist.png")
    else:
        plt.show()
        plt.clf()

def guessN_processData(patient_number,min_counts=None,both_counts=None,stats=False,percentile=None):
    altRatio=[]
    tCount=[]
    rCount=[]
    aCount=[]
    medAltRatio=[]
    for Num in range(1,23):
        df=pd.read_csv('/data/reddylab/scarlett/1000G/result/altRatio/HG00096/tophat_V3'+'/chr'+str(Num)+'_wIntList.ratio',sep="\t",header=0,index_col=False)
        if len(df): 
            if isinstance(both_counts, int) and isinstance(min_counts, int):
                df2 = df[df['refCount'] >= int(both_counts)]
                df3 = df2[df2['altCount'] >= int(both_counts)]
                df4 = df3[df3['totalCount'] >= int(min_counts)]
                if percentile is not None:
                    pct = np.quantile(df4['totalCount'],percentile)
                    df4 = df4[df4['totalCount'] <= pct]
                df5 = df4[df4['if_SNP'] == "Y"]
                df6 = df5[df5['if_biallelic'] == "Y"]
                freq = df6['altCount']/df6['totalCount']
                data = report_hetsRatio(str(patient_number))
                df5['gene']=df5.apply(lambda x: FindKey(str(x['position']),data), axis = 1)
                df5_med = df5.groupby(['gene']).agg({"altRatio":'median'}).reset_index().rename(columns={'altRatio': 'median altRatio'})
                med_atr = np.ndarray.tolist(df5_med['median altRatio'].values)

        altRatio=altRatio+list(freq.values)
        tCount=tCount+list(df6['totalCount'].values)
        rCount=rCount+list(df6['refCount'].values)
        aCount=aCount+list(df6['altCount'].values)
        medAltRatio=medAltRatio+med_atr
    return(medAltRatio,altRatio,tCount,rCount,aCount)


def summary_statistics(data,title):
    print(title+' statistics:')
    if isinstance(data, list):
        print('  #: {:-2} \n  Max: {:-2} \n  Min: {:-2} \n  Mean: {:-2} \n  Median: {:-2} \n  Variance: {:-2} \n  Std: {:-2} \n  25% quantile: {:-2} \n  75% quantile: {:-2}  \n  75% quantile + IQR*1.5: {:-2} \n  90% quantile: {:-2}  \n  91% quantile: {:-2}  \n  95% quantile: {:-2} \n  98% quantile: {:-2} \n  99% quantile: {:-2}'.format(
            len(data),
            max(data),
            min(data),
            statistics.mean(data),
            statistics.median(data),
            statistics.variance(data),
            statistics.stdev(data),
            np.quantile(data,0.25),
            np.quantile(data,0.75),
            np.quantile(data,0.75)+(np.quantile(data,0.75)-np.quantile(data,0.25))*1.5,
            np.quantile(data,0.90),
            np.quantile(data,0.91),
            np.quantile(data,0.95),
            np.quantile(data,0.98),
            np.quantile(data,0.99)))
    else:
        print("Not a list")

if __name__ == "__main__":
    #filedir_t='/Users/scarlett/Desktop/Scarlett_Test/hetsDict/trans/'
    #filedir_g='/Users/scarlett/Desktop/Scarlett_Test/hetsDict/genom/'

    for Num in range(22):
        with open(filedir_g +'chr'+str(Num+1)+'.pickle', 'rb') as handle:
            translen_g = pickle.load(handle)
            if Num+1 ==1:
                translen0_g = translen_g
            else:
                translen_g.update(translen0_g)
                translen0_g =translen_g
