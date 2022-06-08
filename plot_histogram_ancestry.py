#!/usr/bin/env python

def get_data(sample_folder,folder,min_total):
    # 1. beofre alignment bias filtering
    file_path1="/hpc/group/allenlab/scarlett/output/RNAseq/1000Genome/"+sample_folder+"/beastie_SNPs_even/"+folder+"/chr1-22_s0.5_a0.05_sinCov0_totCov1_W1000K1000/result/"
    filename1=sample_folder+"_ASE_all.tsv"
    df=pd.read_csv(file_path1+filename1,sep='\t',header=0)
    df_filtered = df[df["totalCount"]>=min_total]
    return df_filtered

def return_key(sample_ancestry_dict,val):
    sample_list=[]
    for key, value in sample_ancestry_dict.items():
        if value==val:
            sample_list.append(key)
    return(sample_list)


def plot_col_allAncestry(sample_color_dict,sample_ancestry_dict,folder,min_total,col,fontsize=12,vline=None,xlabel=None,title=None,ancestry=None):
    sample_selected = sample_ancestry_dict
    ancestry_chosen=''
    if ancestry is not None:
        sample_selected = return_key(sample_ancestry_dict,ancestry)
        ancestry_chosen=ancestry
    #
    fig = plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    some_list= []
    log2=''
    x_appended=[]
    for sample in sample_selected:
        df_filtered=get_data(sample,folder,min_total)
        x=df_filtered[col].tolist()
        x_appended.extend(x)
        if col=="posterior_median":
            thetas_log2 = [math.log2(s) for s in x]
            x=thetas_log2
            log2='log2 '
        bins = get_bins_num(x)
        plt.hist(x, color=sample_color_dict[sample_ancestry_dict[str(sample)]],density=False,alpha=0.5,bins=bins,label = sample_ancestry_dict[sample] if sample_color_dict[sample_ancestry_dict[str(sample)]] not in some_list else '')
        if sample_color_dict[sample_ancestry_dict[str(sample)]] not in some_list:
            some_list.append(sample_color_dict[sample_ancestry_dict[str(sample)]])
    plt.ylabel("Counts", fontsize=fontsize)
    plt.legend(loc="best", fontsize=12)
    plt.xlabel(log2+xlabel, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    if title is not None:
        plt.title(str(title), fontsize=fontsize+1)
    else:
        plt.title("counts histogram", fontsize=fontsize+1)

    if ancestry is not None:
        print(">>>> "+ ancestry+" column: "+col)
    else:
        print(">>>> all ancestries column: "+col)
    _,_=get_statistics_list(x_appended)
    print("")
    
    plt.subplot(1, 2, 2)
    x_list=[]
    some_list= []
    log2=''
    for sample in sample_selected:
        df_filtered=get_data(sample,folder,min_total)
        x=df_filtered[col].tolist()
        x_list.extend(x)
        if col=="posterior_median":
            thetas_log2 = [math.log2(s) for s in x]
            x=thetas_log2
            log2='log2 '
        bins = get_bins_num(x)
        plt.hist(x, color=sample_color_dict[sample_ancestry_dict[str(sample)]],density=True,alpha=0.5,bins=bins,label = sample_ancestry_dict[sample] if sample_color_dict[sample_ancestry_dict[str(sample)]] not in some_list else '')
        if sample_color_dict[sample_ancestry_dict[str(sample)]] not in some_list:
            some_list.append(sample_color_dict[sample_ancestry_dict[str(sample)]])
    mn, mx = plt.xlim()
    plt.xlim(mn, mx)
    kde_xs = np.linspace(mn, mx, 300)
    kde = st.gaussian_kde(x_list)
    plt.plot(kde_xs, kde.pdf(kde_xs), label="PDF",color='r')
    plt.ylabel("Density", fontsize=fontsize)
    plt.legend(loc="best", fontsize=12)
    plt.xlabel(log2+xlabel, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    if title is not None:
        plt.title(str(title), fontsize=fontsize+1)
    else:
        plt.title("frequency histogram", fontsize=fontsize+1)
    plt.show()


def plot_col_allAncestry_genotypingEr(sample_color_dict,sample_ancestry_dict,min_total,col,fontsize=12,vline=None,xlabel=None,title=None,ancestry=None):
    sample_selected = sample_ancestry_dict
    ancestry_chosen=''
    if ancestry is not None:
        sample_selected = return_key(sample_ancestry_dict,ancestry)
        ancestry_chosen=ancestry
    #
    fig = plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    some_list= []
    log2=''
    x_appended=[]
    for sample in sample_selected:
        df_filtered=get_data_genotypingEr(sample,min_total)
        x=df_filtered[col].tolist()
        x_appended.extend(x)
        if col=="posterior_median":
            thetas_log2 = [math.log2(s) for s in x]
            x=thetas_log2
            log2='log2 '
        bins = get_bins_num(x)
        plt.hist(x, color=sample_color_dict[sample_ancestry_dict[str(sample)]],density=False,alpha=0.5,bins=bins,label = sample_ancestry_dict[sample] if sample_color_dict[sample_ancestry_dict[str(sample)]] not in some_list else '')
        if sample_color_dict[sample_ancestry_dict[str(sample)]] not in some_list:
            some_list.append(sample_color_dict[sample_ancestry_dict[str(sample)]])
    plt.ylabel("Counts", fontsize=fontsize)
    plt.legend(loc="best", fontsize=12)
    plt.xlabel(log2+xlabel, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    if title is not None:
        plt.title(str(title), fontsize=fontsize+1)
    else:
        plt.title("counts histogram", fontsize=fontsize+1)

    if ancestry is not None:
        print(">>>> "+ ancestry+" column: "+col)
    else:
        print(">>>> all ancestries column: "+col)
    _,_=get_statistics_list(x_appended)
    print("")
    
    plt.subplot(1, 2, 2)
    x_list=[]
    some_list= []
    log2=''
    for sample in sample_selected:
        df_filtered=get_data_genotypingEr(sample,min_total)
        x=df_filtered[col].tolist()
        x_list.extend(x)
        if col=="posterior_median":
            thetas_log2 = [math.log2(s) for s in x]
            x=thetas_log2
            log2='log2 '
        bins = get_bins_num(x)
        plt.hist(x, color=sample_color_dict[sample_ancestry_dict[str(sample)]],density=True,alpha=0.5,bins=bins,label = sample_ancestry_dict[sample] if sample_color_dict[sample_ancestry_dict[str(sample)]] not in some_list else '')
        if sample_color_dict[sample_ancestry_dict[str(sample)]] not in some_list:
            some_list.append(sample_color_dict[sample_ancestry_dict[str(sample)]])
    mn, mx = plt.xlim()
    plt.xlim(mn, mx)
    kde_xs = np.linspace(mn, mx, 300)
    kde = st.gaussian_kde(x_list)
    plt.plot(kde_xs, kde.pdf(kde_xs), label="PDF",color='r')
    plt.ylabel("Density", fontsize=fontsize)
    plt.legend(loc="best", fontsize=12)
    plt.xlabel(log2+xlabel, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    if title is not None:
        plt.title(str(title), fontsize=fontsize+1)
    else:
        plt.title("frequency histogram", fontsize=fontsize+1)
    plt.show()

