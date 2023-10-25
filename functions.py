import numpy as np
import sklearn
import os
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import PredictionErrorDisplay
from pygam.datasets import mcycle
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error
import statistics
from math import exp, log,log2
import seaborn as sns
from scipy.special import expit
from . import power

def get_directories(DCC_path):
    path = DCC_path + "/output/RNAseq/1000Genome/"
    sample_list = glob.glob(path + "*/", recursive=True)
    samples = []
    for sample_path in sample_list:
        sample = os.path.basename(os.path.dirname(sample_path))
        samples.append(sample)
    return samples


def get_success_sample(DCC_path):
    path = DCC_path + "/output/RNAseq/1000Genome/"
    sample_list = glob.glob(path + "*/", recursive=True)
    success_list = []
    for sample_path in sample_list:
        sample = os.path.basename(os.path.dirname(sample_path))
        success_file = path + str(sample) + "/success"
        if os.path.isfile(success_file):
            success_list.append(sample)
    return success_list


def plot_pred_vs_actual(
    model, X_train, Y_train, X_test, Y_test
):
    model.fit(X_train, Y_train)
    Y_test_real = Y_test
    Y_test_predicted = model.predict(X_test)

    Y_test_predicted = [expit(y) for y in Y_test_predicted]
    Y_test_real = [expit(y) for y in Y_test_real]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
    # plt.subplot(1, 2, 1)
    ax1.plot(Y_test_real, Y_test_predicted, "o")
    ax1.plot([0, 0.1], [0, 0.1], "--k")
    ax1.set_xlim([0, 0.00001])
    ax1.set_ylim([0, 0.00001])
    ax1.set_title("validation set data < 0.00001")
    ax1.set_xlabel("actual type1error")
    ax1.set_ylabel("predicted type1error")

    ax2.plot(Y_test_real, Y_test_predicted, "o")
    ax2.plot([0, 0.1], [0, 0.1], "--k")
    ax2.set_xlim([0, 0.0001])
    ax2.set_ylim([0, 0.0001])
    ax2.set_title("validation set data < 0.0001")
    ax2.set_xlabel("actual type1error")
    ax2.set_ylabel("predicted type1error")

    ax3.plot(Y_test_real, Y_test_predicted, "o")
    ax3.plot([0, 1], [0, 1], "--k")
    # ax3.set_xlim([0, 0.001])
    # ax3.set_ylim([0, 0.001])
    ax3.set_title("validation set data all")
    ax3.set_xlabel("actual type1error")
    ax3.set_ylabel("predicted type1error")

    # plt.subplot(1, 2, 2)
    ax4.plot(-np.log10(Y_test_real), -np.log10(Y_test_predicted), "o")
    ax4.plot([0, 7], [0, 7], "--k")
    # ax4.set_xlim([0, 1])
    # ax4.set_ylim([0, 1])
    ax4.set_title("validation set data all")
    ax4.set_xlabel("-log10(actual type1error)")
    ax4.set_ylabel("-log10(predicted type1error)")

    plt.tight_layout()
    plt.show()

def plot_self_cv_score(cv, X_train, Y_train, model):
    CV_score = []

    kf = RepeatedKFold(n_splits=cv, n_repeats=1, random_state=None)

    for train_index, test_index in kf.split(X_train):
        # print("Train:", train_index, "Validation:",test_index)
        train_x, test_x = X_train[train_index], X_train[test_index]
        train_y, test_y = Y_train[train_index], Y_train[test_index]

        model.fit(train_x, train_y)
        predictions = model.predict(test_x)
        CV_score.append(mean_squared_error(test_y, predictions))
    mean_mse = statistics.mean(CV_score)
    # print(statistics.mean(CV_score))
    parameter_range = np.arange(1, cv + 1, 1)

    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    ax1.plot(parameter_range, CV_score, label="MSE Score", color="b")
    for i, txt in enumerate(CV_score):
        ax1.annotate(round(txt, 3), (parameter_range[i], CV_score[i]))
    # ax1.plot([0, .1], [0, .1], '--k')
    # ax1.set_xlim([0, .001])
    ax1.set_ylim([0, 2])
    ax1.set_title(str(cv) + " fold CV scores on test")
    ax1.set_xlabel("MSE score on logit(type1error)")
    ax1.set_ylabel("N")
    plt.axhline(
        y=mean_mse, color="r", linestyle="--", label="mean: " + str(round(mean_mse, 2))
    )
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def read_one_df(DCC_path,sample,plk):
    file_path=DCC_path+"/1000Genome/"+str(sample)
    df=pd.read_csv(file_path+"/beastie/runModel_phased_even100/chr1-22_alignBiasp0.05_s0.5_a0.05_sinCov0_totCov1_W1000K1000/result/"+str(sample)+"_ASE_all.tsv",sep="\t")
    sub_df=df[["geneID","number.of.hets","totalCount","gam_lambda","posterior_median","posterior_mass_support_ALT","NaiveSum_pval","MajorSite_pval"]]
    sub_df["sample"]=sample
    size=sub_df.shape[0]
    sub_df["posterior_mass_support_ALT"] = sub_df.apply(
        lambda x: power.calculate_beastie_score(
            plk, x["geneID"], x["gam_lambda"]
        ),
        axis=1,
    )
    def beastie(row):
        if row["posterior_mass_support_ALT"]>0.5:
            return 1
        else:
            return 0
    def NS(row,size):
        if row["NaiveSum_pval"]<=0.05/size:
            return 1
        else:
            return 0
    def MS(row,size):
        if row["MajorSite_pval"]<=0.05/size:
            return 1
        else:
            return 0
    sub_df["beastie_ASE"]=sub_df.apply(lambda row: beastie(row),axis=1)
    sub_df["NS_ASE"]=sub_df.apply(lambda row: NS(row,size),axis=1)
    sub_df["MS_ASE"]=sub_df.apply(lambda row: MS(row,size),axis=1)
    return sub_df

def calculate_max_prob(dict,geneID,Lambda):
    one_over_Lambda = float(1 / float(Lambda))
    thetas=dict.get(geneID)
    thetas_log2 = [log2(x) for x in thetas]
    p_less2 = len([i for i in thetas_log2 if i < log2(one_over_Lambda)]) / len(
        thetas
    )
    p_more2 = len([i for i in thetas_log2 if i > log2(float(Lambda))]) / len(
        thetas
    )
    max_sum2 = max(p_less2,p_more2)
    return max_sum2

def clean_data(filename):
    input_df = pd.read_csv(filename, sep="\t")
    expected_type1error = 0.05 / input_df.shape[0]
    print("expected error: {}".format(expected_type1error))
    # load model
    #input_df["predicted_lambda"]=input_df["predicted_lambda_plus1"]
    selected_df=input_df[["geneID","number.of.hets","totalCount","total.matCount","total.patCount","gam_lambda","NaiveSum_pval","MajorSite_pval","beastie_ASE_gam","NS_ASE","MS_ASE"]]
    selected_df.head()
    return selected_df


def plot_posterior_mass_support_alt(selected_df,ylim=None):
    sns.kdeplot(selected_df["posterior_mass_support_ALT_gam1"].values,label="GAM1")
    sns.kdeplot(selected_df["posterior_mass_support_ALT_gam2"].values,label="GAM2")
    sns.kdeplot(selected_df["posterior_mass_support_ALT_gam3"].values,label="GAM3")
    sns.kdeplot(selected_df["posterior_mass_support_ALT_gam4"].values,label="GAM4")
    sns.kdeplot(selected_df["posterior_mass_support_ALT_gam5"].values,label="GAM5")
    sns.kdeplot(selected_df["posterior_mass_support_ALT_gam6"].values,label="GAM6")
    #sns.kdeplot(selected_df["posterior_mass_support_ALT_linear"].values,label="linear")
    if ylim is not None:
        plt.ylim(0,ylim)
    plt.legend()

def plot_lambda(selected_df):
    sns.kdeplot(selected_df["gam1_lambda"].values,label="GAM1")
    sns.kdeplot(selected_df["gam2_lambda"].values,label="GAM2")
    sns.kdeplot(selected_df["gam3_lambda"].values,label="GAM3")
    sns.kdeplot(selected_df["gam4_lambda"].values,label="GAM4")
    sns.kdeplot(selected_df["gam5_lambda"].values,label="GAM5")
    sns.kdeplot(selected_df["gam6_lambda"].values,label="GAM6")
    plt.legend()

def table_by_hets(output_df,expected_type1error,cutoff):
    output_df["avg_depth"]=output_df["totalCount"]/output_df["number.of.hets"]
    output_df=output_df[output_df["avg_depth"]>=cutoff]
    output_df_MS = output_df[output_df["MajorSite_pval"]<= expected_type1error].groupby("number.of.hets")["MajorSite_pval"].count()
    output_df_NS = output_df[output_df["NaiveSum_pval"]<= expected_type1error].groupby("number.of.hets")["NaiveSum_pval"].count()
    output_df_NS2 = output_df[output_df["NS_ASE"]==1].groupby("number.of.hets")["NS_ASE"].count()
    output_df_GAM1 = output_df[output_df["posterior_mass_support_ALT_gam1"] > 0.5].groupby("number.of.hets")["posterior_mass_support_ALT_gam1"].count()
    output_df_GAM2 = output_df[output_df["posterior_mass_support_ALT_gam2"] > 0.5].groupby("number.of.hets")["posterior_mass_support_ALT_gam2"].count()
    output_df_GAM3 = output_df[output_df["posterior_mass_support_ALT_gam3"] > 0.5].groupby("number.of.hets")["posterior_mass_support_ALT_gam3"].count()
    output_df_GAM4 = output_df[output_df["posterior_mass_support_ALT_gam4"] > 0.5].groupby("number.of.hets")["posterior_mass_support_ALT_gam4"].count()
    output_df_GAM5 = output_df[output_df["posterior_mass_support_ALT_gam5"] > 0.5].groupby("number.of.hets")["posterior_mass_support_ALT_gam5"].count()
    output_df_GAM6 = output_df[output_df["posterior_mass_support_ALT_gam6"] > 0.5].groupby("number.of.hets")["posterior_mass_support_ALT_gam6"].count()
    #output_df_linear = output_df[output_df["posterior_mass_support_ALT_linear"] > 0.5].groupby("number.of.hets")["posterior_mass_support_ALT_linear"].count()
    merged = pd.concat([output_df_MS,output_df_NS,output_df_NS2,output_df_GAM1,output_df_GAM2,output_df_GAM3,output_df_GAM4,output_df_GAM5,output_df_GAM6],axis=1)
    return merged

def table_totalcount(output_df,expected_type1error,cutoff):
    output_df["avg_depth"]=output_df["totalCount"]/output_df["number.of.hets"]
    output_df=output_df[output_df["avg_depth"]>=cutoff]
    output_df_MS = output_df[output_df.MajorSite_pval <= expected_type1error].shape[0]
    output_df_NS = output_df[output_df.NaiveSum_pval <= expected_type1error].shape[0]
    output_df_GAM1 = output_df[output_df.posterior_mass_support_ALT_gam1 > 0.5].shape[0]
    output_df_GAM2 = output_df[output_df.posterior_mass_support_ALT_gam2 > 0.5].shape[0]
    output_df_GAM3 = output_df[output_df.posterior_mass_support_ALT_gam3 > 0.5].shape[0]
    output_df_GAM4 = output_df[output_df.posterior_mass_support_ALT_gam4 > 0.5].shape[0]
    output_df_GAM5 = output_df[output_df.posterior_mass_support_ALT_gam5 > 0.5].shape[0]
    output_df_GAM6 = output_df[output_df.posterior_mass_support_ALT_gam6 > 0.5].shape[0]
    #output_df_linear = output_df[output_df.posterior_mass_support_ALT_linear > 0.5].shape[0]
    dict={"MS":output_df_MS,
    "NS":output_df_NS,
    "GAM1": output_df_GAM1,
    "GAM2": output_df_GAM2,
    "GAM3": output_df_GAM3,
    "GAM4": output_df_GAM4,
    "GAM5": output_df_GAM5,
    "GAM6": output_df_GAM6,
    }
    print(dict)


def plot_evaluation(
    ref_data,model, hets, totalcount, expected_type1error, modelname,
):
    filtered_data=ref_data[(ref_data["hets"]==hets)&(ref_data["totalcount"]==totalcount)]
    ref_lambda=filtered_data[filtered_data["type1error"]<=expected_type1error]
    min_row = ref_lambda.loc[ref_lambda['lambda'].idxmin()]
    ref_lambda=round(min_row["lambda"],2)
    average_count=totalcount/hets
    data = [[np.log(hets), np.log(totalcount), np.log(lam-1+0.001)] for lam in np.linspace(1, 3, 2000)]
    # prediction
    prediction = model.predict(data)
    chosen_lambda = 3
    #print(f"hets: %s totalcount: %s" % (hets, totalcount))
    if min(expit(prediction)) <= expected_type1error:
        chosen_lambda=np.exp(data[np.where(expit(prediction) <= expected_type1error)[0][0]][2])+1

    log_lambdas_minus1 = [x[2] for x in data]
    log_totalcount = [x[1] for x in data]
    log_nhets = [x[0] for x in data]
    lambdas = [exp(y)+1 for y in log_lambdas_minus1]
    totalcount = [exp(y) for y in log_totalcount]
    nhets = [exp(y) for y in log_nhets]

    er = [inv_logit(y) for y in prediction]
    

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))

    ax1.plot(lambdas, er)
    ax1.set_ylim(-expected_type1error * 10, expected_type1error * 10)
    ax1.set_xlim(0.8, 3.1)
    ax1.axvline(
        chosen_lambda,
        color="r",
        linestyle="--",
        label="chosen lambda: "
        + str(round(chosen_lambda, 2))
        + "\n at type1error: "
        + str(expected_type1error),
    )
    ax1.axvline(
        ref_lambda,
        color="b",
        linestyle="--",
        label="reference lambda: "
        + str(round(ref_lambda, 2))
        + "\n at type1error: "
        + str(expected_type1error),
    )
    ax1.set_xlabel("lambda")
    ax1.set_ylabel("expected type1error")
    ax1.set_title(str(modelname) + " model (<" + str(expected_type1error * 10) + ")")
    ax1.legend(loc="upper right")

    ax2.plot(lambdas, er)
    ax2.set_ylim(-expected_type1error * 10, expected_type1error * 100)
    ax2.set_xlim(0.8, 3.1)
    ax2.axvline(
        chosen_lambda,
        color="r",
        linestyle="--",
        label="chosen lambda: "
        + str(round(chosen_lambda, 2))
        + "\n at type1error: "
        + str(expected_type1error),
    )
    ax2.axvline(
        ref_lambda,
        color="b",
        linestyle="--",
        label="reference lambda: "
        + str(round(ref_lambda, 2))
        + "\n at type1error: "
        + str(expected_type1error),
    )
    ax2.set_xlabel("lambda")
    ax2.set_ylabel("expected type1error")
    ax2.set_title(str(modelname) + " model (<" + str(expected_type1error * 100) + ")")
    ax2.legend(loc="upper right")

    ax3.plot(lambdas, er)
    ax3.set_ylim(-expected_type1error * 10, expected_type1error * 1000)
    ax3.set_xlim(0.8, 3.1)
    ax3.axvline(
        chosen_lambda,
        color="r",
        linestyle="--",
        label="chosen lambda: "
        + str(round(chosen_lambda, 2))
        + "\n at type1error: "
        + str(expected_type1error),
    )
    ax3.axvline(
        ref_lambda,
        color="b",
        linestyle="--",
        label="reference lambda: "
        + str(round(ref_lambda, 2))
        + "\n at type1error: "
        + str(expected_type1error),
    )
    ax3.set_xlabel("lambda")
    ax3.set_ylabel("expected type1error")
    ax3.set_title(str(modelname) + " model (<" + str(expected_type1error * 1000) + ")")
    ax3.legend(loc="upper right")

    ax4.plot(lambdas, er)
    # ax3.set_ylim(-expected_type1error*10,expected_type1error*100)
    ax4.set_xlim(0.8, 3.1)
    ax4.axvline(
        chosen_lambda,
        color="r",
        linestyle="--",
        label="chosen lambda: "
        + str(round(chosen_lambda, 2))
        + "\n at type1error: "
        + str(expected_type1error),
    )
    ax4.axvline(
        ref_lambda,
        color="b",
        linestyle="--",
        label="reference lambda: "
        + str(round(ref_lambda, 2))
        + "\n at type1error: "
        + str(expected_type1error),
    )
    ax4.set_xlabel("lambda")
    ax4.set_ylabel("expected type1error")
    ax4.set_title(str(modelname))
    ax4.legend(loc="upper right")
    plt.tight_layout()
    # lt.suptitle("select lambda (for hets 3, totalcount 120) at type1error: "+str(expected_type1error)+"\n\n") # or plt.suptitle('Main title')
    plt.show()

def plot_evaluation_2D(
    ref_data,model, hets, totalcount, expected_type1error, modelname,
):
    filtered_data=ref_data[(ref_data["hets"]==hets)&(ref_data["totalcount"]==totalcount)]
    ref_lambda=filtered_data[filtered_data["type1error"]<=expected_type1error]
    min_row = ref_lambda.loc[ref_lambda['lambda'].idxmin()]
    ref_lambda=round(min_row["lambda"],2)
    average_count=totalcount/hets
    data = [[np.log(totalcount), np.log(lam-1+0.001)] for lam in np.linspace(1, 3, 2000)]
    # prediction
    prediction = model.predict(data)
    chosen_lambda = 3
    #print(f"hets: %s totalcount: %s" % (hets, totalcount))
    if min(expit(prediction)) <= expected_type1error:
        chosen_lambda=np.exp(data[np.where(expit(prediction) <= expected_type1error)[0][0]][1])+1

    log_lambdas_minus1 = [x[1] for x in data]
    log_totalcount = [x[0] for x in data]
    lambdas = [exp(y)+1 for y in log_lambdas_minus1]
    totalcount = [exp(y) for y in log_totalcount]

    er = [inv_logit(y) for y in prediction]
    

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))

    ax1.plot(lambdas, er)
    ax1.set_ylim(-expected_type1error * 10, expected_type1error * 10)
    ax1.set_xlim(0.8, 3.1)
    ax1.axvline(
        chosen_lambda,
        color="r",
        linestyle="--",
        label="chosen lambda: "
        + str(round(chosen_lambda, 2))
        + "\n at type1error: "
        + str(expected_type1error),
    )
    ax1.axvline(
        ref_lambda,
        color="b",
        linestyle="--",
        label="reference lambda: "
        + str(round(ref_lambda, 2))
        + "\n at type1error: "
        + str(expected_type1error),
    )
    ax1.set_xlabel("lambda")
    ax1.set_ylabel("expected type1error")
    ax1.set_title(str(modelname) + " model (<" + str(expected_type1error * 10) + ")")
    ax1.legend(loc="upper right")

    ax2.plot(lambdas, er)
    ax2.set_ylim(-expected_type1error * 10, expected_type1error * 100)
    ax2.set_xlim(0.8, 3.1)
    ax2.axvline(
        chosen_lambda,
        color="r",
        linestyle="--",
        label="chosen lambda: "
        + str(round(chosen_lambda, 2))
        + "\n at type1error: "
        + str(expected_type1error),
    )
    ax2.axvline(
        ref_lambda,
        color="b",
        linestyle="--",
        label="reference lambda: "
        + str(round(ref_lambda, 2))
        + "\n at type1error: "
        + str(expected_type1error),
    )
    ax2.set_xlabel("lambda")
    ax2.set_ylabel("expected type1error")
    ax2.set_title(str(modelname) + " model (<" + str(expected_type1error * 100) + ")")
    ax2.legend(loc="upper right")

    ax3.plot(lambdas, er)
    ax3.set_ylim(-expected_type1error * 10, expected_type1error * 1000)
    ax3.set_xlim(0.8, 3.1)
    ax3.axvline(
        chosen_lambda,
        color="r",
        linestyle="--",
        label="chosen lambda: "
        + str(round(chosen_lambda, 2))
        + "\n at type1error: "
        + str(expected_type1error),
    )
    ax3.axvline(
        ref_lambda,
        color="b",
        linestyle="--",
        label="reference lambda: "
        + str(round(ref_lambda, 2))
        + "\n at type1error: "
        + str(expected_type1error),
    )
    ax3.set_xlabel("lambda")
    ax3.set_ylabel("expected type1error")
    ax3.set_title(str(modelname) + " model (<" + str(expected_type1error * 1000) + ")")
    ax3.legend(loc="upper right")

    ax4.plot(lambdas, er)
    # ax3.set_ylim(-expected_type1error*10,expected_type1error*100)
    ax4.set_xlim(0.8, 3.1)
    ax4.axvline(
        chosen_lambda,
        color="r",
        linestyle="--",
        label="chosen lambda: "
        + str(round(chosen_lambda, 2))
        + "\n at type1error: "
        + str(expected_type1error),
    )
    ax4.axvline(
        ref_lambda,
        color="b",
        linestyle="--",
        label="reference lambda: "
        + str(round(ref_lambda, 2))
        + "\n at type1error: "
        + str(expected_type1error),
    )
    ax4.set_xlabel("lambda")
    ax4.set_ylabel("expected type1error")
    ax4.set_title(str(modelname))
    ax4.legend(loc="upper right")
    plt.tight_layout()
    # lt.suptitle("select lambda (for hets 3, totalcount 120) at type1error: "+str(expected_type1error)+"\n\n") # or plt.suptitle('Main title')
    plt.show()
    
def plot_3_hist(df):
    data1 = df[["type1error"]].values
    data2 = df[["log_type1error"]].values
    data3 = df[["logit_type1error"]].values
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    # plt.subplot(1, 2, 1)
    ax1.hist(data1, label="type1error", color="b")
    ax1.set_xlabel("type1error")
    # ax1.plot([0, .1], [0, .1], '--k')
    # ax1.set_xlim([0, .001])
    # ax1.set_ylim([0.5, 1.1])
    # ax1.set_title(str(cv)+' fold CV scores on test')
    # ax1.set_xlabel("CV score")
    # ax1.set_ylabel('N')

    ax2.hist(data2, label="log(type1error)", color="b")
    ax2.set_xlabel("log(type1error)")
    ax3.hist(data3, label="logit(type1error)", color="b")
    ax3.set_xlabel("logit(type1error)")
    # ax.set_legend(loc = 'best')
    plt.tight_layout()
    plt.show()


def plot_scatter(df1):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
    ax1.scatter(df1["lambda"].values, df1["type1error"].values)
    ax1.set_xlabel("lambda")
    ax1.set_ylabel("type1error")

    ax2.scatter(df1["log_totalcount"].values, df1["logit_type1error"].values)
    ax2.set_xlabel("log(totalcount)")
    ax2.set_ylabel("logit type1error")
    # ax1.plot([0, .1], [0, .1], '--k')
    # ax1.set_xlim([0, .001])
    # ax1.set_ylim([0.5, 1.1])
    # ax1.set_title(str(cv)+' fold CV scores on test')
    # ax1.set_xlabel("CV score")
    # ax1.set_ylabel('N')
    ax3.scatter(df1["log_nhets"].values, df1["logit_type1error"].values)
    ax3.set_xlabel("log(nhets)")
    ax3.set_ylabel("logit type1error")
    # ax.set_legend(loc = 'best')
    ax4.scatter(df1["log_lambda_minus1"].values, df1["logit_type1error"].values)
    ax4.set_xlabel("log(lambda-1)")
    ax4.set_ylabel("logit type1error")
    plt.tight_layout()
    plt.show()

def plot_scatter_totalcount(df1):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    ax1.scatter(df1["totalcount"].values, df1["type1error"].values)
    ax1.set_xlabel("totalcount")
    ax1.set_ylabel("type1error")

    ax2.scatter(df1["totalcount"].values, df1["log_type1error"].values)
    ax2.set_xlabel("totalcount")
    ax2.set_ylabel("log type1error")

    ax3.scatter(df1["totalcount"].values, df1["logit_type1error"].values)
    ax3.set_xlabel("totalcount")
    ax3.set_ylabel("logit type1error")
    plt.tight_layout()
    plt.show()

def plot_scatter_byHets_totalcount(df):
    df = df.sort_values('hets', ascending=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    # Create a scatter plot with 'lambda' on the x-axis, 'type1error' on the y-axis, and color-coded by 'hets'
    for hets in set(df["hets"]):
        subset_df = df[df["hets"] == hets]
        ax1.scatter(subset_df["totalcount"], subset_df["type1error"], label=hets,s=2,alpha=0.5)
    # Add labels and title
    ax1.set_xlabel("totalcount")
    ax1.set_ylabel("type1error")
    ax1.set_title("Type 1 Error by totalcount")
    ax1.legend(title="hets")


    # Create a scatter plot with 'lambda' on the x-axis, 'type1error' on the y-axis, and color-coded by 'hets'
    for hets in set(df["hets"]):
        subset_df = df[df["hets"] == hets]
        ax2.scatter(subset_df["totalcount"], subset_df["log_type1error"], label=hets,s=2,alpha=0.5)
    # Add labels and title
    ax2.set_xlabel("totalcount")
    ax2.set_ylabel("log_type1error")
    ax2.set_title("log(Type 1 Error) by totalcount")
    ax2.legend(title="hets")

    # Create a scatter plot with 'lambda' on the x-axis, 'type1error' on the y-axis, and color-coded by 'hets'
    for hets in set(df["hets"]):
        subset_df = df[df["hets"] == hets]
        ax3.scatter(subset_df["totalcount"], subset_df["logit_type1error"], label=hets,s=2,alpha=0.5)
    # Add labels and title
    ax3.set_xlabel("totalcount")
    ax3.set_ylabel("logit_type1error")
    ax3.set_title("logit(Type 1 Error) by totalcount")
    ax3.legend(title="hets")

def plot_scatter_bytotalcounts_hets(df):
    df = df.sort_values('totalcount', ascending=True)
    # Create 10 evenly-spaced bins for the 'TotalCount' column
    bins = pd.cut(df['totalcount'], 10, labels=["5-100", "100-200", "200-300", "300-400", "400-500", "500-600", "600-700", "700-800", "800-900", "900-1000"])

    # Add the 'totalcount_group' column to the DataFrame
    df['totalcount_group'] = bins

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))

    # Create a scatter plot with 'lambda' on the x-axis, 'type1error' on the y-axis, and color-coded by 'hets'
    for hets in set(df["totalcount_group"]):
        subset_df = df[df["totalcount_group"] == hets]
        ax1.scatter(subset_df["hets"], subset_df["type1error"], label=hets,s=2,alpha=0.5)
    # Add labels and title
    ax1.set_xlabel("hets")
    ax1.set_ylabel("type1error")
    ax1.set_title("Type 1 Error by hets")
    ax1.legend(title="totalcount_group",fontsize=10)


    # Create a scatter plot with 'lambda' on the x-axis, 'type1error' on the y-axis, and color-coded by 'hets'
    for hets in set(df["totalcount_group"]):
        subset_df = df[df["totalcount_group"] == hets]
        ax2.scatter(subset_df["hets"], subset_df["log_type1error"], label=hets,s=2,alpha=0.5)
    # Add labels and title
    ax2.set_xlabel("hets")
    ax2.set_ylabel("log_type1error")
    ax2.set_title("log(Type 1 Error) by hets")
    ax2.legend(title="totalcount_group",fontsize=10)

    # Create a scatter plot with 'lambda' on the x-axis, 'type1error' on the y-axis, and color-coded by 'hets'
    for hets in set(df["totalcount_group"]):
        subset_df = df[df["totalcount_group"] == hets]
        ax3.scatter(subset_df["hets"], subset_df["logit_type1error"], label=hets,s=2,alpha=0.5)
    # Add labels and title
    ax3.set_xlabel("hets")
    ax3.set_ylabel("logit_type1error")
    ax3.set_title("logit(Type 1 Error) by hets")
    ax3.legend(title="totalcount_group",fontsize=10)
    
def plot_scatter_byHets(df):
    df = df.sort_values('hets', ascending=True)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))

    # Create a scatter plot with 'lambda' on the x-axis, 'type1error' on the y-axis, and color-coded by 'hets'
    for hets in set(df["hets"]):
        subset_df = df[df["hets"] == hets]
        ax1.scatter(subset_df["lambda"], subset_df["type1error"], label=hets,s=2,alpha=0.5)
    # Add labels and title
    ax1.set_xlabel("lambda")
    ax1.set_ylabel("type1error")
    ax1.set_title("Type 1 Error by Lambda")
    ax1.legend(title="hets")

    # Create a scatter plot with 'lambda' on the x-axis, 'type1error' on the y-axis, and color-coded by 'hets'
    for hets in set(df["hets"]):
        subset_df = df[df["hets"] == hets]
        ax2.scatter(subset_df["log_lambda_minus1"], subset_df["type1error"], label=hets,s=2,alpha=0.5)
    # Add labels and title
    ax2.set_xlabel("log(lambda-1)")
    ax2.set_ylabel("type1error")
    ax2.set_title("Type 1 Error by log(Lambda-1)")
    ax2.legend(title="hets")

    # Create a scatter plot with 'lambda' on the x-axis, 'type1error' on the y-axis, and color-coded by 'hets'
    for hets in set(df["hets"]):
        subset_df = df[df["hets"] == hets]
        ax3.scatter(subset_df["log_lambda_minus1"], subset_df["log_type1error"], label=hets,s=2,alpha=0.5)
    # Add labels and title
    ax3.set_xlabel("log(lambda-1)")
    ax3.set_ylabel("log(type1error)")
    ax3.set_title("log(Type 1 Error) by log(Lambda-1)")
    ax3.legend(title="hets")

    # Create a scatter plot with 'lambda' on the x-axis, 'type1error' on the y-axis, and color-coded by 'hets'
    for hets in set(df["hets"]):
        subset_df = df[df["hets"] == hets]
        ax4.scatter(subset_df["log_lambda_minus1"], subset_df["logit_type1error"], label=hets,s=2,alpha=0.5)
    # Add labels and title
    ax4.set_xlabel("log(lambda-1)")
    ax4.set_ylabel("logit_type1error")
    ax4.set_title("logit(Type 1 Error) by log(Lambda-1)")
    ax4.legend(title="hets")

def plot_scatter_byCounts(df):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
    df = df.sort_values('totalcount', ascending=True)
    # Create a scatter plot with 'lambda' on the x-axis, 'type1error' on the y-axis, and color-coded by 'hets'
    for hets in set(df["totalcount"]):
        subset_df = df[df["totalcount"] == hets]
        ax1.scatter(subset_df["lambda"], subset_df["type1error"], label=hets,s=2,alpha=0.5)
    # Add labels and title
    ax1.set_xlabel("lambda")
    ax1.set_ylabel("type1error")
    ax1.set_title("Type 1 Error by Lambda")
    ax1.legend(title="totalcount",fontsize=5)

    # Create a scatter plot with 'lambda' on the x-axis, 'type1error' on the y-axis, and color-coded by 'hets'
    for hets in set(df["totalcount"]):
        subset_df = df[df["totalcount"] == hets]
        ax2.scatter(subset_df["log_lambda_minus1"], subset_df["type1error"], label=hets,s=2,alpha=0.5)
    # Add labels and title
    ax2.set_xlabel("log(lambda-1)")
    ax2.set_ylabel("type1error")
    ax2.set_title("Type 1 Error by log(Lambda-1)")
    ax2.legend(title="totalcount",fontsize=5)

    # Create a scatter plot with 'lambda' on the x-axis, 'type1error' on the y-axis, and color-coded by 'hets'
    for hets in set(df["totalcount"]):
        subset_df = df[df["totalcount"] == hets]
        ax3.scatter(subset_df["log_lambda_minus1"], subset_df["log_type1error"], label=hets,s=2,alpha=0.5)
    # Add labels and title
    ax3.set_xlabel("log(lambda-1)")
    ax3.set_ylabel("log(type1error)")
    ax3.set_title("log(Type 1 Error) by log(Lambda-1)")
    ax3.legend(title="totalcount",fontsize=5)

    # Create a scatter plot with 'lambda' on the x-axis, 'type1error' on the y-axis, and color-coded by 'hets'
    for hets in set(df["totalcount"]):
        subset_df = df[df["totalcount"] == hets]
        ax4.scatter(subset_df["log_lambda_minus1"], subset_df["logit_type1error"], label=hets,s=2,alpha=0.5)
    # Add labels and title
    ax4.set_xlabel("log_lambda_minus1")
    ax4.set_ylabel("logit_type1error")
    ax4.set_title("logit(Type 1 Error) by log(Lambda-1)")
    ax4.legend(title="totalcount",fontsize=5)

def plot_cv_pred(model, cv, X_train, y_train):
    y_pred = cross_val_predict(model, X_train, y_train, cv=cv)
    fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
    PredictionErrorDisplay.from_predictions(
        y_train,
        y_pred=y_pred,
        kind="actual_vs_predicted",
        subsample=100,
        ax=axs[0],
        random_state=0,
    )
    axs[0].set_title("Actual vs. Predicted logit(type1error)")
    PredictionErrorDisplay.from_predictions(
        y_train,
        y_pred=y_pred,
        kind="residual_vs_predicted",
        subsample=100,
        ax=axs[1],
        random_state=0,
    )
    axs[1].set_title("Residuals vs. Predicted logit(type1error)")
    fig.suptitle("Plotting cross-validated predictions")
    plt.tight_layout()
    plt.show()


def plot_cv_score(pipe, cv, X_train, y_train):
    CV_score = cross_val_score(pipe, X_train, y_train, cv=cv)
    parameter_range = np.arange(1, cv + 1, 1)

    fig, (ax1) = plt.subplots(1, 1, figsize=(5, 5))
    # plt.subplot(1, 2, 1)
    ax1.plot(parameter_range, CV_score, label="Testing Score", color="b")
    for i, txt in enumerate(CV_score):
        ax1.annotate(round(txt, 3), (parameter_range[i], CV_score[i]))
    # ax1.plot([0, .1], [0, .1], '--k')
    # ax1.set_xlim([0, .001])
    ax1.set_ylim([0.5, 1.1])
    ax1.set_title(str(cv) + " fold CV scores on test")
    ax1.set_xlabel("CV score")
    ax1.set_ylabel("N")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def plot_self_cv_score(cv, X_train, Y_train, model):
    CV_score = []

    kf = RepeatedKFold(n_splits=cv, n_repeats=1, random_state=None)

    for train_index, test_index in kf.split(X_train):
        # print("Train:", train_index, "Validation:",test_index)
        train_x, test_x = X_train[train_index], X_train[test_index]
        train_y, test_y = Y_train[train_index], Y_train[test_index]

        model.fit(train_x, train_y)
        predictions = model.predict(test_x)
        CV_score.append(mean_squared_error(test_y, predictions))
    mean_mse = statistics.mean(CV_score)
    # print(statistics.mean(CV_score))
    parameter_range = np.arange(1, cv + 1, 1)

    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    ax1.plot(parameter_range, CV_score, label="MSE Score", color="b")
    for i, txt in enumerate(CV_score):
        ax1.annotate(round(txt, 3), (parameter_range[i], CV_score[i]))
    # ax1.plot([0, .1], [0, .1], '--k')
    # ax1.set_xlim([0, .001])
    ax1.set_ylim([0, 3])
    ax1.set_title(str(cv) + " fold CV scores on test")
    ax1.set_xlabel("MSE score on log10_type1error)")
    ax1.set_ylabel("N")
    plt.axhline(
        y=mean_mse, color="r", linestyle="--", label="mean: " + str(round(mean_mse, 2))
    )
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

def plot_gam_model_2D(model):
    ## plotting
    plt.figure()
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    titles = ["log(totalcount)", "log(lambda-1)"]
    for i, ax in enumerate(axs):
        XX = model.generate_X_grid(term=i)
        if i != 0:
            XX[:, 0] = 1

        # print(XX)
        ax.plot(XX[:, i], model.partial_dependence(term=i, X=XX))
        ax.plot(
            XX[:, i],
            model.partial_dependence(term=i, X=XX, width=0.95)[1],
            c="r",
            ls="--",
        )
        ax.set_title(titles[i])

def plot_gam_model(model):
    ## plotting
    plt.figure()
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    titles = ["log(nhets)", "log(totalcount)", "log(lambda-1)"]
    for i, ax in enumerate(axs):
        XX = model.generate_X_grid(term=i)
        if i != 0:
            XX[:, 0] = 1

        # print(XX)
        ax.plot(XX[:, i], model.partial_dependence(term=i, X=XX))
        ax.plot(
            XX[:, i],
            model.partial_dependence(term=i, X=XX, width=0.95)[1],
            c="r",
            ls="--",
        )
        ax.set_title(titles[i])
    # XX = gam1.generate_X_grid(term=1, meshgrid=True)
    # # print(XX)
    # XX[:,0] = 1
    # Z = gam1.partial_dependence(term=1, X=XX, meshgrid=True)

    # ax = plt.axes(projection='3d')
    # ax.plot_surface(XX[0], XX[1], Z, cmap='viridis')


def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))


def logit(p):
    return np.log(p) - np.log(1 - p)


def clean_df(df_10M, cutoff):
    df_10M = df_10M[["type1error", "lambda", "hets", "read"]]
    df_10M["totalcount"] = df_10M["read"] * df_10M["hets"]
    df_10M["log_type1error"] = np.log(df_10M["type1error"])
    df_10M["logit_type1error"] = np.log(
        df_10M["type1error"] / (1 - df_10M["type1error"])
    )
    df_10M["log_lambda_minus1"] = np.log(df_10M["lambda"]-1)
    df_10M["log_nhets"] = np.log(df_10M["hets"])
    df_10M["log_totalcount"] = np.log(df_10M["totalcount"])

    # filtering1
    filtered_df = df_10M[df_10M["type1error"] < cutoff]
    selected_variables=filtered_df[["hets","totalcount","lambda","type1error","log_type1error","log_lambda_minus1","logit_type1error","log_nhets","log_totalcount"]]
    selected_variables = selected_variables.groupby(['type1error', 'hets', 'totalcount']).apply(lambda x: x.loc[x['lambda'].idxmin()])
    selected_variables = selected_variables.apply(lambda x: x.reset_index(drop=True))
    return df_10M, selected_variables


def plot_gam_hets(model):
    ## plotting
    plt.figure()
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    titles = ["log(nhets)"]
    XX = model.generate_X_grid(term=0)
    i = 0
    ax.plot(XX[:, i], model.partial_dependence(term=i, X=XX))
    ax.plot(
        XX[:, i], model.partial_dependence(term=i, X=XX, width=0.95)[1], c="r", ls="--"
    )
    ax.set_title(titles[i])


def plot_gam_3d(model):
    ## plotting
    plt.figure()
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    titles = ["log(totalcount) X log(lambda-1)"]
    i = 1
    XX = model.generate_X_grid(term=1, meshgrid=True)
    # XX[:,0] = 1
    Z = model.partial_dependence(term=1, X=XX, meshgrid=True)
    ax = plt.axes(projection="3d")
    ax.plot_surface(XX[0], XX[1], Z, cmap="viridis")
    ax.set_title(titles)

def plot_gam_lambda(model):
    ## plotting
    plt.figure()
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    titles = ["log(lambda-1)"]
    XX = model.generate_X_grid(term=0)
    i = 0
    ax.plot(XX[:, i], model.partial_dependence(term=i, X=XX))
    ax.plot(
        XX[:, i], model.partial_dependence(term=i, X=XX, width=0.95)[1], c="r", ls="--"
    )
    ax.set_title(titles[i])

def plot_gam_3d2(model):
    ## plotting
    plt.figure()
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    titles = ["log(nhets) X log(average count)"]
    i = 1
    XX = model.generate_X_grid(term=1, meshgrid=True)
    # XX[:,0] = 1
    Z = model.partial_dependence(term=1, X=XX, meshgrid=True)
    ax = plt.axes(projection="3d")
    ax.plot_surface(XX[0], XX[1], Z, cmap="viridis")
    ax.set_title(titles)

def plot_extrapolation(model):
    X, y = mcycle()
    XX = model.generate_X_grid(term=0)

    m = X.min()
    M = X.max()
    XX = np.linspace(m - 10, M + 10, 500)
    Xl = np.linspace(m - 10, m, 50)
    Xr = np.linspace(M, M + 10, 50)

    plt.figure()

    plt.plot(XX, model.predict(XX), "k")
    plt.plot(Xl, model.confidence_intervals(Xl), color="b", ls="--")
    plt.plot(Xr, model.confidence_intervals(Xr), color="b", ls="--")
    _ = plt.plot(X, model.confidence_intervals(X), color="r", ls="--")
