"""
This file contains methods to plot the results of our system
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def getScores():
    settings = []
    avgf1_sum = []
    avgf1_ls = []
    avgf1_fs = []
    avglcs_sum = []
    avglcs_ls = []
    avglcs_fs = []
    fileNames = []
    # Obtain all the pickle files with summaries using different Measures
    for file in os.listdir("RougeScores"):
        if file.endswith(".pkl"):
            fileNames.append(os.path.join(os.path.sep+"RougeScores", file))
    for filename in fileNames:
        print("\n")
        MeasureType = getMeasureType(filename=filename)
        df = pd.read_pickle(filename[1:])
        f1s_sum = df["N-SR"].tolist()
        f1s_ls = df["N-LSR"].tolist()
        f1s_fs = df["N-FSR"].tolist()
        lcs_sum = df["L-SR"].tolist()
        lcs_ls = df["L-LSR"].tolist()
        lcs_fs = df["L-FSR"].tolist()

        settings.append(MeasureType)
        avgf1_sum.append(np.mean(f1s_sum))
        avgf1_ls.append(np.mean(f1s_ls))
        avgf1_fs.append(np.mean(f1s_fs))
        avglcs_sum.append(np.mean(lcs_sum))
        avglcs_ls.append(np.mean(lcs_ls))
        avglcs_fs.append(np.mean(lcs_fs))

    return settings,avgf1_sum,avgf1_ls,avgf1_fs,avglcs_sum,avglcs_ls,avglcs_fs

def getMeasureType(filename):
    """
    """
    filename = filename.split(os.path.sep)
    name = filename[-1]
    name = name.split(".")
    return name[0]


def plot(avgf1s,measureType,summaryType,fs,ls,type):
    """
    """
    fig = plt.figure()
    x = [x for x in range(len(measureType))]
    fig.suptitle(summaryType, fontsize=20)
    plt.style.use("fivethirtyeight")
    plt.plot(x,avgf1s,label="Summary_Generated")
    plt.plot(x,fs,label="First-Sentence")
    plt.plot(x,ls,label="Last-Sentence")
    plt.xticks(x, measureType,fontsize = 7)
    plt.xlabel('Measurement-Type', fontsize=10)
    if(type==0):
        plt.ylabel('Rouge-F1-Scores', fontsize=10)
    if(type==1):
        plt.ylabel('Rouge-LCS-Scores', fontsize=10)
    plt.legend(loc='best')
    plt.show()
    fileName = ""
    if(type==0):
        fileName = "ROUGE-N"
    if(type==1):
        fileName = "ROUGE-L"
    fig.savefig(fileName + '.png')

def main():
    settings,avgf1_sum,avgf1_ls,avgf1_fs,avglcs_sum,avglcs_ls,avglcs_fs = getScores()
    avgf1s = [avgf1_sum,avgf1_ls,avgf1_fs]
    # avglcs = [avglcs_sum,avglcs_ls,avglcs_fs]
    settings=[setting.split("_")[0] for setting in settings]
    summary_types = ["Generated-Summary","First-Sentence(Baseline)","Last-Sentence(Baseline)"]
    plot(avgf1s=avgf1_sum,measureType=settings,summaryType="System Vs Baseline",fs=avgf1_fs,ls=avgf1_ls,type = 0)
    plot(avgf1s=avglcs_sum,measureType=settings,summaryType="System Vs Baseline",fs=avglcs_fs,ls=avglcs_ls,type = 1)
    # plot(avgf1s=avglcs_sum,measureType=settings,summaryType="Generated-Summary")
    # plot(avgf1s=avgf1_fs,measureType=settings,summaryType="First-Sentence(Baseline)")
    # plot(avgf1s=avgf1_ls,measureType=settings,summaryType="Last-Sentence(Baseline)")

if __name__ == "__main__":
    main()
