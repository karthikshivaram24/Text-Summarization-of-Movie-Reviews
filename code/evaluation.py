from NlpUtility import NLPWrapper
import pandas as pd
import numpy as np
import glob, os
import configparser as cp

class Evaluation():

    def __init__(self):
        self.nlpWrapper = NLPWrapper()
        self.config = cp.ConfigParser()
        self.config.read("ConfigFile.properties")
        self.ngramSize = int(self.config["RougeN"]["n"])
        pass


    def iter_ngrams(self, sentence, n):
        """
        This method retrieves ngrams of length n from a
        given sentence.

        Params: sentence : a list of tokens of a sentence
                n : length of ngrams
        Returns: A list of ngrams for a given sentence
        """
        return [tuple(sentence[i : i+n]) for i in range(len(sentence)-n+1)]

    def get_recall(self,common_len,RTSummary):
        """
        This method calculates the recall score for rouge-n scores

        Params: common_len: The intersection of ngrams between System generated summary
                            And the given rotten tomatoes summary
                RTSummary: List containing ngrams for the given rotten tomatoes summary

        Returns: recall score for rouge-n
        """
        if len(RTSummary)<=0:
            return 0.0

        return common_len/len(RTSummary)

    def get_precision(self,common_len,SystemSummary):
        """
        This method calculates the precision score for rouge-n scores

        Params: common_len   : The intersection of ngrams between System generated summary
                               And the given rotten tomatoes summary
                SystemSummary: List containing ngrams for the generated system summary

        Returns: precision score for rouge-n
        """
        # print(SystemSummary)
        if len(SystemSummary)<=0:
            return 0.0
        return common_len/len(SystemSummary)

    def get_f1(self,common_len,RTSummary,SystemSummary):
        """
        This method calculates the f1 score for rouge-n scores

        Params: common_len   : The intersection of ngrams between System generated summary
                               And the given rotten tomatoes summary
                SystemSummary: List containing ngrams for the generated system summary
                RTSummary    : List containing ngrams for the given rotten tomatoes summary

        Returns: f1 score for rouge-n
        """
        recall = self.get_recall(common_len,RTSummary)
        precision = self.get_precision(common_len,SystemSummary)
        if precision<=0 or recall <=0:
            return 0.0
        else:
            return  ((2 * recall * precision)/(recall + precision))

    def n_gram_rouge(self,RTSummary,SystemSummary, n = 1):
        """
        Calculates the ROUGE-N Score between 2 summary sentences

        Params:  RTSummary     : The summary sentence picked by Rotten Tomatoes
                 SystemSummary : The summary sentence picked by our system

        Returns: ROUGE-N Score between 2 summary sentences
        """
        rouge_scores = dict()
        RT_tokens = self.nlpWrapper.stemmer(tokens=self.nlpWrapper.tokenize(RTSummary))
        SS_tokens = self.nlpWrapper.stemmer(tokens=self.nlpWrapper.tokenize(SystemSummary))
        RTSummary= set(self.iter_ngrams(RT_tokens,n))
        SystemSummary= set(self.iter_ngrams(SS_tokens,n))
        complete_l = RTSummary.intersection(SystemSummary)
        rouge_score = self.get_f1((len(complete_l)),RTSummary,SystemSummary)

        return rouge_score

    def unit_overlap(self, RTSummary , SystemSummary):
        """
        Calculates The word overlap scores between the 2 summary sentences

        Params:  RTSummary     : The summary sentence picked by Rotten Tomatoes
                 SystemSummary : The summary sentence picked by our system

        Returns: Word_overlap evaluation score between 2 summary sentences

        """

        RTSummary_words = self.nlpWrapper.stemmer(tokens=self.nlpWrapper.tokenize(RTSummary))
        SystemSummary_words = self.nlpWrapper.stemmer(tokens=self.nlpWrapper.tokenize(RTSummary))
        commonWords = set(RTSummary_words)&set(SystemSummary_words)
        commonWords = sorted(commonWords, key = lambda k : RTSummary_words.index(k))

        return (len(commonWords) / (len(RTSummary)+len(SystemSummary)-len(commonWords)))

    """
    Section 3.3.3 : http://www.cai.sk/ojs/index.php/cai/article/viewFile/37/24
    LCS
    """
    def lcs(self, RTSummary , SystemSummary):
        """
        Calculates the Longest common Subsequence between the Rotten Tomatoes
        picked summary sentence and the summary sentence picked by our system.

        Params:  RTSummary     : The summary sentence picked by Rotten Tomatoes
                 SystemSummary : The summary sentence picked by our system

        Returns: The LCS(Longest Common Subsequence) between the 2 summary sentences
        """

        RTSummary_len = len(RTSummary)
        SystemSummary_len = len(SystemSummary)

        LCS = [[None]*(SystemSummary_len+1) for i in range(RTSummary_len+1)]

        for i in range(RTSummary_len+1):
            for j in range(SystemSummary_len+1):
                if i == 0 or j == 0 :
                    LCS[i][j] = 0
                elif RTSummary[i-1] == SystemSummary[j-1]:
                    LCS[i][j] = LCS[i-1][j-1]+1
                else:
                    LCS[i][j] = max(LCS[i-1][j] , LCS[i][j-1])

        return LCS[RTSummary_len][SystemSummary_len]

    def lcs_rouge(self, RTSummary , SystemSummary):
        """
        This method calculates ROUGE-L between the given 2 summary sentences

        Params:  RTSummary     : The summary sentence picked by Rotten Tomatoes
                 SystemSummary : The summary sentence picked by our system

        Returns: ROUGE-L Score between the 2 summary sentences
        """
        return (len(RTSummary)+len(SystemSummary) - (self.lcs(RTSummary,SystemSummary)))/2

    def getMeasureType(self,filename):
        """
        """
        filename = filename.split(os.path.sep)
        name = filename[-1]
        name = name.split(".")
        return name[0]

    def calculateAvgF1(self,f1s):
        """
        """
        if len(f1s) >0:
            return np.mean(f1s)
        else:
            return 0.0

    def main(self,folderName):
        """
        """
        fileNames = []
        # Obtain all the pickle files with summaries using different Measures
        for file in os.listdir(folderName):
            if file.endswith(".pkl"):
                fileNames.append(os.path.join(os.path.sep+folderName, file))
        for filename in fileNames:
            print("\n")
            MeasureType = self.getMeasureType(filename=filename)
            df = pd.read_pickle(filename[1:])
            columns = ["Id","RTSummary","SystemSummary","First-Sent","Last-Sent","N-FSR","N-LSR","N-SR","L-FSR","L-LSR","L-SR"]
            temp = pd.DataFrame(columns = columns)
            temp["Id"] = df["Id"].values
            temp["RTSummary"] = df["RTSummary"].values
            temp["SystemSummary"] = df["SystemSummary"].values
            temp["First-Sent"] = df["First-Sent"].values
            temp["Last-Sent"] = df["Last-Sent"].values
            N_FSR = []
            N_LSR = []
            N_SR =  []
            L_FSR = []
            L_LSR = []
            L_SR =  []
            for index,row in temp.iterrows():
                N_SR.append(self.n_gram_rouge(RTSummary=row["RTSummary"],SystemSummary=row["SystemSummary"], n = self.ngramSize))
                N_FSR.append(self.n_gram_rouge(RTSummary=row["RTSummary"],SystemSummary=row["First-Sent"], n = self.ngramSize))
                N_LSR.append(self.n_gram_rouge(RTSummary=row["RTSummary"],SystemSummary=row["Last-Sent"], n = self.ngramSize))
                L_FSR.append(self.lcs( RTSummary=row["RTSummary"] , SystemSummary=row["First-Sent"]))
                L_LSR.append(self.lcs( RTSummary=row["RTSummary"] , SystemSummary=row["Last-Sent"]))
                L_SR.append(self.lcs( RTSummary=row["RTSummary"] , SystemSummary=row["SystemSummary"]))

            temp["N-FSR"] = pd.Series(N_FSR).values
            temp["N-LSR"] = pd.Series(N_LSR).values
            temp["N-SR"] = pd.Series(N_SR).values
            temp["L-FSR"] = pd.Series(L_FSR).values
            temp["L-LSR"] = pd.Series(L_LSR).values
            temp["L-SR"] = pd.Series(L_SR).values
            temp.to_pickle(path="RougeScores"+os.path.sep+MeasureType+"_ROUGE.pkl")
            print("Measure Used : " + str(MeasureType))
            print("Average F1 Score for System Summary using ROUGE - N : " + str(self.calculateAvgF1(f1s=temp["N-SR"].tolist())))
            print("Average F1 Score for First Sentence using ROUGE - N : " + str(self.calculateAvgF1(f1s=temp["N-FSR"].tolist())))
            print("Average F1 Score for Last Sentence using ROUGE - N : " + str(self.calculateAvgF1(f1s=temp["N-LSR"].tolist())))
            print("Average ROUGE-L score for System Summary : " + str(self.calculateAvgF1(f1s=temp["L-SR"].tolist())) )
            print("Average ROUGE-L score for First Sentence : " + str(self.calculateAvgF1(f1s=temp["L-FSR"].tolist())) )
            print("Average ROUGE-L score for Last Sentence : " + str(self.calculateAvgF1(f1s=temp["L-LSR"].tolist())) )
