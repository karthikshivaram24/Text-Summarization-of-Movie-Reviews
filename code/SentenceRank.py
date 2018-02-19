from similarityMeasures import SimilarityMeasures
from  NlpUtility import NLPWrapper
import networkx as nx
import pandas as pd
import numpy as np
import configparser as cp
import os

class RankSentences(object):
    def __init__(self,Measuretype,summarySize,filename):
        self.Measuretype = Measuretype
        self.SimMeasures = SimilarityMeasures()
        self.nlpWrapper = NLPWrapper()
        self.summarySize = summarySize
        self.filename =filename
        self.config = cp.ConfigParser()
        self.config.read("ConfigFile.properties")
        self.resultsFolder =self.config["ResultsFolderName"]["name"]
        self.ngramSize = self.config["NgramSize"]["n"]

        pass

    def pageRank(self,similarityMatrix,sentences,summarySize):
        """
        """
        pass
        nx_graph = nx.from_scipy_sparse_matrix(similarityMatrix)
        scores = nx.pagerank(nx_graph)
        nx_graph.clear()
        return sorted(((scores[i], s) for i, s in enumerate(sentences)),reverse=True)[:summarySize]

    def getSimMatrix(self,eachReviewSentences,Measuretype):
        """
        """
        pass
        list_csr_Matrix = []
        for sentences in eachReviewSentences:
            list_csr_Matrix.append(self.SimMeasures.createSimilarityMatrix(sentences=sentences,
                                                                            measure=Measuretype,ngramSize=self.ngramSize))
        return list_csr_Matrix

    def getReviewText(self,filename):
        """
        """
        df = pd.read_pickle(filename)
        return df["ReviewText"].tolist()

    def saveScoresAndSummary(self,filename,list_outputs,first,last,measureType):
        """
        """
        scores = []
        summaries = []
        OriginalDf = pd.read_pickle(filename)
        for output in list_outputs:
            scores.append(output[0])
            summaries.append(output[1])

        sc= pd.Series(scores)
        summ = pd.Series(summaries)
        f = pd.Series(first)
        l = pd.Series(last)
        OriginalDf["SystemSummary"] = summ
        OriginalDf["PrScores"] = sc
        OriginalDf["First-Sent"] = f
        OriginalDf["Last-Sent"] = l
        OriginalDf.to_pickle(self.resultsFolder+os.path.sep+measureType+"_"+"FinalDf.pkl")

    def combineReviewSents(self,score_sum_tup_list):
        """
        """
        scores, summary_sents = zip(*score_sum_tup_list)
        return tuple((np.mean(scores)," ".join(summary_sents)))

    def main(self):
        """
        """
        reviewsList = self.getReviewText(filename=self.filename)
        sentences_each_Review =[]
        scoreSummaryResults = []
        first_baseline = []
        last_baseline = []
        # Convert Review Text to sentences
        for review in reviewsList:
            sentences = self.nlpWrapper.sentenceExtractor(review)
            sentences = self.nlpWrapper.removeNAP(sentences)
            first_baseline.append(sentences[0])
            last_baseline.append(sentences[-1])
            sentences_each_Review.append(sentences)

        SimilarityMatrixes =self.getSimMatrix(eachReviewSentences=sentences_each_Review,Measuretype=self.Measuretype)

        for i,matrix in enumerate(SimilarityMatrixes):
            score_summary_tuple = self.pageRank(similarityMatrix=matrix,
                                                sentences=sentences_each_Review[i],
                                                summarySize=self.summarySize)
            if len(score_summary_tuple)>1:
                score_summary_tuple = self.combineReviewSents(score_sum_tup_list=score_summary_tuple)
            elif len(score_summary_tuple)==1:
                score_summary_tuple = score_summary_tuple[0]
            scoreSummaryResults.append(score_summary_tuple)
        self.saveScoresAndSummary(filename=self.filename,
                             list_outputs=scoreSummaryResults,
                             first=first_baseline,
                             last=last_baseline,
                             measureType=self.Measuretype)
