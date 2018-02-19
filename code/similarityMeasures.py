"""
This contains all the methods which measure similarity between 2 sentences,
"""
from collections import Counter
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from gensim.models import Word2Vec

from NlpUtility import NLPWrapper
import math
import itertools
import numpy
import configparser as cp
import ast

numpy.seterr(divide='ignore', invalid='ignore')

class SimilarityMeasures(object):
    def __init__(self):
        self.config = cp.ConfigParser()
        self.config.read("ConfigFile.properties")
        self.nlpWrapper = NLPWrapper()
        # self.origSimThreshold = float(ast.literal_eval(self.config.get("originalSimMeasure","threshold")))
        self.origSimThreshold = float(self.config["originalSimMeasure"]["threshold"])
        # self.nGramSimThreshold = float(ast.literal_eval(self.config.get("nGramSimMeasure","threshold")))
        self.nGramSimThreshold=float(self.config["nGramSimMeasure"]["threshold"])
        # self.w2vSimThreshold = float(ast.literal_eval(self.config.get("word2VecSimMeasure","threshold")))
        self.w2vSimThreshold=float(self.config["word2VecSimMeasure"]["threshold"])
        # self.BM25Threshold = float(ast.literal_eval(self.config.get("BM25Measure","threshold")))
        self.BM25Threshold=float(self.config["BM25Measure"]["threshold"])
        # self.jaccardThreshold = float(ast.literal_eval(self.config.get("JaccardMeasure","threshold")))
        self.jaccardThreshold=float(self.config["JaccardMeasure"]["threshold"])
        # self.w2vModelName = str(ast.literal_eval(self.config.get("W2VModel","modelName")))
        self.NgramSize = int(self.config["NgramSize"]["n"])
        self.w2vModelName = "model"
        pass

    def createSimilarityMatrix(self,sentences,ngramSize,measure="tf-idf"):
        """
        This method creates a similarity matrix between the sentences in the review
        Params: review: text containing the movie reviews
                measure: A string telling us which similarity measure to use
        returns: A matrix contaning similarity measures between each sentence
        """
        if measure=="tf-idf":
            nm = self.tfIdfSim(sentences)
            return nm * nm.T

        elif measure == "original":
            return self.originalSimMeasure(sentences=sentences)

        elif measure == "nGram":
            return self.nGramcountCosineSim(sentences=sentences,k=ngramSize)

        elif measure == "jaccard":
            return self.jaccardSimilarity(sentences=sentences)

        elif measure == "bm25plus":
            return self.bm25PlusScore(sentences,stem=True)
        #
        elif measure == "word2Vec":
            return self.w2vSim(modelName=self.w2vModelName,sentences=sentences)
        pass

    def tfIdfSim(self,sentences):
        """
        This method calculates the tf-Idf Similarity between 2 sentences
        Params: sentences: A list containing all sentences in our review
                reviewLength: The no of sentences in the review containing sentence 1 and sentence 2
        """
        c = CountVectorizer()
        bow_matrix = c.fit_transform(sentences)
        normalized_matrix = TfidfTransformer().fit_transform(bow_matrix)

        # Returning a matrix where row is sentence, columns are words with tf-idf scores
        return normalized_matrix

    # Word Overlap Measure
    def originalSimMeasure(self,sentences):
        """
        This method finds similarity between every sentence pair in the review
        using the original similarity measure used in the TextRank paper.

        link : https://arxiv.org/pdf/1602.03606.pdf

        Params: sentences: A list containing all the sentences
        """
        def simMeasure(sent1,sent2):
            """
            Helper method for similarity calculation
            Params: sent1: List contianing tokens of sentence 1
                    sent2: List containing tokens of sentence 2
            """
            commonWords = set(sent1).intersection(set(sent2))
            num = len(commonWords)
            # print("len1 - " + str(len(sent1))+ " len2 - " + str(len(sent2)))
            denom = 1.0
            if len(sent1) > 1 and len(sent2) > 1:
                denom = math.log(len(sent1)) + math.log(len(sent2))
            return num/denom


        # get tokens list for each sentence
        tokens_list = [self.nlpWrapper.tokenize(sent) for sent in sentences]

        tempMatrix = numpy.zeros(shape=(len(sentences),len(sentences)))

        for combo in itertools.product(range(len(sentences)),repeat = 2):
            sim_Measure = simMeasure(tokens_list[combo[0]],tokens_list[combo[1]])
            if sim_Measure >= self.origSimThreshold:
                tempMatrix[combo[0],combo[1]] = sim_Measure

        return csr_matrix(tempMatrix)

    def cos_sim(self,v1,v2):
        """
        This is just a helper function to calculate the cosine similarity between 2 vectors
        """
        return dot(v1, v2.T)/(norm(v1)*norm(v2))

    def nGramcountCosineSim(self,sentences,k=2):
        """
        This methods returns a csr-matrix where each row and column are represented
        by sentences in the review and the values are the cosine similarity between
        the count of ngrams of the sentences.

        Params: sentences: A list containing sentences found in the review
                k        : Size of ngram (default value of 2 given)
        returns: A csr Matrix containing similarity measures between each sentence pair
        """
        c = CountVectorizer(ngram_range=tuple((int(k),int(k))),dtype=numpy.float32)
        ngramMatrix = c.fit_transform(raw_documents=sentences)
        # Normalize matrix by row sum
        ngramMatrix = normalize(ngramMatrix, axis=1, norm='l1')
        tempMatrix = numpy.zeros(shape=(len(sentences),len(sentences)))
        for combo in itertools.product(range(len(sentences)),repeat = 2):
            cs = self.cos_sim(ngramMatrix[combo[0],].toarray(),ngramMatrix[combo[1],].toarray())
            if cs >= self.nGramSimThreshold:
                tempMatrix[combo[0],combo[1]] = cs

        return csr_matrix(tempMatrix)

    def jaccardSimilarity(self,sentences,stem=True):
        """
        This method contains the method to calculate jaccardSimilarity between sentences.

        Params:    sentences: A list containing all the sentences present in the review
                   stem: A Flag to check if we are stemming the words in the sentence or not

        Returns:   A csr Matrix containing jaccard similarity measures between each sentence pair
        """
        tokens = [self.nlpWrapper.tokenize(sentence) for sentence in sentences]
        temp =[]
        if stem==True:
            for t in tokens:
                temp.append(self.nlpWrapper.stemmer(tokens=t))
        elif stem==False:
            temp = tokens

        tempMatrix = numpy.zeros(shape=(len(sentences),len(sentences)))

        for combo in itertools.product(range(len(sentences)),repeat=2):
            num = len(set(temp[combo[0]]).intersection(set(temp[combo[1]])))
            denom = len(set(temp[combo[0]]) | set(temp[combo[1]]))
            if denom<=0 or num<=0:
                tempMatrix[combo[0],combo[1]] = 0.0
            else:
                if float(num/denom) >= self.jaccardThreshold:
                    tempMatrix[combo[0],combo[1]] = num/denom

        return csr_matrix(tempMatrix)

    def token_sent_mapper(self,tokens):
        """
        This is a helper method that creates a default dictionary with
        Key--> token , value--> sentences containing that token

        Params: tokens: A list of list of sentence tokens
        Returns: A defaultdict containing above mentioned key value pairs
        """
        result = defaultdict(list)

        for sent_id in range(len(tokens)):
            for tok in tokens[sent_id]:
                result[tok].append(sent_id)

        return result

    def idfCal(self,N,term,token_sent_dic):
        """
        This is a inverse document frequency calculater, helper method for bm25scorer
        Params: N - Total no of documents
                term - the word to calculate the idf of
                token_sent_dic -  A dictionary containing the  token to sentences  mapping
        Returns: The inverse document frequency score of the given term
        """
        temp = (N /( 1 + len(token_sent_dic[term])))
        if temp >0:
            return math.log(temp)
        else:
            # Since natural log of zero is undefined
            return 0


    def bm25PlusScore(self,sentences,stem=True,k1=1.2,b=0.75):
        """
        This method computes the bm25+ scores between 2 given pairs of sentences, usually bm25 is
        a ranking function that calculates a rank score for documents in search engines related to
        the search query asked, here we are using it on sentences pairs

        link: https://en.wikipedia.org/wiki/Okapi_BM25

        Params: sentences: A list containing all the sentences in our review
                k1       : A user defined parameter for bm25 ( in range[1.2,2.0])
                b        : A user defined parameter for bm25 ( usualy set to 0.75)

        Returns: A csr Matrix containing the bm25 scores between each sentence pair
        """
        # Needs IDF Scores, Freq, avgdl(average document length in the corpus)

        tokens = [self.nlpWrapper.tokenize(sentence) for sentence in sentences]
        temp =[]
        if stem==True:
            for t in tokens:
                temp.append(self.nlpWrapper.stemmer(tokens=t))
        elif stem==False:
            temp = tokens

        token_sent_dic = self.token_sent_mapper(temp)

        # Average Sentence Length in the review
        sum =0
        for sent in temp:
            sum+=len(sent)
        avdl = sum/len(temp)

        tempMatrix = numpy.zeros(shape=(len(sentences),len(sentences)))

        for combo in itertools.product(range(len(sentences)),repeat=2):
            document = combo[0]
            query = combo[1]
            count_q_t = Counter(temp[combo[0]])
            score_sum = 0
            for i in range(len(temp[combo[1]])):
                idf = self.idfCal(N=len(temp),term=temp[combo[1]][i],token_sent_dic=token_sent_dic)
                t = 0
                if temp[combo[1]][i] in count_q_t:
                    t = count_q_t[temp[combo[1]][i]]

                num =  t * (k1 + 1)
                denom = t +(k1 *((1-b)+(b*(len(temp[combo[0]])/avdl))))
                score_sum+= idf * (num/denom)
            if score_sum >= self.BM25Threshold:
                tempMatrix[combo[0],combo[1]] = score_sum

        return csr_matrix(tempMatrix)

    def calcDoc2Vec(self,w2vModel,sentence_tokens):
        """
        Gets the average vector for the sentence using word2vec scores
        from our trained model

        Params: w2vModel: Our pre-trained word2vec model
                sentence_tokens: list of tokens for a given sentence

        Returns: A vector for the given sentence
        """
        def makeFeatureVec(words, model, num_features):
            featureVec = numpy.zeros((num_features,),dtype="float64")
            nwords = 0.
            index2word_set = set(model.wv.vocab)
            for word in words:
                if word in index2word_set:
                    nwords = nwords + 1.
                    featureVec = numpy.add(featureVec,model[word])
            if nwords != 0.:
                featureVec = numpy.divide(featureVec,nwords)
            return featureVec

        return makeFeatureVec(words=sentence_tokens,model=w2vModel,num_features=300)


    def w2vSim (self,modelName,sentences):
        """
        Creates a csr matrix containing similarity between sentences of a review scores
        using cosine similarity between 2 sentence vectors that were obtained using
        word2vec

        Params: modelName : Name of the pretrained word2vec model
                sentences: List of sentences of a review

        Returns: A csr Matrix containing cosine sim between 2 sentence pairs whose
                 vectors were obtained using word2vec.
        """

        tokens = [self.nlpWrapper.tokenize(sentence) for sentence in sentences]
        m = Word2Vec.load(modelName)

        tempMatrix = numpy.zeros(shape=(len(sentences),len(sentences)))
        for combo in itertools.product(range(len(sentences)),repeat=2):
            sen1_vec = self.calcDoc2Vec(w2vModel=m,sentence_tokens=tokens[combo[0]])
            sen2_vec = self.calcDoc2Vec(w2vModel=m,sentence_tokens=tokens[combo[1]])
            if numpy.count_nonzero(sen1_vec)>=1 and numpy.count_nonzero(sen2_vec)>=1 :
                cosine_sim = cosine_similarity(sen1_vec.reshape(1,-1) ,sen2_vec.reshape(1,-1) )
                if cosine_sim >= self.w2vSimThreshold:
                    tempMatrix[combo[0],combo[1]] = cosine_sim

        return csr_matrix(tempMatrix)

    # def lexRankSimilarity(self,sentences,stem=True):
    #     """
    #     """
    #     tokens = [self.nlpWrapper.tokenize(sentence) for sentence in sentences]
    #     temp =[]
    #     if stem==True:
    #         for t in tokens:
    #             temp.append(self.nlpWrapper.stemmer(tokens=t))
    #     elif stem==False:
    #         temp = tokens
    #
    #     token_sent_dic = self.token_sent_mapper(temp)
    #
    #     tempMatrix = numpy.zeros(shape=(len(sentences),len(sentences)))
    #
    #     for combo in itertools.product(range(len(sentences)),repeat=2):
    #         commonWords = set(temp[combo[0]]).intersection(temp[combo[1]])
    #         sent1Counter = Counter(temp[combo[0]])
    #         sent2Counter = Counter(temp[combo[1]])
    #         sum_n = 0
    #         for w in commonWords:
    #             sum_n+= sent1Counter[w] * sent2Counter[w] * (self.idfCal(N=len(temp),
    #                                                                     term=w,
    #                                                                     token_sent_dic=token_sent_dic)) **2
    #
    #         denom = () * ()
    #     pass
