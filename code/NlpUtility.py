"""
This file contains all the nlp utility methods used on the extracted data from the
webscrapper, its mostly just a wrapper class for nltk's nlp utility methods.
"""
import nltk
from nltk import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
import re
import string

nltk.download('punkt')
nltk.download("stopwords")

class NLPWrapper(object):

    stopset = set(stopwords.words('english'))
    ps = nltk.stem.porter.PorterStemmer()

    def init(self,):
        self.punc_dic = str.maketrans({key: None for key in string.punctuation})
        pass

    def sentenceExtractor(self,text):
        """
        This method finds the sentence boundary in the text and returns a list
        of sentences.

        Params : text     : a document, paragraph in the form of a string
        returns: Sentences: a list of sentences
        """
        sentences =[]
        for sentence in  sent_tokenize(text):
            if len(sentence.strip())>1:
                sentences.append(sentence.lower())

        return sentences


    def removeNAP(self,sentences):
        """
        This method removes all instance of non alphabetic characters and multiple
        spaces.

        Params : sentences: List of sentences
        Returns: cleanSentences: List of cleaned sentences
        """
        punc_dic = {key: " " for key in string.punctuation}
        table = str.maketrans(punc_dic)

        for sent_index in range(len(sentences)):
            sentences[sent_index] = sentences[sent_index].translate(table)
            sentences[sent_index] = re.sub(r'[0-9|]',"",sentences[sent_index])
            sentences[sent_index] = re.sub(r' +'," ",sentences[sent_index])

        return sentences

    def tokenize(self,sentence,removeStopWords = True):
        """
        This method is just a wrapper for the nltk tokenizer
        Params: sentence: Clean sentence
        Returns: words: List of words present in the sentence
        """
        if removeStopWords == True:
            tokens = word_tokenize(sentence.lower())
            tokens = [w for w in tokens if not w in NLPWrapper.stopset]
            return tokens
        else:
            return  word_tokenize(sentence)

    def stemmer(self,tokens):
        """
        Wrapper for nltk's porter stemmer

        Params: tokens: A list of tokens of a given sentence
        Returns: A list of stemmed Tokens
        """
        # return [NLPWrapper.ps.stem(x) for x in tokens]
        return tokens
