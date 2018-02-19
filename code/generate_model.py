import pandas as pd 
from bs4 import BeautifulSoup
import re
import nltk.data 
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim.models.keyedvectors import KeyedVectors

def review_to_wordlist(review):
	review_text = BeautifulSoup(review).get_text()
	review_text = re.sub("[^a-zA-Z]"," ", review_text)
	words = review_text.lower().split()
	return (words)

def review_to_sentences(review, tokenizer):
	# use Nlt tokenizer to split the paragraph into sentences
	raw_sentences = tokenizer.tokenize(review.strip())
	sentences = []
	for raw_sentence in raw_sentences:
		# If a sentence is empty, skip it
		if len(raw_sentence) > 0:
			# Otherwise, call review_to_wordlist to get a list of words
			sentences.append(review_to_wordlist(raw_sentence))
	return sentences 

def generate_sentences(trainobj, unlabeledtrainobj):
	sentences = []
	print ("Parsing sentences from training set\n")
	for review in trainobj:
		sentences += review_to_sentences(review, tokenizer)

	print ("Parsing sentences from unlabeled set\n")
	for review in unlabeledtrainobj:
		sentences += review_to_sentences(review, tokenizer)
	return sentences 


if __name__ == '__main__':
	train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t",quoting=3)
	test = pd.read_csv("testData.tsv", header=0, delimiter="\t",quoting=3)
	unlabeled_train = pd.read_csv("unlabeledTrainData.tsv", header=0, delimiter="\t",quoting=3)
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	print("generating sentences...\n")
	sents = generate_sentences(train["review"], unlabeled_train["review"])
	print("training model...\n")
	model = gensim.models.Word2Vec(sents, size=300, min_count=40, workers=4, window=10, sample=1e-3)
	model.save("model")
	w2v = model.wv 
	w2v.save_word2vec_format('model_w2v_corpus.txt')
