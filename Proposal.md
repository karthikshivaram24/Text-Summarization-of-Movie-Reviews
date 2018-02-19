## Overview

Sentiment Analysis of Movie Reviews using Text Summarization. We will be working on the problem on Text Summarization.  We can use text summarization techniques to get a complete picture of how a movie is actually doing by summarizing multiple reviews into one. And we can get a sentiment of the movie if it was positively accepted or not , so it could actually aid us if we wanted to do perform sentiment analysis of movie reviews.

## Data

1) Will Be using movie reviews for the initial period then move on to news articles

2) Will scrap it from the web using web scrapping

3) That movie reviews might not be long enough for the summarization to be done effectively

## Method

Will actually be looking into Few algorithms, haven't implemented them yet so still have to perform the trial and error phase. But the main algorithms we are investigating are basic summarization techniques using tf-idf scores, Text Rank based on Page Rank , RNN's and Sentence Compression. Mostly will only use library functions for text processing and for the RNN Implementations, other than that  our algorithm will be implemented by us. And for the Sentiment Analysis part we might modify existing libraries a little bit.

## Related Work

A Survey on Automatic Text Summarization -

https://www.cs.cmu.edu/~afm/Home_files/Das_Martins_survey_summarization.pdf



Text Summarization Using Lexical Chains

http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.15.9903&rep=rep1&type=pdf



Neural Text Summarization -

https://cs224d.stanford.edu/reports/urvashik.pdf



Text Summarization using Deep Learning and Ridge Regression - https://arxiv.org/ftp/arxiv/papers/1612/1612.08333.pdf



Graph-based Ranking Algorithms for Sentence Extraction,

Applied to Text Summarization -

http://delivery.acm.org/10.1145/1220000/1219064/a20-mihalcea.pdf?ip=69.246.204.126&id=1219064&acc=OPEN&key=4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E6D218144511F3437&CFID=898837596&CFTOKEN=45074316&__acm__=1486687469_183eec95b97c93af3ea56f93c4f2e327



SentiSummary: Sentiment Summarization for User

Product Reviews - 

http://www.seas.upenn.edu/~cse400/CSE400_2009_2010/final_report/Schaye_Feczko.pdf



Movie Review Mining and Summarization

https://pdfs.semanticscholar.org/d576/d9ea5cc898d2fb4e833a630e59ff02edc7a8.pdf



Automatic Text Summarization using a Machine

Learnig Approach

https://www.cs.kent.ac.uk/people/staff/aaf/pub_papers.dir/SBIA-2002-Joel.pdf



Large-Scale Sentiment Analysis for News and Blogs

http://www.uvm.edu/pdodds/files/papers/others/2007/godbole2007a.pdf

## Evaluation

Will be using Rouge - Recall-Oriented Understudy for Gisting Evaluation to evaluate our text summaries and will compare them to the online text summarizers and will check the precision and recall of our system compared to their's and will plot f1-measure for the sentiment analysis part and check the overall accuracy of our system compared to a normal sentiment analysis performed using a support vector machine.

