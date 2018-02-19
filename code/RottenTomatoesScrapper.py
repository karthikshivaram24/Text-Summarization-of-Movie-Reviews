"""
This file contains the code to scrape rotten tomatoes for moview reviews as well as the corresponding links of the critic reviews.
It is the starting phase of our text summarization project.
"""
from newspaper import Article
import urllib # For url response
import bs4 # Beautiful Soup for html parsing
import pandas as pd # Pandas for organization of data
import time # To make the crawler sleep to prevent ip ban
import os # To create file and folder structures for our data
import math
import sys
import configparser as cp
import ast

class RTScrapper(object):

    def __init__(self,):
        """
        This is just the constructor for the RTScrapper object where we set up our user Agent for our headers parameters.
        This is done so that if any site prevents a program from automatically accessing it or if it has a robot.txt and you
        get an error like an "HTTP 403 Error - Forbidden" this will fix it.
        """
        self.headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'}
        self.base_url = "https://www.rottentomatoes.com"

    def mainUrls(self,movie_names,pickleFilename):
        """
        This method reads the given file with url's to rotten tomatoes movie reviews and calls the scrapper on each url to collect
        the links to all critic reviews as well as scrape the given one sentence summary of the critic review.

        Params:   MovieNames       : Names of movies to scrape reviews for.
                  pickleFilename   : The name of the file to store the dataframe

        Returns:  Nothing (Saves the pandas dataframe as a .pkl file)
        """
        # with open(filename,'r') as fp:
        #     movie_names = fp.readlines()

        No_of_Movies = len(movie_names)

        column_names = ["Id","MovieName","ReviewsLink","Total_No_of_Reviews"]

        df = pd.DataFrame(index = [x for x in range(No_of_Movies)] ,columns = column_names)
        df = df.fillna(0)

        for movie_index in range(No_of_Movies):
            req = urllib.request.Request(self.base_url+"/m/"+movie_names[movie_index], None, self.headers)
            opener = urllib.request.build_opener()
            response = opener.open(req)
            soup = bs4.BeautifulSoup(response,'lxml')
            linkAndReviewNos = soup.find('a',{"class":"view_all_critic_reviews"})
            link =  linkAndReviewNos['href']
            ReviewNos = linkAndReviewNos.text.split()[-1].replace("(","").replace(")","")
            df.loc[movie_index] = [movie_index,movie_names[movie_index],link,ReviewNos]

        movie2review = dict()
        total_reviews = 0
        for index, row in df.iterrows():
            link,page_nos = row["ReviewsLink"],row["Total_No_of_Reviews"]
            movie_name = row["MovieName"]
            pages = math.ceil(int(page_nos)/20)
            summary = []
            reviewLink = []
            print("Scrapping "+ movie_name.strip() + ":")
            for i in range(1,pages+1,1):
                req = urllib.request.Request(self.base_url+link+"?page="+str(i)+"&sort=", None, self.headers)
                opener = urllib.request.build_opener()
                response = opener.open(req)
                soup = bs4.BeautifulSoup(response,'lxml')
                cont = soup.find('div',{'class':'content'}).find('div',{'class':'review_table'})
                page_no = soup.find('div',{'class':'content'}).find('span',{'class':'pageInfo'}).text
                for row in cont.find_all('div',{'class':'row review_table_row'}):
                    if row != None:
                        content = row.find('div',{'class':'col-xs-16 review_container'}).find('div',{'class':'review_area'}).find('div',{'class':'review_desc'})
                        date = row.find('div',{'class':'col-xs-16 review_container'}).find('div',{'class':'review_area'}).find('div',{'class':'review_date subtle small'}).text
                        year = date.split(",")[-1].strip()
                        line_summary = content.find('div',{'class':'the_review'}).text
                        if(content.find('a') != None):
                            article_link = content.find('a')['href']
                        # Check date (select if only after 2008) because older links might not exist anymore
                        if ( int(year)>=2008 and (content.find('a') != None and "[Full review in Spanish]" not in line_summary)):
                            summary.append(content.find('div',{'class':'the_review'}).text)
                            reviewLink.append(content.find('a')['href'])
                            total_reviews+=1
            movie2review[movie_name]= {"summary":summary,"reviewLink":reviewLink}

        self.writeDftoFile(movie2review=movie2review,filename=pickleFilename)

        print("Retrieved " +str(total_reviews) + " reviews for " + str(No_of_Movies)+" Movies" )

    def scrapeCriticsReview(self,filename,sleepTime = 5):
        """
        This method basically scrapes the critic's review on the critics main page

        Params:   filename - The file containing a pandas dataframe stored as a pickle object

        Returns:  Nothing (Saves the summary to a .txt file)
        """
        overall_df = pd.read_pickle(filename)

        if not os.path.exists("ScrappedData"):
            os.makedirs("ScrappedData")

        for index, row in overall_df.iterrows():
            if index >= 25 and index % 25 == 0:
                print("Crawler is sleeping for "+str(sleepTime)+" seconds")
                time.sleep(sleepTime)
            url = row["ReviewLink"]
            if not url.startswith("http"):
                url = "http://"+url
            try:
                print("Scrapping -->  " + url)
                article = Article(url)
                article.download()
                article.parse()
                summary = article.text
                summary = self.cleanContent(reviewStripped=summary)
                self.saveReview(id=row["Id"],movieName=row["Movie"],Review=summary)
            except Exception as e:
                print(str(e))
            # if index >= 25 and index % 25 == 0:
            #     print("Crawler is sleeping for "+str(sleepTime)+" seconds")
            #     time.sleep(sleepTime)
            # url = row["ReviewLink"]
            # if not url.startswith("http"):
            #     url = "http://"+url
            # req = urllib.request.Request(url, None, self.headers)
            # try:
            #     print("Scrapping -->  " + url)
            #     opener = urllib.request.build_opener()
            #     response = opener.open(req)
            #     soup = bs4.BeautifulSoup(response,'lxml')
            #
            #     # Remove JavaScript and other unwanted Style elements
            #     for script in soup(["script", "style"]):
            #         script.extract()
            #
            #     body = soup.body
            #     para_list =[]
            #     for para in body.find_all('p'):
            #         para_list.append(para.text.strip())
            #     summary =  " ".join(para_list)
            #     summary = self.cleanContent(reviewStripped=summary)
            #     self.saveReview(id=row["Id"],movieName=row["Movie"],Review=summary)
            #
            # except Exception as e:
            #     print(str(e))

    def writeDftoFile(self,movie2review,filename):
        """
        This method just converts a dictionar to  a pandas dataframe and then to a pickle file.

        Params: movie2review : The dictionary containing movie to reviews pairs
                filename     : The name of the file to save as

        Returns: Nothing
        """
        overall_df = pd.DataFrame()
        id_ = 0
        for movie in movie2review:
            for entry1,entry2 in zip(movie2review[movie]["summary"],movie2review[movie]["reviewLink"]):
                    overall_df = overall_df.append({"Id":int(id_),"Movie":movie,"ReviewLink":entry2,"Summary":entry1},ignore_index=True)
                    id_+=1

        print("Saving DataFrame to File .........")
        overall_df.to_pickle(filename)

    def saveReview(self,id,movieName,Review):
        """
        This method saves the review string to a .txt file.

        Params: id        : The id of the review for the movie
                movieName : The name of the movie for which we are saving the review
                Review    : The string containing the review of the movie

        Returns: Nothing
        """
        if not os.path.exists("ScrappedData"+os.sep+movieName.strip("\n")):
            os.makedirs("ScrappedData"+os.sep+movieName.strip("\n"))

        with open("ScrappedData"+os.sep+movieName.strip("\n")+os.sep+str(int(id))+"_"+movieName.strip("\n")+".txt",'w') as fp:
            fp.write(Review)
            print("Written Summary to --> " + "ScrappedData"+os.sep+movieName.strip("\n")+os.sep+str(int(id))+"_"+movieName.strip("\n")+".txt")
        fp.close()

    def cleanContent(self,reviewStripped):
        """
        This method removes \\n,\\t and other space related characters
        from the scraped data.
        """
        wordList = reviewStripped.split()
        return " ".join(wordList)


    def readMovies(self):
        """
        This reads a .properties file to get the list of movies we need to scrape
        reviews for.

        Params : None
        Returns: movieNames: A list containing names of movies we want to scrape
        """
        config = cp.ConfigParser()
        config.read("ConfigFile.properties")
        movieNames = ast.literal_eval(config.get("MovieNames","moviesName"))
        return movieNames


    def main(self):
        print("\t-------------------------- Starting Rotten Tomatoes Scrapper --------------------------")
        movie_names = self.readMovies()
        self.mainUrls(movie_names=movie_names,pickleFilename="DFM2R.pkl")
        self.scrapeCriticsReview(filename="DFM2R.pkl")
        print("\t-------------------------- Finished Scrapping Rotten Tomatoes For Movie Reviews --------------------------")

if __name__=="__main__":
    scrapper = RTScrapper()
    scrapper.main()
