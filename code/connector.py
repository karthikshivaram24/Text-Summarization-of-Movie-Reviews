import pandas as pd
import os
import configparser as cp
import ast

class Organizer(object):
    def __init__(self,):
        self.config = cp.ConfigParser()
        self.config.read("ConfigFile.properties")
        self.minchar = int(ast.literal_eval(self.config.get("MinimumCharacReview","minchar")))
        pass

    def readReviews(self,id,movieName,path):
        """
        Reads the review file of a given movie then returns the text in it
        """
        fileName = path+os.path.sep+movieName+os.path.sep+str(int(id))+"_"+movieName+".txt"
        summary = ""
        if os.path.exists(fileName):
            with open(fileName,"r") as fp:
                summary =fp.read()
            fp.close()
        return summary

    def connectDFtoReview(self,df_filename,reviewFolder):
        """
        This method reads a pandas dataframe from a pickle file and gets
        the corresponding summary sentence for each extracted review
        Params: df_filename : Name of pickle file containing the dataframe saved during collection
                reviewFolder: The name of the folder containing the movie reviews
        """
        pass
        df_without_reviews = pd.read_pickle(df_filename)
        column_names = ["Id","MovieName","ReviewText","RTSummary"]
        df_with_reviews = pd.DataFrame(columns = column_names)
        row_count = 0
        for index, row in df_without_reviews.iterrows() :
            summary = self.readReviews(id=row["Id"],movieName=row["Movie"],path="ScrappedData")
            if len(summary)>=self.minchar:
                df_with_reviews.loc[row_count] = [row["Id"],row["Movie"],summary,row["Summary"]]
                row_count+=1

        df_with_reviews.to_pickle("CompleteData.pkl")
