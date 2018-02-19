from RottenTomatoesScrapper import RTScrapper
from SentenceRank import RankSentences
from connector import Organizer
from evaluation import Evaluation
import configparser as cp
import ast

MeaureTypes = ["tf-idf","original","nGram","jaccard","bm25plus","word2Vec"]

# MeaureTypes = ["tf-idf","original","nGram","jaccard","bm25plus"]
# MeaureTypes = ["nGram"]

def getConfigParams():
    """
    Method retrieves the summarySize from config file
    """
    config = cp.ConfigParser()
    config.read("ConfigFile.properties")
    # summarySize = ast.literal_eval(config.get("SummarySize","size"))
    summarySize = int(config["SummarySize"]["size"])
    Scores_folder = config["ResultsFolderName"]["name"]
    return summarySize,Scores_folder

def main():
    # Step1: Collect Data (Uncomment this, if you dont have data)
    # Change movie names in config.properties for the ones you want
    print("---- Starting Scrapping Module ---")
    scrapper = RTScrapper()
    scrapper.main()
    print("\n")
    print("---- Completed Scrapping of Movie Reviews ----")
    # Step2: Create Complete DataFrame with all the information
    connector = Organizer()
    connector.connectDFtoReview(df_filename="DFM2R.pkl",reviewFolder="ScrappedData")
    print("\n")
    print("---- Starting Raking Module ----")
    # Step3: Run Weighted Page Rank for scores
    # ss = getConfigParams()
    summarySize,Scores_folder=getConfigParams()
    for Measure in MeaureTypes:
        print("-- Using " + str(Measure)+" --")
        ranker = RankSentences(Measuretype=Measure,summarySize=summarySize,filename="CompleteData.pkl")
        ranker.main()
    print("\n")
    print("---- Completed Ranking of Sentences ----")
    # Final Step evaluate
    print("\n")
    print("---- Evaluating Summaries ----")
    evaluate = Evaluation()
    evaluate.main(folderName=Scores_folder)
    print("\n")
    print("---- Completed Evaluation ----")


if __name__=="__main__":
    main()
