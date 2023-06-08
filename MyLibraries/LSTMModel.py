# Core Libraries
import pandas as pd 

from neuralforecast import NeuralForecast
import pickle

from MyLibraries.DataSets import *


submission = pd.read_csv("shell-datathon-cash-flow-coderspace/sample_submission.csv")

Target_df_new = Target()


h = Target_df_new.index.nunique()

train = Target_df_new[:-23]
test = Target_df_new[-23:]

def LSTM_model():

    target_model = NeuralForecast.load("./MLModel/")

    prediction = target_model.predict(test).reset_index()

    predicition_limitied = prediction[:70]

    predicition_new = predicition_limitied
    predicition_new.rename(columns={"ds":"Date"}, inplace=True)
    predicition_new["Date"] = predicition_new["Date"].dt.strftime("%Y-%m-%d")

    submission_new = predicition_new[["Date","LSTM-median"]]

    target_submission = pd.merge(submission_new,submission, left_index=True,right_index=True, how="inner")
    target_submission = target_submission[["Date_y","LSTM-median"]]
    target_submission.columns = ["Date","Net Cashflow from Operations"]


    return target_submission