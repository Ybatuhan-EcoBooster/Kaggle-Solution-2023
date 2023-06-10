import pandas as pd
from neuralforecast import NeuralForecast
from MyLibraries.DataSets import *

Target_df_new = Target()

train = Target_df_new[:-23]
test = Target_df_new[-23:]

def LSTM_model150():
    target_model = NeuralForecast.load('MLModel150')

    prediction = target_model.predict(test).reset_index()

    predicition_limitied = prediction[:70]

    predicition_new = predicition_limitied
    predicition_new.rename(columns={"ds":"Date"}, inplace=True)
    predicition_new["Date"] = predicition_new["Date"].dt.strftime("%Y-%m-%d")

    submission_new = predicition_new[["Date","LSTM-median"]]
   
    submission = pd.read_csv("shell-datathon-cash-flow-coderspace\sample_submission.csv")
    target_submission = pd.merge(submission_new,submission, left_index=True,right_index=True, how="inner")
    target_submission = target_submission[["Date_y","LSTM-median"]]
    target_submission.columns = ["Date","Net Cashflow from Operations"]

    return target_submission

def LSTM_model100():
    target_model = NeuralForecast.load('MLModel100')

    prediction = target_model.predict(test).reset_index()

    predicition_limitied = prediction[:70]

    predicition_new = predicition_limitied
    predicition_new.rename(columns={"ds":"Date"}, inplace=True)
    predicition_new["Date"] = predicition_new["Date"].dt.strftime("%Y-%m-%d")

    submission_new = predicition_new[["Date","LSTM-median"]]
   
    submission = pd.read_csv("shell-datathon-cash-flow-coderspace\sample_submission.csv")
    target_submission = pd.merge(submission_new,submission, left_index=True,right_index=True, how="inner")
    target_submission = target_submission[["Date_y","LSTM-median"]]
    target_submission.columns = ["Date","Net Cashflow from Operations"]
    
    return target_submission


