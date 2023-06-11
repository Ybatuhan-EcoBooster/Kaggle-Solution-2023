import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM
from neuralforecast.losses.pytorch import DistributionLoss

from MyLibraries.DataSets import *

Target_df_new = Target()

train = Target_df_new[:-23]
test = Target_df_new[-23:]
h = Target_df_new.index.nunique()


def LSTMModelMain(max_step,submissioncsv = None):
        h = Target_df_new.index.nunique()
        models = [LSTM(h=h,input_size=-1,
                        loss=DistributionLoss(distribution='Normal', level=[90]),
                        max_steps=max_step,
                        encoder_n_layers=2,
                        encoder_hidden_size=200,
                        context_size=10,
                        decoder_hidden_size=200,
                        decoder_layers=2,
                        learning_rate=1e-3,
                        scaler_type='standard',
                        futr_exog_list=['onpromotion'])]

        target_model = NeuralForecast(models=models, freq='D')
        target_model.fit(train)

        prediction = target_model.predict(test).reset_index()

        predicition_limitied = prediction[:70]

        predicition_new = predicition_limitied
        predicition_new.rename(columns={"ds":"Date"}, inplace=True)
        predicition_new["Date"] = predicition_new["Date"].dt.strftime("%Y-%m-%d")

        submission_new = predicition_new[["Date","LSTM-median"]]
        submission = pd.read_csv(submissioncsv)
        target_submission = pd.merge(submission_new,submission, left_index=True,right_index=True, how="inner")
        target_submission = target_submission[["Date_y","LSTM-median"]]
        target_submission.columns = ["Date","Net Cashflow from Operations"]

        return target_submission