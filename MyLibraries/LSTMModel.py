# Core Libraries
import pandas as pd

#Prediction Libraries 
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM
from neuralforecast.losses.pytorch import DistributionLoss

# Main Table 
from MyLibraries.DataSets import *
from MyLibraries.Platts import *


''' 
Model Web Url = https://nixtla.github.io/neuralforecast/models.lstm.html

- max_step = if laearning_rate is low, max_step should be high.
- Available scaler_type = 'standard','None' or 'robust'
- Available distributions = 'Poisson','Normal','StudentT','NegativeBinomial','Tweedie' or 'Bernoulli (Temporal Classifiers)'

Recommended Features are writing in here
'''
def LSTMModelMain(max_step,submissioncsv = None,target = None):
        
        Target_df_new = target

        train = Target_df_new[:-23] # Train and test size aranged in here
        test = Target_df_new[-23:]

        h = Target_df_new.index.nunique() #Forecast Horizon, Recommendation range is Target_df_new.index.nunique()

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

        target_model = NeuralForecast(models=models, freq='D') # freq = Daily Prediciton
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