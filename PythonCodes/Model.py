# Core Libraries
import pandas as pd

#Prediction Libraries 
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM
from neuralforecast.losses.pytorch import DistributionLoss

# Main Table 
from PythonCodes.ModelData import *
from PythonCodes.PlattsPrediction import *

''' 
Model Web Url = https://nixtla.github.io/neuralforecast/models.lstm.html

- max_step = if laearning_rate is low, max_step should be high.
- Available scaler_type = 'standard','None' or 'robust'
- Available distributions = 'Poisson','Normal','StudentT','NegativeBinomial','Tweedie' or 'Bernoulli (Temporal Classifiers)'

Recommended Features are writing in here, But You can cahnge to improve your data set score
'''
def LSTMModelMain(TrainSize,Savecsv = None,TargetModel = None):
        
        ModelTargetDF_New = TargetModel

        ModelTrain = ModelTargetDF_New[:-30] # Train and test size aranged in here
        test = ModelTargetDF_New[-30:]

        ModelTrainSize = ModelTargetDF_New.index.nunique() #Forecast Horizon, Recommendation range is Target_df_new.index.nunique()

        LSTMModel = [LSTM(h=ModelTrainSize,input_size=-1,
                        loss=DistributionLoss(distribution='Normal', level=[80,90]),
                        max_steps=TrainSize,
                        encoder_n_layers=2,
                        encoder_hidden_size=200,
                        context_size=10,
                        decoder_hidden_size=200,
                        decoder_layers=2,
                        learning_rate=1e-4,
                        scaler_type='standard',
                        futr_exog_list=['y_[lag12]'])]

        TargetModelDF = NeuralForecast(models=LSTMModel, freq='D') # freq = Daily Prediciton
        TargetModelDF.fit(ModelTrain)

        ModelPrediction = TargetModelDF.predict(test).reset_index()

        PredicitionLimittation = ModelPrediction[:70]

        ModelPredicitionNew = PredicitionLimittation
        ModelPredicitionNew.rename(columns={"ds":"Date"}, inplace=True)
        ModelPredicitionNew["Date"] = ModelPredicitionNew["Date"].dt.strftime("%Y-%m-%d")

        SaveDf_new = ModelPredicitionNew[["Date","LSTM-median"]]
        submission = pd.read_csv(Savecsv)
        TargetModel_SaveDF = pd.merge(SaveDf_new,submission, left_index=True,right_index=True, how="inner")
        TargetModel_SaveDF = TargetModel_SaveDF[["Date_y","LSTM-median"]]
        TargetModel_SaveDF.columns = ["Date","Net Cashflow from Operations"]

        return TargetModel_SaveDF