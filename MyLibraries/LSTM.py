from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM
from neuralforecast.losses.pytorch import DistributionLoss


import DataSets


Target_df_new = DataSets.Target()
h = Target_df_new.index.nunique()
train = Target_df_new[:-23]
test = Target_df_new[-23:]

def E150():
    h = Target_df_new.index.nunique()
    models = [LSTM(h=h,input_size=-1,
                loss=DistributionLoss(distribution='Normal', level=[90]),
                max_steps=150,
                encoder_n_layers=2,
                encoder_hidden_size=200,
                context_size=10,
                decoder_hidden_size=200,
                decoder_layers=2,
                learning_rate=1e-3,
                scaler_type='standard',
                futr_exog_list=['onpromotion'])]

    target_model_150 = NeuralForecast(models=models, freq='D')
    target_model_150.fit(train)
    target_model_150.save(path="MLModel150",model_index=None,overwrite=True,save_dataset=True)
    return target_model_150

def E100():
    h = Target_df_new.index.nunique()
    models = [LSTM(h=h,input_size=-1,
                loss=DistributionLoss(distribution='Normal', level=[90]),
                max_steps=100,
                encoder_n_layers=2,
                encoder_hidden_size=200,
                context_size=10,
                decoder_hidden_size=200,
                decoder_layers=2,
                learning_rate=1e-3,
                scaler_type='standard',
                futr_exog_list=['onpromotion'])]

    target_model_100 = NeuralForecast(models=models, freq='D')
    target_model_100.fit(train)
    target_model_100.save(path="MLModel100", model_index=None,overwrite=True,save_dataset=True)
    return target_model_100
