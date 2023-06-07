# Core Libraries
import pandas as pd 
import numpy as np

# Prediction Libraries
from xgboost import XGBRegressor
from mlforecast import MLForecast
from window_ops.rolling import rolling_mean, rolling_max, rolling_min

def Platts():
    Platts = pd.read_csv("shell-datathon-cash-flow-coderspace/platts.csv")
    Platts["Tarih"] = pd.to_datetime(Platts['Tarih'])
    Platts['Tarih'] = pd.DatetimeIndex(Platts["Tarih"].dt.strftime("%Y-%m-%d"))

    Platts = Platts.iloc[:,:-1].copy()
    Platts.rename(columns={"Tarih":"Date"}, inplace=True)
    Platts = Platts[::-1]
    Platts.set_index("Date", inplace=True)
    Platts10PPM = Platts[Platts["Ürün"] == "10 ppm ULSD CIF Med (Genova/Lavera)" ]
    PlattsPremunl10PPM = Platts[Platts["Ürün"] == "Prem Unl 10 ppm CIF Med (Genova/Lavera)"]


    Platts10PPM = Platts10PPM.reset_index()
    Platts10PPM = Platts10PPM[["Date","Ürün","AB Piyasa Fiyatı","AB Piyasa Fiyatı- Yüksek","AB Piyasa Fiyatı- Düşük"]]
    Platts10PPM.columns = ["ds","unique_id","y","High","Low",]

    Platts10PPM_1 = Platts10PPM[["ds","unique_id","y"]]
    Platts10PPM_2 = Platts10PPM[["ds","unique_id","High"]]
    Platts10PPM_2.columns = ["ds","unique_id","y"]
    Platts10PPM_3 = Platts10PPM[["ds","unique_id","Low"]]
    Platts10PPM_3.columns = ["ds","unique_id","y"]

    PlattsPremunl10PPM = PlattsPremunl10PPM.reset_index()
    PlattsPremunl10PPM = PlattsPremunl10PPM[["Date","Ürün","AB Piyasa Fiyatı","AB Piyasa Fiyatı- Yüksek","AB Piyasa Fiyatı- Düşük"]]
    PlattsPremunl10PPM.columns = ["ds","unique_id","y","High","Low",]

    PlattsPremunl10PPM_1 = PlattsPremunl10PPM[["ds","unique_id","y"]]
    PlattsPremunl10PPM_2 = PlattsPremunl10PPM[["ds","unique_id","High"]]
    PlattsPremunl10PPM_2.columns = ["ds","unique_id","y"]
    PlattsPremunl10PPM_3 = PlattsPremunl10PPM[["ds","unique_id","Low"]]
    PlattsPremunl10PPM_3.columns = ["ds","unique_id","y"]

    Platts_model_list = [Platts10PPM_1,Platts10PPM_2,Platts10PPM_3,PlattsPremunl10PPM_1,PlattsPremunl10PPM_2,PlattsPremunl10PPM_3]

    models = [XGBRegressor(random_state=0, n_estimators=100)]

    model = MLForecast(models=models,
                    freq='D',
                    lags=[1,7,14],
                    lag_transforms={
                        1: [(rolling_mean, 7), (rolling_max, 7), (rolling_min, 7)],
                    },
                    date_features=['dayofweek', 'day'],
                    num_threads=6)


    Plattes_Models = [] 
    for i in Platts_model_list:
        models_list = model.fit(i, id_col='unique_id', time_col='ds', target_col='y', static_features=[])
        Plattes_Models.append(models_list)

    Platts10PPM_1_prediciton =  Plattes_Models[0].predict(horizon=32)
    Platts10PPM_1_prediciton.columns = ["unique_id","ds","y"]
    Platts10PPM_2_prediciton =  Plattes_Models[1].predict(horizon=32)
    Platts10PPM_2_prediciton.columns = ["unique_id","ds","y"]
    Platts10PPM_3_prediciton =  Plattes_Models[2].predict(horizon=32)
    Platts10PPM_3_prediciton.columns = ["unique_id","ds","y"]

    Platts10PPM_1 = Platts10PPM_1.append(Platts10PPM_1_prediciton,ignore_index = True)
    Platts10PPM_1.columns = ["ds","unique_id","AB Piyasa Fiyatı"]
    Platts10PPM_2 = Platts10PPM_2.append(Platts10PPM_2_prediciton,ignore_index = True)
    Platts10PPM_2.columns = ["ds","unique_id","AB Piyasa Fiyatı-Yüksek"]
    Platts10PPM_3 = Platts10PPM_3.append(Platts10PPM_3_prediciton,ignore_index = True)
    Platts10PPM_3.columns = ["ds","unique_id","AB Piyasa Fiyatı-Düşük"]

    Platts10PPM = pd.merge(Platts10PPM_1, Platts10PPM_2,on=["ds","unique_id"], how='inner')
    Platts10PPM = pd.merge(Platts10PPM,Platts10PPM_3,on=["ds","unique_id"], how='inner')
    Platts10PPM = Platts10PPM[["ds","AB Piyasa Fiyatı","AB Piyasa Fiyatı-Yüksek","AB Piyasa Fiyatı-Düşük"]]
    Platts10PPM.columns = ["Date","AB Piyasa FiyatıPP","AB Piyasa Fiyatı-YüksekPP","AB Piyasa Fiyatı-DüşükPP"]

    #############################################################################################################

    PlattsPremunl10PPM_1_prediciton =  Plattes_Models[3].predict(horizon=32)
    PlattsPremunl10PPM_1_prediciton.columns = ["unique_id","ds","y"]
    PlattsPremunl10PPM_2_prediciton =  Plattes_Models[4].predict(horizon=32)
    PlattsPremunl10PPM_2_prediciton.columns = ["unique_id","ds","y"]
    PlattsPremunl10PPM_3_prediciton =  Plattes_Models[5].predict(horizon=32)
    PlattsPremunl10PPM_3_prediciton.columns = ["unique_id","ds","y"]

    PlattsPremunl10PPM_1 = PlattsPremunl10PPM_1.append(PlattsPremunl10PPM_1_prediciton,ignore_index = True)
    PlattsPremunl10PPM_1.columns = ["ds","unique_id","AB Piyasa Fiyatı"]
    PlattsPremunl10PPM_2 = PlattsPremunl10PPM_2.append(PlattsPremunl10PPM_2_prediciton,ignore_index = True)
    PlattsPremunl10PPM_2.columns = ["ds","unique_id","AB Piyasa Fiyatı-Yüksek"]
    PlattsPremunl10PPM_3 = PlattsPremunl10PPM_3.append(PlattsPremunl10PPM_3_prediciton,ignore_index = True)
    PlattsPremunl10PPM_3.columns = ["ds","unique_id","AB Piyasa Fiyatı-Düşük"]

    PlattsPremunl10PPM = pd.merge(PlattsPremunl10PPM_1, PlattsPremunl10PPM_2,on=["ds","unique_id"], how='inner')
    PlattsPremunl10PPM = pd.merge(PlattsPremunl10PPM,PlattsPremunl10PPM_3,on=["ds","unique_id"], how='inner')
    PlattsPremunl10PPM= PlattsPremunl10PPM[["ds","AB Piyasa Fiyatı","AB Piyasa Fiyatı-Yüksek","AB Piyasa Fiyatı-Düşük"]]
    PlattsPremunl10PPM.columns = ["Date","AB Piyasa FiyatıPPrem","AB Piyasa Fiyatı-YüksekPPrem","AB Piyasa Fiyatı-DüşükPPrem"]


    Platts = PlattsPremunl10PPM.merge(Platts10PPM, on=["Date"], how="inner")
    
    return Platts
    
Brent = pd.read_csv("shell-datathon-cash-flow-coderspace/brent.csv")
Brent["Tarih"] = pd.to_datetime(Brent["Tarih"])
Brent["Tarih"] = pd.DatetimeIndex(Brent["Tarih"].dt.strftime("%Y-%m-%d"))
Brent = Brent.iloc[:,:-1]
Brent.set_index("Tarih", inplace=True)
Brent = Brent.reset_index()
Brent.rename(columns={"Tarih":"Date"},inplace= True)
    

def CashFlow():  
    CashFlow = pd.read_csv("shell-datathon-cash-flow-coderspace/cash_flow_train.csv")
    CashFlow["Inflows- currency"] = CashFlow["Inflows- currency"].replace(np.nan,0)
    CashFlow["Date"] = pd.to_datetime(CashFlow["Date"])
    CashFlow["Date"] = pd.DatetimeIndex(CashFlow["Date"].dt.strftime("%Y-%m-%d"))
    CashFlow.set_index("Date", inplace=True)
    CashFlow = CashFlow.reset_index()
    
    return CashFlow

def Currency():
    Currency = pd.read_csv("shell-datathon-cash-flow-coderspace/usd.csv")
    Currency["Tarih"] = pd.to_datetime(Currency["Tarih"])
    Currency["Tarih"] = pd.DatetimeIndex(Currency["Tarih"].dt.strftime("%Y-%m-%d"))
    Currency.rename(columns={"Tarih":"Date"},inplace=True)
    Currency.set_index("Date", inplace=True)
    Currency = Currency.fillna(method="ffill")
    Currency = Currency.reset_index()

    return Currency



df = pd.merge(CashFlow(), Currency(),on="Date", how='inner')
df = pd.merge(df,Brent,on="Date",how='inner')



platts = Platts()

def Target():
    
    df['Date'] = pd.to_datetime(df['Date'])
    platts['Date'] = pd.to_datetime(platts['Date'])

    merged_df = df.merge(platts, on='Date', how='inner')

    Target_df_new = merged_df[["Date",'Customers - DDS','Customers - EFT','T&S Collections','FX Sales','Other operations','Tüpraş',
                'Other Oil','Gas','Import payments (FX purchases)','Tax','Operatioınal and Admin. Expenses','VIS Buyback Payments','Net Cashflow from Operations',
                'Inflows- currency','USD ALIŞ','USD SATIŞ','EUR ALIŞ','EUR SATIŞ', 'GBP ALIŞ', 'GBP SATIŞ','Ürün',
                'AB Piyasa Fiyatı','AB Piyasa Fiyatı- Yüksek', 'AB Piyasa Fiyatı- Düşük','AB Piyasa FiyatıPP', 'AB Piyasa Fiyatı-YüksekPP','AB Piyasa Fiyatı-DüşükPP',
                "AB Piyasa FiyatıPPrem","AB Piyasa Fiyatı-YüksekPPrem","AB Piyasa Fiyatı-DüşükPPrem"]].copy()

    Target_df_new.rename(columns={"Date":"ds","Ürün":"unique_id","Net Cashflow from Operations":"y"}, inplace= True) #y = predicton Column
    Target_df_new["unique_id"] = "Shell"

    return Target_df_new
