#Core Libraries
import pandas as pd

# Prediction Libraries
from xgboost import XGBRegressor
from mlforecast import MLForecast
from window_ops.rolling import rolling_mean, rolling_max, rolling_min


# Prediciton of Platts
def Platts(plattscsv = None):
    PlattsDF = pd.read_csv(plattscsv)
    PlattsDF["Tarih"] = pd.to_datetime(PlattsDF['Tarih'])
    PlattsDF['Tarih'] = pd.DatetimeIndex(PlattsDF["Tarih"].dt.strftime("%Y-%m-%d"))

    PlattsDF = PlattsDF.iloc[:,:-1].copy()
    PlattsDF.rename(columns={"Tarih":"Date"}, inplace=True)
    PlattsDF = PlattsDF[::-1]
    PlattsDF.set_index("Date", inplace=True)
    Platts10PPMDF = PlattsDF[PlattsDF["Ürün"] == "10 ppm ULSD CIF Med (Genova/Lavera)" ]
    PlattsPremunl10PPMDF = PlattsDF[PlattsDF["Ürün"] == "Prem Unl 10 ppm CIF Med (Genova/Lavera)"]


    Platts10PPMDF = Platts10PPMDF.reset_index()
    Platts10PPMDF = Platts10PPMDF[["Date","Ürün","AB Piyasa Fiyatı","AB Piyasa Fiyatı- Yüksek","AB Piyasa Fiyatı- Düşük"]]
    Platts10PPMDF.columns = ["ds","unique_id","y","High","Low",]

    Platts10PPM_1_ModelDF = Platts10PPMDF[["ds","unique_id","y"]]
    Platts10PPM_2_ModelDF = Platts10PPMDF[["ds","unique_id","High"]]
    Platts10PPM_2_ModelDF.columns = ["ds","unique_id","y"]
    Platts10PPM_3_ModelDF = Platts10PPMDF[["ds","unique_id","Low"]]
    Platts10PPM_3_ModelDF.columns = ["ds","unique_id","y"]

    PlattsPremunl10PPMDF = PlattsPremunl10PPMDF.reset_index()
    PlattsPremunl10PPMDF = PlattsPremunl10PPMDF[["Date","Ürün","AB Piyasa Fiyatı","AB Piyasa Fiyatı- Yüksek","AB Piyasa Fiyatı- Düşük"]]
    PlattsPremunl10PPMDF.columns = ["ds","unique_id","y","High","Low",]

    PlattsPremunl10PPM_1_ModelDF = PlattsPremunl10PPMDF[["ds","unique_id","y"]]
    PlattsPremunl10PPM_2_ModelDF = PlattsPremunl10PPMDF[["ds","unique_id","High"]]
    PlattsPremunl10PPM_2_ModelDF.columns = ["ds","unique_id","y"]
    PlattsPremunl10PPM_3_ModelDF = PlattsPremunl10PPMDF[["ds","unique_id","Low"]]
    PlattsPremunl10PPM_3_ModelDF.columns = ["ds","unique_id","y"]

    PlattsModelList = [Platts10PPM_1_ModelDF,Platts10PPM_2_ModelDF,Platts10PPM_3_ModelDF,PlattsPremunl10PPM_1_ModelDF,PlattsPremunl10PPM_2_ModelDF,PlattsPremunl10PPM_3_ModelDF]

    ''' XGBoost web url = https://xgboost.readthedocs.io/en/stable/
        MlForecast web url = https://nixtla.github.io/mlforecast/
    '''

    PlattModels = [XGBRegressor(random_state=0, n_estimators=100)]

    PlattesTargetModel = MLForecast(models=PlattModels,
                    freq='D',
                    lags=[1,7,14],
                    lag_transforms={
                        1: [(rolling_mean, 7), (rolling_max, 7), (rolling_min, 7)],
                    },
                    date_features=['dayofweek', 'day'],
                    num_threads=6)


    PlattesModels_List = [] 
    for i in PlattsModelList:
        # uniqe id = It should Be Platts10PPM and PlattsPremunl10PPM
        # y  =   prediciton Column
        # ds = Date of estiamtion
        models_list = PlattesTargetModel.fit(i, id_col='unique_id', time_col='ds', target_col='y', static_features=[])
        PlattesModels_List.append(models_list)

    Platts10PPM_1_prediciton =  PlattesModels_List[0].predict(horizon=32)
    Platts10PPM_1_prediciton.columns = ["unique_id","ds","y"]
    Platts10PPM_2_prediciton =  PlattesModels_List[1].predict(horizon=32)
    Platts10PPM_2_prediciton.columns = ["unique_id","ds","y"]
    Platts10PPM_3_prediciton =  PlattesModels_List[2].predict(horizon=32)
    Platts10PPM_3_prediciton.columns = ["unique_id","ds","y"]

    Platts10PPM_1_ModelDF = pd.concat([Platts10PPM_1_ModelDF,Platts10PPM_1_prediciton],ignore_index = True)
    Platts10PPM_1_ModelDF.columns = ["ds","unique_id","AB Piyasa Fiyatı"]
    Platts10PPM_2_ModelDF = pd.concat([Platts10PPM_2_ModelDF,Platts10PPM_2_prediciton],ignore_index = True)
    Platts10PPM_2_ModelDF.columns = ["ds","unique_id","AB Piyasa Fiyatı-Yüksek"]
    Platts10PPM_3_ModelDF = pd.concat([Platts10PPM_3_ModelDF,Platts10PPM_3_prediciton],ignore_index = True)
    Platts10PPM_3_ModelDF.columns = ["ds","unique_id","AB Piyasa Fiyatı-Düşük"]

    Platts10PPMDF = pd.merge(Platts10PPM_1_ModelDF, Platts10PPM_2_ModelDF,on=["ds","unique_id"], how='inner')
    Platts10PPMDF = pd.merge(Platts10PPMDF,Platts10PPM_3_ModelDF,on=["ds","unique_id"], how='inner')
    Platts10PPMDF = Platts10PPMDF[["ds","AB Piyasa Fiyatı","AB Piyasa Fiyatı-Yüksek","AB Piyasa Fiyatı-Düşük"]]
    Platts10PPMDF.columns = ["Date","AB Piyasa FiyatıPP","AB Piyasa Fiyatı-YüksekPP","AB Piyasa Fiyatı-DüşükPP"]

    #############################################################################################################

    PlattsPremunl10PPM_1_prediciton =  PlattesModels_List[3].predict(horizon=32)
    PlattsPremunl10PPM_1_prediciton.columns = ["unique_id","ds","y"]
    PlattsPremunl10PPM_2_prediciton =  PlattesModels_List[4].predict(horizon=32)
    PlattsPremunl10PPM_2_prediciton.columns = ["unique_id","ds","y"]
    PlattsPremunl10PPM_3_prediciton =  PlattesModels_List[5].predict(horizon=32)
    PlattsPremunl10PPM_3_prediciton.columns = ["unique_id","ds","y"]

    PlattsPremunl10PPM_1_ModelDF = pd.concat([PlattsPremunl10PPM_1_ModelDF,PlattsPremunl10PPM_1_prediciton],ignore_index = True)
    PlattsPremunl10PPM_1_ModelDF.columns = ["ds","unique_id","AB Piyasa Fiyatı"]
    PlattsPremunl10PPM_2_ModelDF = pd.concat([PlattsPremunl10PPM_2_ModelDF,PlattsPremunl10PPM_2_prediciton],ignore_index = True)
    PlattsPremunl10PPM_2_ModelDF.columns = ["ds","unique_id","AB Piyasa Fiyatı-Yüksek"]
    PlattsPremunl10PPM_3_ModelDF = pd.concat([PlattsPremunl10PPM_3_ModelDF,PlattsPremunl10PPM_3_prediciton],ignore_index = True)
    PlattsPremunl10PPM_3_ModelDF.columns = ["ds","unique_id","AB Piyasa Fiyatı-Düşük"]

    PlattsPremunl10PPMDF = pd.merge(PlattsPremunl10PPM_1_ModelDF, PlattsPremunl10PPM_2_ModelDF,on=["ds","unique_id"], how='inner')
    PlattsPremunl10PPMDF = pd.merge(PlattsPremunl10PPMDF,PlattsPremunl10PPM_3_ModelDF,on=["ds","unique_id"], how='inner')
    PlattsPremunl10PPMDF= PlattsPremunl10PPMDF[["ds","AB Piyasa Fiyatı","AB Piyasa Fiyatı-Yüksek","AB Piyasa Fiyatı-Düşük"]]
    PlattsPremunl10PPMDF.columns = ["Date","AB Piyasa FiyatıPPrem","AB Piyasa Fiyatı-YüksekPPrem","AB Piyasa Fiyatı-DüşükPPrem"]


    PlattsDF = PlattsPremunl10PPMDF.merge(Platts10PPMDF, on=["Date"], how="inner")
    
    return PlattsDF
    