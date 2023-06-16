''' ALL data sets came from DataSet files
'''

# Core Libraries
import pandas as pd 
import numpy as np


# Brent File 
def Brent(brentcsv = None):
    BrentDF = pd.read_csv(brentcsv)
    BrentDF["Tarih"] = pd.to_datetime(BrentDF["Tarih"])
    BrentDF["Tarih"] = pd.DatetimeIndex(BrentDF["Tarih"].dt.strftime("%Y-%m-%d"))
    BrentDF = BrentDF.iloc[:,:-1]
    BrentDF.set_index("Tarih", inplace=True)
    BrentDF = BrentDF.reset_index()
    BrentDF.rename(columns={"Tarih":"Date"},inplace= True)

    return BrentDF
    
# Cashflow Train File
def CashFlow(cashflowcsv=None):  
    CashFlowDF = pd.read_csv(cashflowcsv)
    CashFlowDF["Inflows- currency"] = CashFlowDF["Inflows- currency"].replace(np.nan,0)
    CashFlowDF["Date"] = pd.to_datetime(CashFlowDF["Date"])
    CashFlowDF["Date"] = pd.DatetimeIndex(CashFlowDF["Date"].dt.strftime("%Y-%m-%d"))
    CashFlowDF.set_index("Date", inplace=True)
    CashFlowDF = CashFlowDF.reset_index()
    
    return CashFlowDF


#USD File 
def Currency(Currencycsv = None):
    CurrencyDF = pd.read_csv(Currencycsv)
    CurrencyDF["Tarih"] = pd.to_datetime(CurrencyDF["Tarih"])
    CurrencyDF["Tarih"] = pd.DatetimeIndex(CurrencyDF["Tarih"].dt.strftime("%Y-%m-%d"))
    CurrencyDF.rename(columns={"Tarih":"Date"},inplace=True)
    CurrencyDF.set_index("Date", inplace=True)
    CurrencyDF = CurrencyDF.fillna(method="ffill")
    CurrencyDF = CurrencyDF.reset_index()

    return CurrencyDF

# Target is final dataframe of estimation
''' Target Dataframe is inclueds:
    -Platts
    -Brent
    -Cashflow
    -Usd
    All is decided with my decision
'''
#Merged DataSets for Target Data Set

def Target(plattscsv = None,Cashflowcsv = None,currencycsv = None,brentcsv = None):
    PlattsDF = plattscsv
    ModelDF = pd.merge(Cashflowcsv, currencycsv,on="Date", how='inner')
    ModelDF = pd.merge(ModelDF,brentcsv,on="Date",how='inner')  
    ModelDF['Date'] = pd.to_datetime(ModelDF['Date'])
    PlattsDF['Date'] = pd.to_datetime(PlattsDF['Date'])

    ModelMerged_DF = ModelDF.merge(PlattsDF, on='Date', how='inner')

    # Selected Columns
    TargetModelDF_New = ModelMerged_DF[["Date",'Customers - DDS','Customers - EFT','T&S Collections','FX Sales','Other operations','Tüpraş',
                'Other Oil','Gas','Import payments (FX purchases)','Tax','Operatioınal and Admin. Expenses','VIS Buyback Payments','Net Cashflow from Operations',
                'Inflows- currency','USD ALIŞ','USD SATIŞ','EUR ALIŞ','EUR SATIŞ', 'GBP ALIŞ', 'GBP SATIŞ','Ürün',
                'AB Piyasa Fiyatı','AB Piyasa Fiyatı- Yüksek', 'AB Piyasa Fiyatı- Düşük','AB Piyasa FiyatıPP', 'AB Piyasa Fiyatı-YüksekPP','AB Piyasa Fiyatı-DüşükPP',
                "AB Piyasa FiyatıPPrem","AB Piyasa Fiyatı-YüksekPPrem","AB Piyasa Fiyatı-DüşükPPrem"]].copy()

    TargetModelDF_New.rename(columns={"Date":"ds","Ürün":"unique_id","Net Cashflow from Operations":"y"}, inplace= True) #y = predicton Column
    TargetModelDF_New["unique_id"] = "Production"  # NeuralForecast need unique_id, Shell is represent of it.

    return TargetModelDF_New

