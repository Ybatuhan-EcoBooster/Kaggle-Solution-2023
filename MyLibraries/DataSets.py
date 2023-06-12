''' shell-datathon-cash-flow-coderspace files
'''

# Core Libraries
import pandas as pd 
import numpy as np

# New Platts DataSet 
from MyLibraries.Platts import *


# Brent File
def Brent():
    Brent = pd.read_csv("shell-datathon-cash-flow-coderspace/brent.csv")
    Brent["Tarih"] = pd.to_datetime(Brent["Tarih"])
    Brent["Tarih"] = pd.DatetimeIndex(Brent["Tarih"].dt.strftime("%Y-%m-%d"))
    Brent = Brent.iloc[:,:-1]
    Brent.set_index("Tarih", inplace=True)
    Brent = Brent.reset_index()
    Brent.rename(columns={"Tarih":"Date"},inplace= True)

    return Brent
    
# Cashflow Train File
def CashFlow():  
    CashFlow = pd.read_csv("shell-datathon-cash-flow-coderspace\cash_flow_train.csv")
    CashFlow["Inflows- currency"] = CashFlow["Inflows- currency"].replace(np.nan,0)
    CashFlow["Date"] = pd.to_datetime(CashFlow["Date"])
    CashFlow["Date"] = pd.DatetimeIndex(CashFlow["Date"].dt.strftime("%Y-%m-%d"))
    CashFlow.set_index("Date", inplace=True)
    CashFlow = CashFlow.reset_index()
    
    return CashFlow


#USD File 
def Currency():
    Currency = pd.read_csv("shell-datathon-cash-flow-coderspace/usd.csv")
    Currency["Tarih"] = pd.to_datetime(Currency["Tarih"])
    Currency["Tarih"] = pd.DatetimeIndex(Currency["Tarih"].dt.strftime("%Y-%m-%d"))
    Currency.rename(columns={"Tarih":"Date"},inplace=True)
    Currency.set_index("Date", inplace=True)
    Currency = Currency.fillna(method="ffill")
    Currency = Currency.reset_index()

    return Currency



# Targe is final dataframe of estimation
''' Target Dataframe is inclueds:
    -Platts
    -Brent
    -Cashflow
    -Usd
'''
def Target():
    #Merged DataSets
    platts = Platts()
    df = pd.merge(CashFlow(), Currency(),on="Date", how='inner')
    df = pd.merge(df,Brent(),on="Date",how='inner')  
    df['Date'] = pd.to_datetime(df['Date'])
    platts['Date'] = pd.to_datetime(platts['Date'])

    merged_df = df.merge(platts, on='Date', how='inner')

    # Selected Columns
    Target_df_new = merged_df[["Date",'Customers - DDS','Customers - EFT','T&S Collections','FX Sales','Other operations','Tüpraş',
                'Other Oil','Gas','Import payments (FX purchases)','Tax','Operatioınal and Admin. Expenses','VIS Buyback Payments','Net Cashflow from Operations',
                'Inflows- currency','USD ALIŞ','USD SATIŞ','EUR ALIŞ','EUR SATIŞ', 'GBP ALIŞ', 'GBP SATIŞ','Ürün',
                'AB Piyasa Fiyatı','AB Piyasa Fiyatı- Yüksek', 'AB Piyasa Fiyatı- Düşük','AB Piyasa FiyatıPP', 'AB Piyasa Fiyatı-YüksekPP','AB Piyasa Fiyatı-DüşükPP',
                "AB Piyasa FiyatıPPrem","AB Piyasa Fiyatı-YüksekPPrem","AB Piyasa Fiyatı-DüşükPPrem"]].copy()

    Target_df_new.rename(columns={"Date":"ds","Ürün":"unique_id","Net Cashflow from Operations":"y"}, inplace= True) #y = predicton Column
    Target_df_new["unique_id"] = "Shell"  # NeuralForecast need unique_id, Shell is represent of it.

    return Target_df_new
