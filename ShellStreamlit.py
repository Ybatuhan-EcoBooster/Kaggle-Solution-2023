####################################################### Data Sets ############################################################################
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

########################################################## ML Model ####################################################################
from window_ops.rolling import rolling_mean, rolling_max, rolling_min
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM
from neuralforecast.losses.pytorch import DistributionLoss


submission = pd.read_csv("shell-datathon-cash-flow-coderspace/sample_submission.csv")

Target_df_new = Target()


h = Target_df_new.index.nunique()

train = Target_df_new[:-23]
test = Target_df_new[-23:]

def LSTM_model():

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

    target_model = NeuralForecast(models=models, freq='D')

    target_model.fit(train)
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



################################################################### WEB ##############################################################################

# Streamlit Web
import streamlit as st
from streamlit_extras.colored_header import colored_header

#Graphs
import plotly.graph_objects as go


### Page Configure ###
st.set_page_config(page_title = "Sehll Cash Flow Dashboard",
                    page_icon ='✅',
                    layout = 'wide')

# ML Model
@st.cache_resource
def LSTM_Model():
    LSTM_df = LSTM_model()
    st.balloons()        
    return LSTM_df

# Containers
Main_target_graph = st.container()
Tables = st.empty()
Currency_area = st.container()
Text = st.container()
Information = st.container()
##########################################################################
# Main Graph
with Main_target_graph:


    colored_header(label="Welcome To Sehll Cash Flow Datathon 2023 Dashboard Project!",description= None ,color_name="light-blue-70")
    Target_df_new = Target()
    train = Target_df_new[:-23]
    test = Target_df_new[-23:]
    


    LSTM_checkbox = st.checkbox("📥 Apply the Model!")
    CashFlow_df = CashFlow()

    if LSTM_checkbox:

        
        
        with st.spinner('⚙️ Working Progress - \t This Progress takes couple minutes!⏱'):
            Model_df = LSTM_Model()
            Main_graph = go.Figure()
            Main_graph.add_trace(go.Line(x= train["ds"], marker_color="blue",y= train["y"], name = "Train"))
            Main_graph.add_trace(go.Line(x=test["ds"],marker_color="blue" ,y = test["y"], name = "Test"))
            Main_graph.add_trace(go.Line(x= Model_df["Date"], y= Model_df["Net Cashflow from Operations"],marker_color="gold", name = "Prediction"))
            Main_graph.add_trace(go.Line(x= CashFlow_df["Date"],y=CashFlow_df["Total Inflows"],name ="Total Inflows",marker_color="blue",line=dict(dash ="dashdot")))
            Main_graph.add_trace(go.Line(x= CashFlow_df["Date"],y=CashFlow_df["Total Outflows"],name ="Total Outflows",marker_color="blue",line=dict(dash ="dashdot")))
            Main_graph.update_xaxes(title_text = 'Date')
            Main_graph.update_yaxes(title_text = 'Price')
            Main_graph.update_layout(autosize=False, width = 1400, height = 500,title = "Forecasting Net Cashflow from Operations",legend_title_text='Parameters')
            Main_target_graph.plotly_chart(Main_graph)
        
        st.success('The model is completed!')    
    else:    
        Main_graph = go.Figure()
        Main_graph.add_trace(go.Line(x= train["ds"], y= train["y"],marker_color="blue", name = "Train"))
        Main_graph.add_trace(go.Line(x=test["ds"], y = test["y"],marker_color="yellow",name = "Test"))
        Main_graph.add_trace(go.Line(x= CashFlow_df["Date"],y=CashFlow_df["Total Inflows"],name ="Total Inflows",marker_color="magenta",line=dict(dash ="dashdot")))
        Main_graph.add_trace(go.Line(x= CashFlow_df["Date"],y=CashFlow_df["Total Outflows"],name ="Total Outflows",marker_color="orange",line=dict(dash ="dashdot")))
        Main_graph.update_xaxes(title_text = 'Date')
        Main_graph.update_yaxes(title_text = 'Price')
        Main_graph.update_layout(autosize=False, width = 1400, height = 500,title = "Net Cashflow from Operations",legend_title_text='Parameters')
        Main_target_graph.plotly_chart(Main_graph)
        
##########################################################################
with Tables.container():

    Dataframes,Dataframes_model = st.columns(2)

    with Dataframes.container():
        
        main_df = Target_df_new[["ds","y"]]
        main_df.columns = ["Date","Net Cashflow from Operations"]
        
        st.title("Real Net Cashflow from Operations")
        st.dataframe(main_df,use_container_width=True)
    
    with Dataframes_model.container():
        
        if LSTM_checkbox:
            st.title("Forecasting Net Cashflow from Operations")
            st.dataframe(LSTM_Model(),use_container_width=True)

##########################################################################
with Currency_area:
    
    colored_header(label = "Currency Exchange Price 💱",description= None ,color_name="light-blue-70")

    option = st.selectbox(
    'How would you like to see?',
    ('Graph', 'Table'))

    st.write('You selected:', option)
    CurrencyBuySell = Currency()
    
    if option == "Graph":
        tab1,tab2= st.tabs([f"💱 Currency Buy", "💱 Currency Sell "])
        
        CurrencyBuy = go.Figure()
        CurrencyBuy.add_trace(go.Line(x=CurrencyBuySell["Date"],y = CurrencyBuySell["USD ALIŞ"], name = "USD"))
        CurrencyBuy.add_trace(go.Line(x=CurrencyBuySell["Date"],y = CurrencyBuySell["EUR ALIŞ"], name = "EUR"))
        CurrencyBuy.add_trace(go.Line(x=CurrencyBuySell["Date"],y = CurrencyBuySell["GBP ALIŞ"], name = "GBP"))
        CurrencyBuy.update_xaxes(title_text = 'Date')
        CurrencyBuy.update_yaxes(title_text = 'Price')
        CurrencyBuy.update_layout(autosize=False, width = 1400, height = 500,legend_title_text='Currency')
        tab1.plotly_chart(CurrencyBuy)

        CurrencySell = go.Figure()
        CurrencySell.add_trace(go.Line(x=CurrencyBuySell["Date"],y = CurrencyBuySell["USD SATIŞ"], name = "USD"))
        CurrencySell.add_trace(go.Line(x=CurrencyBuySell["Date"],y = CurrencyBuySell["EUR SATIŞ"], name = "EUR"))
        CurrencySell.add_trace(go.Line(x=CurrencyBuySell["Date"],y = CurrencyBuySell["GBP SATIŞ"], name = "GBP"))
        CurrencySell.update_xaxes(title_text = 'Date')
        CurrencySell.update_yaxes(title_text = 'Price')
        CurrencySell.update_layout(autosize=False, width = 1400, height = 500,legend_title_text='Currency')
        tab2.plotly_chart(CurrencySell)

    else:
        tab3,tab4 = st.tabs([f"💱 Currency Buy", "💱 Currency Sell "])
        
        CurrencyBuy  = CurrencyBuySell[["Date","USD ALIŞ","EUR ALIŞ","GBP ALIŞ"]]
        CurrencyBuy.columns  = ["Date","USD Buy","EUR Buy","GBP Buy"]
        CurrencySell = CurrencyBuySell[["Date","USD SATIŞ","EUR SATIŞ","GBP SATIŞ"]]
        CurrencySell.columns = ["Date","USD Sell","EUR Sell","GBP Sell"]
        
        tab3.dataframe(CurrencyBuy,use_container_width=True)
        tab4.dataframe(CurrencySell,use_container_width=True)

##########################################################################
with Text:
    
    colored_header(label ="Model:",description= None ,color_name="light-blue-70")
    st.write('''Model is the Long Short-Term Memory Recurrent Neural Network (LSTM), uses a multilayer LSTM encoder and an MLP decoder.
    It builds upon the LSTM-cell that improves the exploding and vanishing gradients of classic RNN’s. 
    This network has been extensively used in sequential prediction tasks like language modeling, phonetic labeling, and forecasting.
 
    ''')
    colored_header(label="📍Referance",description= None ,color_name="light-blue-70")
    st.write('''https://nixtla.github.io/neuralforecast/models.lstm.html''')
    st.write('''https://www.kaggle.com/competitions/new-shell-cashflow-datathon-2023/overview''')

##########################################################################
with Information:
        
        html_info = '<h2 align="center"> INFORMATION </h2>'
        st.sidebar.markdown(html_info,unsafe_allow_html=True)
        html = ''' <h2 align="center"> I'm a Finance and Data Analyst 💻 and Financer 🏦! </h2> '''
        st.sidebar.markdown(html,unsafe_allow_html=True)
        st.sidebar.markdown("### 🤝 Connect with me:")
        html_2 = '''<a href="https://www.linkedin.com/in/batuhannyildirim/"><img align="left" src="https://raw.githubusercontent.com/yushi1007/yushi1007/main/images/linkedin.svg" alt="Batuhan YILDIRIM | LinkedIn" width="21px"/></a>
<a href="https://twitter.com/batuhan1148"><img align="left" src="https://camo.githubusercontent.com/ac6e1101f110e5f500287cf70dac72519687620deefb5e8de1fa7ba6a3ba2407/68747470733a2f2f6564656e742e6769746875622e696f2f537570657254696e7949636f6e732f696d616765732f706e672f747769747465722e706e67" alt="Batuhan YILDIRIM | Twitter" width="22px"/></a>
<a href="https://medium.com/@BatuhanYildirim1148"><img align="left" src="https://raw.githubusercontent.com/yushi1007/yushi1007/main/images/medium.svg" alt="Batuhan YILDIRIM | Medium" width="21px"/></a><a href="https://github.com/Ybatuhan-EcoBooster"><img align="left" src="https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white" alt="Batuhan YILDIRIM | GitHub"/>
<a href="https://www.kaggle.com/ecobooster"><img align="left" src="https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white" alt="Batuhan YILDIRIM | Kaggle" /></br>'''
        st.sidebar.markdown(html_2,unsafe_allow_html=True)
        st.sidebar.markdown("## 💼 Technical Skills")
        st.sidebar.markdown("### 📋 Languages")
        html_3 = '''![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)![MicrosoftSQLServer](https://img.shields.io/badge/Microsoft%20SQL%20Server-CC2927?style=for-the-badge&logo=microsoft%20sql%20server&logoColor=white)![R](https://img.shields.io/badge/r-%23276DC3.svg?style=for-the-badge&logo=r&logoColor=white)'''
        st.sidebar.markdown(html_3,unsafe_allow_html=True)
        st.sidebar.markdown("### 💻 IDEs/Editors")
        html_4 = "![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)"
        st.sidebar.markdown(html_4, unsafe_allow_html=True)
        st.sidebar.markdown("### 🧭 ML/DL")
        html_5 = "![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)"
        st.sidebar.markdown(html_5,unsafe_allow_html=True)

    
