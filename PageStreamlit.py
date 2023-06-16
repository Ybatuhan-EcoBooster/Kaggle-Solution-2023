
''' 
- -> run commend is = streamlit run PageStreamlit.py 

it work on LocalHost

'''

# Streamlit Web
import streamlit as st
from streamlit_extras.colored_header import colored_header

#Graphs
import plotly.graph_objects as go

# Python Codes file
from PythonCodes.ModelData import *
from PythonCodes.Model import *
from PythonCodes.PlattsPrediction import *

### Page Configure ###
st.set_page_config(page_title = "LSTM Prediction Model",
                    page_icon ='✅',
                    layout = 'wide')


# Containers
FirstFrame = st.empty()
MainTargetModelVis = st.container()
Tables = st.container()
SecondFrame = st.empty()
ModelCurrencyArea = st.container()
Referance = st.container()
Information = st.container()
ModelDataSets = st.container()
########################################################################## Second Frame ###################################################

def CurrencyTableFrame(DataFrame = None):
    with SecondFrame.container():
        with ModelCurrencyArea:
            
            colored_header(label = "Currency Exchange Price 💱",description= None ,color_name="red-70")

            SelectBox2 = st.selectbox(
            'How would you like to see Visulazation?',
            ('Graph', 'Table'))

            st.write('You selected:', SelectBox2)
            CurrencyData = DataFrame
            
            if SelectBox2 == "Graph":
                tab1,tab2= st.tabs([f"💱 Currency Buy", "💱 Currency Sell "])
                
                CurrencyBuyDF = go.Figure()
                CurrencyBuyDF.add_trace(go.Line(x=CurrencyData["Date"],y = CurrencyData["USD ALIŞ"], name = "USD"))
                CurrencyBuyDF.add_trace(go.Line(x=CurrencyData["Date"],y = CurrencyData["EUR ALIŞ"], name = "EUR"))
                CurrencyBuyDF.add_trace(go.Line(x=CurrencyData["Date"],y = CurrencyData["GBP ALIŞ"], name = "GBP"))
                CurrencyBuyDF.update_xaxes(title_text = 'Date')
                CurrencyBuyDF.update_yaxes(title_text = 'Price')
                CurrencyBuyDF.update_layout(autosize=False, width = 1400, height = 500,legend_title_text='Currency')
                tab1.plotly_chart(CurrencyBuyDF)

                CurrencySellDF = go.Figure()
                CurrencySellDF.add_trace(go.Line(x=CurrencyData["Date"],y = CurrencyData["USD SATIŞ"], name = "USD"))
                CurrencySellDF.add_trace(go.Line(x=CurrencyData["Date"],y = CurrencyData["EUR SATIŞ"], name = "EUR"))
                CurrencySellDF.add_trace(go.Line(x=CurrencyData["Date"],y = CurrencyData["GBP SATIŞ"], name = "GBP"))
                CurrencySellDF.update_xaxes(title_text = 'Date')
                CurrencySellDF.update_yaxes(title_text = 'Price')
                CurrencySellDF.update_layout(autosize=False, width = 1400, height = 500,legend_title_text='Currency')
                tab2.plotly_chart(CurrencySellDF)


            else:
                tab3,tab4 = st.tabs([f"💱 Currency Buy", "💱 Currency Sell "])
                
                CurrencyBuyDF  = CurrencyData[["Date","USD ALIŞ","EUR ALIŞ","GBP ALIŞ"]]
                CurrencyBuyDF.columns  = ["Date","USD Buy","EUR Buy","GBP Buy"]
                CurrencySellDF = CurrencyData[["Date","USD SATIŞ","EUR SATIŞ","GBP SATIŞ"]]
                CurrencySellDF.columns = ["Date","USD Sell","EUR Sell","GBP Sell"]
                
                tab3.dataframe(CurrencyBuyDF,use_container_width=True)
                col1, col2, col3 = tab3.columns(3)
                
                DeltaUSD = CurrencyData["USD ALIŞ"]
                DeltaUSDLastValue = DeltaUSD.iloc[-1]
                DeltaUSDPrevious = DeltaUSD.iloc[-2]
                
                if DeltaUSDLastValue > DeltaUSDPrevious:
                    col1.metric(label="USD",value=DeltaUSDLastValue,delta=DeltaUSDPrevious,delta_color = "normal")
                else:
                    col1.metric(label="USD",value=DeltaUSDLastValue,delta=DeltaUSDPrevious,delta_color = "inverse")
                
                DeltaEUR = CurrencyData["EUR ALIŞ"]
                DeltaEURLastValue = DeltaEUR.iloc[-1]
                DeltaEURPrevious = DeltaEUR.iloc[-2]
                
                if DeltaEURLastValue > DeltaEURPrevious:
                    col2.metric(label="EUR",value=DeltaEURLastValue,delta=DeltaEURPrevious,delta_color = "normal")
                else:
                    col2.metric(label="EUR",value=DeltaEURLastValue,delta=DeltaEURPrevious,delta_color = "inverse")
                
                DeltaGBP = CurrencyData["GBP ALIŞ"]
                DeltaGBPLastValue = DeltaGBP.iloc[-1]
                DeltaGBPPrevious = DeltaGBP.iloc[-2]
                
                if DeltaGBPLastValue > DeltaGBPPrevious:
                    col3.metric(label="EUR",value=DeltaGBPLastValue,delta=DeltaGBPPrevious,delta_color = "normal")
                else:
                    col3.metric(label="EUR",value=DeltaGBPLastValue,delta=DeltaGBPPrevious,delta_color = "inverse")
                tab4.dataframe(CurrencySellDF,use_container_width=True)
                
                col1, col2, col3 = tab4.columns(3)
                
                DeltaUSD = CurrencyData["USD SATIŞ"]
                DeltaUSDLastValue = DeltaUSD.iloc[-1]
                DeltaUSDPrevious = DeltaUSD.iloc[-2]
                
                if DeltaUSDLastValue > DeltaUSDPrevious:
                    col1.metric(label="USD",value=DeltaUSDLastValue,delta=DeltaUSDPrevious,delta_color = "normal")
                else:
                    col1.metric(label="USD",value=DeltaUSDLastValue,delta=DeltaUSDPrevious,delta_color = "inverse")
                
                DeltaEUR = CurrencyData["EUR SATIŞ"]
                DeltaEURLastValue = DeltaEUR.iloc[-1]
                DeltaEURPrevious = DeltaEUR.iloc[-2]
                
                if DeltaEURLastValue > DeltaEURPrevious:
                    col2.metric(label="EUR",value=DeltaEURLastValue,delta=DeltaEURPrevious,delta_color = "normal")
                else:
                    col2.metric(label="EUR",value=DeltaEURLastValue,delta=DeltaEURPrevious,delta_color = "inverse")
                
                DeltaGBP = CurrencyData["GBP SATIŞ"]
                DeltaGBPLastValue = DeltaGBP.iloc[-1]
                DeltaGBPPrevious = DeltaGBP.iloc[-2]
                
                if DeltaGBPLastValue > DeltaGBPPrevious:
                    col3.metric(label="EUR",value=DeltaGBPLastValue,delta=DeltaGBPPrevious,delta_color = "normal")
                else:
                    col3.metric(label="EUR",value=DeltaGBPLastValue,delta=DeltaGBPPrevious,delta_color = "inverse")

    ########################################################################## Refernce ###########################################################
    with Referance:
        
        colored_header(label ="Model Description:",description= None ,color_name="red-70")
        st.write('''Model is the Long Short-Term Memory Recurrent Neural Network (LSTM), uses a multilayer LSTM encoder and an MLP decoder.
        It builds upon the LSTM-cell that improves the exploding and vanishing gradients of classic RNN’s. 
        This network has been extensively used in sequential prediction tasks like language modeling, phonetic labeling, and forecasting.
    
        ''')
        colored_header(label="📍Referance",description= None ,color_name="light-blue-70")
        st.write('''https://nixtla.github.io/neuralforecast/models.lstm.html''')
        st.write('''https://www.kaggle.com/competitions/new-shell-cashflow-datathon-2023/overview''')

    ########################################################################## Information ##########################################################
    with st.empty():
            with Information:
                html_info = '<h2 align="center"> INFORMATION </h2>'
                st.markdown(html_info,unsafe_allow_html=True)
                html_name = '<h2 align="center"> Designed By Batuhan YILDIRIM </h2>'
                st.markdown(html_name,unsafe_allow_html=True)
                html = ''' <h2 align="center"> I'm a Finance and Data Analyst 💻 </h2> '''
                st.markdown(html,unsafe_allow_html=True)
                col1,col2 = st.columns(2)
                with col1:
                    st.markdown("### 🤝 Connect with me:")
                    html_2 = '''<a href="https://www.linkedin.com/in/batuhannyildirim/"><img align="left" src="https://raw.githubusercontent.com/yushi1007/yushi1007/main/images/linkedin.svg" alt="Batuhan YILDIRIM | LinkedIn" width="21px"/></a>
            <a href="https://twitter.com/batuhan1148"><img align="left" src="https://camo.githubusercontent.com/ac6e1101f110e5f500287cf70dac72519687620deefb5e8de1fa7ba6a3ba2407/68747470733a2f2f6564656e742e6769746875622e696f2f537570657254696e7949636f6e732f696d616765732f706e672f747769747465722e706e67" alt="Batuhan YILDIRIM | Twitter" width="22px"/></a>
            <a href="https://medium.com/@BatuhanYildirim1148"><img align="left" src="https://raw.githubusercontent.com/yushi1007/yushi1007/main/images/medium.svg" alt="Batuhan YILDIRIM | Medium" width="21px"/></a><a href="https://github.com/Ybatuhan-EcoBooster"><img align="left" src="https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white" alt="Batuhan YILDIRIM | GitHub"/>
            <a href="https://www.kaggle.com/ecobooster"><img align="left" src="https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white" alt="Batuhan YILDIRIM | Kaggle" /></br>'''
                    st.markdown(html_2,unsafe_allow_html=True)
                    st.markdown("### 🤓 About Me:")
                    st.write('''I am a Finance and Accounting student at Akademia Ekonomiczno-Humanistyczna w Warszawie. I was a student of the Turkish Aetheraeronauticall Association University Industrial Engineering and Anadolu University Finance Department. I gained business and cultural experience by participating in the Work and Travel program, which is held every year from June 22 to September 28, 2019. In 2019-2020, I became the student representative of the IEEE Turkish Aeronautical Association University Student Society IAS (Industrial Applications Association). During my student representation, I conducted 3 different workshops and 2 projects. "Horizon Solar Rover", one of these projects, participated in the competition organized by Turkey's IEEE PES (Power and Energy Community) by leading a project team of 26 people, and we came second out of 20 teams. IEEE Turkey is preparing for the competition I started my job as Entrepreneur Network and continued to strengthen entrepreneurs. In 2020, I increased my business and cultural experience by participating in the work and travel program in America again. At the same time, I started secondary education by applying to Anadolu University's second university records. I love being involved in organizations, speaking in public, leading and working in a team, analyzing and solving problems, researching startups and being inspired by their stories, producing and marketing, and sharing. One of my biggest goals is to share my knowledge and experience with everyone and to be inspired by them. 
    Taking part in organizations, speaking in public, leading a team and working in a team, analyzing problems and producing solutions, researching startups and being inspired by their stories, producing and marketing them, and sharing the knowledge I have gained are among my favorite qualities. 
    One of my biggest goals is to tell everyone about the experiences I have had and to be inspired by them.''')
                with col2:
                    st.markdown("## 💼 Technical Skills")
                    st.markdown("### 📋 Languages")
                    html_3 = '''![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)![MicrosoftSQLServer](https://img.shields.io/badge/Microsoft%20SQL%20Server-CC2927?style=for-the-badge&logo=microsoft%20sql%20server&logoColor=white)![R](https://img.shields.io/badge/r-%23276DC3.svg?style=for-the-badge&logo=r&logoColor=white)'''
                    st.markdown(html_3,unsafe_allow_html=True)
                    st.markdown("### 💻 IDEs/Editors")
                    html_4 = "![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)"
                    st.markdown(html_4, unsafe_allow_html=True)
                    st.markdown("### 🧭 ML/DL")
                    html_5 = "![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)"
                    st.markdown(html_5,unsafe_allow_html=True)
#####################################################################################################################################################

########################################################################## First Model #############################################

with FirstFrame.container():
   
    colored_header(label="Welcome To LSTM Model Dashboard Project!",description= None ,color_name="red-70")
    with ModelDataSets:
        st.sidebar.title("Files")

        SaveDF_CSV = st.sidebar.file_uploader("Please Choose a Save file",accept_multiple_files=False)
        if SaveDF_CSV is not None:
            LastSave_DF = SaveDF_CSV 
        
        BrentDF_CSV = st.sidebar.file_uploader("Please Choose a Brent file",accept_multiple_files=False)
        if BrentDF_CSV is not None:
            LastBrent_DF = Brent(BrentDF_CSV)
        
        TrainCashFlow_CSV = st.sidebar.file_uploader("Please Choose a Train CashFlow file",accept_multiple_files=False)
        if TrainCashFlow_CSV is not None:
            LastCashFlow_DF = CashFlow(TrainCashFlow_CSV)
        
        USD_CSV = st.sidebar.file_uploader("Please Choose a USD file",accept_multiple_files=False)
        if USD_CSV is not None:
            LastUSD_DF = Currency(USD_CSV)
        
        PlattsDF_DF = st.sidebar.file_uploader("Please Choose a Platts file",accept_multiple_files=False)
        if PlattsDF_DF is not None:
            LastPlatts_DF = Platts(PlattsDF_DF)
            ModelTargetDF_New = Target(plattscsv=LastPlatts_DF,Cashflowcsv=LastCashFlow_DF,currencycsv=LastUSD_DF,brentcsv=LastBrent_DF)
            ModelTrain = ModelTargetDF_New[:-30]
            ModelTest = ModelTargetDF_New[-30:] 
        else:
            st.info("Please Upload Save-Brent-CashFlow-Platts-USD csv file !!!! You Can Find Inside of Data Set File")
            
        @st.cache_resource(show_spinner="⚙️ Working Progress... This Progress takes couple minutes!⏱")
        def LSTM_Model():
            LSTMModelDF = LSTMModelMain(int(SelectBox1),LastSave_DF,ModelTargetDF_New)
            return LSTMModelDF

        # Main Graph
        with MainTargetModelVis:
                ModelTargetDF_New = Target(plattscsv=LastPlatts_DF,Cashflowcsv=LastCashFlow_DF,currencycsv=LastUSD_DF,brentcsv=LastBrent_DF)

                col_1,col_2 = st.columns(2)
                with col_1:            
                    SelectBox1 = st.text_input( "Please Enter Model Size !!! 👇")
                    
                    st.write('You selected:',SelectBox1)

                with col_2:
                    LSTMCheckbox = st.checkbox("📥 Apply the Model!")

                CashFlowDF_LSTM = LastCashFlow_DF   
                if LSTMCheckbox:
                    ModelSaveStreamlit = LSTM_Model()
                    MainModelGraph = go.Figure()
                    MainModelGraph.add_trace(go.Line(x= ModelTrain["ds"], marker_color="blue",y= ModelTrain["y"], name = "Train",visible='legendonly'))
                    MainModelGraph.add_trace(go.Line(x=ModelTest["ds"],marker_color="blue" ,y = ModelTest["y"], name = "Test"))
                    MainModelGraph.add_trace(go.Line(x= ModelSaveStreamlit["Date"], y= ModelSaveStreamlit["Net Cashflow from Operations"],marker_color="white", name = "Prediction"))
                    MainModelGraph.add_trace(go.Line(x= CashFlowDF_LSTM["Date"],y=CashFlowDF_LSTM["Total Inflows"],name ="Total Inflows",marker_color="blue",line=dict(dash ="dashdot"),visible='legendonly'))
                    MainModelGraph.add_trace(go.Line(x= CashFlowDF_LSTM["Date"],y=CashFlowDF_LSTM["Total Outflows"],name ="Total Outflows",marker_color="blue",line=dict(dash ="dashdot"),visible='legendonly'))
                    MainModelGraph.update_xaxes(title_text = 'Date')
                    MainModelGraph.update_yaxes(title_text = 'Price')
                    MainModelGraph.update_layout(autosize=False, width = 1400, height = 500,title = "Forecasting Net Cashflow from Operations",legend_title_text='Parameters')
                    MainTargetModelVis.plotly_chart(MainModelGraph)
                    with Tables:
                        st.title("Forecasting Net Cashflow from Operations")
                        st.dataframe(ModelSaveStreamlit.style.highlight_max("Net Cashflow from Operations"),use_container_width=True)
                        def ConvertCSV(SaveDF):
                            return SaveDF.to_csv().encode('utf-8')
                        ModelCSV = ModelSaveStreamlit
                        ModelCSV = ConvertCSV(ModelCSV)
                        st.download_button('Download the file',ModelCSV,file_name ="Prediction.csv")
                    CurrencyTableFrame(LastUSD_DF)
                    st.success("Model is Completed !!!")
                

                else:
                    MainModelGraph = go.Figure()
                    MainModelGraph.add_trace(go.Line(x= ModelTrain["ds"], y= ModelTrain["y"],marker_color="gold", name = "Main"))
                    MainModelGraph.add_trace(go.Line(x=ModelTest["ds"], y = ModelTest["y"],marker_color="gold",name = "Main"))
                    MainModelGraph.add_trace(go.Line(x= CashFlowDF_LSTM["Date"],y=CashFlowDF_LSTM["Total Inflows"],name ="Total Inflows",marker_color="magenta",line=dict(dash ="dashdot")))
                    MainModelGraph.add_trace(go.Line(x= CashFlowDF_LSTM["Date"],y=CashFlowDF_LSTM["Total Outflows"],name ="Total Outflows",marker_color="blue",line=dict(dash ="dashdot")))
                    MainModelGraph.update_xaxes(title_text = 'Date')
                    MainModelGraph.update_yaxes(title_text = 'Price')
                    MainModelGraph.update_layout(autosize=False, width = 1400, height = 500,title = "Net Cashflow from Operations",legend_title_text='Parameters')
                    MainTargetModelVis.plotly_chart(MainModelGraph)
                    with Tables:   
                        RealCashflow = ModelTargetDF_New[["ds","y"]]
                        RealCashflow.columns = ["Date","Net Cashflow from Operations"]
                                        
                        st.title("Real Net Cashflow from Operations")
                        st.dataframe(RealCashflow,use_container_width=True)
                    CurrencyTableFrame(LastUSD_DF)
