# Streamlit Web
import streamlit as st
from streamlit_extras.colored_header import colored_header

#Graphs
import plotly.graph_objects as go

from MyLibraries.DataSets import *
from MyLibraries.LSTMModel import *
### Page Configure ###
st.set_page_config(page_title = "Sehll Cash Flow Dashboard 2023",
                    page_icon ='✅',
                    layout = 'wide')

Target_df_new = Target()
train = Target_df_new[:-23]
test = Target_df_new[-23:]

# Containers
placeholder = st.empty()
Main_target_graph = st.container()
Tables = st.container()
placeholder_2 = st.empty()
Currency_area = st.container()
Text = st.container()
Information = st.container()
DataSet = st.container()

##########################################################################
def CurrencyTable():
    with placeholder_2.container():
        with Currency_area:
            
            colored_header(label = "Currency Exchange Price 💱",description= None ,color_name="light-blue-70")

            option_2 = st.selectbox(
            'How would you like to see?',
            ('Graph', 'Table'))

            st.write('You selected:', option_2)
            CurrencyBuySell = Currency()
            
            if option_2 == "Graph":
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
                col1, col2, col3 = tab3.columns(3)
                
                delta_usd = CurrencyBuySell["USD ALIŞ"]
                delta_usd_value = delta_usd.iloc[-1]
                delta_usd_valueP = delta_usd.iloc[-2]
                
                if delta_usd_value > delta_usd_valueP:
                    col1.metric(label="USD",value=delta_usd_value,delta=delta_usd_valueP,delta_color = "normal")
                else:
                    col1.metric(label="USD",value=delta_usd_value,delta=delta_usd_valueP,delta_color = "inverse")
                
                delta_eur = CurrencyBuySell["EUR ALIŞ"]
                delta_eur_value = delta_eur.iloc[-1]
                delta_eur_valueP = delta_eur.iloc[-2]
                
                if delta_eur_value > delta_eur_valueP:
                    col2.metric(label="EUR",value=delta_eur_value,delta=delta_eur_valueP,delta_color = "normal")
                else:
                    col2.metric(label="EUR",value=delta_eur_value,delta=delta_eur_valueP,delta_color = "inverse")
                
                delta_gbp = CurrencyBuySell["GBP ALIŞ"]
                delta_gbp_value = delta_gbp.iloc[-1]
                delta_gbp_valueP = delta_gbp.iloc[-2]
                
                if delta_gbp_value > delta_gbp_valueP:
                    col3.metric(label="EUR",value=delta_gbp_value,delta=delta_gbp_valueP,delta_color = "normal")
                else:
                    col3.metric(label="EUR",value=delta_gbp_value,delta=delta_gbp_valueP,delta_color = "inverse")
                tab4.dataframe(CurrencySell,use_container_width=True)
                
                col1, col2, col3 = tab4.columns(3)
                
                delta_usd = CurrencyBuySell["USD SATIŞ"]
                delta_usd_value = delta_usd.iloc[-1]
                delta_usd_valueP = delta_usd.iloc[-2]
                
                if delta_usd_value > delta_usd_valueP:
                    col1.metric(label="USD",value=delta_usd_value,delta=delta_usd_valueP,delta_color = "normal")
                else:
                    col1.metric(label="USD",value=delta_usd_value,delta=delta_usd_valueP,delta_color = "inverse")
                
                delta_eur = CurrencyBuySell["EUR SATIŞ"]
                delta_eur_value = delta_eur.iloc[-1]
                delta_eur_valueP = delta_eur.iloc[-2]
                
                if delta_eur_value > delta_eur_valueP:
                    col2.metric(label="EUR",value=delta_eur_value,delta=delta_eur_valueP,delta_color = "normal")
                else:
                    col2.metric(label="EUR",value=delta_eur_value,delta=delta_eur_valueP,delta_color = "inverse")
                
                delta_gbp = CurrencyBuySell["GBP SATIŞ"]
                delta_gbp_value = delta_gbp.iloc[-1]
                delta_gbp_valueP = delta_gbp.iloc[-2]
                
                if delta_gbp_value > delta_gbp_valueP:
                    col3.metric(label="EUR",value=delta_gbp_value,delta=delta_gbp_valueP,delta_color = "normal")
                else:
                    col3.metric(label="EUR",value=delta_gbp_value,delta=delta_gbp_valueP,delta_color = "inverse")

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

########################################################################## ML Model #############################################
with placeholder.container():
    colored_header(label="Welcome To Sehll Cash Flow Datathon 2023 Dashboard Project!",description= None ,color_name="light-blue-70")
    with st.spinner("⚙️ Working Progress... This Progress takes couple minutes!⏱"):
        with DataSet:
            Submission_csv = st.sidebar.file_uploader("Choose a Submission file",accept_multiple_files=False)
            if Submission_csv is not None:
                Submission_df = Submission_csv  
            
            option = st.sidebar.text_input( "Please Enter Train Size Number!!! 👇")
            st.sidebar.write('You selected:',option)
                
            def LSTM_Model():
                model = LSTMModelMain(int(option),Submission_df)
                return model


        # Main Graph
        with Main_target_graph:

                LSTM_checkbox = st.checkbox("📥 Apply the Model!")
                CashFlow_df = CashFlow()
                
                if LSTM_checkbox:
                    Model_df = LSTM_Model()
                    Main_graph = go.Figure()
                    Main_graph.add_trace(go.Line(x= train["ds"], marker_color="ghostwhite",y= train["y"], name = "Train"))
                    Main_graph.add_trace(go.Line(x=test["ds"],marker_color="ghostwhite" ,y = test["y"], name = "Test"))
                    Main_graph.add_trace(go.Line(x= Model_df["Date"], y= Model_df["Net Cashflow from Operations"],marker_color="gold", name = "Prediction"))
                    Main_graph.add_trace(go.Line(x= CashFlow_df["Date"],y=CashFlow_df["Total Inflows"],name ="Total Inflows",marker_color="ghostwhite",line=dict(dash ="dashdot")))
                    Main_graph.add_trace(go.Line(x= CashFlow_df["Date"],y=CashFlow_df["Total Outflows"],name ="Total Outflows",marker_color="ghostwhite",line=dict(dash ="dashdot")))
                    Main_graph.update_xaxes(title_text = 'Date')
                    Main_graph.update_yaxes(title_text = 'Price')
                    Main_graph.update_layout(autosize=False, width = 1400, height = 500,title = "Forecasting Net Cashflow from Operations",legend_title_text='Parameters')
                    Main_target_graph.plotly_chart(Main_graph)
                    with Tables:
                        st.title("Forecasting Net Cashflow from Operations")
                        st.dataframe(Model_df.style.highlight_max("Net Cashflow from Operations"),use_container_width=True)
                        def convert_df(df):
                            return df.to_csv().encode('utf-8')
                        model_csv_df150 = Model_df
                        model_csv_150 = convert_df(model_csv_df150)
                        st.download_button('Download the file',model_csv_150,file_name ="ModelForecast.csv")
                    st.success('The model is completed!')
                    CurrencyTable()

                else:    
                    Main_graph = go.Figure()
                    Main_graph.add_trace(go.Line(x= train["ds"], y= train["y"],marker_color="blue", name = "Train"))
                    Main_graph.add_trace(go.Line(x=test["ds"], y = test["y"],marker_color="red",name = "Test"))
                    Main_graph.add_trace(go.Line(x= CashFlow_df["Date"],y=CashFlow_df["Total Inflows"],name ="Total Inflows",marker_color="magenta",line=dict(dash ="dashdot")))
                    Main_graph.add_trace(go.Line(x= CashFlow_df["Date"],y=CashFlow_df["Total Outflows"],name ="Total Outflows",marker_color="orange",line=dict(dash ="dashdot")))
                    Main_graph.update_xaxes(title_text = 'Date')
                    Main_graph.update_yaxes(title_text = 'Price')
                    Main_graph.update_layout(autosize=False, width = 1400, height = 500,title = "Net Cashflow from Operations",legend_title_text='Parameters')
                    Main_target_graph.plotly_chart(Main_graph)
                    with Tables:   
                        main_df = Target_df_new[["ds","y"]]
                        main_df.columns = ["Date","Net Cashflow from Operations"]
                                
                        st.title("Real Net Cashflow from Operations")
                        st.dataframe(main_df,use_container_width=True)
                    CurrencyTable()
