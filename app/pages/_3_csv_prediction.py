import streamlit as st
import requests
import pandas as pd

def app():

    st.markdown("""
        ###### Model returns segments for each client in CSV file
        *Please upload CSV file*
    """)
    #upload file
    uploaded_file = st.file_uploader(
    "Upload your csv file", type=["csv"], accept_multiple_files=False)

    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)
        json = dataframe.to_json()
        url_post = 'http://localhost:8000/uploadfile/'
        file = {"file": json}
        response = requests.post(url_post,files=file).json()
        response_df = pd.read_json(response)
        st.write(response_df)
