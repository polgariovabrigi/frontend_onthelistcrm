import streamlit as st
import requests
import pandas as pd
    
# url = 'http://localhost:8000/predict'
# response = requests.get(url)
# response.json()

'''
# OnTheList Segmentation Model

Model returns segments for each client in CSV file
'''

'''
Please upload CSV file
'''
#upload file
uploaded_file = st.file_uploader(
    "Upload your csv file", type=["csv"], accept_multiple_files=False)

#read file
#try/except
if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  # st.write(df)
  
  
# enter here the address of your flask api
url = ''
response = requests.get(url)
prediction = response.json()
# pred = prediction['prediction']
# pred

