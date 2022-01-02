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

if uploaded_file is not None:
  
    file_details = {"filename":uploaded_file.name, "filetype":uploaded_file.type,
                      "filesize":uploaded_file.size}
    st.write(file_details)
    #api sending info
    file_name = uploaded_file.name
    url_post = 'https://on-the-list-crm-sqpnwxjv3a-df.a.run.app//uploadfile/'
    response = requests.post(url_post, files={'file': uploaded_file.getvalue()})
    api_answer = response.json()
    st.write(api_answer)
    # prediction = Image.open(img_path.get("name"))
    #uploaded data visualization
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)
    
    
        # res = requests.post(f"http://backend:8080/{style}", files=files)
        # img_path = res.json()
        # image = Image.open(img_path.get("name"))
        # st.image(image, width=500)
  
# enter here the address of your flask api
url = 'https://on-the-list-crm-sqpnwxjv3a-df.a.run.app'
response = requests.get(url)
prediction = response.json()
st.write(prediction)


