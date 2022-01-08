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
    # d = dict()
    # with open(uploaded_file.name, 'rb') as f:
    #   for line in uploaded_file :
    #     line = line.strip('\n')
    #     (key, val) = line.split(",")
    #     d[key] = val
    # st.write(d)
    #api sending info
    # st.write(uploaded_file.getvalue())
    files = {'csv_file': uploaded_file.read()}

    url_post = 'http://localhost:8001/uploadfile/'

    # url_post = 'https://on-the-list-crm-sqpnwxjv3a-df.a.run.app//uploadfile/'
    # response = requests.post(url_post,files=uploaded_file)
    response = requests.post(url_post,files=files)
    api_answer = response.json()
    st.write(api_answer)
    # prediction = Image.open(img_path.get("name"))
    #uploaded data visualization
    # df = pd.read_csv(uploaded_file)
    # st.dataframe(df)




# enter here the address of your flask api
# url = 'https://on-the-list-crm-sqpnwxjv3a-df.a.run.app'
# response = requests.get(url)
# prediction = response.json()
# st.write(prediction)

