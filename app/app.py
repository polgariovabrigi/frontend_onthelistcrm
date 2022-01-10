import streamlit as st

# Custom imports
from multipage import MultiPage
from pages import _1_general_dashboard, _2_segmentation_dashboard, _3_csv_prediction, _4_customer_prediction  # import your pages here

# Create an instance of the app
app = MultiPage()

# Title of the main page
st.title("OnTheList Data Alalys and Prediction")
st.markdown("""
    ##### Extraction of keys elements from the database and prediction about segmentation and customers
""")

# Add all your applications (pages) here
app.add_page("General dashboard", _1_general_dashboard.app)
app.add_page("Segmentation dashboard", _2_segmentation_dashboard.app)
app.add_page("Upload CSV and analyse", _3_csv_prediction.app)
app.add_page("Customer prediction", _4_customer_prediction.app)

# The main app
app.run()





















    # files = {'csv_file': uploaded_file.read()}
    # api_answer = response.json()
    # st.write(api_answer)
    # file_details = {"filename":uploaded_file.name, "filetype":uploaded_file.type,
    #                 "filesize":uploaded_file.size}
    # # d = dict()
    # # with open(uploaded_file.name, 'rb') as f:
    # #   for line in uploaded_file :
    # #     line = line.strip('\n')
    # #     (key, val) = line.split(",")
    # #     d[key] = val
    # # st.write(d)
    # #api sending info
    # # st.write(uploaded_file.getvalue())

    # # url_post = 'https://on-the-list-crm-sqpnwxjv3a-df.a.run.app//uploadfile/'
    # # response = requests.post(url_post,files=uploaded_file)
    # response = requests.post(url_post,files=files)
    # api_answer = response.json()
    # st.write(api_answer)
    # # prediction = Image.open(img_path.get("name"))
    # #uploaded data visualization
    # # df = pd.read_csv(uploaded_file)
    # # st.dataframe(df)




# enter here the address of your flask api
# url = 'https://on-the-list-crm-sqpnwxjv3a-df.a.run.app'
# response = requests.get(url)
# prediction = response.json()
# st.write(prediction)
