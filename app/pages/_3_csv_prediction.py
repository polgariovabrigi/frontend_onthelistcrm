import streamlit as st
import requests
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def app():

    data_df = pd.read_pickle('on_the_list_crm/data/segmentation_all_df_07_01_2022.pkl')
    # seg_df = pd.read_pickle('on_the_list_crm/data/segmentation_07_01_2022.pkl')
    data_df_cust = data_df.drop_duplicates(subset=['customer_ID'], keep='last')
    table_final_price = pd.pivot_table(data_df, values='final_price', index=['customer_ID'], aggfunc=np.sum)
    spend_per_purchase = pd.read_pickle('on_the_list_crm/data/spend_per_purchase.pkl')
    nb_of_itm_per_purchase = pd.read_pickle('on_the_list_crm/data/nb_of_itm.pkl')
    seg = 0

    st.markdown("""
        ###### Model returns segments for each client in CSV file
        *Please upload CSV file*
    """)
    #upload file
    uploaded_file = st.file_uploader(
    "Upload your csv file", type=["csv"], accept_multiple_files=False)

    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file)
        # st.write(dataframe)
        json = dataframe.to_json()
        url_post = 'http://localhost:8000/uploadfile/'
        file = {"file": json}
        response = requests.post(url_post,files=file).json()
        response_df = pd.read_json(response)
        # st.write(response_df)
        response_df_cust = response_df.drop_duplicates(subset=['customer_ID'], keep='last')
        cust_list = response_df_cust['customer_ID'].unique().tolist()
        table_final_price_response = pd.pivot_table(response_df, values='final_price', index=['customer_ID'], aggfunc=np.sum)

        st.markdown("""
            ## Customer informations
        """)

        col1, col2, col3, col4 = st.columns((1,1,1,1))
        col1.metric("Number of unique customer", len(response_df['customer_ID'].unique().tolist()),f"{round(response_df_cust['segmentation'].count()/data_df_cust['segmentation'].count(),4)*100}% of total",delta_color="off")
        col2.metric("Average age", round(response_df_cust['age'].mean(),2),f"{round(response_df_cust['age'].mean()-data_df_cust['age'].mean(),2)} years")
        col3.metric("", 0)
        col4.metric("pourcentage of premium customers", f"{round(response_df_cust[response_df_cust['premium_status']=='y']['premium_status'].count()/response_df_cust['premium_status'].count()*100,2)}%",f"{round((response_df_cust[response_df_cust['premium_status']=='y']['premium_status'].count()/response_df_cust['premium_status'].count()*100)-(data_df_cust[data_df_cust['premium_status']=='y']['premium_status'].count()/data_df_cust['premium_status'].count()*100),2)} points")

        col1, spc, col2 = st.columns((2,.1,2))
        with col1:
            fig, ax = plt.subplots(figsize=(7,3))
            ax = sns.histplot(data=response_df_cust, x='age', ax=ax, bins=10)
            ax.set(xlabel='age', ylabel='count')
            ax.set_title('Age distribution', fontsize=20)
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(7,3))
            ax = sns.barplot(x=['Men','women'],
                            y=[round(response_df_cust[response_df_cust['gender']=='male']['gender'].count()/response_df_cust['gender'].count()*100,2),round(response_df_cust[response_df_cust['gender']=='female']['gender'].count()/response_df_cust['gender'].count()*100,2)],
                            ax=ax)
            ax.set(xlabel='gender', ylabel='%')
            ax.set_title('Gender distribution', fontsize=20)
            st.pyplot(fig)

        st.markdown("""
            ## Customer district and nationality
        """)

        col1, col2, col3 = st.columns((1,1,2))
        with col1:
            st.markdown("""
                ###### District in Hong Kong
            """)
            tmp_dict = {}
            for district in response_df_cust['district'].unique():
                count=response_df_cust[response_df_cust['district'] == district]['district'].count()
                tmp_dict[district] = round(    count   /   len(response_df_cust)  *  100     ,2)
                tmp_dict = dict(sorted(tmp_dict.items(), key=lambda item: item[1], reverse=True))
            i = 0
            for key,value in tmp_dict.items():
                if i < 8:
                    st.text(f"- {key} : {value}%")
                i += 1

        with col2:
            st.markdown("""
                ###### Nationality
            """)
            tmp_dict = {}
            for country in response_df_cust['nationality'].unique():
                count=response_df_cust[response_df_cust['nationality'] == country]['nationality'].count()
                tmp_dict[country] = round(    count   /   len(response_df_cust)  *  100     ,2)
                tmp_dict = dict(sorted(tmp_dict.items(), key=lambda item: item[1], reverse=True))
            i = 0
            for key,value in tmp_dict.items():
                if i < 8:
                    st.text(f"- {key} : {value}%")
                i += 1
            # for country in data_df_cust['nationality'].unique():
            #     count=data_df_cust[data_df_cust['nationality'] == country]['nationality'].count()
            #     if count/len(data_df_cust)*100 > 0.7:
            #         st.text(f"- {country} : {round(    count   /   len(data_df_cust)  *  100     ,2)}%")

        st.markdown("""
            ## Customer behavior
        """)
        col1, col2, col3, col4 = st.columns((1,1,1,1))
        col1.metric("Average spending per item", f"{round(response_df['final_price'].mean(),2)} HKD",f"{round(response_df['final_price'].mean()-data_df['final_price'].mean(),2)} HKD")
        col2.metric("Average items per purchase", round(nb_of_itm_per_purchase[nb_of_itm_per_purchase['customer_ID'].isin(cust_list)]['money_spend'].mean(),2),f"{round(nb_of_itm_per_purchase[nb_of_itm_per_purchase['customer_ID'].isin(cust_list)]['money_spend'].mean()-nb_of_itm_per_purchase['money_spend'].mean(),2)}")
        col3.metric("Average spending per purchase", f"{round(spend_per_purchase[spend_per_purchase['customer_ID'].isin(cust_list)]['money_spend'].mean(),2)} HKD", f"{round(spend_per_purchase[spend_per_purchase['customer_ID'].isin(cust_list)]['money_spend'].mean()-spend_per_purchase['money_spend'].mean(),2)} HKD")
        col4.metric("Median expenditure per consumer", f"{table_final_price_response['final_price'].quantile(.5)} HKD", f"{round(table_final_price_response['final_price'].quantile(.5)-table_final_price['final_price'].quantile(.5),2)} HKD")

        st.markdown("""
            ## About product
        """)
        col1, col2, col3, col4 = st.columns((1,.1,1,.1))
        with col1:
            data = response_df.groupby('product_cat')[['final_price']].agg('count').reset_index()
            fig, ax = plt.subplots(figsize=(7,3))
            ax = sns.barplot(data=data, x='product_cat', y='final_price', ax=ax, order=data['product_cat'])
            ax.set(xlabel ="",ylabel='count')
            ax.tick_params(axis='x', rotation=70)
            ax.set_title('Purchase per product categorie', fontsize=20)
            st.pyplot(fig)

        with col3:
            data = response_df.groupby('product_cat')[['final_price']].agg('sum').reset_index()
            fig, ax = plt.subplots(figsize=(7,3))
            ax = sns.barplot(data=data, x='product_cat', y='final_price', ax=ax, order=data['product_cat'])
            ax.set(xlabel ="",ylabel='count')
            ax.tick_params(axis='x', rotation=70)
            ax.set_title('Incom per product categorie', fontsize=20)
            st.pyplot(fig)

        st.markdown("""
            ## About vendor
        """)

        col1, col2, col3, col4 = st.columns((1,.1,1,.1))
        with col1:
            data = response_df.groupby('vendor_cat')[['final_price']].agg('count').reset_index()
            fig, ax = plt.subplots(figsize=(7,3))
            ax = sns.barplot(data=data, x='vendor_cat', y='final_price', ax=ax, order=data['vendor_cat'])
            ax.set(xlabel ="",ylabel='count')
            ax.tick_params(axis='x', rotation=70)
            ax.set_title('Purchase per vendor categorie', fontsize=20)
            st.pyplot(fig)

        with col3:
            data = response_df.groupby('vendor_cat')[['final_price']].agg('sum').reset_index()
            fig, ax = plt.subplots(figsize=(7,3))
            ax = sns.barplot(data=data, x='vendor_cat', y='final_price', ax=ax, order=data['vendor_cat'])
            ax.set(xlabel ="",ylabel='count')
            ax.tick_params(axis='x', rotation=70)
            ax.set_title('Incom per vendor categorie', fontsize=20)
            st.pyplot(fig)
