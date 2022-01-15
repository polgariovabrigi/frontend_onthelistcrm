import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


@st.cache
def get_histo(df,col,bins,range):
    return plt.hist(df[col],bins=bins,range=range)

def app():

    data_df = pd.read_pickle('app/data/segmentation_all_df_07_01_2022_3.pkl')
    # seg_df = pd.read_pickle('/home/benjamin/code/Benjaminbhk/on_the_list_crm/on_the_list_crm/data/segmentation_07_01_2022.pkl')
    data_df_cust = data_df.drop_duplicates(subset=['customer_ID'], keep='last')
    table_final_price = pd.pivot_table(data_df, values='final_price', index=['customer_ID'], aggfunc=np.sum)
    spend_per_purchase = pd.read_pickle('app/data/spend_per_purchase.pkl')
    nb_of_itm_per_purchase = pd.read_pickle('app/data/nb_of_itm.pkl')

    sns.set_style("white")

    st.markdown("""
        ## Customer informations
    """)

    col1, col2, col3, col4 = st.columns((1,1,1,1))
    col1.metric("Number of unique customer", len(data_df['customer_ID'].unique().tolist()))
    col2.metric("Average age", round(data_df_cust['age'].mean(),2))
    col3.metric("Average number of time they came", round(data_df.drop_duplicates(subset=['date','customer_ID'], keep='last').groupby('customer_ID')[['date']].agg('count').reset_index()['date'].mean(),2))
    col4.metric("pourcentage of premium customers", f"{round(data_df_cust[data_df_cust['premium_status']=='y']['premium_status'].count()/data_df_cust['premium_status'].count()*100,2)}%")

    col1, spc, col2 = st.columns((2,.1,2))
    with col1:
        fig, ax = plt.subplots(figsize=(7,3))
        ax = sns.histplot(data=data_df_cust, x='age', ax=ax, bins=10)
        ax.set(xlabel='age', ylabel='count')
        ax.set_title('Age distribution', fontsize=20)
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(7,3))
        ax = sns.barplot(x=['Men','women'],
                          y=[round(data_df_cust[data_df_cust['gender']=='male']['gender'].count()/data_df_cust['gender'].count()*100,2),round(data_df_cust[data_df_cust['gender']=='female']['gender'].count()/data_df_cust['gender'].count()*100,2)],
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
        for district in data_df_cust['district'].unique():
            count=data_df_cust[data_df_cust['district'] == district]['district'].count()
            tmp_dict[district] = round(    count   /   len(data_df_cust)  *  100     ,2)
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
        for country in data_df_cust['nationality'].unique():
            count=data_df_cust[data_df_cust['nationality'] == country]['nationality'].count()
            tmp_dict[country] = round(    count   /   len(data_df_cust)  *  100     ,2)
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
    col1.metric("Average spending per item", f"{round(data_df['final_price'].mean(),2)} HKD")
    col2.metric("Average items per purchase", round(nb_of_itm_per_purchase['money_spend'].mean(),2))
    col3.metric("Average spending per purchase", f"{round(spend_per_purchase['money_spend'].mean(),2)} HKD")
    col4.metric("Median expenditure per consumer", f"{table_final_price['final_price'].quantile(.5)} HKD")

    st.markdown("""
        ## About product
    """)
    col1, col2, col3, col4 = st.columns((1,.1,1,.1))
    with col1:
        data = data_df.groupby('product_cat')[['final_price']].agg('count').reset_index()
        fig, ax = plt.subplots(figsize=(7,3))
        ax = sns.barplot(data=data, x='product_cat', y='final_price', ax=ax, order=data['product_cat'])
        ax.set(xlabel ="",ylabel='count')
        ax.tick_params(axis='x', rotation=70)
        ax.set_title('Purchase per product categorie', fontsize=20)
        st.pyplot(fig)

    with col3:
        data = data_df.groupby('product_cat')[['final_price']].agg('sum').reset_index()
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
        data = data_df.groupby('vendor_cat')[['final_price']].agg('count').reset_index()
        fig, ax = plt.subplots(figsize=(7,3))
        ax = sns.barplot(data=data, x='vendor_cat', y='final_price', ax=ax, order=data['vendor_cat'])
        ax.set(xlabel ="",ylabel='count')
        ax.tick_params(axis='x', rotation=70)
        ax.set_title('Purchase per vendor categorie', fontsize=20)
        st.pyplot(fig)

    with col3:
        data = data_df.groupby('vendor_cat')[['final_price']].agg('sum').reset_index()
        fig, ax = plt.subplots(figsize=(7,3))
        ax = sns.barplot(data=data, x='vendor_cat', y='final_price', ax=ax, order=data['vendor_cat'])
        ax.set(xlabel ="",ylabel='count')
        ax.tick_params(axis='x', rotation=70)
        ax.set_title('Incom per vendor categorie', fontsize=20)
        st.pyplot(fig)







    # col3.metric("Age repartition",
    #             st.bar_chart(get_histo(data_df_cust,'age',bins=50,range=(1,80)))
    #             )
    # st.bar_chart(data_df_cust['age'])

    # row1_space1, row1_1, row1_space2, row1_2, row1_space3, row1_3, row1_3_space4 = st.columns((.1, 1, .1, 1, .1, 1, .1))

    # with row1_1:

    #     seg_rev_df = data_df.groupby('segmentation')[['final_price']].agg('sum').reset_index()
    #     fig, ax = plt.subplots(figsize=(7,5))

    #     ax = sns.barplot(x=seg_rev_df['segmentation'],y=seg_rev_df['final_price'], ax=ax)
    #     ax.set(xlabel='Customer Segments', ylabel='Revenue in 10 million HKD')
    #     ax.set_title('Segments Revenue Breakdown', fontweight="bold", fontsize=20)
    #     st.pyplot(fig)
