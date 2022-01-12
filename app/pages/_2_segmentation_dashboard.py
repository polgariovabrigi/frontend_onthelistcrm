import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def app():
    '''
    # Segmentation dashboard

    Work in progress
    '''

    data_df = pd.read_pickle('on_the_list_crm/data/segmentation_all_df_07_01_2022.pkl')
    seg_df = pd.read_pickle('on_the_list_crm/data/segmentation_07_01_2022.pkl')
    data_df_cust = data_df.drop_duplicates(subset=['customer_ID'], keep='last')
    table_final_price = pd.pivot_table(data_df, values='final_price', index=['customer_ID'], aggfunc=np.sum)
    spend_per_purchase = pd.read_pickle('on_the_list_crm/data/spend_per_purchase.pkl')
    nb_of_itm_per_purchase = pd.read_pickle('on_the_list_crm/data/nb_of_itm.pkl')

    row1_space1, row1_1, row1_space2, row1_2, row1_space3, row1_3, row1_3_space4 = st.columns((.1, 1, .1, 1, .1, 1, .1))
    with row1_1:

        seg_rev_df = data_df.groupby('segmentation')[['final_price']].agg('sum').reset_index()
        fig, ax = plt.subplots(figsize=(7,5))

        ax = sns.barplot(x=seg_rev_df['segmentation'],y=seg_rev_df['final_price'], ax=ax)
        ax.set(xlabel='Customer Segments', ylabel='Revenue in 10 million HKD')
        ax.set_title('Segments Revenue Breakdown', fontweight="bold", fontsize=20)
        st.pyplot(fig)

    seg = -1

    col0, col1, col2, col3, col4= st.columns((1,1,1,1,1))
    with col0:
        if st.button('-Accessory-'):
            seg = 0
    with col1:
        if st.button('-Cosmetics-'):
            seg = 1
    with col2:
        if st.button('-Cloth/Acc-'):
            seg = 2
    with col3:
        if st.button('--Premium+-'):
            seg = 3
    with col4:
        if st.button('--Premium--'):
            seg = 4

    col5, col6, col7, col8, col9 = st.columns((1,1,1,1,1))
    with col5:
        if st.button('--Kitchen--'):
            seg = 5
    with col6:
        if st.button('--Clothes--'):
            seg = 6
    with col7:
        if st.button('---Shoes---'):
            seg = 7
    with col8:
        if st.button('-Old-Cloth-'):
            seg = 8
    with col9:
        if st.button('-Old-cosme-'):
            seg = 9

    if seg == 0:
        st.text("Buy principaly accessories - spend about 10% more than the average per item but less items - slightly younger than the average")
    elif seg == 1:
        st.text("Buy principaly cosmetics - spend about 30% less than the average per items - younger than the average")
    elif seg == 2:
        st.text("Buy principaly clothes and accessories - spend about 10% more than the average per items - slightly younger than the average")
    elif seg == 3:
        st.text("Buy principaly accessories and some clothes - spend way more than the average per items and more items - important population of premium")
    elif seg == 4:
        st.text("Buy principaly clothes and some accessories - spend about 10% more than the average per items and more items - important population of premium")
    elif seg == 5:
        st.text("Buy principaly kitchen product and some home product - spend about 15% less than the average per items - almost no premium")
    elif seg == 6:
        st.text("Buy principaly clothes - spend 10% less than the average per items - almost no premium -")
    elif seg == 7:
        st.text("Buy principaly shoes - spend 20% more than the average per items but less product - almost no premium")
    elif seg == 8:
        st.text("Buy principaly clothes and some accessories - spend about 15% more than the average per item - way older than the average")
    elif seg == 9:
        st.text("Buy principaly cosmetics - spend about 15% more than the average per item - older than the average")

    if seg != -1:

        st.markdown("""
                ## info
            """)

        seg_0_df = seg_df[seg_df['customer_segmentation'] == seg]
        seg_0_list = seg_0_df['customer_ID'].unique().tolist()
        data_0_df = data_df[data_df['customer_ID'].isin(seg_0_list)]
        data_0_df_cust = data_0_df.drop_duplicates(subset=['customer_ID'], keep='last')
        table_final_price_0 = pd.pivot_table(data_0_df, values='final_price', index=['customer_ID'], aggfunc=np.sum)

        col1, col2, col3, col4 = st.columns((1,1,1,1))
        with col1:
            st.metric("Number of unique customer", len(seg_0_df['customer_ID'].unique().tolist()),f"{round(data_0_df_cust['segmentation'].count()/data_df_cust['segmentation'].count(),4)*100}% of total",delta_color="off")
        with col2:
            st.metric("Average age", round(data_0_df_cust['age'].mean(),2),f"{round(data_0_df_cust['age'].mean()-data_df_cust['age'].mean(),2)} years")
        with col3:
            st.metric("Average number of time they came", round(data_0_df.drop_duplicates(subset=['date','customer_ID'], keep='last').groupby('customer_ID')[['date']].agg('count').reset_index()['date'].mean(),2) ,round(data_0_df.drop_duplicates(subset=['date','customer_ID'], keep='last').groupby('customer_ID')[['date']].agg('count').reset_index()['date'].mean() - data_df.drop_duplicates(subset=['date','customer_ID'], keep='last').groupby('customer_ID')[['date']].agg('count').reset_index()['date'].mean(),2))
        with col4:
            st.metric("Pourcentage of premium customers", f"{round(data_0_df_cust[data_0_df_cust['premium_status']=='y']['premium_status'].count()/data_0_df_cust['premium_status'].count()*100,2)}%",f"{round((data_0_df_cust[data_0_df_cust['premium_status']=='y']['premium_status'].count()/data_0_df_cust['premium_status'].count()*100)-(data_df_cust[data_df_cust['premium_status']=='y']['premium_status'].count()/data_df_cust['premium_status'].count()*100),2)} points")

        col1, col2, col3, col4 = st.columns((1,1,1,1))
        with col1:
            st.metric("fidelity to this group", f"{round(data_0_df[data_0_df['segmentation'] == seg]['segmentation'].count()/data_0_df['segmentation'].count()*100,2)}%")
        with col2:
            st.metric("weight of this group in term of purchase ", f"{round(data_0_df['segmentation'].count()/data_df['segmentation'].count()*100,2)}%")
        with col3:
            st.metric("weight of this group in term of sepnding", f"{round(data_0_df['final_price'].sum()/data_df['final_price'].sum()*100,2)}%")
        with col4:
            st.metric("spending ratio", round((data_0_df['final_price'].sum()/data_df['final_price'].sum()*100)/(data_0_df_cust['segmentation'].count()/data_df_cust['segmentation'].count()*100),2))

        col1, spc, col2 = st.columns((2,.1,2))
        with col1:
            fig, ax = plt.subplots(figsize=(7,3))
            ax = sns.histplot(data=data_0_df_cust, x='age', ax=ax, bins=10)
            ax.set(xlabel='age', ylabel='count')
            ax.set_title('Age distribution', fontsize=20)
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(7,3))
            ax = sns.barplot(x=['Men','women'],
                            y=[round(data_0_df_cust[data_0_df_cust['gender']=='male']['gender'].count()/data_0_df_cust['gender'].count()*100,2),round(data_0_df_cust[data_0_df_cust['gender']=='female']['gender'].count()/data_0_df_cust['gender'].count()*100,2)],
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
            for district in data_0_df_cust['district'].unique():
                count=data_0_df_cust[data_0_df_cust['district'] == district]['district'].count()
                tmp_dict[district] = round(    count   /   len(data_0_df_cust)  *  100     ,2)
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
            for country in data_0_df_cust['nationality'].unique():
                count=data_0_df_cust[data_0_df_cust['nationality'] == country]['nationality'].count()
                tmp_dict[country] = round(    count   /   len(data_0_df_cust)  *  100     ,2)
                tmp_dict = dict(sorted(tmp_dict.items(), key=lambda item: item[1], reverse=True))
            i = 0
            for key,value in tmp_dict.items():
                if i < 8:
                    st.text(f"- {key} : {value}%")
                i += 1

        st.markdown("""
            ## Customer behavior
        """)
        col1, col2, col3, col4 = st.columns((1,1,1,1))
        col1.metric("Average spending per item", f"{round(data_0_df['final_price'].mean(),2)} HKD",f"{round(data_0_df['final_price'].mean()-data_df['final_price'].mean(),2)} HKD")
        col2.metric("Average items per purchase", round(nb_of_itm_per_purchase[nb_of_itm_per_purchase['customer_ID'].isin(seg_0_list)]['money_spend'].mean(),2),f"{round(nb_of_itm_per_purchase[nb_of_itm_per_purchase['customer_ID'].isin(seg_0_list)]['money_spend'].mean()-nb_of_itm_per_purchase['money_spend'].mean(),2)}")
        col3.metric("Average spending per purchase", f"{round(spend_per_purchase[spend_per_purchase['customer_ID'].isin(seg_0_list)]['money_spend'].mean(),2)} HKD", f"{round(spend_per_purchase[spend_per_purchase['customer_ID'].isin(seg_0_list)]['money_spend'].mean()-spend_per_purchase['money_spend'].mean(),2)} HKD")
        col4.metric("Median expenditure per consumer", f"{table_final_price_0['final_price'].quantile(.5)} HKD", f"{round(table_final_price_0['final_price'].quantile(.5)-table_final_price['final_price'].quantile(.5),2)} HKD")

        st.markdown("""
            ## About product
        """)
        col1, col2, col3, col4 = st.columns((1,.1,1,.1))
        with col1:
            data = data_0_df.groupby('product_cat')[['final_price']].agg('count').reset_index()
            fig, ax = plt.subplots(figsize=(7,3))
            ax = sns.barplot(data=data, x='product_cat', y='final_price', ax=ax, order=data['product_cat'])
            ax.set(xlabel ="",ylabel='count')
            ax.tick_params(axis='x', rotation=70)
            ax.set_title('Purchase per product categorie', fontsize=20)
            st.pyplot(fig)

        with col3:
            data = data_0_df.groupby('product_cat')[['final_price']].agg('sum').reset_index()
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
            data = data_0_df.groupby('vendor_cat')[['final_price']].agg('count').reset_index()
            fig, ax = plt.subplots(figsize=(7,3))
            ax = sns.barplot(data=data, x='vendor_cat', y='final_price', ax=ax, order=data['vendor_cat'])
            ax.set(xlabel ="",ylabel='count')
            ax.tick_params(axis='x', rotation=70)
            ax.set_title('Purchase per vendor categorie', fontsize=20)
            st.pyplot(fig)

        with col3:
            data = data_0_df.groupby('vendor_cat')[['final_price']].agg('sum').reset_index()
            fig, ax = plt.subplots(figsize=(7,3))
            ax = sns.barplot(data=data, x='vendor_cat', y='final_price', ax=ax, order=data['vendor_cat'])
            ax.set(xlabel ="",ylabel='count')
            ax.tick_params(axis='x', rotation=70)
            ax.set_title('Incom per vendor categorie', fontsize=20)
            st.pyplot(fig)
