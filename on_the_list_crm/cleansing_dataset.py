import pandas as pd
import numpy as np

#read data
def get_data():
    #load data
    data_df = pd.read_csv('../raw_data/data_400_000_rows_clean_df.csv')
    #change the data
    for column in data_df.columns:
        if data_df[column].dtype == 'O':
            data_df[column] = data_df[column].str.lower()
    return data_df


def rename_columns(data_df):
    #rename existing columns
    data_df = data_df.rename(columns={'_sdc_source_key_id':'order_ID',
                        'sku':'item_ID',
                        'item price':'item_price',
                        'item quantity':'item_quantity',
                        'line-item discount':'discount',
                        'price_qty':'final_price',
                        'email':'customer_ID',
                        'Nationality':'nationality'}, inplace = True)
    return data_df

def cleanse_featureš(data_df):
    #drop unimportant features as itemdid, customerid.
    data_df = data_df.dropna(subset=['item_ID', 'customer_ID','vendor'],inplace=True)
    #impute missing nationality with HKSAR
    data_df['nationality'] = data_df['nationality'].fillna('hong kong sar')
    #impute missing gender with Female gender
    data_df['gender'] = data_df['gender'].fillna('female')
    #impute missing age with with age mean
    data_df['age'] = data_df['age'].fillna(round(data_df['age'].mean()))
    #handling age outliers
    data_df.loc[data_df['age'] < 18, 'age'] = round(data_df['age'].mean())
    data_df.loc[data_df['age'] > 90, 'age'] = round(data_df['age'].mean())
    #remove gift sales from db
    data_df = data_df[data_df['price'] != 0]
    #OTL membership sales
    data_df = data_df[data_df['title'] != 'onthelist premium membership']
    #Other OTL sales
    data_df = data_df[data_df['vendor'] != 'onthelist']
    data_df = data_df[data_df['vendor'] != 'onthelist hk']
    data_df = data_df[data_df['vendor'] != 'onthelisttest']
    #add new column on/off
    data_df['on_off'] = '__'
    data_df.loc[data_df["vendor"].str.contains('-online'),'on_off'] = 'online'
    data_df.loc[data_df["vendor"].str.contains('- online'),'on_off'] = 'online'
    data_df.loc[data_df["vendor"].str.contains('-onlione'),'on_off'] = 'online'
    data_df.loc[data_df["vendor"].str.contains('-offline'),'on_off'] = 'offline'
    data_df.loc[data_df["vendor"].str.contains('- offline'),'on_off'] = 'offline'
    data_df = data_df[data_df['on_off'] != '__']

    # Creat the tmp with vendor, title, product and tags for the NLP
    data_df['vendor_tmp'] = data_df['vendor'].astype(str)
    data_df['title_tmp'] = data_df['title'].astype(str)
    data_df['product_type_tmp'] = data_df['product_type'].astype(str)
    data_df['tags_tmp'] = data_df['tags'].astype(str)
    data_df['tmp_NLP'] = data_df[['vendor_tmp', 'title_tmp', 'product_type_tmp', 'tags_tmp']].agg(' '.join, axis=1)
    data_df = data_df.drop(columns=['vendor_tmp', 'title_tmp', 'product_type_tmp', 'tags_tmp'])

    return data_df
