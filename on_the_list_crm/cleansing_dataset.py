import pandas as pd


def get_raw_data(path='data/2020-2021_V2.csv', rows = None):
    #load data
    data_df = pd.read_csv(path, nrows=rows)
    return data_df

class BasicCleaner():

    def __init__(self):
        pass

    def fit(self, data_df, y=None):
        return self

    def transform(self, data_df, y=None):
        assert isinstance(data_df, pd.DataFrame)
        # work for 2020-2021_V2.csv
        if 'vendor_line' in data_df.columns:
            data_df['vendor'] = data_df['vendor_line']
            data_df['title'] = data_df['title_line']
            data_df.drop(columns=['vendor_line','title_line'],inplace=True)

        # lowercas all the text
        for column in data_df.columns:
            if data_df[column].dtype == 'O':
                data_df[column] = data_df[column].str.lower()
        # renaming the columns
        data_df.rename(columns={'_sdc_source_key_id':'order_ID',
                                'sku':'item_ID',
                                'price':'item_price',
                                'quantity':'item_quantity',
                                'discount' : 'item_discount',
                                'price_qty':'final_price',
                                'email':'customer_ID',
                                'Nationality':'nationality'},
                       inplace = True)
        #drop unimportant features as itemdid, customerid.
        data_df.dropna(subset=['item_ID', 'customer_ID','vendor','premium_status'],inplace = True)
        #impute missing district with HKSAR
        data_df['district'].fillna('central & western',inplace = True)
        #impute missing nationality with HKSAR
        data_df['nationality'].fillna('hong kong sar',inplace = True)
        #impute missing gender with Female gender
        data_df['gender'].fillna('female',inplace = True)
        #impute missing age with with age mean
        data_df['age'].fillna(round(data_df['age'].mean()),inplace = True)
        #handling age outliers
        data_df.loc[data_df['age'] < 18, 'age'] = round(data_df['age'].mean())
        data_df.loc[data_df['age'] > 90, 'age'] = round(data_df['age'].mean())
        #remove gift sales from db
        data_df = data_df[data_df['item_price'] != 0]
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
        data_df.loc[data_df["vendor"].str.contains('-offline1'),'on_off'] = 'offline'
        data_df['vendor'] = data_df['vendor'].map(lambda x: x.rstrip('offline'))
        data_df['vendor'] = data_df['vendor'].map(lambda x: x.rstrip('online'))
        data_df['vendor'] = data_df['vendor'].map(lambda x: x.rstrip('-'))
        data_df['vendor'] = data_df['vendor'].map(lambda x: x.rstrip(' - '))
        # data_df['vendor'] = data_df['vendor'].map(lambda x: x.rstrip('-offline1'))
        data_df.loc[data_df["vendor"].str.contains('ferragamo-online-m'),'on_off'] = 'online'
        data_df.loc[data_df["vendor"].str.contains('ferragamo-online-m'),'vendor'] = 'ferragamo-men'
        data_df.loc[data_df["vendor"].str.contains('ferragamo-online-wom'),'on_off'] = 'online'
        data_df.loc[data_df["vendor"].str.contains('ferragamo-online-wom'),'vendor'] = 'ferragamo-women'
        data_df.loc[data_df["vendor"].str.contains('tedbaker-offline1/centra'),'on_off'] = 'offline'
        data_df.loc[data_df["vendor"].str.contains('tedbaker-offline1/centra'),'vendor'] = 'tedbaker'
        data_df.loc[data_df["vendor"].str.contains('tedbaker-offline1/kow'),'on_off'] = 'offline'
        data_df.loc[data_df["vendor"].str.contains('tedbaker-offline1/kow'),'vendor'] = 'tedbaker'
        data_df.loc[data_df["vendor"].str.contains('tedbaker-offline1/c'),'on_off'] = 'offline'
        data_df.loc[data_df["vendor"].str.contains('tedbaker-offline1/c'),'vendor'] = 'tedbaker'
        data_df = data_df[data_df['on_off'] != '__']

        # # Creat the tmp with vendor, title, product and tags for the NLP
        # data_df['item_price_tmp'] = data_df['item_price'].astype(str)
        # data_df['vendor_tmp'] = data_df['vendor'].astype(str)
        # data_df['title_tmp'] = data_df['title'].astype(str)
        # data_df['product_type_tmp'] = data_df['product_type'].astype(str)
        # data_df['tags_tmp'] = data_df['tags'].astype(str)
        # data_df['tmp_NLP'] = data_df[['vendor_tmp', 'title_tmp', 'product_type_tmp', 'tags_tmp']].agg(' '.join, axis=1)
        # data_df.drop(columns=['vendor_tmp','title_tmp','product_type_tmp','tags_tmp','vendor_line','title_line','item_price_tmp'],inplace=True)
        # creating product_cat and product_gender
        data_df['product_cat'] = '__'
        data_df['product_gender'] = '__'
        data_df['vendor_cat'] = '__'
        return data_df

if __name__ == "__main__":
    data_df = get_raw_data(path='/content/2020-2021_V2.csv')
    data_df = BasicCleaner().transform(data_df)
    print(data_df.shape)
