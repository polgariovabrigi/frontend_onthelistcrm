import pandas as pd


def get_raw_data(path='data/2020-2021_V2.csv', rows = None):
    #load data
    data_df = pd.read_csv(path, nrows=rows,encoding='latin1')
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
        data_df = data_df[data_df['on_off'] != '__']
        data_df['vendor'] = data_df['vendor'].map(lambda x: x.rstrip('-offline'))
        data_df['vendor'] = data_df['vendor'].map(lambda x: x.rstrip('-online'))
        data_df['vendor'] = data_df['vendor'].map(lambda x: x.rstrip(' - online'))
        data_df['vendor'] = data_df['vendor'].map(lambda x: x.rstrip(' - offline'))

        # Creat the tmp with vendor, title, product and tags for the NLP
        data_df['item_price_tmp'] = data_df['item_price'].astype(str)
        data_df['vendor_tmp'] = data_df['vendor'].astype(str)
        data_df['title_tmp'] = data_df['title'].astype(str)
        data_df['product_type_tmp'] = data_df['product_type'].astype(str)
        data_df['tags_tmp'] = data_df['tags'].astype(str)
        data_df['tmp_NLP'] = data_df[['vendor_tmp', 'title_tmp', 'product_type_tmp', 'tags_tmp']].agg(' '.join, axis=1)
        data_df['tmp_NLP_2'] = data_df[['item_price_tmp', 'title_tmp', 'product_type_tmp', 'tags_tmp']].agg(' '.join, axis=1)
        data_df.drop(columns=['vendor_tmp','title_tmp','product_type_tmp','tags_tmp','vendor_line','title_line','item_price_tmp'],inplace=True)
        # creating product_cat and product_gender
        data_df['product_cat'] = '__'
        data_df['product_gender'] = '__'
        data_df['vendor_cat'] = '__'
        return data_df

class GetDataFrameToTrainNLP():

    def __init__(self):
        pass

    def fit(self, data_df, y=None):
        assert isinstance(data_df, pd.DataFrame)
        # selecting 200_000 rows for the training
        data_sample_200_000_df = data_df.sample(n = 200_000)
        data_sample_200_000_df = data_sample_200_000_df.dropna()
        # cleaning all the ","
        data_sample_200_000_df['tags'] = data_sample_200_000_df["tags"].str.replace(',',' ')

        # manual encoding for product_cat
        ### clothes
        data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('\d{2}'),'product_cat'] = 'clothes'
        data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('[\s|]x*s[\s|]'),'product_cat'] = 'clothes'
        data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('[\s|]m[\s|]'),'product_cat'] = 'clothes'
        data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('[\s|]x*l[\s|]'),'product_cat'] = 'clothes'
        data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('shirt'),'product_cat'] = 'clothes'
        data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('dress'),'product_cat'] = 'clothes'
        data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('underwear'),'product_cat'] = 'clothes'
        data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('trousers'),'product_cat'] = 'clothes'

        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('watershorts'),'product_cat'] = 'clothes'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('top'),'product_cat'] = 'clothes'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('dresses'),'product_cat'] = 'clothes'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('trousers'),'product_cat'] = 'clothes'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('clothing'),'product_cat'] = 'clothes'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('apparel'),'product_cat'] = 'clothes'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('ready-to-wear'),'product_cat'] = 'clothes'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('underwear'),'product_cat'] = 'clothes'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('bra'),'product_cat'] = 'clothes'

        data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('risinglotus'),'product_cat'] = 'clothes'
        data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('aliceolivia'),'product_cat'] = 'clothes'
        data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('casacalvin'),'product_cat'] = 'clothes'
        data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('risinglotus'),'product_cat'] = 'clothes'

        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('shoes'),'product_cat'] = 'clothes'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('footwear'),'product_cat'] = 'clothes'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('boot'),'product_cat'] = 'clothes'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('slippers'),'product_cat'] = 'clothes'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('sandal'),'product_cat'] = 'clothes'
        ### bath_body
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('bath & body'),'product_cat'] = 'bath_body'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('fragrance'),'product_cat'] = 'bath_body'
        data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('the body shop'),'product_cat'] = 'bath_body'
        data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('dermatory'),'product_cat'] = 'bath_body'
        data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('bathing'),'product_cat'] = 'bath_body'
        data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('sheet mask'),'product_cat'] = 'bath_body'
        data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('cleansing'),'product_cat'] = 'bath_body'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('serum'),'product_cat'] = 'bath_body'
        data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('skincare set'),'product_cat'] = 'bath_body'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('makeup'),'product_cat'] = 'bath_body'
        data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('shower gel'),'product_cat'] = 'bath_body'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('defrisant'),'product_cat'] = 'bath_body'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('skincare'),'product_cat'] = 'bath_body'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('shampoo'),'product_cat'] = 'bath_body'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('leave-in beauty'),'product_cat'] = 'bath_body'
        data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('phyto'),'product_cat'] = 'bath_body'
        data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('shiseido'),'product_cat'] = 'bath_body'
        data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('bareminerals'),'product_cat'] = 'bath_body'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('lip'),'product_cat'] = 'bath_body'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('skin'),'product_cat'] = 'bath_body'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('personal care appliance'),'product_cat'] = 'bath_body'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('make-up'),'product_cat'] = 'bath_body'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('make up'),'product_cat'] = 'bath_body'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('hand care'),'product_cat'] = 'bath_body'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('anti-aging'),'product_cat'] = 'bath_body'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('face'),'product_cat'] = 'bath_body'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('eyes'),'product_cat'] = 'bath_body'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('hair'),'product_cat'] = 'bath_body'
        data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('moisturising'),'product_cat'] = 'bath_body'
        data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('aromatherapy associates'),'product_cat'] = 'bath_body'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('bodycare'),'product_cat'] = 'bath_body'
        ### accessory
        data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('[\s|]bag'),'product_cat'] = 'accessory'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('goggles'),'product_cat'] = 'accessory'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('tech accessories'),'product_cat'] = 'accessory'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('accessories'),'product_cat'] = 'accessory'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('bag'),'product_cat'] = 'accessory'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('bags'),'product_cat'] = 'accessory'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('cap'),'product_cat'] = 'accessory'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('jewelry'),'product_cat'] = 'accessory'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('jewellery'),'product_cat'] = 'accessory'
        data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('accessories'),'product_cat'] = 'accessory'
        data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('wallets'),'product_cat'] = 'accessory'
        data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('belts'),'product_cat'] = 'accessory'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('rings'),'product_cat'] = 'accessory'
        data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('scarf'),'product_cat'] = 'accessory'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('goods'),'product_cat'] = 'accessory'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('handbags'),'product_cat'] = 'accessory'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('wallet'),'product_cat'] = 'accessory'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('sunglasses'),'product_cat'] = 'accessory'
        # food
        data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('qwehli'),'product_cat'] = 'food'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('seafood'),'product_cat'] = 'food'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('beef'),'product_cat'] = 'food'
        data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('plantin kaviari'),'product_cat'] = 'food'
        ### wine
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('wine'),'product_cat'] = 'wine'
        data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('altaya'),'product_cat'] = 'wine'
        ### drink
        data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('or tea'),'product_cat'] = 'drink'
        data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('ortea'),'product_cat'] = 'drink'
        data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('tea'),'product_cat'] = 'drink'
        ### kitchen
        data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('lecreuset'),'product_cat'] = 'kitchen'
        data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('le creuset'),'product_cat'] = 'kitchen'
        data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('stojostasher'),'product_cat'] = 'kitchen'
        data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('kitchenware'),'product_cat'] = 'kitchen'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('kitchenware'),'product_cat'] = 'kitchen'
        ### home_appliance
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('home electronics'),'product_cat'] = 'home_appliance'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('homeware'),'product_cat'] = 'home_appliance'
        ### home
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('candle'),'product_cat'] = 'home'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('towel'),'product_cat'] = 'home'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('bedding set'),'product_cat'] = 'home'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('writitng'),'product_cat'] = 'home'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('writing'),'product_cat'] = 'home'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('luggage'),'product_cat'] = 'home'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('cuddledry'),'product_cat'] = 'home'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('paper product'),'product_cat'] = 'home'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('pillow case'),'product_cat'] = 'home'
        data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('riedel'),'product_cat'] = 'home'
        data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('moleskine'),'product_cat'] = 'home'
        data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('waldmann'),'product_cat'] = 'home'
        # manual encoding for product_gender
        ### men
        data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('[\s|]men'),'product_gender'] = 'men'
        data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('^men'),'product_gender'] = 'men'
        data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('[\s|]male'),'product_gender'] = 'men'
        data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('^male'),'product_gender'] = 'men'
        data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('[\s|]man'),'product_gender'] = 'men'
        data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('^man'),'product_gender'] = 'men'
        ### women
        data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('[\s|]women'),'product_gender'] = 'women'
        data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('^women'),'product_gender'] = 'women'
        data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('[\s|]female'),'product_gender'] = 'women'
        data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('^female'),'product_gender'] = 'women'
        data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('[\s|]woman'),'product_gender'] = 'women'
        data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('^woman'),'product_gender'] = 'women'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('^jewel'),'product_gender'] = 'women'
        ### children
        data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('[\s|]junior'),'product_gender'] = 'children'
        data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('^junior'),'product_gender'] = 'children'
        data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('[\s|]baby'),'product_gender'] = 'children'
        data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('^baby'),'product_gender'] = 'children'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('.*baby.*'),'product_gender'] = 'children'
        data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('.*kids.*'),'product_gender'] = 'children'
        # removing all the non_tag rows
        data_for_train_test = data_sample_200_000_df[(data_sample_200_000_df['product_cat'] != '__')]
        # creating the features and target
        X_train = data_for_train_test['tmp_NLP']
        y_train = data_for_train_test['product_cat']
        return X_train,y_train


if __name__ == "__main__":
    data_df = get_raw_data(path='/content/2020-2021_V2.csv')
    data_df = BasicCleaner().transform(data_df)
    data_df
