import pandas as pd
import pickle
from datetime import datetime
import pytz

from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


class Segmentation():

    def __init__(self,data_df):
        self.data_df = data_df
        self.id_df = data_df[['order_ID','item_ID','date','customer_ID']]

    def set_pipeline(self, n_cluster=10):
        preproc_pipe = ColumnTransformer([('normal_distributed_encoded', StandardScaler(), ['age']),
                                          ('num_minmax_encoded', MinMaxScaler(), ['item_price', 'final_price','item_quantity','item_discount']),
                                          ('categorical_encoded', OneHotEncoder(sparse = False, handle_unknown ='ignore'), ['product_cat','gender','vendor_cat','premium_status','district','nationality','on_off'])
                                        ])

        self.pipe = Pipeline([('preproc', preproc_pipe),
                              ('pca', PCA(n_components=10,random_state=42)), #81% explained
                              ('kmeans',KMeans(n_clusters=n_cluster,random_state=42))
                              ])
        return self.pipe

    def fit(self):
        self.set_pipeline()
        self.km_model = self.pipe.fit(self.data_df)
        return self.km_model

    def predict(self, data_df):
        # returning the segmntation df with ID (one customer_ID can epear many times)
        self.id_df['segmentation'] = self.km_model.predict(data_df)
        # creating empty list to creat the final DF with unique customer_ID
        seg = []
        cust = []
        # feeding the list with the most common segmentation for every unique customer_ID
        num_customer=0
        total_num_customer=len(self.id_df['customer_ID'].unique())
        for customer in self.id_df['customer_ID'].unique():
            tmp_df = self.id_df[self.id_df['customer_ID'] == customer]
            seg.append(tmp_df['segmentation'].mode()[0])
            cust.append(customer)
            num_customer += 1
            if num_customer%1000 == 0:
                print(f'quantity of customer done : {num_customer}/{total_num_customer}')
        # creating and returning the segmntation df with ID (one customer_ID appears only one time)
        self.segment_df = pd.DataFrame({"customer_ID": cust, "customer_segmentation": seg})
        return self.segment_df

    def predict_all_df(self):
        # returning the segmntation df with ID (one customer_ID can epear many times)
        self.data_df['segmentation'] = self.km_model.predict(self.data_df)
        return self.data_df

    def save_km_model(self):
        d = datetime.now(pytz.timezone('Asia/Hong_Kong')).strftime("%d_%m_%Y_%Hh%M")
        filename = f'kmean_model_{d}.sav'
        pickle.dump(self.km_model, open(filename, 'wb'))
        print(f"model save as : {filename}")
        return self

    def load_km_model(self):

        self.km_model = pickle.load(open('on_the_list_crm/kmean_model_07_01_2022_09h15.sav', 'rb'))

        return self.km_model


if __name__ == "__main__":

    data_df = pd.read_pickle('data/03_clean_+_vendor_and_product_cat_done.pkl')
    # data_df = data_df.sample(n=500_000, random_state=42)

    print('init the segmentation')
    print('...')
    segmentation = Segmentation(data_df)
    print('init done')

    # print('fit the model')
    # print('...')
    # segmentation.fit()
    # print('fit done')

    # print('saving the km model')
    # print('...')
    # segmentation.save_km_model()
    # print('model saved')


    print('loading the km model')
    print('...')
    segmentation.load_km_model()
    print('model loaded')

    print('prediction')
    print('...')
    # segment_df = segmentation.predict()
    # segment_df.to_csv('data/segmentation_07_01_2022.csv')
    # segment_df.to_pickle('data/segmentation_07_01_2022.pkl')

    data_df = segmentation.predict_all_df()
    data_df.to_csv('data/segmentation_all_df_07_01_2022.csv')
    data_df.to_pickle('data/segmentation_all_df_07_01_2022.pkl')
    print('prediction done')
