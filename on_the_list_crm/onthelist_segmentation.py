import pandas as pd
import pickle
from datetime import datetime
import pytz

from product_cat_and_gender import transform_dataset

from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


class Segmentation():

    # def __init__(self, sample = False, from_file = False, raw_data_path='data/2020-2021.csv' , file_path='data/all_clean.csv', model_path='kmean_model_28_12_2021_16h44.sav'):
        # self.sample = sample
        # self.from_file = from_file
        # self.file_path = file_path
        # self.model_path = model_path
        # self.raw_data_path = raw_data_path

    def __init__(self,data_df):
        self.data_df = data_df
        self.id_df = data_df[['order_ID','item_ID','date','customer_ID']]

    # def get_data_and_clean(self, rows_nlp = 200_000):
    #     # if we are working with a origine csv from Onthelist
    #     # this part will clean and creat the columns we need
    #     # for the kmean model
    #     if self.from_file == False:
    #         self.data_df = transform_dataset(data_path = self.raw_data_path, rows = None, verbose=0)
    #     # if we are working with a csv already clean and ready for the kmean model
    #     else:
    #         self.data_df = pd.read_csv(self.file_path)
    #     # if work only on a part of the data
    #     if self.sample != False:
    #         self.data_df = self.data_df.sample(n=self.sample,random_state=42)
    #     self.id_df = self.data_df[['order_ID','item_ID','date','customer_ID']]
    #     return self.data_df

    def set_pipeline(self, n_cluster=10):
        preproc_pipe = ColumnTransformer([('normal_distributed_encoded', StandardScaler(), ['age']),
                                          ('num_minmax_encoded', MinMaxScaler(), ['item_price', 'final_price','item_quantity','item_discount']),
                                        #   ('categorical_encoded', OneHotEncoder(sparse = False), ['vendor','product_cat','gender','product_gender','premium_status','district','nationality','on_off'])])
                                          ('categorical_encoded', OneHotEncoder(sparse = False), ['product_cat','gender','product_gender','premium_status','district','nationality','on_off'])])
        self.pipe = Pipeline([('preproc', preproc_pipe),
                              ('pca', PCA(n_components=10,random_state=42)), #81% explained
                              ('kmeans',KMeans(n_clusters=n_cluster,random_state=42,))
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
        for customer in self.id_df['customer_ID'].unique():
            tmp_df = self.id_df[self.id_df['customer_ID'] == customer]
            seg.append(tmp_df['segmentation'].mode()[0])
            cust.append(customer)
        # creating and returning the segmntation df with ID (one customer_ID appears only one time)
        self.segment_df = pd.DataFrame({"customer_ID": cust, "customer_segmentation": seg})
        return self.segment_df

    def save_km_model(self):
        d = datetime.now(pytz.timezone('Asia/Hong_Kong')).strftime("%d_%m_%Y_%Hh%M")
        filename = f'kmean_model_{d}.sav'
        pickle.dump(self.km_model, open(filename, 'wb'))
        return self

    def load_km_model(self):
        self.km_model = pickle.load(open('kmean_model_28_12_2021_16h44.sav', 'rb'))
        return self.km_model


if __name__ == "__main__":

    # init the model
    segmentation = Segmentation(from_file = True, sample = False)
    segmentation.get_data_and_clean()
    # pipe = segmentation.fit()
    # segmentation.save_km_model(pipe)
    segmentation.load_km_model()
    segment_df = segmentation.predict()
    print(segment_df)
