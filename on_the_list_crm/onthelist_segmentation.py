import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# from sklearn.pipeline import Pipeline #only for the if name=main
# from sklearn.compose import ColumnTransformer #only for the if name=main
# from sklearn.preprocessing import OneHotEncoder, MinMaxScaler #only for the if name=main
from product_cat_and_gender import fit,transform
from preprocessing import Encoder

### class Segmentation :
###   input :
###      you need to give the encoded df WITHOUT the customer_ID colomn
###      and the customer_ID on a separate df
###   output :
###      return a df with 2 rows, the customer_ID (unique) and the segmentation categorie
###
###
### To use this class :
###
###   1. init the class
###        segmentation = Segmentation( <encoded_df> , <customer_ID_df> )
###
###   2. fit the class
###        segmentation.fit()
###
###   3. predict the class return a df with two columns
###        final_df = segmentation.predict()




class Segmentation():

    def __init__(self, from_file = 'yes'):
        self.from_file = from_file

    #### For later ... need to find a way to predict the perfect k dynamicaly. we will take 10 for now
    #create the function with the raw_dataset using cleansing (Brigita's file)

    def clean_transform_data(self, rows=None):
        if self.from_file == 'no':
            data_df,model,dict_label,tokenizer_ = fit(batch_size=2048,verbose=1,rows=rows)
            data_df = transform(data_df,model,dict_label,tokenizer_,verbose=1)
            preprocessor = Encoder()
            preprocessor.init()
            preprocessor.fit(data_df)
            self.data_array_transformed,self.id_df = preprocessor.transform(data_df)
            return self.data_array_transformed,self.id_df

        data_df = pd.read_csv('data/all_clean.csv')
        data_df = data_df.sample(n=20_000,random_state=42) #  <<================= to delet later
        data_df.dropna()
        preprocessor = Encoder()
        preprocessor.init()
        preprocessor.fit(data_df)
        self.data_array_transformed,self.id_df = preprocessor.transform(data_df)

        self.X_train = self.data_array_transformed
        # self.X_train = self.X_train.astype(dtype = 'float')

        # for column in self.X_train.columns:
        #     if self.X_train[column].dtypes != 'float64':
        #         print(f'erreur {column}')

        # self.X_train = self.X_train.to_numpy()
        # print(self.X_train.dtype)
        # self.X_train = pd.DataFrame(self.X_train)
        # self.X_train = self.X_train.astype('float64')
        # self.X_train_final = self.X_train[:,0:2]
        # self.X_train_final = pd.DataFrame(self.X_train_final)
        # print(self.X_train_final.dtypes)
        return self.X_train

    # def k_finder(self):
    #     # inertias = []
    #     # ks = range(1,50,2)
    #     # for k in ks:
    #     #     km_test = KMeans(n_clusters=k).fit(X_train_transformed)
    #     #     inertias.append(km_test.inertia_)
    #     self.n_cluster=10
    #     return self.n_cluster

    def fit(self,n_cluster=10):
        # doing the kmeans model
        km = KMeans(n_clusters=n_cluster)


        km.fit(self.X_train)

        # returning the segmntation df with ID (one customer_ID can epear many times)
        customer_segmentation_list = km.labels_.tolist()
        self.customer_ID_df['segmentation'] = customer_segmentation_list
        return None

    def predict(self):
        # creating empty list to creat the final DF with unique customer_ID
        seg = []
        cust = []

        # feeding the list with the most common segmentation for every unique customer_ID
        for customer in self.customer_ID_df['customer_ID'].unique():
            tmp_df = self.customer_ID_df[self.customer_ID_df['customer_ID'] == customer]
            seg.append(tmp_df['segmentation'].mode()[0])
            cust.append(customer)

        # creating and returning the segmntation df with ID (one customer_ID appears only one time)
        self.segment_df = pd.DataFrame({"customer_ID": cust, "customer_segmentation": seg})

        return self.segment_df




if __name__ == "__main__":


    segmentation = Segmentation(from_file = 'yes')

    #get the data
    data_df = segmentation.clean_transform_data()

    segmentation.fit()
    # segment_df = segmentation.predict()

    # print(segment_df)



    # fit the class
    # segmentation.fit()

    # # predict the class
    # final_df = segmentation.predict()
    # print(final_df)
