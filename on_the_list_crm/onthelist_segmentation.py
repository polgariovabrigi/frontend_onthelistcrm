import pandas as pd
from sklearn.cluster import KMeans
from product_cat_and_gender import fit,transform
from preprocessing import Encoder



class Segmentation():

    def __init__(self, from_file = True, sample = False):
        self.from_file = from_file
        self.sample = sample


    def clean_transform_data(self, rows=200_000):
        # if we are working with a origine csv from Onthelist
        # this part will clean and creat the columns we need
        # for the kmean model
        if self.from_file == False:
            # importing cleaning and creating the rows product_cat and product_gender
            # here rows indicat the number of rows to take to generate the
            data_df,model,dict_label,tokenizer_ = fit(batch_size=2048,verbose=1,rows=rows)
            data_df = transform(data_df,model,dict_label,tokenizer_,verbose=1)
        # if we are working with a csv already clean and ready for the kmean model
        else:
            data_df = pd.read_csv('data/all_clean.csv')
        # if work only on a part of the data
        if self.sample != False:
            data_df = data_df.sample(n=self.sample,random_state=42)
        # encode the data
        preprocessor = Encoder()
        preprocessor.init()
        preprocessor.fit(data_df)
        self.data_array_transformed,self.id_df = preprocessor.transform(data_df)
        # at the end we have a array with all the non_id/non_date colomns encoded
        # and a dataframe with all the id/date columns
        # Now we return the array with the name X_train
        self.X_train = self.data_array_transformed
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
        customer_segmentation_list = km.labels_
        self.id_df['segmentation'] = customer_segmentation_list
        return km

    def predict(self):
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


if __name__ == "__main__":

    # init the model
    segmentation = Segmentation(from_file = False, sample = 20_000)
    # get the data
    X_train = segmentation.clean_transform_data()
    # fir the model
    segmentation.fit()
    # predict
    segment_df = segmentation.predict()
    print(segment_df)
