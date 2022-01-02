from pandas.core.arrays.integer import Int8Dtype
from six import print_
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from product_cat_and_gender import fit,transform
from sklearn.neighbors import NearestNeighbors
import pandas as pd

class CustomerPrediction():

    def __init__(self, from_file = True, sample = False):
        self.from_file = from_file
        self.sample = sample


    def clean_transform_data(self, rows=200_000):
        # if we are working with a origine csv from Onthelist
        # this part will clean and creat the columns we need
        if self.from_file == False:
            # importing cleaning and creating the rows product_cat and product_gender
            # here rows indicat the number of rows to take to generate the
            data_df,model,dict_label,tokenizer_ = fit(batch_size=2048,verbose=1,rows=rows)
            data_df = transform(data_df,model,dict_label,tokenizer_,verbose=1)
        # if we are working with a csv already clean and ready for the kmean model
        else:
            data_df = pd.read_csv('data/data_400_000_rows_clean_df.csv')
            self.data_sel_df = data_df[['customer_ID','price','quantity','discount','final_price','product_cat','product_gender','on_off','premium_status']]
        # if work only on a part of the data
        if self.sample != False:
            data_df = data_df.sample(n=self.sample,random_state=42)

        # encode the data
        categorical_ord_enc = OneHotEncoder(handle_unknown='ignore', sparse = False)
        numerical_enc = MinMaxScaler()

        X = self.data_sel_df.drop(['customer_ID'], axis=1)
        y = self.data_sel_df[['customer_ID']]

        self.preprocessor = ColumnTransformer([
            ('num_minmax_encoded', numerical_enc, ['price', 'final_price','quantity','discount']),
            ('categorical_encoded', categorical_ord_enc, ['product_cat','product_gender','premium_status','on_off'])])

        self.X_transformed = self.preprocessor.fit_transform(X)
        return self.X_transformed


    def predict(self, input, n_neighbors=1000):

        # input_df = pd.DataFrame([[200,1,0,200,'bath_body','unisex','online','y']],columns=['price','quantity','discount','final_price','product_cat','product_gender','on_off','premium_status'])
        input_df = pd.DataFrame([input],columns=['price','quantity','discount','final_price','product_cat','product_gender','on_off','premium_status'])
        self.input_transformed = self.preprocessor.transform(input_df)
        self.n_neighbors = n_neighbors

        neigh = NearestNeighbors(n_neighbors=n_neighbors)
        neigh.fit(self.X_transformed)
        NearestNeighbors(n_neighbors=n_neighbors)

        index_df = neigh.kneighbors(self.input_transformed)[1][0]
        pred_df = self.data_sel_df.iloc[index_df,:]
        pred_sorted_purchase_vol_df = pred_df.groupby(['customer_ID'])['final_price'].sum().reset_index().rename(columns={'final_price':'total_purchase_amount'}).sort_values(by=['total_purchase_amount'], ascending=False)
        return pred_sorted_purchase_vol_df


if __name__ == "__main__":

    # init the model
    customerprediction = CustomerPrediction(from_file = True, sample = 200_000)
    # get the data
    X_transformed = customerprediction.clean_transform_data()
    # predict
    pred_df = customerprediction.predict([200,1,0,200,'bath_body','unisex','online','y'])
    print(pred_df)
