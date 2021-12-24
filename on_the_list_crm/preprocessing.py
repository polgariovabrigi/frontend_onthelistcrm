from pandas.core.arrays.integer import Int8Dtype
from six import print_
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn import set_config; set_config(display='diagram')

class Encoder():

    def __init__(self):
        self.preprocessor = None

    def init(self):

        categorical_ord_enc = OneHotEncoder(sparse = False,dtype='int8')
        numerical_enc = MinMaxScaler()
        scaler = StandardScaler()

        self.preprocessor = ColumnTransformer([
            ('normal_distributed_encoded', scaler, ['age']),
            ('num_minmax_encoded', numerical_enc, ['item_price', 'final_price','item_quantity','item_discount']),
            ('categorical_encoded', categorical_ord_enc, ['vendor','product_cat','gender','product_gender','premium_status','district','nationality','on_off'])])
            # remainder = "passthrough")

        return self.preprocessor

    def fit(self,data_df):
        self.preprocessor.fit(data_df)
        return self.preprocessor

    def transform(self,data_df,dtype='float64'):
        self.id_df = data_df[['Unnamed: 0','order_ID','item_ID','date','customer_ID']]
        self.data_array_transformed = self.preprocessor.transform(data_df)
        # data_df_transformed = pd.DataFrame(data_array_transformed)
        # data_df_transformed = data_df_transformed.astype(dtype)
        # for column in id_df.columns:
        #     data_df_transformed[column]=id_df[column]
        return self.data_array_transformed,self.id_df


if __name__ == "__main__" :

    data_df = pd.read_csv('data/all_clean.csv')
    data_df = data_df.sample(n=20000,random_state=42)
    preprocessor = Encoder()
    preprocessor.init()
    preprocessor.fit(data_df)
    data_array_transformed,id_df = preprocessor.transform(data_df)

    # print(data_array_transformed.dtype)
    # print(id_df)
