from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import cleansing_dataset

def encode_data(rows = None):

    data_df = cleansing_dataset.get_name_clean(rows=rows)

    categorical_ord_enc = OneHotEncoder(handle_unknown='ignore')
    numerical_enc = MinMaxScaler()
    scaler = StandardScaler()

    preprocessor = ColumnTransformer([
        ('normal_distributed_encoded', scaler, data_df[['age']]),
        ('num_minmax_encoded', numerical_enc, data_df[['item_price', 'final_price','item_quantity','item_discount']]),
        ('categorical_encoded', categorical_ord_enc, data_df[['vendor','product_cat','gender','product_gender','premium_status','district','nationality','on_off']])],
        remainder = "passthrough")
    return preprocessor

if __name__ == "__main__" :

    # data_df = cleansing_dataset.get_name_clean(rows=50000)
    print(encode_data(rows = 50000))
