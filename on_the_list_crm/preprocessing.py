from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

def encode_data(df):
    categorical_ord_enc = OneHotEncoder(handle_unknown='ignore')
    numerical_enc = MinMaxScaler()
    scaler = StandardScaler()
    preprocessor = ColumnTransformer([
        ('normal_distributed_encoded', scaler, ['age']),
        ('num_minmax_encoded', numerical_enc, ['price', 'final_price','quantity']),
        ('categorical_encoded', categorical_ord_enc, ['vendor','product_cat','gender','product_gender','premium_status'])],
        remainder = "passthrough")
    return preprocessor 