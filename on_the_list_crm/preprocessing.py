from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

def encode_data(df):
    categorical_ord_enc = OneHotEncoder(handle_unknown='ignore')
    numerical_enc = MinMaxScaler()
    preprocessor = ColumnTransformer([
        ('num_minmax_encoded', numerical_enc, ['price', 'final_price','quantity','age']),
        ('categorical_encoded', categorical_ord_enc, ['vendor','product_cat','gender','product_gender','premium_status'])],
        remainder = "passthrough")
    return preprocessor 
