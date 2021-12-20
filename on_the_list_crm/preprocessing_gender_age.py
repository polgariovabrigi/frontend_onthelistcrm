import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

# import numpy as np 

def get_data():
    data = pd.read_csv("../raw_data/Makefile.csv")
    return data

def process_gender(df):
    #encoded data
    ord_enc = OrdinalEncoder()
    df['gender_encoded'] = ord_enc.fit_transform(df[["gender"]])
    #balanced data 
    
    return df

def procecss_age(df):
    #age below 15 and above 90 was imputed with age mean
    age_mean = df['age'].mean().astype(int)
    df['age'] = df['age'].map(lambda x: age_mean if (x < 10 or x > 90) else x)
    #replace null values with mean value
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(df[['age']])
    df['age'] = imputer.transform(df[['age']]) 
    return df 
