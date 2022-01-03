import pandas as pd

def get_sample():
    df = pd.read_csv('data/2020-2021.csv')
    #2020-2021.csv is not in data folder yet.
    sample_dt = df.sample(10)
    return sample_dt

if __name__ == "__main__":
    sample = get_sample()
    sample
