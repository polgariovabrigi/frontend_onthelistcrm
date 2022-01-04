import cleansing_dataset

def vendor_cat(data_df):
    with open( 'vendor_cat_original_list.txt', 'rb' ) as f:
        vendor_cat_original_dictionary = f.read().decode()
        vendor_cat_original_dictionary = eval(vendor_cat_original_dictionary)
    for vendor,cat in vendor_cat_original_dictionary.items():
        data_df.loc[data_df["vendor"].str.contains(vendor),'vendor_cat'] = cat
    data_df = data_df[data_df['vendor_cat'] != '__']
    return(data_df)

if __name__ == "__main__":

    print('get the data')
    print('...')
    data_df = cleansing_dataset.get_raw_data(path='data/2020-2021_V2.csv', rows = None)
    print(f'data loaded : {data_df.shape}')

    print('cleaning the data')
    print('...')
    data_df = cleansing_dataset.BasicCleaner().transform(data_df)
    print(f'data clean : {data_df.shape}')

    print('computing vendor cat')
    print('...')
    data_df = vendor_cat(data_df)
    print(f'vendor cat ok : {data_df.shape}')

    data_df.to_csv('data/clean_and_vendor_cat_done.csv')
    # print(f'new data_df saved as : clean_and_vendor_cat_done.csv')
    print(f'new data_df saved as : clean_and_vendor_cat_done.csv',encoding='utf-8-sig')
