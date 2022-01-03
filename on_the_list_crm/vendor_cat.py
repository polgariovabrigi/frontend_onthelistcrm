

def vendor_cat(data_df):
    with open( 'vendor_cat_original_list.txt', 'rb' ) as f:
        vendor_cat_original_dictionary = f.read().decode()
        vendor_cat_original_dictionary = eval(vendor_cat_original_dictionary)
    for vendor,cat in vendor_cat_original_dictionary.items():
        data_df.loc[data_df["vendor"].str.contains(vendor),'vendor_cat'] = cat
    data_df = data_df[data_df['vendor_cat'] != '__']
    return(data_df)
