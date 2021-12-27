from os import ctermid
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import cleansing_dataset
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.models import load_model
# import json
import datetime
import pickle


def fit(batch_size=32,verbose=0, rows=200_000):
    data_df,data_sample_200_000_df = get_data(rows=rows)
    data_sample_200_000_df = manual_encoding_data_for_product_cat(data_sample_200_000_df)
    data_sample_200_000_df = manual_encoding_data_for_product_gender(data_sample_200_000_df)
    X_train,y_train,dict_label = clean_training_data(data_sample_200_000_df)
    X_train_token,max_size,vocab_size,tokenizer_ = tokenizer(X_train)
    X_train_token_pad = padding(X_train_token,max_size)
    model = initiat_the_model(max_size,vocab_size)
    model = fit_the_model(model,X_train_token_pad,y_train,batch_size=batch_size,verbose=verbose)
    model_saved = save_the_model_pickle(model)
    return(data_df,model,dict_label,tokenizer_, model_saved)

def get_data(rows = 200_000):
    # get the data
    data_df = cleansing_dataset.get_name_clean()

    # extracting 200_000 rows
    data_sample_200_000_df = data_df.sample(n = rows,random_state=42)
    data_sample_200_000_df = data_sample_200_000_df.dropna()

    data_sample_200_000_df['tags'] = data_sample_200_000_df["tags"].str.replace(',',' ')

    # creating product_cat and product_gender
    data_sample_200_000_df['product_cat'] = '__'
    data_df['product_cat'] = '__'
    data_sample_200_000_df['product_gender'] = 'unisex'
    data_df['product_gender'] = 'unisex'

    return data_df,data_sample_200_000_df

def manual_encoding_data_for_product_cat(data_sample_200_000_df):

    # clothes
    data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('\d{2}'),'product_cat'] = 'clothes'
    data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('[\s|]x*s[\s|]'),'product_cat'] = 'clothes'
    data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('[\s|]m[\s|]'),'product_cat'] = 'clothes'
    data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('[\s|]x*l[\s|]'),'product_cat'] = 'clothes'
    data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('shirt'),'product_cat'] = 'clothes'
    data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('dress'),'product_cat'] = 'clothes'
    data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('underwear'),'product_cat'] = 'clothes'
    data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('trousers'),'product_cat'] = 'clothes'

    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('watershorts'),'product_cat'] = 'clothes'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('top'),'product_cat'] = 'clothes'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('dresses'),'product_cat'] = 'clothes'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('trousers'),'product_cat'] = 'clothes'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('clothing'),'product_cat'] = 'clothes'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('apparel'),'product_cat'] = 'clothes'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('ready-to-wear'),'product_cat'] = 'clothes'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('underwear'),'product_cat'] = 'clothes'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('bra'),'product_cat'] = 'clothes'

    data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('risinglotus'),'product_cat'] = 'clothes'
    data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('aliceolivia'),'product_cat'] = 'clothes'
    data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('casacalvin'),'product_cat'] = 'clothes'
    data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('risinglotus'),'product_cat'] = 'clothes'

    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('shoes'),'product_cat'] = 'clothes'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('footwear'),'product_cat'] = 'clothes'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('boot'),'product_cat'] = 'clothes'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('slippers'),'product_cat'] = 'clothes'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('sandal'),'product_cat'] = 'clothes'

    # bath_body
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('bath & body'),'product_cat'] = 'bath_body'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('fragrance'),'product_cat'] = 'bath_body'
    data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('the body shop'),'product_cat'] = 'bath_body'
    data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('dermatory'),'product_cat'] = 'bath_body'
    data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('bathing'),'product_cat'] = 'bath_body'
    data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('sheet mask'),'product_cat'] = 'bath_body'
    data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('cleansing'),'product_cat'] = 'bath_body'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('serum'),'product_cat'] = 'bath_body'
    data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('skincare set'),'product_cat'] = 'bath_body'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('makeup'),'product_cat'] = 'bath_body'
    data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('shower gel'),'product_cat'] = 'bath_body'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('defrisant'),'product_cat'] = 'bath_body'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('skincare'),'product_cat'] = 'bath_body'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('shampoo'),'product_cat'] = 'bath_body'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('leave-in beauty'),'product_cat'] = 'bath_body'
    data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('phyto'),'product_cat'] = 'bath_body'
    data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('shiseido'),'product_cat'] = 'bath_body'
    data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('bareminerals'),'product_cat'] = 'bath_body'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('lip'),'product_cat'] = 'bath_body'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('skin'),'product_cat'] = 'bath_body'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('personal care appliance'),'product_cat'] = 'bath_body'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('make-up'),'product_cat'] = 'bath_body'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('make up'),'product_cat'] = 'bath_body'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('hand care'),'product_cat'] = 'bath_body'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('anti-aging'),'product_cat'] = 'bath_body'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('face'),'product_cat'] = 'bath_body'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('eyes'),'product_cat'] = 'bath_body'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('hair'),'product_cat'] = 'bath_body'
    data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('moisturising'),'product_cat'] = 'bath_body'
    data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('aromatherapy associates'),'product_cat'] = 'bath_body'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('bodycare'),'product_cat'] = 'bath_body'

    # accessory
    data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('[\s|]bag'),'product_cat'] = 'accessory'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('goggles'),'product_cat'] = 'accessory'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('tech accessories'),'product_cat'] = 'accessory'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('accessories'),'product_cat'] = 'accessory'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('bag'),'product_cat'] = 'accessory'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('bags'),'product_cat'] = 'accessory'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('cap'),'product_cat'] = 'accessory'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('jewelry'),'product_cat'] = 'accessory'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('jewellery'),'product_cat'] = 'accessory'
    data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('accessories'),'product_cat'] = 'accessory'
    data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('wallets'),'product_cat'] = 'accessory'
    data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('belts'),'product_cat'] = 'accessory'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('rings'),'product_cat'] = 'accessory'
    data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('scarf'),'product_cat'] = 'accessory'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('goods'),'product_cat'] = 'accessory'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('handbags'),'product_cat'] = 'accessory'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('wallet'),'product_cat'] = 'accessory'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('sunglasses'),'product_cat'] = 'accessory'

    # food
    data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('qwehli'),'product_cat'] = 'food'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('seafood'),'product_cat'] = 'food'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('beef'),'product_cat'] = 'food'
    data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('plantin kaviari'),'product_cat'] = 'food'

    # wine
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('wine'),'product_cat'] = 'wine'
    data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('altaya'),'product_cat'] = 'wine'

    # drink
    data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('or tea'),'product_cat'] = 'drink'
    data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('ortea'),'product_cat'] = 'drink'
    data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('tea'),'product_cat'] = 'drink'

    # kitchen
    data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('lecreuset'),'product_cat'] = 'kitchen'
    data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('le creuset'),'product_cat'] = 'kitchen'
    data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('stojostasher'),'product_cat'] = 'kitchen'
    data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('kitchenware'),'product_cat'] = 'kitchen'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('kitchenware'),'product_cat'] = 'kitchen'

    # home_appliance
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('home electronics'),'product_cat'] = 'home_appliance'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('homeware'),'product_cat'] = 'home_appliance'

    # home
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('candle'),'product_cat'] = 'home'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('towel'),'product_cat'] = 'home'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('bedding set'),'product_cat'] = 'home'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('writitng'),'product_cat'] = 'home'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('writing'),'product_cat'] = 'home'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('luggage'),'product_cat'] = 'home'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('cuddledry'),'product_cat'] = 'home'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('paper product'),'product_cat'] = 'home'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('pillow case'),'product_cat'] = 'home'
    data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('riedel'),'product_cat'] = 'home'
    data_sample_200_000_df.loc[data_sample_200_000_df["vendor"].str.contains('moleskine'),'product_cat'] = 'home'

    return data_sample_200_000_df

def manual_encoding_data_for_product_gender(data_sample_200_000_df):

    # men
    data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('[\s|]men'),'product_gender'] = 'men'
    data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('^men'),'product_gender'] = 'men'
    data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('[\s|]male'),'product_gender'] = 'men'
    data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('^male'),'product_gender'] = 'men'
    data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('[\s|]man'),'product_gender'] = 'men'
    data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('^man'),'product_gender'] = 'men'

    # women
    data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('[\s|]women'),'product_gender'] = 'women'
    data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('^women'),'product_gender'] = 'women'
    data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('[\s|]female'),'product_gender'] = 'women'
    data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('^female'),'product_gender'] = 'women'
    data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('[\s|]woman'),'product_gender'] = 'women'
    data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('^woman'),'product_gender'] = 'women'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('^jewel'),'product_gender'] = 'women'

    # children
    data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('[\s|]junior'),'product_gender'] = 'children'
    data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('^junior'),'product_gender'] = 'children'
    data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('[\s|]baby'),'product_gender'] = 'children'
    data_sample_200_000_df.loc[data_sample_200_000_df["tags"].str.contains('^baby'),'product_gender'] = 'children'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('.*baby.*'),'product_gender'] = 'children'
    data_sample_200_000_df.loc[data_sample_200_000_df["product_type"].str.contains('.*kids.*'),'product_gender'] = 'children'

    return data_sample_200_000_df

def clean_training_data(data_sample_200_000_df):
    # removing all the non_tag rows
    data_for_train_test = data_sample_200_000_df[(data_sample_200_000_df['product_cat'] != '__')]

    # creating the features and target
    X_train = data_for_train_test['tmp_NLP']
    y_train = data_for_train_test['product_cat']

    # encoding the target and creat a dictionary to identify theme
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)

    # nb_cat = len(data_for_train_test['product_cat'].unique().tolist())
    # lst_cat = list(range(0,nb_cat-1))
    # lst = list(label_encoder.inverse_transform(lst_cat))
    lst = list(label_encoder.inverse_transform([0,1,2,3,4,5,6,7,8]))
    dict_label = {}
    for i in range (9):
        dict_label[i] = lst[i]
    dict_label

    # save the dict_label
    with open('product_cat_dict.txt', 'w') as f:
        print(dict_label, file=f)
    # with open('product_cat_dict.txt', 'w') as file:
    #     file.write(json.dumps(dict_label))

    return(X_train,y_train,dict_label)

def tokenizer(X_train):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    X_train_token = tokenizer.texts_to_sequences(X_train)
    # get token max size
    max_size = 78
    # for element in X_train_token:
    #     if len(element) > max_size:
    #         max_size = len(element)

    # get vocab_size
    vocab_size = max([i for i in tokenizer.word_index.values()])

    return X_train_token,max_size,vocab_size,tokenizer

def padding(X_train_token,max_size):
    X_train_token_pad = pad_sequences(X_train_token,
                                      maxlen=max_size,
                                      dtype='int32',
                                      padding='post',
                                      value=0.0)
    return X_train_token_pad

def initiat_the_model(max_size,vocab_size):
    model = Sequential()

    # creating the layers
    model.add(layers.Embedding(input_dim=vocab_size+1,
                               input_length=max_size,
                               output_dim=100,
                               mask_zero=True,
                               ))
    model.add(layers.LSTM(units=20))
    model.add(layers.Dense(10,activation="relu"))
    model.add(layers.Dense(10,activation="relu"))
    model.add(layers.Dense(9,activation="softmax"))
    return model

def fit_the_model(model,X_train_token_pad,y_train,batch_size=32,verbose=0):
    # computing the model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics='accuracy')
    es = EarlyStopping(patience=4,restore_best_weights=True)
    # history = model.fit(...   to store the epoch
    model.fit(X_train_token_pad, y_train,
                        validation_split=0.2,
                        epochs=20,
                        callbacks=[es],
                        batch_size=batch_size,
                        verbose=verbose)

    return model

def save_the_model(model, path="./knn_model_for_product_cat"):
    """Save the NLP model to disk, and return the
    name and location of the file.
    """
    ct = datetime.datetime.now().strftime("%d/%m_%H_%M")
    model_name = 'model' + '_' + ct
    model.save(f"{path}/{model_name}")
    return model_name


def save_the_model_pickle(model, path="./knn_model_for_product_cat"):
    # Export model as pickle file
    ct = datetime.datetime.now().strftime("%d/%m_%H_%M")
    modelname = 'model' + '_' + ct
    with open(f"{path}/{modelname}.pkl", "wb") as file:
        pickle.dump(model, file)
    # Load model from pickle file
    model = pickle.load(open(f"{path}/{modelname}.pkl", "rb"))
    return model

def transform(data_df,model,dict_label,tokenizer,verbose=0):
    # applying the model to the whole dataset

    # extracting the tmp_NLP column
    x = data_df['tmp_NLP']
    # creating the list to receive the prediction
    pred_list_tmp = []

    # comput the size of x
    x_size = x.shape[0]

    # creating the loop that will transform x step by step
    count = 0
    while count < x_size-9999:
        x_tmp = x.iloc[count:count+10000]
        x_token = tokenizer.texts_to_sequences(x_tmp)
        x_token_pad = pad_sequences(x_token,
                                    maxlen=78,
                                    dtype='int32',
                                    padding='post',
                                    value=0.0)
        prediction = model.predict(x_token_pad,verbose=verbose)
        # storing the prediction into pred_list_tmp
        for sub_list in prediction:
            sub_list = sub_list.tolist()
            maximum = max(sub_list)
            index = sub_list.index(maximum)
            pred_list_tmp.append(index)
        count = count+10000

    x_tmp = x.iloc[count:x_size+1]
    x_tmp = x.iloc[count:count+9999]
    x_token = tokenizer.texts_to_sequences(x_tmp)
    x_token_pad = pad_sequences(x_token,
                                maxlen=78,
                                dtype='int32',
                                padding='post',
                                value=0.0)
    prediction = model.predict(x_token_pad,verbose=verbose)
    # storing the prediction into pred_list_tmp
    for sub_list in prediction:
        sub_list = sub_list.tolist()
        maximum = max(sub_list)
        index = sub_list.index(maximum)
        pred_list_tmp.append(index)


    # changing numbers to the actual name of the category
    pred_list = []
    for element in pred_list_tmp :
        pred_list.append(dict_label[element])


    data_df['product_cat'] = pred_list

    # removing the tmp_NLP, tags, product_type and title columns
    data_df = data_df.drop(columns=['tmp_NLP','tags','product_type','title'])

    return data_df

if __name__ == "__main__" :

    # quick try #
    data_df,model,dict_label,tokenizer_,model_saved = fit(batch_size=2048,verbose=1,rows=200)
    # long run #
    # data_df,model,dict_label,tokenizer_ = fit(batch_size=32,verbose=1)

    data_df = transform(data_df,model,dict_label,tokenizer_,verbose=1)
    print(data_df)
    # data_df.to_csv('data/all_clean.csv')
