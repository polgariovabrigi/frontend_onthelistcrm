import pandas as pd
from datetime import datetime
import pytz
import pickle


from on_the_list_crm.cleansing_dataset import get_raw_data, BasicCleaner, GetDataFrameToTrainNLP

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping



def fit_and_save_nlp_model(data_df, batch_size=32,verbose=1):
    data_df = creating_column_for_nlp_train(data_df)
    X_train, y_train = creating_X_and_y_train(data_df)
    y_train,dict_label = y_train_encoder(y_train)
    X_train_token,max_size,vocab_size,tokenizer_ = tokenizer(X_train)
    X_train_token_pad = padding(X_train_token,max_size)
    nlp_model = initiat_nlp_model(max_size,vocab_size)
    nlp_model = fit_nlp_model(nlp_model,X_train_token_pad,y_train,batch_size=batch_size,verbose=verbose)
    filename = save_nlp_model(nlp_model)
    print(f"model save as : {filename}")
    return(data_df,nlp_model,dict_label,tokenizer_)

def creating_column_for_nlp_train(data_df):
    data_df['vendor_tmp'] = data_df['vendor'].astype(str)
    data_df['vendor_cat_tmp'] = data_df['vendor_cat'].astype(str)
    data_df['title_tmp'] = data_df['title'].astype(str)
    data_df['product_type_tmp'] = data_df['product_type'].astype(str)
    data_df['tags_tmp'] = data_df['tags'].astype(str)
    data_df['tmp_NLP'] = data_df[['vendor_cat_tmp', 'title_tmp', 'product_type_tmp', 'tags_tmp', 'vendor_tmp']].agg(' '.join, axis=1)
    data_df.drop(columns=['vendor_cat_tmp','title_tmp','product_type_tmp','tags_tmp','vendor_tmp'],inplace=True)
    return data_df

def creating_X_and_y_train(data_df):

    # selecting 200_000 rows for the training
    data_sample_400_000_df = data_df.sample(n = 400_000)
    data_sample_400_000_df = data_sample_400_000_df.dropna()
    # cleaning all the ","
    data_sample_400_000_df['tags'] = data_sample_400_000_df["tags"].str.replace(',',' ')

    # manual encoding for product_cat
    ### general encoding
    data_sample_400_000_df.loc[data_sample_400_000_df["vendor_cat"].str.contains('home'),'product_cat'] = 'home'
    data_sample_400_000_df.loc[data_sample_400_000_df["vendor_cat"].str.contains('shoes'),'product_cat'] = 'shoes'
    data_sample_400_000_df.loc[data_sample_400_000_df["vendor_cat"].str.contains('cosmetic'),'product_cat'] = 'bath_body'
    data_sample_400_000_df.loc[data_sample_400_000_df["vendor_cat"].str.contains('bags'),'product_cat'] = 'accessory'
    data_sample_400_000_df.loc[data_sample_400_000_df["vendor_cat"].str.contains('watches & jewellery'),'product_cat'] = 'accessory'
    data_sample_400_000_df.loc[data_sample_400_000_df["vendor_cat"].str.contains('f&b'),'product_cat'] = 'food'
    data_sample_400_000_df.loc[data_sample_400_000_df["vendor_cat"].str.contains('wine'),'product_cat'] = 'wine'


    ### clothes
    # data_sample_400_000_df.loc[data_sample_400_000_df["tags"].str.contains('\d{2}'),'product_cat'] = 'clothes'
    data_sample_400_000_df.loc[data_sample_400_000_df["tags"].str.contains('[\s|]x*s[\s|]'),'product_cat'] = 'clothes'
    data_sample_400_000_df.loc[data_sample_400_000_df["tags"].str.contains('[\s|]m[\s|]'),'product_cat'] = 'clothes'
    data_sample_400_000_df.loc[data_sample_400_000_df["tags"].str.contains('[\s|]x*l[\s|]'),'product_cat'] = 'clothes'
    data_sample_400_000_df.loc[data_sample_400_000_df["tags"].str.contains('shirt'),'product_cat'] = 'clothes'
    data_sample_400_000_df.loc[data_sample_400_000_df["tags"].str.contains('dress'),'product_cat'] = 'clothes'
    data_sample_400_000_df.loc[data_sample_400_000_df["tags"].str.contains('underwear'),'product_cat'] = 'clothes'
    data_sample_400_000_df.loc[data_sample_400_000_df["tags"].str.contains('trousers'),'product_cat'] = 'clothes'
    data_sample_400_000_df.loc[data_sample_400_000_df["tags"].str.contains('swimsuit'),'product_cat'] = 'clothes'

    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('watershorts'),'product_cat'] = 'clothes'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('top'),'product_cat'] = 'clothes'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('dresses'),'product_cat'] = 'clothes'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('trousers'),'product_cat'] = 'clothes'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('clothing'),'product_cat'] = 'clothes'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('apparel'),'product_cat'] = 'clothes'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('ready-to-wear'),'product_cat'] = 'clothes'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('underwear'),'product_cat'] = 'clothes'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('bra'),'product_cat'] = 'clothes'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('swimsuit'),'product_cat'] = 'clothes'

    # data_sample_400_000_df.loc[data_sample_400_000_df["vendor"].str.contains('risinglotus'),'product_cat'] = 'clothes'
    # data_sample_400_000_df.loc[data_sample_400_000_df["vendor"].str.contains('aliceolivia'),'product_cat'] = 'clothes'
    # data_sample_400_000_df.loc[data_sample_400_000_df["vendor"].str.contains('casacalvin'),'product_cat'] = 'clothes'
    # data_sample_400_000_df.loc[data_sample_400_000_df["vendor"].str.contains('risinglotus'),'product_cat'] = 'clothes'

    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('shoes'),'product_cat'] = 'shoes'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('footwear'),'product_cat'] = 'shoes'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('boot'),'product_cat'] = 'shoes'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('slippers'),'product_cat'] = 'shoes'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('sandal'),'product_cat'] = 'shoes'

    ### bath_body
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('bath & body'),'product_cat'] = 'bath_body'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('fragrance'),'product_cat'] = 'bath_body'
    # data_sample_400_000_df.loc[data_sample_400_000_df["vendor"].str.contains('the body shop'),'product_cat'] = 'bath_body'
    # data_sample_400_000_df.loc[data_sample_400_000_df["vendor"].str.contains('dermatory'),'product_cat'] = 'bath_body'
    data_sample_400_000_df.loc[data_sample_400_000_df["tags"].str.contains('bathing'),'product_cat'] = 'bath_body'
    data_sample_400_000_df.loc[data_sample_400_000_df["tags"].str.contains('sheet mask'),'product_cat'] = 'bath_body'
    data_sample_400_000_df.loc[data_sample_400_000_df["tags"].str.contains('cleansing'),'product_cat'] = 'bath_body'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('serum'),'product_cat'] = 'bath_body'
    data_sample_400_000_df.loc[data_sample_400_000_df["tags"].str.contains('skincare set'),'product_cat'] = 'bath_body'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('makeup'),'product_cat'] = 'bath_body'
    data_sample_400_000_df.loc[data_sample_400_000_df["tags"].str.contains('shower gel'),'product_cat'] = 'bath_body'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('defrisant'),'product_cat'] = 'bath_body'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('skincare'),'product_cat'] = 'bath_body'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('shampoo'),'product_cat'] = 'bath_body'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('leave-in beauty'),'product_cat'] = 'bath_body'
    data_sample_400_000_df.loc[data_sample_400_000_df["vendor"].str.contains('phyto'),'product_cat'] = 'bath_body'
    data_sample_400_000_df.loc[data_sample_400_000_df["vendor"].str.contains('shiseido'),'product_cat'] = 'bath_body'
    data_sample_400_000_df.loc[data_sample_400_000_df["vendor"].str.contains('bareminerals'),'product_cat'] = 'bath_body'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('lip'),'product_cat'] = 'bath_body'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('skin'),'product_cat'] = 'bath_body'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('personal care appliance'),'product_cat'] = 'bath_body'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('make-up'),'product_cat'] = 'bath_body'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('make up'),'product_cat'] = 'bath_body'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('hand care'),'product_cat'] = 'bath_body'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('anti-aging'),'product_cat'] = 'bath_body'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('face'),'product_cat'] = 'bath_body'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('eyes'),'product_cat'] = 'bath_body'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('hair'),'product_cat'] = 'bath_body'
    data_sample_400_000_df.loc[data_sample_400_000_df["tags"].str.contains('moisturising'),'product_cat'] = 'bath_body'
    data_sample_400_000_df.loc[data_sample_400_000_df["vendor"].str.contains('aromatherapy associates'),'product_cat'] = 'bath_body'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('bodycare'),'product_cat'] = 'bath_body'


    ### accessory
    data_sample_400_000_df.loc[data_sample_400_000_df["tags"].str.contains('[\s|]bag'),'product_cat'] = 'accessory'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('goggles'),'product_cat'] = 'accessory'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('tech accessories'),'product_cat'] = 'accessory'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('accessories'),'product_cat'] = 'accessory'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('bag'),'product_cat'] = 'accessory'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('bags'),'product_cat'] = 'accessory'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('cap'),'product_cat'] = 'accessory'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('jewelry'),'product_cat'] = 'accessory'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('jewellery'),'product_cat'] = 'accessory'
    data_sample_400_000_df.loc[data_sample_400_000_df["tags"].str.contains('accessories'),'product_cat'] = 'accessory'
    data_sample_400_000_df.loc[data_sample_400_000_df["tags"].str.contains('wallets'),'product_cat'] = 'accessory'
    data_sample_400_000_df.loc[data_sample_400_000_df["tags"].str.contains('belts'),'product_cat'] = 'accessory'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('rings'),'product_cat'] = 'accessory'
    data_sample_400_000_df.loc[data_sample_400_000_df["tags"].str.contains('scarf'),'product_cat'] = 'accessory'
    # data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('goods'),'product_cat'] = 'accessory'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('handbags'),'product_cat'] = 'accessory'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('wallet'),'product_cat'] = 'accessory'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('sunglasses'),'product_cat'] = 'accessory'

    # food
    # data_sample_400_000_df.loc[data_sample_400_000_df["vendor"].str.contains('qwehli'),'product_cat'] = 'food'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('seafood'),'product_cat'] = 'food'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('beef'),'product_cat'] = 'food'
    # data_sample_400_000_df.loc[data_sample_400_000_df["vendor"].str.contains('plantin kaviari'),'product_cat'] = 'food'
    data_sample_400_000_df.loc[data_sample_400_000_df["tags"].str.contains('tea'),'product_cat'] = 'food'

    ### wine
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('wine'),'product_cat'] = 'wine'
    # data_sample_400_000_df.loc[data_sample_400_000_df["vendor"].str.contains('altaya'),'product_cat'] = 'wine'

    ### drink
    # data_sample_400_000_df.loc[data_sample_400_000_df["vendor"].str.contains('or tea'),'product_cat'] = 'drink'
    # data_sample_400_000_df.loc[data_sample_400_000_df["vendor"].str.contains('ortea'),'product_cat'] = 'drink'
    # data_sample_400_000_df.loc[data_sample_400_000_df["tags"].str.contains('tea'),'product_cat'] = 'drink'

    ### kitchen
    data_sample_400_000_df.loc[data_sample_400_000_df["vendor"].str.contains('lecreuset'),'product_cat'] = 'kitchen'
    data_sample_400_000_df.loc[data_sample_400_000_df["vendor"].str.contains('le creuset'),'product_cat'] = 'kitchen'
    # data_sample_400_000_df.loc[data_sample_400_000_df["vendor"].str.contains('stojostasher'),'product_cat'] = 'kitchen'
    data_sample_400_000_df.loc[data_sample_400_000_df["tags"].str.contains('kitchenware'),'product_cat'] = 'kitchen'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('kitchenware'),'product_cat'] = 'kitchen'

    ### home
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('home electronics'),'product_cat'] = 'home'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('homeware'),'product_cat'] = 'home'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('candle'),'product_cat'] = 'home'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('towel'),'product_cat'] = 'home'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('bedding set'),'product_cat'] = 'home'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('writitng'),'product_cat'] = 'home'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('writing'),'product_cat'] = 'home'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('luggage'),'product_cat'] = 'home'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('cuddledry'),'product_cat'] = 'home'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('paper product'),'product_cat'] = 'home'
    data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('pillow case'),'product_cat'] = 'home'
    # data_sample_400_000_df.loc[data_sample_400_000_df["vendor"].str.contains('riedel'),'product_cat'] = 'home'
    # data_sample_400_000_df.loc[data_sample_400_000_df["vendor"].str.contains('moleskine'),'product_cat'] = 'home'
    # data_sample_400_000_df.loc[data_sample_400_000_df["vendor"].str.contains('waldmann'),'product_cat'] = 'home'


    # # manual encoding for product_gender
    # ### men
    # data_sample_400_000_df.loc[data_sample_400_000_df["tags"].str.contains('[\s|]men'),'product_gender'] = 'men'
    # data_sample_400_000_df.loc[data_sample_400_000_df["tags"].str.contains('^men'),'product_gender'] = 'men'
    # data_sample_400_000_df.loc[data_sample_400_000_df["tags"].str.contains('[\s|]male'),'product_gender'] = 'men'
    # data_sample_400_000_df.loc[data_sample_400_000_df["tags"].str.contains('^male'),'product_gender'] = 'men'
    # data_sample_400_000_df.loc[data_sample_400_000_df["tags"].str.contains('[\s|]man'),'product_gender'] = 'men'
    # data_sample_400_000_df.loc[data_sample_400_000_df["tags"].str.contains('^man'),'product_gender'] = 'men'
    # ### women
    # data_sample_400_000_df.loc[data_sample_400_000_df["tags"].str.contains('[\s|]women'),'product_gender'] = 'women'
    # data_sample_400_000_df.loc[data_sample_400_000_df["tags"].str.contains('^women'),'product_gender'] = 'women'
    # data_sample_400_000_df.loc[data_sample_400_000_df["tags"].str.contains('[\s|]female'),'product_gender'] = 'women'
    # data_sample_400_000_df.loc[data_sample_400_000_df["tags"].str.contains('^female'),'product_gender'] = 'women'
    # data_sample_400_000_df.loc[data_sample_400_000_df["tags"].str.contains('[\s|]woman'),'product_gender'] = 'women'
    # data_sample_400_000_df.loc[data_sample_400_000_df["tags"].str.contains('^woman'),'product_gender'] = 'women'
    # data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('^jewel'),'product_gender'] = 'women'
    # ### children
    # data_sample_400_000_df.loc[data_sample_400_000_df["tags"].str.contains('[\s|]junior'),'product_gender'] = 'children'
    # data_sample_400_000_df.loc[data_sample_400_000_df["tags"].str.contains('^junior'),'product_gender'] = 'children'
    # data_sample_400_000_df.loc[data_sample_400_000_df["tags"].str.contains('[\s|]baby'),'product_gender'] = 'children'
    # data_sample_400_000_df.loc[data_sample_400_000_df["tags"].str.contains('^baby'),'product_gender'] = 'children'
    # data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('.*baby.*'),'product_gender'] = 'children'
    # data_sample_400_000_df.loc[data_sample_400_000_df["product_type"].str.contains('.*kids.*'),'product_gender'] = 'children'

    # removing all the non_tag rows
    data_for_train_test = data_sample_400_000_df[(data_sample_400_000_df['product_cat'] != '__')]
    # creating the features and target
    X_train = data_for_train_test['tmp_NLP']
    y_train = data_for_train_test['product_cat']
    return X_train, y_train

def y_train_encoder(y_train):

    # encoding the target and creat a dictionary to identify theme
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    # creation of a dict_label to save the encoding
    lst = list(label_encoder.inverse_transform([0,1,2,3,4,5,6,7]))
    dict_label = {}
    for i in range (8):
        dict_label[i] = lst[i]
    dict_label
    # save the dict_label
    d = datetime.now(pytz.timezone('Asia/Hong_Kong')).strftime("%d_%m_%Y_%Hh%M")
    filename = f'product_cat_dict_{d}.txt'
    with open(filename, 'w') as f:
        print(dict_label, file=f)
    return y_train,dict_label

def tokenizer(X_train):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)

    # save the tokenizer
    d = datetime.now(pytz.timezone('Asia/Hong_Kong')).strftime("%d_%m_%Y_%Hh%M")
    filename = f'tokenizer_for_nlp_model_{d}.sav'
    pickle.dump(tokenizer, open(filename, 'wb'))

    # encode the text
    X_train_token = tokenizer.texts_to_sequences(X_train)
    # get token max size
    max_size = 100
    # get vocab_size
    vocab_size = max([i for i in tokenizer.word_index.values()])
    print(vocab_size)

    return X_train_token,max_size,vocab_size,tokenizer

def padding(X_train_token,max_size):
    X_train_token_pad = pad_sequences(X_train_token,
                                      maxlen=max_size,
                                      dtype='int32',
                                      padding='post',
                                      value=0.0)
    return X_train_token_pad

def initiat_nlp_model(max_size,vocab_size):
    nlp_model = Sequential()

    # creating the layers
    nlp_model.add(layers.Embedding(input_dim=vocab_size+1,
                               input_length=max_size,
                               output_dim=100,
                               mask_zero=True,
                               ))
    nlp_model.add(layers.LSTM(units=20))
    nlp_model.add(layers.Dense(10,activation="relu"))
    nlp_model.add(layers.Dense(10,activation="relu"))
    nlp_model.add(layers.Dense(8,activation="softmax"))
    return nlp_model

def fit_nlp_model(nlp_model,X_train_token_pad,y_train,batch_size=32,verbose=0):
    # computing the model
    nlp_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics='accuracy')
    es = EarlyStopping(patience=4,restore_best_weights=True)
    # history = model.fit(...   to store the epoch
    nlp_model.fit(X_train_token_pad,
                  y_train,
                  validation_split=0.2,
                  epochs=20,
                  callbacks=[es],
                  batch_size=batch_size,
                  verbose=verbose)

    return nlp_model

def save_nlp_model(nlp_model):
    d = datetime.now(pytz.timezone('Asia/Hong_Kong')).strftime("%d_%m_%Y_%Hh%M")
    filename = f'nlp_model_{d}.sav'
    pickle.dump(nlp_model, open(filename, 'wb'))
    return filename

# ______________________________________________________________________

def transform_dataset(data_df, nlp_model, tokenizer, verbose=0):
    data_df['vendor_tmp'] = data_df['vendor'].astype(str)
    data_df['vendor_cat_tmp'] = data_df['vendor_cat'].astype(str)
    data_df['title_tmp'] = data_df['title'].astype(str)
    data_df['product_type_tmp'] = data_df['product_type'].astype(str)
    data_df['tags_tmp'] = data_df['tags'].astype(str)
    data_df['tmp_NLP'] = data_df[['vendor_cat_tmp', 'title_tmp', 'product_type_tmp', 'tags_tmp', 'vendor_tmp']].agg(' '.join, axis=1)
    data_df.drop(columns=['vendor_cat_tmp','title_tmp','product_type_tmp','tags_tmp','vendor_tmp'],inplace=True)

    # loading the models and the product_cat_dict
    with open( 'on_the_list_crm/product_cat_dict_05_01_2022_15h43.txt', 'rb' ) as f:
        dict_label = f.read().decode()
        dict_label = eval(dict_label)

    # extracting the tmp_NLP column
    X = data_df['tmp_NLP']
    # creating the list to receive the prediction
    pred_list_tmp = []
    # comput the size of x
    X_size = X.shape[0]
    # creating the loop that will transform x step by step
    count = 0

    while count < X_size-9999:
        print(f'comput from line {count} to line {count + 10000 -1}')
        X_tmp = X.iloc[count:count+10000]
        X_token = tokenizer.texts_to_sequences(X_tmp)
        X_token_pad = pad_sequences(X_token,
                                maxlen=100,
                                dtype='int32',
                                padding='post',
                                value=0.0)
        prediction = nlp_model.predict(X_token_pad,verbose=verbose)
        # storing the prediction into pred_list_tmp
        for sub_list in prediction:
            sub_list = sub_list.tolist()
            maximum = max(sub_list)
            index = sub_list.index(maximum)
            pred_list_tmp.append(index)
        count = count+10000

    # X_tmp = X.iloc[count:X_size+1]
    print(f'comput from line {count} to final line')
    X_tmp = X.iloc[count:]
    X_token = tokenizer.texts_to_sequences(X_tmp)
    X_token_pad = pad_sequences(X_token,
                            maxlen=100,
                            dtype='int32',
                            padding='post',
                            value=0.0)
    prediction = nlp_model.predict(X_token_pad,verbose=verbose)
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
    # data_df = data_df.drop(columns=['tmp_NLP','vendor_tmp', 'title_tmp', 'product_type_tmp', 'tags_tmp'])
    return data_df

def load_nlp_model(model_path='on_the_list_crm/nlp_model_05_01_2022_16h16.sav'):
    nlp_model = pickle.load(open(model_path, 'rb'))
    return nlp_model

def load_tokenizer(model_path='on_the_list_crm/tokenizer_for_nlp_model_05_01_2022_15h43.sav'):
    tokenizer = pickle.load(open(model_path, 'rb'))
    return tokenizer

if __name__ == "__main__" :

    #### to save the model ####
    # print('loading the data')
    # print('...')
    # data_df = pd.read_pickle('data/02_clean_+_vendor_cat_done.pkl')

    # print('creating and saving the model')
    # print('...')
    # fit_and_save_nlp_model(data_df, batch_size=1024,verbose=1)


    #### to comput product cat ####
    data_df = pd.read_pickle('data/02_clean_+_vendor_cat_done.pkl')
    print('computing product cat')
    print('...')
    data_df = transform_dataset(data_df ,verbose=1)
    data_df.to_pickle('data/03_clean_+_vendor_and_product_cat_done.pkl')
