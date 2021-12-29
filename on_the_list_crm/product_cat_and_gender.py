from datetime import datetime
import pytz
import pickle

from cleansing_dataset import get_raw_data, BasicCleaner, GetDataFrameToTrainNLP

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping



def fit_and_save_nlp_model(batch_size=32,verbose=1):
    data_df = get_raw_data(path='data/2020-2021.csv', rows = None)
    data_df = BasicCleaner().transform(data_df)
    X_train,y_train = GetDataFrameToTrainNLP().fit(data_df)
    y_train,dict_label = y_train_encoder(y_train)
    X_train_token,max_size,vocab_size,tokenizer_ = tokenizer(X_train)

    X_train_token_pad = padding(X_train_token,max_size)
    nlp_model = initiat_nlp_model(max_size,vocab_size)
    nlp_model = fit_nlp_model(nlp_model,X_train_token_pad,y_train,batch_size=batch_size,verbose=verbose)
    filename = save_nlp_model(nlp_model)
    print(f"model save as : {filename}")
    return(data_df,nlp_model,dict_label,tokenizer_)


def y_train_encoder(y_train):

    # encoding the target and creat a dictionary to identify theme
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    # creation of a dict_label to save the encoding
    lst = list(label_encoder.inverse_transform([0,1,2,3,4,5,6,7,8]))
    dict_label = {}
    for i in range (9):
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
    max_size = 78
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
    nlp_model.add(layers.Dense(9,activation="softmax"))
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
    filename = f'kmean_model_{d}.sav'
    pickle.dump(nlp_model, open(filename, 'wb'))
    return filename

# ______________________________________________________________________

def transform_dataset(data_path = 'data/2020-2021.csv',rows = None,verbose=0):
    # applying the model to the whole dataset

    # loading the models and the product_cat_dict
    nlp_model = load_km_model(model_path='nlp_model_29_12_2021_01h16.sav')
    tokenizer = load_tokenizer(model_path='tokenizer_for_nlp_model_29_12_2021_20h20.sav')
    with open( 'product_cat_dict.txt', 'rb' ) as f:
        dict_label = f.read().decode()
        dict_label = eval(dict_label)

    # loading the data
    data_df = get_raw_data(path=data_path, rows = rows)
    data_df = BasicCleaner().transform(data_df)

    # extracting the tmp_NLP column
    X = data_df['tmp_NLP']
    # creating the list to receive the prediction
    pred_list_tmp = []
    # comput the size of x
    X_size = X.shape[0]
    # creating the loop that will transform x step by step
    count = 0

    while count < X_size-9999:
        X_tmp = X.iloc[count:count+10000]
        X_token = tokenizer.texts_to_sequences(X_tmp)
        X_token_pad = pad_sequences(X_token,
                                maxlen=78,
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

    X_tmp = X.iloc[count:X_size+1]
    X_token = tokenizer.texts_to_sequences(X_tmp)
    X_token_pad = pad_sequences(X_token,
                            maxlen=78,
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
    data_df = data_df.drop(columns=['vendor_tmp', 'title_tmp', 'product_type_tmp', 'tags_tmp'])
    return data_df

def load_km_model(model_path='nlp_model_29_12_2021_01h16.sav'):
    nlp_model = pickle.load(open(model_path, 'rb'))
    return nlp_model

def load_tokenizer(model_path='tokenizer_for_nlp_model_29_12_2021_20h20.sav'):
    tokenizer = pickle.load(open(model_path, 'rb'))
    return tokenizer

if __name__ == "__main__" :

    # quick try #
    # data_df,model,dict_label,tokenizer_ = fit_and_save_nlp_model(batch_size=2048,verbose=1)
    # long run #
    data_df,model,dict_label,tokenizer_ = fit_and_save_nlp_model(batch_size=32,verbose=1)

    # data_df = transform_dataset(data_path = 'data/2020-2021.csv',rows = 100_000,verbose=1)
    # print(data_df)
    # data_df.to_csv('data/verif.csv')
