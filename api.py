from fastapi import FastAPI, File, UploadFile
import pandas as pd

from on_the_list_crm.cleansing_dataset import BasicCleaner
from on_the_list_crm.vendor_cat import vendor_cat
from on_the_list_crm.product_cat_and_gender import load_nlp_model , load_tokenizer, transform_dataset
from on_the_list_crm.onthelist_segmentation import Segmentation

# from on_the_list_crm.onthelist_segmentation import Segmentation
# from on_the_list_crm.product_cat_and_gender import transform_dataset

#main page for testing API
app = FastAPI()
@app.get("/")
def root():
    return "Hello from Cloud Run CD"

# #segmentation route
@app.post("/uploadfile/")
async def upload_file(csv_file: UploadFile = File(...)):
    contents = await csv_file.read()
    data_df = pd.read_dict(contents)
    return data_df
    # data_df = BasicCleaner(data_df)
    # data_df = vendor_cat(data_df)
    # nlp_model = load_nlp_model(model_path='on_the_list_crm/kmean_model_05_01_2022_19h19.sav')
    # tokenizer = load_tokenizer(model_path='on_the_list_crm/tokenizer_for_nlp_model_05_01_2022_15h43.sav')
    # transform_dataset(data_df, nlp_model, tokenizer)
    # segmentation = Segmentation(data_df)
    # segmentation.load_km_model()
    # segment_df = segmentation.predict()
    # return segment_df

    #return prediction
    # return {"filename": csv_file.filename,
    #         "filetype": csv_file.content_type,
    #         "content": contents
    #         }

#Loading the trained model
# with open("./finalized_model.pkl", "rb") as f:
#     loaded_model = pickle.load(f)


#customerß
# @app.get("/predict_product")
# def predict_customer():
#     return "Testing our deployment"
