from fastapi import FastAPI, File, UploadFile
import pandas as pd

import on_the_list_crm.cleansing_dataset
import on_the_list_crm.vendor_cat
import on_the_list_crm.product_cat_and_gender
import on_the_list_crm.onthelist_segmentation

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
    # data_df = transform_dataset(data_df)
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
