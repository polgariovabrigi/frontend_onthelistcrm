from fastapi import FastAPI, File, UploadFile
import pandas as pd
from on_the_list_crm.onthelist_segmentation import Segmentation
from on_the_list_crm.product_cat_and_gender import transform_dataset
from io import StringIO

#main page for testing API
app = FastAPI()
@app.get("/")
def root():
    return { "greet": "Hello from Testing" }

# #segmentation route
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    content = await file.read()
    # df = pd.read_csv(StringIO(str(content)), encoding='utf-8')
    # print(type(df))
    print(type(content))
    return content

# #segmentation route
# @app.post("/uploadfile")
# async def upload_file(file: UploadFile = File(...)):
#     contents = await file.file
#     return {"filename": contents}
    # df = pd.read_csv(csv_file.file)

    # data_df = pd.read_dict(contents)
    # data_df = transform_dataset(data_df)
    # segmentation = Segmentation(data_df)
    # segmentation.load_km_model()
    # segment_df = segmentation.predict()


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
