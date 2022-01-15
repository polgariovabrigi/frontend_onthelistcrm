from fastapi import FastAPI, File, UploadFile
import pandas as pd
from on_the_list_crm.vendor_cat import vendor_cat
from on_the_list_crm.cleansing_dataset import BasicCleaner
from on_the_list_crm.onthelist_segmentation import Segmentation
from on_the_list_crm.product_cat_and_gender import transform_dataset, load_nlp_model, load_tokenizer
from io import StringIO
import json

#main page for testing API
app = FastAPI()
@app.get("/")
def root():
    return {"greet": "Hello from Testing"}

# #segmentation route
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    print(file)
    content = await file.read()
    print(content)
    json_data = json.loads(content.decode('utf-8'))
    print(json_data)
    data_df =pd.DataFrame(json_data)
    data_df = BasicCleaner().transform(data_df)
    data_df = vendor_cat(data_df)
    nlp_model = load_nlp_model(model_path='on_the_list_crm/nlp_model_05_01_2022_16h16.sav')
    tokenizer = load_tokenizer(model_path='on_the_list_crm/tokenizer_for_nlp_model_05_01_2022_15h43.sav')
    data_df = transform_dataset(data_df, nlp_model, tokenizer, verbose=1)
    segmentation = Segmentation(data_df)
    segmentation.load_km_model()
    data_df = segmentation.predict_all_df()
    js = data_df.to_json(orient = 'columns')
    return js
