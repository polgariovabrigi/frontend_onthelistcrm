from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
# import pickle 
# import pandas as pd
# from io import File

#main page to select what type of prediction client wishes for

app = FastAPI()
@app.get("/")
def root():
    return "Hello from Cloud Run CD"

# #segmentation route
# #get the uploaded files
# @app.post("/uploadfile/")
# async def create_upload_file(file: UploadFile = File(...)):
#    return {
#       "filename": file.filename
#    }

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
   return {
      "filename" : file.filename
   }


# UploadFileapp = FastAPI()
# @app.post("/image") 
# async def image(image: UploadFile = File(...)): 
#     return {
#    "filename": image.filename
# }

# @app.post("/get_csv", response_class = StreamingResponse)
# def get_csv(file: bytes = File(...)):
#     return "Hello"
#     # file as str
#     inputFileAsStr = io.StringIO(str(file,'utf-8'))
#     # dataframe
#     df = pd.read_csv(inputFileAsStr)
#     # output file
#     outFileAsStr = StringIO()
#     df.to_csv(outFileAsStr, index = False)
#     response = StreamingResponse(io.StringIO(df.to_csv(index=False), media_type="csv"),
#         headers={
#             'Content-Disposition': 'attachment;filename=dataset.csv',
#             'Access-Control-Expose-Headers': 'Content-Disposition'
#         }
#     )
#     return response

#run the model 
# @app.get("/predict_segment")
# def predict_segment(csv):
#     model = pickle.load(open('kmean_model_28_12_2021_16h_44.sav', 'rb'))
#     prediction = model.predict(data)
#     return prediction 


#customerß
# @app.get("/predict_product")
# def predict_customer():
#     return "Testing our deployment"


