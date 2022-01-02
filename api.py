from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import pandas as pd



#main page for testing API
app = FastAPI()
@app.get("/")
def root():
    return "Hello from Cloud Run CD"

# #segmentation route
@app.post("/uploadfile/")
async def upload_file(csv_file: UploadFile = File(...)):
    # dataframe = pd.read_csv(csv_file.file)
    #get the file
    #clean
    #return prediction
    return {"Hello"}
  


#Loading the trained model
# with open("./finalized_model.pkl", "rb") as f:
#     loaded_model = pickle.load(f)

# @app.post("/get_csv", response_class = StreamingResponse)
# def get_csv(file: bytes = File(...)):
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


