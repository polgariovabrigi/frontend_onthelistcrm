from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return "On The List AI Prediction"

#segmentation route
@app.get("/predict_segment")
def predict_segment(csv):
    return "Testing our deployment"

#customerß
@app.get("/predict_product")
def predict_customer():
    return "Testing our deployment"