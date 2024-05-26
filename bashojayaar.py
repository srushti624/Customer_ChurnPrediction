from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from tensorflow.keras.models import load_model

app = FastAPI()

# Define the data model using Pydantic
class Customer(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    PhoneService: int
    MultipleLines: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    PaperlessBilling: int
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract_Month_to_month: int
    Contract_One_year: int
    Contract_Two_year: int
    PaymentMethod_Electronic_check: int
    PaymentMethod_Mailed_check: int
    PaymentMethod_Bank_transfer_automatic: int
    PaymentMethod_Credit_card_automatic: int
    InternetService_DSL: int
    InternetService_Fiber_optic: int
    InternetService_No: int

# Load the ANN model
model = load_model(r'C:\Users\srush\Downloads\customer_churn\customer_churn_ANNmodel.h5')

@app.post("/predict/")
async def predict_churn(customer: Customer):
    try:
        # Convert the Pydantic model to a DataFrame
        input_data = pd.DataFrame([dict(customer)])
        # Predict using the loaded model
        prediction = model.predict(input_data)
        # Return the prediction
        return {"probability_of_churn": float(prediction[0, 0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))