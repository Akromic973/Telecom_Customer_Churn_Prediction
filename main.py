#Import Serializing Library
import pickle

#Import Graphical User Interface Library
import gradio as gr

#Import Filesystem Paths Library
from pathlib import Path

#Import Asynchronous Context Manager Library
from contextlib import asynccontextmanager

#Import Pandas Library
import pandas as pd

#Import FastAPI Class
from fastapi import FastAPI, HTTPException, status
#Import Pydantic Type Verification Class and Function
from pydantic import BaseModel

#________________________________________GLOBAL_VARIABLE____________________________________________#
#Global variable stocking the machine learning model
model = None

#________________________________________LIFESPAN_MANAGER____________________________________________#
#Operation at the start of the application
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model_path = "mlp_threshold_pipeline.pkl"
    print(f"🚀 Loading model from {Path(model_path)}")

    try:
        if not Path(model_path).exists():
            print(f"❌ Error: {model_path} not found.")
            model = None
        else:
            with open(model_path, "rb") as file:
                model = pickle.load(file)
            print(f"✅ Model loaded successfully.")

        #Run the application
        yield

    except Exception as e:
        raise HTTPException(status_code=500, detail= f"❌ Critical error during startup: {e}")

    finally:
        print("🛑 Shutting down app...")
        if "model" in globals():
            del model


#____________________________________________FASTAPI____________________________________________#
#Main application, instance of the FastAPI class
app = FastAPI(
    lifespan= lifespan,
    title="Telecom Customer Churn Prediction App",
    description= """
    Predicts customer churn using Machine Learning. 
    
    - **Inputs**: Customer information such as demographics, service usage, and contract details.  
    - **Output**: Probability of churn (0.0 to 1.0).
    
    """,
    version="0.1.0",
    contact={
        "name": "PAUL Stanley",
        "email": "stanley.paul97300@gmail.com",
    }
)

class CustomerData(BaseModel):
    #ID
    customerID:         str

    #Demographic
    Gender:             str
    SeniorCitizen:      str
    Partner:            str
    Dependents:         str

    #Web Services
    PhoneService:       str
    MultipleLines:      str
    InternetService:    str
    OnlineSecurity:     str
    OnlineBackup:       str
    DeviceProtection:   str
    TechSupport:        str
    StreamingTV:        str
    StreamingMovies:    str

    #Categorical Account Information
    Contract:           str
    PaperlessBilling:   str
    PaymentMethod:      str


    #Numerical Account Information
    Tenure:             int
    MonthlyCharges:     float
    TotalCharges:       float


#Application Health Check
@app.get("/")
async def root():
    """
    Health check endpoint for monitoring and load balancer.

    :return: JSON status message.
    """
    return {"status": "Online", "model_loaded": model is not None}

#Churn Customer Prediction
@app.post("/predict",
          summary="Predicts customer churn using Machine Learning",
          status_code=status.HTTP_201_CREATED,
          )
async def predict(customerData: CustomerData):
    """
        Perform real-time churn analysis based on customer behavior and contract details.

        - **Inputs**: Tenure, Monthly Charges, Contract Type, Internet Service, etc.
        - **Calculations**: Runs a preloaded Multi-layer perceptron model with a fix threshold.
        - **Output**:
            - **Probability**: 0.0 to 1.0 (Higher means more likely to churn)
            - **Decision**: Boolean based on the model's internal threshold
    """

    if model is None:
        raise HTTPException(status_code= 503, detail= "Model not initialized")

    try:
        #Convert single input to DataFrame to maintain feature names
        input_df = pd.DataFrame([customerData.model_dump()])

        #Perform probabilities prediction
        probs = model.predict_proba(input_df)
        churn_prob = float(probs[0][1])

        #Prediction
        prediciton = model.predict(input_df)
        churn = bool(prediciton[0])

        #Customer Prediction Information
        return {
            "customerID": customerData.customerID,
            churn_probability : round(churn_prob, 4),
            "prediciton" : "Churn" if churn else "Stay",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail= f"Error predicting customer churn: {e}")


#____________________________________________GRADIO_GUI_LOGIC____________________________________________#
def gradio_predict(*agrs):
    if model is None:
        raise HTTPException(status_code= 503, detail= "Model not initialized")

    columns = [
        "customerID", "Gender", "SeniorCitizen", "Partner", "Dependents",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "Tenure", "MonthlyCharges", "TotalCharges",
    ]

    try:
        # Pass raw Gradio inputs through Pydantic for cleaning
        raw_data = dict(zip(columns, agrs))
        validated_data = CustomerData(**raw_data)

        #Convert single input to DataFrame to maintain feature names
        input_df = pd.DataFrame([validated_data.model_dump()])

        #Prediction probability
        probs = model.predict_proba(input_df)
        churn_prob = float(probs[0][1])
        label =  "⚠️ CHURN" if bool(model.predict(input_df)[0]) else "✅ STAY"

        #Customer Result
        return f"Result: {label}\nChurn Probability: {round(churn_prob, 4)}"

    except Exception as e:
        raise HTTPException(status_code=500, detail= f"Error predicting customer churn: {e}")



# Define the Gradio Layout
with gr.Blocks(title= "Machine Learning Churn Prediction GUI") as block:
    gr.Markdown("# 📊 Telecom Customer Churn Portal")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 👤 Customer Profile")
            cid = gr.Textbox(label="Customer ID")
            gen = gr.Dropdown(label="Gender", choices=["Male", "Female"])
            sen = gr.Radio(label="Senior Citizen", choices=["Yes", "No"])
            par = gr.Radio(label="Partner", choices=["Yes", "No"])
            dep = gr.Radio(label="Dependents", choices=["Yes", "No"])

        with gr.Column():
            gr.Markdown("### 🛠️ Services")
            ps  = gr.Radio(label="Phone Service", choices=["Yes", "No"])
            ml  = gr.Radio(label="Multiple Lines", choices=["Yes", "No"])
            isv = gr.Dropdown(label="Internet Service", choices=['DSL', 'Fiber optic', 'No'])
            os  = gr.Radio(label="Online Security", choices=['Yes', 'No'])
            ob  = gr.Radio(label="Online Backup", choices=['Yes', 'No'])
            dp  = gr.Radio(label="Device Protection", choices=['Yes', 'No'])
            ts  = gr.Radio(label="Tech Support", choices=['Yes', 'No'])
            stv = gr.Radio(label="Streaming TV", choices=['Yes', 'No'])
            smv = gr.Radio(label="Streaming Movies", choices=['Yes', 'No'])

        with gr.Column():
            gr.Markdown("### 📄 Contract & Billing")
            con = gr.Dropdown(label="Contract", choices=["Month-to-month", "One year", "Two year"])
            pb  = gr.Radio(label="Paperless Billing", choices=["Yes", "No"])
            pm  = gr.Dropdown(label="Paperless Method", choices=['Manual', 'Bank transfer (automatic)', 'Credit card (automatic)'])
            ten = gr.Number(label="Tenure", minimum=0, maximum=600)
            mc  = gr.Number(label="Monthly Charges", minimum=0.0)
            tc  = gr.Number(label="Total Charges", minimum=0.0)

            btn = gr.Button(value="Analyze Risk", variant= "primary")
            output = gr.Textbox(label="Analysis Result")

    btn.click(
        fn= gradio_predict,
        inputs= [cid, gen, sen, par, dep,
                 ps, ml, isv, os, ob, dp, ts, stv, smv,
                 con, pb, pm, ten, mc, tc,
                 ],
        outputs= output,
    )

#___________________________________MOUNT_GRADIO_INTO_FASTAPI_________________________________#
app = gr.mount_gradio_app(app, block, path= "/gui")