from fastapi import FastAPI, UploadFile, File, Response
import pandas as pd
from fastapi.responses import HTMLResponse, FileResponse
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import io

app = FastAPI()

# Load the model at the start of the app
model = joblib.load('rf_trained_model.joblib')

@app.get("/predict", response_class=HTMLResponse)
async def form_post():
    return """
    <html>
        <head>
            <style>
                body { 
                    display: flex; 
                    justify-content: center; 
                    align-items: center; 
                    height: 100vh; 
                    margin: 0; 
                    font-family: Arial, sans-serif; 
                }
                form {
                    border: 1px solid #ccc;
                    padding: 20px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                }
                input[type=file] {
                    margin-bottom: 10px;
                }
                input[type=submit] {
                    background-color: #4CAF50;
                    color: white;
                    padding: 10px 20px;
                    border: none;
                    cursor: pointer;
                }
                input[type=submit]:hover {
                    background-color: #45a049;
                }
            </style>
        </head>
        <body>
            <form action="/predict" enctype="multipart/form-data" method="post">
                <input name="file" type="file">
                <input type="submit">
            </form>
        </body>
    </html>
    """



@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    # Read the uploaded file directly into a DataFrame
    data_test = pd.read_csv(file.file, sep=";")

    # Preprocessing steps
    data_test['poutcome'].replace("nonexistent", np.nan, inplace=True)
    data_test['loan'].replace("unknown", np.nan, inplace=True)
    data_test['housing'].replace("unknown", np.nan, inplace=True)
    data_test['default'].replace("unknown", np.nan, inplace=True)
    data_test['education'].replace("unknown", np.nan, inplace=True)
    data_test['marital'].replace("unknown", np.nan, inplace=True)
    data_test['job'].replace("unknown", np.nan, inplace=True)
    data_test = data_test.drop(['pdays', 'poutcome', 'subscribed'], axis = 1)
    data_test = data_test.dropna()

    # Select categorical columns
    categorical_columns = data_test.select_dtypes(include=['object']).columns
    # Create a label encoder object
    label_encoder = LabelEncoder()

    # Iterate through categorical columns and apply label encoding
    for column in categorical_columns:
        data_test[column] = label_encoder.fit_transform(data_test[column])

    # data_test.head()

    # Make predictions
    predictions = model.predict(data_test)
    data_test['Prediction'] = predictions

    # Create an in-memory buffer
    buffer = io.StringIO()
    data_test.to_csv(buffer, index=False)
    buffer.seek(0)  # Rewind the buffer to the beginning

    # Return the buffer as a response
    return Response(content=buffer.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=predictions.csv"})

@app.get('/')
def basic_view():
    return {"WELCOME": "GO TO /docs route, or /post or send post request to /predict "}
