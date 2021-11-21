from flask import Flask, render_template, request
import jsonify
import requests
import joblib
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from heart import sc
from heart import heart_pred

app = Flask(__name__)


model = joblib.load(r'heart_model.joblib')
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    
    if request.method == 'POST':
        
        data = list(request.form.to_dict().values())
        if(data[1] == 'female'):
            data[1] = "0"
        else:
            data[1] = "1"
        
        if(data[2] == 'typical'):
            data[2] = "1"
        elif data[2] == "atypical":
            data[2] = "2"
        elif data[2] == "nonagigal":
            data[2] =="3"
        else:
            data[2] = "4"
        
        input_data = [float(x.strip()) for x in data]
        print(input_data)

        prediction=heart_pred(input_data)
        output=prediction[0]
        
        if output==0:
            return render_template('result.html',prediction_text="You don't have a Heart Disease!")
        else:
            return render_template('result.html',prediction_text="You may have a Heart Disease!")
    else:
        return render_template('result.html',prediction_text="Enter Correct DATA")


if __name__=="__main__":
    app.run(debug=True)
