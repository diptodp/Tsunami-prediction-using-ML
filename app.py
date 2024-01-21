

from flask import Flask, render_template, request
import pickle
import numpy as np
from joblib import dump, load
from sklearn.preprocessing import StandardScaler

# Load the model using joblib
loaded_model = load('random_forest.joblib')
scaler = load('scaler.joblib')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def index():
     if request.method == "POST":
        data1 = request.form['significance']
        data2 = request.form['magnitudo']
        data3 = request.form['latitude']
        data4 = request.form['longitude']
        data5 = request.form['depth']

        data1 = float(data1)
        data2 = float(data2)
        data3 = float(data3)
        data4 = float(data4)
        data5 = float(data5)

         

        # Convert data to a numpy array
        input_values = np.array([[data1, data2, data3, data4, data5]])

        # Scale the input data using the loaded scaler
        scaled_values = scaler.transform(input_values)

        # Use the loaded model for prediction
        prediction = loaded_model.predict(scaled_values)
        output = prediction[0]

    
        if output == 0:
            prediction_text = '<div style="color: #02A064;background-color: #E2E2E3;text-align: center; padding: 8px; border-radius: 5px;">You are safe! This earthquake will not cause a Tsunami :)</div>'
        else:
            prediction_text = '<div style="color: #fd0404; text-align: center;background-color: #E2E2E3;padding: 8px;border-radius: 5px;">This earthquake will cause a Tsunami :(</div>'

        return render_template('index.html', 
                               prediction_text=prediction_text, 
                               significance=data1, 
                               magnitude=data2, 
                               latitude=data3, 
                               longitude=data4, 
                               depth=data5)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
