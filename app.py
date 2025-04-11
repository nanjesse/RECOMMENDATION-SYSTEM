import os.path
import logging
from flask import Flask, request, render_template
import numpy as np
import pandas
import sklearn
import pickle



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_requests.log"),
        logging.StreamHandler() 
    ]
)

# ========== Load Model & Scalers ==========
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

# ========== Flask App ==========
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    # Extract form values
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']

    # Build feature list
    feature_list = [N, P, K, temp, humidity, ph, rainfall]

    # Log request
    logging.info(f"Received Input: {feature_list} from IP: {request.remote_addr}")

    # Convert inputs to floats and reshape
    single_pred = np.array(feature_list, dtype=float).reshape(1, -1)

    # Transform features
    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)

    # Make prediction
    prediction = model.predict(final_features)

    # Crop mapping
    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
        8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
        14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
        19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
    }

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = f"{crop} is the best crop to be cultivated right there"
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

    # Log response
    logging.info(f"Prediction Output: {result}")

    return render_template('index.html', result=result)


if __name__ == "__main__":
    import sys
    print(sys.executable)  
    app.run(debug=True)

