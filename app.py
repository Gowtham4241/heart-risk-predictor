from flask import Flask, request, render_template
import pickle
import numpy as np
import os
import requests
from requests.exceptions import RequestException

app = Flask(__name__)

def download_model_from_env():
    """If MODEL_URL is set in env, try to download model.pkl to working dir."""
    model_url = os.getenv('MODEL_URL')
    if not model_url:
        return False, 'MODEL_URL not set'

    try:
        resp = requests.get(model_url, timeout=10)
        resp.raise_for_status()
        with open('model.pkl', 'wb') as f:
            f.write(resp.content)
        return True, 'Downloaded'
    except RequestException as e:
        return False, str(e)


# Load model (try local file first, then optional download)
model = None
if os.path.exists('model.pkl'):
    try:
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
            print('✅ Loaded model.pkl from local file')
    except Exception as e:
        print(f'❌ Failed to load local model.pkl: {e}')

if model is None:
    ok, msg = download_model_from_env()
    if ok:
        try:
            with open('model.pkl', 'rb') as file:
                model = pickle.load(file)
                print('✅ Loaded model.pkl after download')
        except Exception as e:
            print(f'❌ Downloaded model but failed to load: {e}')
    else:
        print(f'⚠️ Model not available locally and download skipped/failed: {msg}')

@app.route('/')
def home():
    return  render_template('index.html')
    #return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        Age = float(request.form['age'])
        cigsPerDay = float(request.form['cigsPerDay'])
        Diabetes = float(request.form['diabetes'])
        TotalCholastrol = float(request.form['totChol'])
        DiaBP = float(request.form['DiaBP'])
        BMI = float(request.form['BMI'])
        glucose = float(request.form['glucose'])

        # Create feature array
        features = np.array([[Age, cigsPerDay, Diabetes, TotalCholastrol, DiaBP, BMI, glucose]])

        # Check if model is loaded
        if model is None:
            return render_template('index.html', prediction_text="Model not loaded. Please check 'model.pkl'.")

        # Make prediction
        prediction = model.predict(features)[0]
        result = "RISK ✅" if prediction == 1 else "NO RISK ❌"

        return render_template('index.html', prediction_text=f'RISK of coronary heart disease: {result}')

    except ValueError as e:
        return render_template('index.html', prediction_text=f'ValueError: {str(e)}')
    except KeyError as e:
        return render_template('index.html', prediction_text=f'KeyError: {str(e)}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
