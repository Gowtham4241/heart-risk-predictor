import os # <-- ADDED: To read the dynamic port assigned by Render
from flask import Flask, request, render_template
import pickle
import numpy as np

# Removed the unused "import requests" that caused the error

app = Flask(__name__)

# Load model
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print("❌ Model file not found. Please ensure 'model.pkl' is in the root directory.")
    model = None
except Exception as e:
    print(f"❌ An error occurred while loading the model: {e}")
    model = None

@app.route('/')
def home():
    # Flask looks for 'index.html' inside the 'templates' folder
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get data from form (input names must match 'index.html')
        # Ensure all data is converted to float for the model
        Age = float(request.form['age'])
        cigsPerDay = float(request.form['cigsPerDay'])
        Diabetes = float(request.form['diabetes'])
        TotalCholastrol = float(request.form['totChol'])
        DiaBP = float(request.form['DiaBP'])
        BMI = float(request.form['BMI'])
        glucose = float(request.form['glucose'])

        # 2. Create feature array for the model
        features = np.array([[Age, cigsPerDay, Diabetes, TotalCholastrol, DiaBP, BMI, glucose]])

        # 3. Check if model is loaded
        if model is None:
            return render_template('index.html', prediction_text="Model not loaded. Deployment error.")

        # 4. Make prediction
        prediction = model.predict(features)[0]
        
        # Determine the result text
        if prediction == 1:
            result = "HIGH RISK ✅"
            message = "Based on the provided data, the model predicts a high risk of coronary heart disease."
        else:
            result = "LOW RISK ❌"
            message = "Based on the provided data, the model predicts a low risk of coronary heart disease."

        return render_template(
            'index.html', 
            prediction_text=f'Predicted Result: {result}',
            detail_message=message
        )

    except ValueError:
        return render_template('index.html', prediction_text='Error: Please ensure all fields are filled correctly with numbers.')
    except KeyError as e:
        return render_template('index.html', prediction_text=f'Error: Missing form field. Please check if "{e.args[0]}" exists in index.html.')
    except Exception as e:
        # Catch any other unexpected errors
        return render_template('index.html', prediction_text=f'An unexpected error occurred: {str(e)}')

if __name__ == '__main__':
    # Use the port provided by the environment (Render), or default to 5000 for local development
    # This is CRITICAL for deployment platforms like Render.
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
