import os
from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    print("✅ Model loaded successfully.")
    print(f"📦 Model type: {type(model).__name__}")
    
    # Check model expectations
    if hasattr(model, 'n_features_in_'):
        print(f"📊 Model expects {model.n_features_in_} features")
    if hasattr(model, 'feature_names_in_'):
        print(f"🏷️ Feature names: {list(model.feature_names_in_)}")
    if hasattr(model, 'classes_'):
        print(f"🎯 Model classes: {model.classes_} (0=LOW risk, 1=HIGH risk)")
    
except FileNotFoundError:
    print("❌ Model file not found. Please ensure 'model.pkl' is in the root directory.")
    model = None
except Exception as e:
    print(f"❌ An error occurred while loading the model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        Age = float(request.form['age'])
        cigsPerDay = float(request.form['cigsPerDay'])
        Diabetes = float(request.form['diabetes'])
        TotalCholastrol = float(request.form['totChol'])
        DiaBP = float(request.form['DiaBP'])
        BMI = float(request.form['BMI'])
        glucose = float(request.form['glucose'])
        
        # Create feature array
        features = np.array([[Age, cigsPerDay, Diabetes, TotalCholastrol, DiaBP, BMI, glucose]])
        
        print(f"\n🔍 INPUT FEATURES:")
        print(f"   Age: {Age}")
        print(f"   Cigarettes/day: {cigsPerDay}")
        print(f"   Diabetes: {Diabetes}")
        print(f"   Total Cholesterol: {TotalCholastrol}")
        print(f"   Diastolic BP: {DiaBP}")
        print(f"   BMI: {BMI}")
        print(f"   Glucose: {glucose}")
        print(f"   Feature array shape: {features.shape}")
        
        if model is None:
            return render_template('index.html', prediction_text="Model not loaded. Deployment error.")
        
        # Check feature count mismatch
        if hasattr(model, 'n_features_in_'):
            if features.shape[1] != model.n_features_in_:
                return render_template(
                    'index.html', 
                    prediction_text=f"ERROR: Model expects {model.n_features_in_} features but got {features.shape[1]}. Please check training data."
                )
        
        # Make prediction
        prediction = model.predict(features)[0]
        print(f"🎯 Raw prediction: {prediction}")
        
        # Try to get prediction probability if available
        proba = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            print(f"📊 Prediction probabilities: LOW risk: {proba[0]:.3f}, HIGH risk: {proba[1]:.3f}")
        
        # Test with multiple random inputs to check if model always predicts same
        test_random = np.random.rand(5, 7) * 100
        random_preds = model.predict(test_random)
        print(f"🧪 Random test predictions: {random_preds}")
        
        if len(set(random_preds)) == 1:
            print("⚠️ WARNING: Model predicts the SAME class for all random inputs!")
        
        # Determine result
        if prediction == 1:
            result = "HIGH RISK ⚠️"
            message = "Based on the provided data, the model predicts a high risk of coronary heart disease."
            if proba:
                message += f" (Confidence: {proba[1]*100:.1f}%)"
        else:
            result = "LOW RISK ✅"
            message = "Based on the provided data, the model predicts a low risk of coronary heart disease."
            if proba:
                message += f" (Confidence: {proba[0]*100:.1f}%)"
        
        # Add warning if model seems broken
        if proba and (proba[1] > 0.99 or proba[0] > 0.99):
            message += " ⚠️ NOTE: Model shows very high confidence (>99%) - may need calibration."
        
        return render_template(
            'index.html', 
            prediction_text=f'Predicted Result: {result}',
            detail_message=message
        )
        
    except ValueError as e:
        return render_template('index.html', prediction_text=f'Error: Please ensure all fields are filled correctly with numbers. Details: {str(e)}')
    except KeyError as e:
        return render_template('index.html', prediction_text=f'Error: Missing form field. Please check if "{e.args[0]}" exists in index.html.')
    except Exception as e:
        return render_template('index.html', prediction_text=f'An unexpected error occurred: {str(e)}')

@app.route('/debug-model')
def debug_model():
    """Debug endpoint to analyze model behavior"""
    if model is None:
        return "Model not loaded"
    
    import numpy as np
    import json
    
    debug_info = {
        "model_type": str(type(model).__name__),
        "model_loaded": True,
    }
    
    # Get model expectations
    if hasattr(model, 'n_features_in_'):
        debug_info["expected_features"] = model.n_features_in_
    if hasattr(model, 'feature_names_in_'):
        debug_info["feature_names"] = list(model.feature_names_in_)
    if hasattr(model, 'classes_'):
        debug_info["classes"] = model.classes_.tolist()
    
    # Test with different input ranges
    test_results = []
    
    # Test case 1: Very low risk
    low_risk_input = np.array([[25, 0, 0, 150, 70, 20, 80]])
    pred1 = model.predict(low_risk_input)[0]
    prob1 = model.predict_proba(low_risk_input)[0].tolist() if hasattr(model, 'predict_proba') else None
    
    # Test case 2: Very high risk
    high_risk_input = np.array([[70, 30, 1, 300, 100, 40, 200]])
    pred2 = model.predict(high_risk_input)[0]
    prob2 = model.predict_proba(high_risk_input)[0].tolist() if hasattr(model, 'predict_proba') else None
    
    # Test with random inputs
    random_inputs = np.random.rand(10, 7) * 100
    random_preds = model.predict(random_inputs).tolist()
    
    debug_info["test_cases"] = {
        "low_risk_input (25,0,0,150,70,20,80)": {
            "prediction": int(pred1),
            "probabilities": prob1
        },
        "high_risk_input (70,30,1,300,100,40,200)": {
            "prediction": int(pred2),
            "probabilities": prob2
        },
        "random_predictions": random_preds,
        "all_predictions_same": len(set(random_preds)) == 1,
        "unique_predictions": list(set(random_preds))
    }
    
    # Check if model might need scaling
    debug_info["possible_issues"] = []
    if len(set(random_preds)) == 1:
        debug_info["possible_issues"].append("Model predicts SAME class for ALL random inputs - likely needs feature scaling")
    if hasattr(model, 'predict_proba') and prob1 and (prob1[1] > 0.99 or prob1[0] > 0.99):
        debug_info["possible_issues"].append("Model shows extreme confidence (>99%) - possible overfitting or scaling issue")
    
    return debug_info

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
