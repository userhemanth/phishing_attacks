import sys
import os
import pickle
import numpy as np

# Import feature extraction from feature.py
from feature import FeatureExtraction

def test_prediction(url):
    print(f"Testing URL: {url}")
    
    # Load the model
    model_path = os.path.join("pickle", "model.pkl")
    try:
        with open(model_path, "rb") as file:
            gbc = pickle.load(file)
    except FileNotFoundError:
        print(f"Error: Model not found at {model_path}")
        return

    print("Model loaded successfully.")
    
    try:
        # Extract features
        print("Extracting features...")
        obj = FeatureExtraction(url)
        features = obj.getFeaturesList()
        x = np.array(features).reshape(1,30)
        
        # Predict
        y_pred = gbc.predict(x)[0]
        y_pro_phishing = gbc.predict_proba(x)[0,0]
        y_pro_non_phishing = gbc.predict_proba(x)[0,1]
        
        print("-" * 30)
        print(f"Prediction: {'Safe' if y_pred == 1 else 'Phishing'}")
        print(f"Probability Safe: {y_pro_non_phishing*100:.2f}%")
        print(f"Probability Phishing: {y_pro_phishing*100:.2f}%")
        print("-" * 30)
        
    except Exception as e:
        print(f"Prediction failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_prediction(sys.argv[1])
    else:
        test_prediction("https://www.google.com")
