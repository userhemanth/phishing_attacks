import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')

def train():
    print("Loading data...")
    data = pd.read_csv("phishing.csv")
    
    # Preprocessing
    if 'Index' in data.columns:
        data = data.drop(['Index'], axis=1)
        
    X = data.drop(["class"], axis=1)
    y = data["class"]
    
    print("Training GradientBoostingClassifier...")
    # Using parameters found in the notebook or defaults if not specified
    # Default parameters are usually safe: learning_rate=0.1, n_estimators=100, max_depth=3
    gbc = GradientBoostingClassifier(random_state=42) 
    gbc.fit(X, y)
    
    print("Saving model to pickle/model.pkl...")
    import os
    if not os.path.exists("pickle"):
        os.makedirs("pickle")
        
    with open("pickle/model.pkl", "wb") as f:
        pickle.dump(gbc, f)
        
    print("Done!")

if __name__ == "__main__":
    train()
