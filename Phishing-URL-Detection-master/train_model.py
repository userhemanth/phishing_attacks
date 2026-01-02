
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import os

# Load dataset
data = pd.read_csv('phishing.csv')

# Drop Index column if it exists, actually based on header 'Index' it seems it does.
# Features are all columns except 'Index' and 'class'
# Target is 'class'

# It seems the first column is Index.
X = data.drop(['Index', 'class'], axis=1)
y = data['class']

# Verify shape
print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# Train model
# app.py uses gbc, let's assume default parameters or similar to what's typical.
gbc = GradientBoostingClassifier()
gbc.fit(X, y)

print("Model trained.")

# Ensure pickle directory exists
if not os.path.exists('pickle'):
    os.mkdir('pickle')

# Save model
with open('pickle/model.pkl', 'wb') as f:
    pickle.dump(gbc, f)

print("Model saved to pickle/model.pkl")
