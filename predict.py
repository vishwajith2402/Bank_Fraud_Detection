import joblib
import numpy as np

model = joblib.load("model.pkl")

def predict_fraud(features):

    features = np.array(features).reshape(1, -1)

    result = model.predict(features)

    probability = model.predict_proba(features)

    return result[0], probability[0][1]