import numpy as np

def predict_rul(model, X):
    predictions = model.predict(X)
    return predictions