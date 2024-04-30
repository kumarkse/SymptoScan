import pickle
import numpy as np
import pandas as pd

Xtrainall = pd.read_csv("Training.csv")
disease = Xtrainall.iloc[:,-1]
disease = np.unique(disease)

labels = Xtrainall.columns[:-1]


def do_prediction(selected_symptoms):
    if (len(selected_symptoms)==0):
        return ["Please make atleast one selection"]
    with open("modelllist.pkl", "rb") as f:
        models = pickle.load(f)

    pred_list = np.array([0] * len(labels))

    for symptom in selected_symptoms:
        if symptom != "None":
            pred_list[np.where(labels==symptom)[0]] = 1

    pred_list = pred_list.reshape(1,-1)
    results = []

    for model in models:
        try:
            res = model.predict(pred_list) 
            results.append(res)
        except Exception as e:
            results.append(f"Error: {e}")

    with open("encoder.pkl","rb") as f:
        encoder= pickle.load(f)

    results = encoder.inverse_transform(results)

    return np.unique(results)
