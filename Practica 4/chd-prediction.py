import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from data_analysis import *


def train(dataframe, target):
    """
    Given a datframe and a target variable
    return a logistic regression model
    """

    x = dataframe.drop(target, axis=1);
    y = dataframe[target]
    rlog = LogisticRegression().fit(x, y)

    return rlog

def predict(x, model, threshold=0.5):
    """
    Using the proability that our model return for each
    possibility (0 or 1), create the proper predictions according
    the given threshold. Return the array with the results.
    """

    predicciones = model.predict_proba(x)
    predicciones_umbral = (predicciones[:, 0] >= threshold).astype(int)

    return predicciones_umbral


def report(x, predictions, target):
    """
    Return a CSV file that has the input data X and the proper
    prediction.
    """

    x[target] = predictions
    x.to_csv('resultados.csv', index=False)


def main():

    df = create_dataframe('framingham.csv')
    df_pacientes = create_dataframe('pacientes.csv')

    regr = train(df, 'TenYearCHD')

    prediccion = predict(df_pacientes, regr, 0.75)

    listaPrediccion = []
    for h in range(len(prediccion)):
        listaPrediccion.append(prediccion[h])

    report(df_pacientes, listaPrediccion, 'TenYearCHD')

if __name__ == '__main__':
    main()