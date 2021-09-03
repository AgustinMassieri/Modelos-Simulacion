import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from data_analysis import create_dataframe, plot_scatter

PARAMS = ['R&D Spend', 'Administration', 'Marketing Spend', 'State', 'Profit']
# Complete with the Independent variables that you will use

#MIRANDO LOS GRAFICOS GENERADOS CON EL ARCHIVO ANTERIOR Y TENIENDO EN CUENTA QUE STATE ES 'CATEGORICA' POR ENDE YA NO LA TENDRIA EN CUENTA
#DECIDI USAR COMO VARIABLES INDEPENDIENTES A 'R&D Spend' Y 'Marketing Spend'

IND_VAR = ['R&D Spend', 'Marketing Spend']
DEP_VAR = ['Profit']


def train_model(dataframe):
    """
    Train a linear regression model with
    its dependent and indepent variables
    from a specific dataframe
    """

    regr = linear_model.LinearRegression()

    var_explicativa = dataframe[IND_VAR]
    var_objetivo = dataframe[DEP_VAR]

    regr.fit(var_explicativa, var_objetivo)

    return regr

def betas(regr_model):
    """
    Return beta values given a
    specific model
    """
    return (regr_model.coef_[0][0], regr_model.coef_[0][1], regr_model.intercept_[0])


def mse(regr_model, dataframe):
    """
    Calculate the Sum of Square Errors
    and then divide them by the amount of instances
    Do not use mean_squared_errors.
    """
    error = 0

    profit = np.transpose(dataframe[DEP_VAR])

    for i in range(0, profit.size):
        error += np.power(profit[i] - expected_profit(regr_model, dataframe[IND_VAR])[i], 2)

    error = error / profit.size

    return error

def expected_profit(regr_model, datos):
    """"
    Predict the profit of a certain startup given the
    variables that you use in your model
    """
    prediccion_profit = regr_model.predict(datos)

    return prediccion_profit

def main():
    # Main function, this will be useful to test your program
    df = create_dataframe('50_Startups.csv')
    model = train_model(df)
    print(df.describe())
    print("\nBetas: ", betas(model))
    print("\nMSE: ", mse(model, df)[0])

    new_data = pd.DataFrame({'R&D Spend': 40000, 'Marketing Spend': 129300}, index = [0])

    print("\nExpected Profit: ", expected_profit(model, new_data))

    print("\nMSE (Formula): ", mean_squared_error(df[DEP_VAR], expected_profit(model, df[IND_VAR])))
    print("\nR2: ", r2_score(df[DEP_VAR], expected_profit(model, df[IND_VAR])))

if __name__ == '__main__':
    main()
