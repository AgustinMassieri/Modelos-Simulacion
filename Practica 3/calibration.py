from data_analysis import *
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore")

"""
Reference:	Chwirut, D., NIST (1979). 
Ultrasonic Reference Block Study.
"""


def train_model(df, dep_variables, target, mode, degree=1):

    if mode == 'p':

        polynomial_features = PolynomialFeatures(degree=degree)
        xp = polynomial_features.fit_transform(dep_variables)

        model = sm.OLS(target, xp)
        regr = model.fit()

        x = np.arange(start=0.4, stop=6, step=0.1)
        x = x[:, None]
        xNuevo = polynomial_features.fit_transform(x)

        y = regr.predict(xNuevo)

        plt.scatter(dep_variables, target)
        plt.plot(x, y, color='red')
        plt.title('Ajuste polinomico de grado: ' + str(degree))
        plt.text(4.5, 40, 'R2 adj: ' + str(round(regr.rsquared_adj, 3)))
        plt.show()

    elif mode == 'l':

        x_log = np.log10(dep_variables)
        y_log = np.log10(target)

        x_log = sm.add_constant(x_log)
        model = sm.OLS(y_log, x_log)
        regr = model.fit()

        a = 10 ** regr.params[0]
        b = regr.params[1]

        x = np.arange(start=0.4, stop=6, step=0.1)
        x = x[:, None]

        y = a * np.power(x, b)

        plt.scatter(dep_variables, target)
        plt.plot(x, y, color='red')
        plt.title('Ajuste Logaritmico ')
        plt.text(4.5, 40, 'R2 adj: ' + str(round(regr.rsquared_adj, 3)))
        plt.show()

    return regr



def is_calibrated(x, y, regr, mode, degree=1):

    if ( mode == 'p'):

        prediccion = regr.predict(PolynomialFeatures(degree=degree).fit_transform([[x]]))

        print("> Usando el ajuste polinomico: \n")

    elif ( mode == 'l' ):

        beta_0 = regr.params[0]
        beta_1 = regr.params[1]

        a = np.power(10, beta_0)

        prediccion = a * np.power(x, beta_1)

        print("\n> Usando el ajuste logaritmico: ")

    if ( (prediccion < (y * 1.05)) and (prediccion > (y * 0.95)) ):
        print("Esta calibrado! Valor predecido: ", prediccion, " - Valor 'y': ", y)
    else:
        print("No esta calibrado! Valor predecido: ", prediccion, " - Valor 'y': ", y)



def main():

    df = create_dataframe('Chwirut1.csv')


    print("\n> Modelo Ajuste Polinomico Grado 2\n")
    modeloPolinomicoGrad2 = train_model(df, df[['metal_distance']], df[['ultrasonic_response']], 'p', 2)
    print(modeloPolinomicoGrad2.summary())


    print("\n\n> Modelo Ajuste Polinomico Grado 3\n")
    modeloPolinomicoGrad3 = train_model(df, df[['metal_distance']], df[['ultrasonic_response']], 'p', 3)
    print(modeloPolinomicoGrad3.summary())

    print("\n\n> Modelo Ajuste Logaritmico\n")
    modeloLogaritmico = train_model(df, df[['metal_distance']], df[['ultrasonic_response']], 'l')
    print(modeloLogaritmico.summary())

    print("\n\n")

    i = 1
    while i:
        x = float(input('Metal distance: '))

        y = float(input('Ultrasonic Value: '))

        is_calibrated(x, y, modeloLogaritmico, 'l')

        i = int(input('\nContinue? (1 YES - 0 NO): '))


if __name__ == '__main__':
    main()