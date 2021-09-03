import pandas as pd

from data_analysis import *


def set_dummy_variable(dataframe, column):

    dummy = pd.get_dummies(dataframe[column], prefix=column) #Formo mi nueva variable dummy

    if (len(dummy.columns) == 2):

        dummy = dummy.iloc[:, 1:]

    n_col = dataframe.columns.get_loc(column) #Averiguo el numero de columna donde se encuentra la variable Cualitativa a modificar

    for i in range(len(dummy.columns)): #Por cada columna de la variable dummy...

        dataframe.insert(n_col + i, dummy.columns[i], dummy[dummy.columns[i]], True) #Inserto la nueva columna ya modificada en el dataframe

    del dataframe[column] #Elimino la columna Cualitativa original del dataframe

    #print(dataframe.head(), "\n") #Para verificar el funcionamiento

    return dataframe #Devuelvo el dataframe modificado


def add_dummies(dataframe, columns):
    """
    Add dummy variables for every column given
    return the dataframe
    """

    for var in columns: #Por cada variable Cualitativa que esta en la lista, se la paso como parametro a la funcion junto con el dataframe para pasarla a Cuantitativa

        set_dummy_variable(dataframe, var) #Uso la funcion creada para hacer el cambio

    return dataframe #Devuelvo el dataframe modificado

def main():

    df = create_dataframe('insurance.csv') #Creo el dataframe

    df = add_dummies(df, ['sex', 'smoker', 'region']) #Hago el cambio de las variables Cualitativas --> Cuantitativas

    df.to_csv('insurance-ready.csv', index=False) #Vuelvo los datos en un archivo .csv

if __name__ == '__main__':
    main()
