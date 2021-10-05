#############################
# Pto 1: MODELOS DE REGRESIÓN
#############################

# En el mercado del real estate, el precio de los inmuebles es una variable crucial a la
# hora de realizar una operación. Conocer el valor de mercado de una propiedad es clave
# tanto para comprarla y venderla como así también para alquilarla. Este datasetr
# permite tasar un inmueble para conocer los valores en los que se mueve el mercado
# o averiguar el precio promedio en la zona en la que estás buscando mudarte.


# Cargar dataset "propiedades.csv" con función read.csv

# a) Ver estructura del dataset

# b) Seleccionar propiedades cuyo precio sea mayor a U$S50.000 y menor a U$S800.000

# c) Entrenar un modelo de regresión lineal simple para predecir la variable precio
# para cada una las variables predictoras numéricas

# d) Indicar una medida de  performance predictiva para cada modelo utilizando validación cruzada 5K-CV,
# presentando un dataset con los resultados de cada fold para cada modelo.


import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

def regresionLineal(data_frame_var_explicativa, data_frame_var_objetivo):

    regr = linear_model.LinearRegression()
    modelo = regr.fit(data_frame_var_explicativa, data_frame_var_objetivo)

    return modelo


def main():

#a)
    data_frame = pd.read_csv('propiedades.csv') #Cargo el dataset

    print("\n> Muestro el dataset original: \n", data_frame.head(), "\n") #Muestro el dataset

    #Como el dataset contiene variables Categoricas voy a transformarlas en dummy

    #Convierto a dummy la variable barrio
    barrio = pd.factorize(data_frame['barrio'])
    print("\n> Conversion de variable barrio: ", barrio, "\n")
    index_barrio = data_frame.columns.get_loc('barrio')
    data_frame = data_frame.drop('barrio', axis=1)
    data_frame.insert(index_barrio, 'barrio', barrio[0])


    #Convierto a dummy la variable tipo_prop
    tipo_prop = pd.factorize(data_frame['tipo_prop'])
    print("\n> Conversion de variable tipo_prop: ", tipo_prop, "\n")
    index_tipo_prop = data_frame.columns.get_loc('tipo_prop')
    data_frame = data_frame.drop('tipo_prop', axis=1)
    data_frame.insert(index_tipo_prop, 'tipo_prop', tipo_prop[0])


    print("\n> Muestro el dataset modificado: \n", data_frame.head(), "\n") #Muestro el dataset ya con el cambio de variables realizado
    data_frame.to_csv('propiedades_nuevo.csv', index=False) #Guardo el nuevo dataset en un csv para poder visualizar los cambios


    #Nuestra VAR_OBJETIVO sera el "PRECIO" a lo largo del ejercicio
    var_objetivo = 'precio'

#b)
    #Tengo que seleccionar propiedades cuyo precio sea mayor a U$S50.000 y menor a U$S800.000
    data_frame_nuevo = data_frame[ (data_frame['precio'] > 50000) & (data_frame['precio'] < 800000) ]

    print("\n> Muestro el dataset filtrado por la condicion de precio:\n ", data_frame.head(), "\n") #Muestro el dataset filtrado por la condicion de precio
    data_frame_nuevo.to_csv('propiedades_nuevo_filtrado.csv', index=False) #Guardo el nuevo dataset en un csv para poder visualizar los cambios

#c)
    listaDeVariablesNumericas = ['cant_amb', 'cant_banos', 'sup_total', 'sup_cub']
    listaConModelosEntrenados = []

    for variable in listaDeVariablesNumericas:

        aux = regresionLineal(data_frame_nuevo[[variable]], data_frame_nuevo[[var_objetivo]])
        listaConModelosEntrenados.append(aux)

#d)
    mse = {}

    #MODELO 1
    X = data_frame_nuevo[['cant_amb']]
    y = data_frame_nuevo[[var_objetivo]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_1 = linear_model.LinearRegression().fit(X_train, y_train)
    scores = cross_val_score(model_1, X, y, cv=5, scoring='neg_mean_squared_error')
    mse['model1'] = abs(scores.mean())

    #MODELO 2
    X = data_frame_nuevo[['cant_banos']]
    y = data_frame_nuevo[[var_objetivo]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_2 = linear_model.LinearRegression().fit(X_train, y_train)
    scores = cross_val_score(model_2, X, y, cv=5, scoring='neg_mean_squared_error')
    mse['model2'] = abs(scores.mean())

    #MODELO 3
    X = data_frame_nuevo[['sup_total']]
    y = data_frame_nuevo[[var_objetivo]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_3 = linear_model.LinearRegression().fit(X_train, y_train)
    scores = cross_val_score(model_3, X, y, cv=5, scoring='neg_mean_squared_error')
    mse['model3'] = abs(scores.mean())

    #MODELO 4
    X = data_frame_nuevo[['sup_cub']]
    y = data_frame_nuevo[[var_objetivo]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_4 = linear_model.LinearRegression().fit(X_train, y_train)
    scores = cross_val_score(model_4, X, y, cv=5, scoring='neg_mean_squared_error')
    mse['model4'] = abs(scores.mean())

    print("\n> MSE: ", mse, "\n")

if __name__ == '__main__':
    main()