#################################
# Pto 2: MODELOS DE CLASIFICACION
#################################
# El dataset mark_banco.csv registra datos reales de campañas de telemarketing para vender depósitos
# a plazo fijo llevadas a cabo por banco minorista portugués.
# Dentro de una campaña, los operadores ofrecen los productos y/o servicios de dos maneras y
# en momentos diferentes: realizando llamadas telefónicas (saliente) o
# aprovechando la llamada al call center de un cliente por cualquier otra razón (entrante).
# Por lo tanto, el resultado es un contacto de resultado binario: éxito/positivo (compra el producto)
# o fracaso/negativo (no lo compra).
# Los datos corresponden a contactos con clientes realizados mayo de 2008 hasta junio de 2013 y
# totalizan  35.010 registros. Cada registro incluye el resultado del contacto en la cariable "y"
# y diferentes variables predictoras o atributos. Estos incluyen atributos de telemarketing
# ,detalles del producto (por ejemplo tasa de interés ofrecida) e información del cliente
# (por ejemplo edad). Estos registros se enriquecieron con datos de
# índole de socio-económica (por ejemplo la tasa de desempleo).
#

# a) Ver estructura del dataset
#
# b) Se pide:
#   i) crear un conjunto de entrenamiento y otro de prueba con el dataset

#   ii) Entrenar un modelo de Regresion Logistica utilizando todas las variables sobre el
#      conjunto de entrenamiento.

#   iii) Para umbrales entre 0.1 y 0.9 (con pasos de a 0.1), calcular la exactitud, precision y
#       recall/sensitividad sobre el conjunto de prueba.

#   iv) Graficar la exactitud con respecto al umbral para el rango entre 0.1 y 0.9 con los valores
#     obtenidos. ¿Como se puede explicar el comportamiento de la curva en base a la presencia de FN y FP?


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def dummy_maker(data_frame, nombre):
    variable = pd.factorize(data_frame[nombre])
    print('\n> Conversion de variable "',nombre,'": ', variable, '\n')
    index_variable = data_frame.columns.get_loc(nombre)
    data_frame = data_frame.drop(nombre, axis=1)
    data_frame.insert(index_variable, nombre, variable[0])
    return data_frame

def prediccion(rlog, X_test, i):

    probabilidadPrediccion = rlog.predict_proba(X_test)

    predictions = []

    for elemento in probabilidadPrediccion:

        if elemento[1] >= i:
            predictions.append('yes')
        else:
            predictions.append('no')

    return predictions

def metricaParaMatrizDeConfusion(matrizDeConfusion):

    VN, FP, FN, VP = matrizDeConfusion.ravel()

    metrics = {}

    metrics['accuracy'] = (VP + VN) / (VP + FN + VN + FN)
    metrics['precision'] = VP / (VP + FP)
    metrics['sensitivity'] = VP / (VP + FN)

    return metrics

def main():

#a)

    data_frame = pd.read_csv('mark_banco.csv', delimiter=';') #Cargo el dataset

    print("\n> Muestro el dataset original: \n", data_frame.head(), "\n") #Muestro el dataset

    lista_index = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'day_of_week']

    for n in range(len(lista_index)):
        data_frame = dummy_maker(data_frame, lista_index[n])

    print("\n> Muestro el dataset modificado: \n", data_frame.head(), "\n")  # Muestro el dataset ya con el cambio de variables realizado
    data_frame.to_csv('mark_banco_nuevo.csv', index=False)  # Guardo el nuevo dataset en un csv para poder visualizar los cambios

#b)
#i)

    target_variable = 'y'
    y = data_frame[[target_variable]]
    X = data_frame.drop(target_variable, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#ii)

    rlog = LogisticRegression().fit(X_train, y_train.values.ravel())

#iii)

    listaParaUmbrales = []
    i = 0.1

    while (i <= 0.9):

        listaParaUmbrales.append(prediccion(rlog, X_test, i))
        i = i + 0.1

    #Agrego a una lista, las matrices de confusion generadas para cada threshold
    listaParaMatrices = []


    print("\nMatrices para cada umbral:")

    for elemento in listaParaUmbrales:

        listaParaMatrices.append(confusion_matrix(y_test.values.ravel(), elemento))
        print("\n", confusion_matrix(y_test.values.ravel(), elemento))


    listaDeMetricasParaMatrices =  []

    print("\nMetricas para cada umbral:")

    for elemento in listaParaMatrices:

        listaDeMetricasParaMatrices.append(metricaParaMatrizDeConfusion(elemento))
        print("\n", metricaParaMatrizDeConfusion(elemento))

if __name__ == '__main__':
    main()