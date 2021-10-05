import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import numpy as np
#############################
# Pto 1: MODELOS DE REGRESI�N
#############################

# En el mercado del real estate, el precio de los inmuebles es una variable crucial a la 
# hora de realizar una operaci�n. Conocer el valor de mercado de una propiedad es clave 
# tanto para comprarla y venderla como as� tambi�n para alquilarla. Este datasetr 
# permite tasar un inmueble para conocer los valores en los que se mueve el mercado 
# o averiguar el precio promedio en la zona en la que est�s buscando mudarte.


# Cargar dataset "propiedades.csv" con funci�n read.csv

# a) Ver estructura del dataset

# b) Seleccionar propiedades cuyo precio sea mayor a U$S50.000 y menor a U$S800.000

# c) Entrenar un modelo de regresi�n lineal simple para predecir la variable precio 
# para cada una las variables predictoras num�ricas

# d) Indicar una medida de  performance predictiva para cada modelo utilizando validaci�n cruzada 5K-CV, 
# presentando un dataset con los resultados de cada fold para cada modelo. 

print("MODELOS DE REGRESIÓN")
# a)
dataset = pd.read_csv("propiedades.csv")
print(dataset)

# b)
datasetAfterFilter = dataset[ (dataset['precio'] > 50000) & (dataset['precio'] < 800000 )]
print(datasetAfterFilter)

# c)

#Viendo el dataset, las numéricas son cant_amb, cant_banos, sup_total, sup_cub, precio (precio no la tenemos enc cuenta
#por que justamente es la variable objetivo)
numeric_columns = ['cant_amb', 'cant_banos', 'sup_total', 'sup_cub']
target_variable = 'precio'

variableWithModelDictionary = {}

#Imprimimos el R2 y el MSE
for numeric_column in numeric_columns:
    variableWithModelDictionary[numeric_column] = linear_model.LinearRegression().fit(datasetAfterFilter[[numeric_column]], datasetAfterFilter[[target_variable]])
    model = variableWithModelDictionary[numeric_column]
    true_output = datasetAfterFilter[[target_variable]]
    predicted_output = variableWithModelDictionary[numeric_column].predict(datasetAfterFilter[[numeric_column]])
    
    print("El R2 con '%s' nos queda %s y el MSE queda %s" % (numeric_column, r2_score(true_output, predicted_output), mean_squared_error(true_output, predicted_output)))

# d)

X_train, X_test, Y_train, Y_test = train_test_split(datasetAfterFilter[numeric_columns], datasetAfterFilter[[target_variable]], test_size=0.2)
print("Tamaño del set de entrenamiento: ", len(X_train))
print("Tamaño del set de testeo: ", len(X_test))

variableWithCrossValidationModel = {}

for numeric_column in numeric_columns:
    variableWithCrossValidationModel[numeric_column] = linear_model.LinearRegression().fit(X_train[[numeric_column]], Y_train)
    scores = cross_val_score(linear_model.LinearRegression(), X_train[[numeric_column]], Y_train[[target_variable]], cv=5, scoring='neg_mean_squared_error')
    print("Mean of Scores with '%s': %s" % (numeric_column, abs(scores.mean())))


#################################
# Pto 1: MODELOS DE CLASIFICACION
#################################
# El dataset mark_banco.csv registra datos reales de campa�as de telemarketing para vender dep�sitos 
# a plazo fijo llevadas a cabo por banco minorista portugu�s. 
# Dentro de una campa�a, los operadores ofrecen los productos y/o servicios de dos maneras y 
# en momentos diferentes: realizando llamadas telef�nicas (saliente) o 
# aprovechando la llamada al call center de un cliente por cualquier otra raz�n (entrante). 
# Por lo tanto, el resultado es un contacto de resultado binario: �xito/positivo (compra el producto) 
# o fracaso/negativo (no lo compra).
# Los datos corresponden a contactos con clientes realizados mayo de 2008 hasta junio de 2013 y 
# totalizan  35.010 registros. Cada registro incluye el resultado del contacto en la cariable "y"
# y diferentes variables predictoras o atributos. Estos incluyen atributos de telemarketing 
# ,detalles del producto (por ejemplo tasa de inter�s ofrecida) e informaci�n del cliente 
# (por ejemplo edad). Estos registros se enriquecieron con datos de 
# �ndole de socio-econ�mica (por ejemplo la tasa de desempleo). 
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
#     obtenidos. �Como se puede explicar el comportamiento de la curva en base a la presencia de FN y FP?

def set_dummy_variable(dataframe, column, label):
    new_columns = pd.get_dummies(column, prefix=label)
    column_location = dataframe.columns.get_loc(label)
    dataframe = dataframe.drop([label], axis=1)
    for index in range(len(new_columns.columns)):
        dataframe.insert(column_location + index, new_columns.columns[index], new_columns[[new_columns.columns[index]]])
    return dataframe


def add_dummies(dataframe, columns):
    for column_label in columns:
        dataframe = set_dummy_variable(dataframe, dataframe[[column_label]], column_label)
    return dataframe




print("MODELOS DE REGRESIÓN")
# a)
dataset = pd.read_csv("mark_banco.csv", delimiter=';')
print(dataset)

#Procesamos las variables dummies:
dataset = add_dummies(dataset, ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'day_of_week'])
dataset = dataset.drop(labels = ['job_unemployed', 'marital_single', 'education_illiterate', 'default_no', 'housing_no', 'loan_no', 'contact_telephone', 'month_dec', 'day_of_week_fri'], axis=1)
print(dataset)


def predictWithThreshold(model, dataset, threshold):
    predictionsWithProbability = model.predict_proba(dataset)
    predictions = []

    for predictionWithProbability in predictionsWithProbability:
        if predictionWithProbability[1] >= threshold:
            predictions.append('yes')
        else:
            predictions.append('no')

    return predictions

def buildConfusionMatrix(realOutput, predictions):
    confusionMatrix = {
        'VP': 0,
        'FP': 0,
        'VN': 0,
        'FN': 0
    }

    for index in range(len(realOutput)):
        real = realOutput[index]
        prediction = predictions[index]

        if real == 'no' and prediction == 'no':
            confusionMatrix['VN'] += 1
        elif real == 'yes' and prediction == 'yes':
            confusionMatrix['VP'] += 1
        elif real == 'yes' and prediction == 'no':
            confusionMatrix['FN'] += 1
        elif real == 'no' and prediction == 'yes':
            confusionMatrix['FP'] += 1
           
    return confusionMatrix



# b)
# b)i)
target_variable = 'y'
Y = dataset[[target_variable]]
X = dataset.drop(target_variable, 1)

print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
print("Tamaño del set de entrenamiento: ", len(X_train))
print("Tamaño del set de testeo: ", len(X_test))

# b)ii)

logisticRegressionModel = linear_model.LogisticRegression(solver='lbfgs', max_iter=10000).fit(X_train, Y_train.values.ravel())


# b)iii)

def calculateMetricsOnConfusionMatrix(confusionMatrix):
    metrics = {}
    VP = confusionMatrix['VP']
    FP = confusionMatrix['FP']
    VN = confusionMatrix['VN']
    FN = confusionMatrix['FN']
    
    metrics['accuracy'] = (VP+VN)/(VP+FN+VN+FN)
    metrics['precision'] = (VP)/(VP+FP)
    metrics['sensitivity'] = (VP)/(VP+FN)

    return metrics


predictionsWithLowThreshold = predictWithThreshold(logisticRegressionModel, X_test, 0.1)
predictionsWithHighThreshold = predictWithThreshold(logisticRegressionModel, X_test, 0.9)

lowThresholdConfusionMatrix = buildConfusionMatrix(Y_test.values.ravel(), predictionsWithLowThreshold)
highThresholdConfusionMatrix = buildConfusionMatrix(Y_test.values.ravel(), predictionsWithHighThreshold)

print(calculateMetricsOnConfusionMatrix(lowThresholdConfusionMatrix))
print(calculateMetricsOnConfusionMatrix(highThresholdConfusionMatrix))

# b)iiii)

threshold_axis = []
precision_axis = []
accuracy_axis = []
sensitivity_axis = []


for threshold in list(np.arange(0.1,1,0.1)):
    threshold_axis.append(threshold)
    metrics = calculateMetricsOnConfusionMatrix(buildConfusionMatrix(Y_test.values.ravel(), predictWithThreshold(logisticRegressionModel, X_test, threshold)))
    precision_axis.append(metrics['precision'])
    accuracy_axis.append(metrics['accuracy'])
    sensitivity_axis.append(metrics['sensitivity'])

fig, ax = plt.subplots()

ax.plot(threshold_axis, precision_axis, label = "Precision")
ax.plot(threshold_axis, accuracy_axis, label = "Accuracy")
ax.plot(threshold_axis, sensitivity_axis, label = "Sensitivity")
ax.axis("equal")
leg = ax.legend()
plt.show()