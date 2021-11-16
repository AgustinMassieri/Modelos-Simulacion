
#|--------------------------------------------|
#|   TRABAJO PRACTICO: MODELOS Y SIMULACIÓN   |
#| PONTIFICIA UNIVERSIDAD CATÓLICA ARGENTINA  |
#|                  2021                      |
#|--------------------------------------------|
#| ALUMNOS:                                   |
#|              MASSIERI AGUSTÍN              |
#|              JOAQUÍN VELAZQUEZ             |
#|--------------------------------------------|
#| PROFESORES:                                |
#|              CARLOS ARANA                  |
#|              TOMÁS NOZICA                  |
#|--------------------------------------------|

from parte1 import *
from parte2 import *
from typing import Counter
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


def bootstrap_sample(df, n, p = 0.5): #Create n bootstrap samples which size will be p times the original dataframe df. 
                                      #p between 0 and 1. Return an array with the bootstrap samples.

    bootstrap_samples = []

    for iteration in range(n):
        bootstrap_samples.append(df.sample(frac=p,replace=True))

    return bootstrap_samples
    
        
def bagging_predict(b_sample, x, target): #Train as many decision trees as b_samples exists.
                                          #Store its prediction for an observation x and then with voting function return either 1 or 0

    predictions = []

    for sample in b_sample:
        
        X = sample.drop(target, axis = 1)
        y = sample[[target]]
        
        aDecisionTree = tree.DecisionTreeClassifier().fit(X.values,y)

        predictions += list(aDecisionTree.predict(x))

    return voting(predictions)


def voting(lst): #Given a list with 0s an 1s return the number with higher prevalence

    counter = Counter(lst)

    if counter[0] > counter[1]:
        return 0
    else: 
        return 1


def bagging_score(b_sample, df, target, p = 0.2): #Return the accuracy of a given bagging model

    values = {}

    for sample in b_sample:
        outOfBagSamples = sample.sample(frac=p, replace=False)

        train = sample.drop(outOfBagSamples.index)

        Y = train[[target]]
        X = train.drop(Y, axis=1)
        aDecisionTree = tree.DecisionTreeClassifier().fit(X,Y)

        Y_TEST = outOfBagSamples[[target]]
        X_TEST = outOfBagSamples.drop(Y_TEST, axis = 1)

        predictions = aDecisionTree.predict(X_TEST)

        for predictionIndex,index in enumerate(list(outOfBagSamples.index)):
            
            if index not in values:
                values[index] = []
            values[index].append(predictions[predictionIndex])

    return count_hits(values, df, target) / len(values.keys())


def count_hits(values, df, target): #Given a dictionary with predictions for certain indexes of the original dataset, ...
                                    #... apply voting function and return the count of the ones that are the same to real data.
    hits = 0

    for index in values.keys():
        votingResult = voting(values[index])

        if votingResult == df.loc[index][target]:
            hits += 1

    return hits


def obtenerListaDeAccuracy(dataset): #Obtenemos el accuracy para cada modelo mientras aumentamos de a 10 el N° de muestras en cada vuelta

    i = 10

    listaDeAccuracy = []

    while(i <= 100 ):

        b_sample = bootstrap_sample(dataset, i, 0.5)

        listaDeAccuracy.append(bagging_score(b_sample, dataset, 'Life expectancy'))

        i += 10
    
    return listaDeAccuracy


def graficarListaAccuracy(listaDeAccuracy): #Vemos graficamente como varia el Accuracy a medida que aumentamos de a 10 el N° de muestras

    listaCantidadDeMuestras = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    fig, ax = plt.subplots()

    plt.scatter(listaCantidadDeMuestras, listaDeAccuracy, c="red")
    plt.plot(listaCantidadDeMuestras, listaDeAccuracy, c="red")
    plt.xlabel("N° de muestras")
    plt.ylabel("Accuracy")
    plt.title("Accuracy VS N° de muestras")
    ax.grid(axis = 'both', color = 'gray', linestyle = 'dashed')
    plt.show()

    plt.savefig('graficoListaAccuracy.png')


def auxiliar(b_sample, target, p = 0.2):

    values = {}

    for sample in b_sample:
        outOfBagSamples = sample.sample(frac=p, replace=False)

        train = sample.drop(outOfBagSamples.index)

        Y = train[[target]]
        X = train.drop(Y, axis=1)
        aDecisionTree = tree.DecisionTreeClassifier().fit(X,Y)

        Y_TEST = outOfBagSamples[[target]]
        X_TEST = outOfBagSamples.drop(Y_TEST, axis = 1)

        predictions = aDecisionTree.predict(X_TEST)

        for predictionIndex,index in enumerate(list(outOfBagSamples.index)):
            
            if index not in values:
                values[index] = []
            values[index].append(predictions[predictionIndex])
    
    return values


def votacionUmbral(dicc, umbral = 0.5):

    i = 0
    diccReturn = {}

    for clave in dicc:
        
        cantidadDeUnos = Counter(dicc[clave])[1]
        cantidadTotal = len(dicc[clave])
        probabilidadUno = cantidadDeUnos / cantidadTotal

        if (probabilidadUno >= umbral):
            diccReturn[clave] = 1
        else:
            diccReturn[clave] = 0
        i+=1

    return diccReturn


def votacionRangoCompletoDeUmbrales(dicc):

    print("\n")

    for i in np.arange(0, 1, 0.1):

        print("\nUmbral ", round(i, 2), ":\n", sorted(votacionUmbral(dicc, i).items()))
    

def matrizDeConfusionUmbrales(dataset, target, dicc):

    print("\n")

    eje_x = []
    eje_y = []
    eje_z = []

    for i in np.arange(0, 1, 0.1):

        diccionario = votacionUmbral(dicc, i)
        print("-----------------------\nUMRBAL ", round(i, 2), ": \n")
        
        tn, fp, fn, tp = confusion_matrix(dataset[target], list(diccionario.values())).ravel()
        print("\t> TN = ", tn)
        print("\t> FP = ", fp)
        print("\t> FN = ", fn)
        print("\t> TP = ", tp)

        eje_x.append(tp)
        eje_y.append(fp)
        eje_z.append(i)
        
    plt.bar(eje_x, eje_z, color = "red", label = "TP")
    plt.bar(eje_y, eje_z, color = "blue", label = "FP")
    plt.ylabel('Umbrales')
    plt.xlabel('Valores para TP y FP')
    plt.title('Umbrales vs TP - FP')
    plt.xlim(12, 55)
    plt.legend()
    plt.show()

    plt.savefig('UmbralesContraTP-FP.png')

        
#AdaBoost
def adaBoost(dataset, target):

    X = dataset.drop(target, axis = 1)
    y = dataset[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    clf = AdaBoostClassifier(n_estimators=50,learning_rate=1).fit(X_train, y_train)

    print("\n> Score sobre set de prueba - Ada Boost: ", clf.score(X_test, y_test))

    y_predicciones = clf.predict(X_test)

    print("\n> Confusion Matrix: ")

    tn, fp, fn, tp = confusion_matrix(y_test, y_predicciones).ravel()

    print("\t> TN = ", tn)
    print("\t> FP = ", fp)
    print("\t> FN = ", fn)
    print("\t> TP = ", tp)

    print("\n> Clasificadores debiles: \n\n", clf.estimators_)
     
    sub_tree = clf.estimators_[2]
    plot_tree(sub_tree)
    plt.show()

    plt.savefig('adaBoostTree.png')


#GradientBoosting
def gradientBoosting(dataset, target):

    X = dataset.drop(target, axis = 1)
    y = dataset[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    clf = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, y_train)
    
    print("\n> Score sobre set de prueba - Gradient Boosting: ", clf.score(X_test, y_test))

    y_predicciones = clf.predict(X_test)

    print("\n> Confusion Matrix: ")

    tn, fp, fn, tp = confusion_matrix(y_test, y_predicciones).ravel()

    print("\t> TN = ", tn)
    print("\t> FP = ", fp)
    print("\t> FN = ", fn)
    print("\t> TP = ", tp)

    print("\n> Clasificadores debiles: \n\n", clf.estimators_)

    sub_tree = clf.estimators_[2,0]
    plot_tree(sub_tree)
    plt.show()

    plt.savefig('gradientBoostingTree.png')


def main():

    print("\n ---------------- TP FINAL - MODELOS Y SIMULACION - MASSIERI, VELAZQUEZ ----------------\n")

    dataset = obtenerDatasetLimpio('Life_Expectancy.csv')
    print(dataset.head())

    transformarDataset(dataset)

    print("\n > Muestro el dataset transformado: \n", dataset.head())

    b_sample = bootstrap_sample(dataset, 100, 0.5)
    
    print("\n> Quantity of bootstrap samples: ", len(b_sample))

    print("\n> Accuracy (100 muestras): ", bagging_score(b_sample, dataset, 'Life expectancy'), "\n")

    listaDeAccuracy = obtenerListaDeAccuracy(dataset)
    graficarListaAccuracy(listaDeAccuracy)

    diccionarioPredicciones = auxiliar(b_sample,  'Life expectancy')
    print("\n> Diccionario Predicciones Ordenado: \n", sorted(diccionarioPredicciones.items()))

    votingDiccionarioPredicciones = votacionUmbral(diccionarioPredicciones, 0.5)
    print("\n> Diccionario Predicciones Voting Ordenado Para Umbral de 0.5: \n", sorted(votingDiccionarioPredicciones.items()))

    votacionRangoCompletoDeUmbrales(diccionarioPredicciones)
    
    matrizDeConfusionUmbrales(dataset, "Life expectancy", diccionarioPredicciones)

    print("\n\n---------------------- ADA BOOST ----------------------\n")
    adaBoost(dataset, 'Life expectancy')

    print("\n------------------- GRADIENT BOOSTING -------------------\n")
    gradientBoosting(dataset, 'Life expectancy')

    print("\n\n---> ANTE UNA SITUACION DE OVERFITTING EN NUESTRO MODELO (SCORE > 0.92) DEBERIAMOS PENSAR EN AUMENTAR O DISNMINUIR LOS HIPERPARAMETROS (LEARNING_RATE - MAX_DEPTH) PARA INTENTAR CONSEGUIR UN SCORE NI MUY BAJO NI TAN ALTO QUE NOS CAUSE OVERFITTING!\n")

if __name__ == '__main__':
    main()       
