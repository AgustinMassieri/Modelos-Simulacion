
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

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('once')

def obtenerDatasetLimpio(nombre):

    dataset = pd.read_csv(nombre, delimiter=',')
    dataset = dataset.drop(['Columna_A_Borrar'], axis = 1) #Limpiamos el dataset eliminando al primer columna que no servia
    dataset = dataset.dropna(axis=0)                       #Limpiamos el dataset eliminando las filas en las que habia campos vacios     

    return dataset

def mostrarArbol(dataset, modelo):
    
    print("Profundidad del árbol: ", modelo.get_depth())
    print("Número de nodos terminales: ", modelo.get_n_leaves())

    plot = plot_tree(
                decision_tree = modelo,
                feature_names = dataset.drop(columns = "Life expectancy").columns,
                class_names   = 'Life expectancy',
                filled        = True,
                impurity      = False,
                fontsize      = 6,
                precision     = 2,
    )

    plt.show()

    plt.savefig('arbolDeDecisión.png')


def getIndicesDataset(dataset): #Al limpiar el dataset lo que hicimos fue "vaciar" por completo las filas en las que faltaba algun campo, pero ...
                                #... no se hace un "desplazamiento hacia arriba" para que quede bien ordenado por lo que, para luego poder iterar bien ...
                                #... vamos a usar esta lista de indices del dataset
    return dataset.index.tolist()


def scores(X, y):

    #Seccion de Prueba para max-depth
    #Por default --> max_depth: int, default=None. If None, then nodes are expanded until all leaves are... 
    #... pure or until all leaves contain less than min_samples_split samples.

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

    dicc = {}
    i = 1
    while (i <= 10):
        
        modelo = DecisionTreeRegressor(max_depth = i, random_state = 123) 
        modelo.fit(X_train, y_train)

        scoreEntrenamiento = modelo.score(X_train, y_train)
        scoreTest = modelo.score(X_test, y_test)

        dicc[i] = [scoreEntrenamiento-scoreTest, scoreEntrenamiento, scoreTest]
        i+=1
         
    return dicc
    

def muestraScores(dicc):

    print("\n> Diccionario con los valores obtenidos para cada vuelta: \n")
    
    print("(N°, [scoreEntrenamiento-scoreTest, scoreEntrenamiento, scoreTest])")

    dicc_items = dicc.items()
    for item in dicc_items:
        print(item)

    print("\n")

def graficosDiccionario(dicc):

    fig, ax = plt.subplots()

    listaDeRestaScores = []
    listaDeScoresEntrenamiento = []
    listaDeScoresTest = []
    listaProfundidades = [1,2,3,4,5,6,7,8,9,10]

    for elemento in dicc:

        listaDeRestaScores.append(dicc[elemento][0])
        listaDeScoresEntrenamiento.append(dicc[elemento][1])
        listaDeScoresTest.append(dicc[elemento][2])
 
    plt.scatter(listaProfundidades, listaDeRestaScores, c="red", label = 'scoreEntrenamiento - scoreTest')
    plt.plot(listaProfundidades, listaDeRestaScores, c="red")
    plt.scatter(listaProfundidades, listaDeScoresEntrenamiento, c="black", label = 'scoreEntrenamiento')
    plt.plot(listaProfundidades, listaDeScoresEntrenamiento, c="black")
    plt.scatter(listaProfundidades, listaDeScoresTest, c="green", label = 'scoreTest')
    plt.plot(listaProfundidades, listaDeScoresTest, c="green")
    plt.xlabel("max_depth")
    ax.grid(axis = 'both', color = 'gray', linestyle = 'dashed')
    plt.title("Scores VS max_depth")
    plt.legend()
    plt.show()

    plt.savefig('graficoDiccionario.png')

    

def main():

    #Cargamos el dataset
    dataset = obtenerDatasetLimpio('Life_Expectancy.csv')

    #Mostramos el dataset
    print("\nDataset original: \n\n",dataset.head())

    X = dataset.drop('Life expectancy', axis = 1)
    y = dataset[['Life expectancy']]

    #Obtenemos un diccionario para cada iteracion de 1 a 10 para el parametro "max_depth"
    dicc = scores(X, y)

    #Visualizamos el diccionario obtenido luego de las 10 iteraciones
    muestraScores(dicc)

    #Graficos diccionarios
    graficosDiccionario(dicc)

    #Vemos que el scoreEntrenamiento sube a medida que aumenta la profundidad y que el scoreTest aumenta hasta max_depth = 5 y luego comienza a disminuir
    #Elegimos max_depth = 5 como optimo

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

    modelo = DecisionTreeRegressor(max_depth = 5, random_state = 123) 
    modelo.fit(X_train, y_train) 

    #Mostramos el arbol
    mostrarArbol(dataset, modelo)


if __name__ == '__main__':
    main()
    