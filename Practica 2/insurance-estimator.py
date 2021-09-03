from data_analysis import *
from data_preparation import *
import statsmodels.api as sm
import operator
import numpy as np



def forward_stepwise_selection(dataframe, target):

    var_explicativas = dataframe.drop(target, axis=1)
    var_explicativas = sm.add_constant(var_explicativas)
    var_objetivo = dataframe[[target]]

    variables = ['const']
    iterate_columns = var_explicativas.columns.drop('const', 1)
    r2_adj = []
    vars_size = iterate_columns.size
    var_model = {}

    for k in range(0, vars_size):

        r2 = {}

        for var in iterate_columns:

            var_explicativa = var_explicativas[variables + [var]]
            model = sm.OLS(var_objetivo, var_explicativa)
            regr = model.fit()
            r2[var] = regr.rsquared_adj

        var_max_r2 = max(r2.items(), key=operator.itemgetter(1))[0]
        var_model[k] = var_max_r2
        r2_adj.append(r2[var_max_r2])
        iterate_columns = iterate_columns.drop(var_max_r2, 1)
        variables.append(var_max_r2)

    r2_max_index = r2_adj.index(max(r2_adj))

    #r2_variation(vars_size, r2_adj, 'Forward Stepwise Selection', 'k', 'R2') #Grafico los R2 ajustados para ver donde se produce el maximo

    #print("\n> R2 Ajustados: ", r2_adj, "\n") #Muestro el arreglo de los R2 ajustado para cada variable

    variables = ['const']
    i = 0
    while i <= r2_max_index:
        variables.append(var_model[i])
        i += 1

    #print("> Variables seleccionadas: ", variables, "\n") #Muestro el arreglo de las variables seleccionadas

    return r2_adj, variables



def backward_stepwise_selection(dataframe, target):

    var_explicativas = dataframe.drop(target, axis=1)
    var_explicativas = sm.add_constant(var_explicativas)
    var_objetivo = dataframe[[target]]

    variables = var_explicativas.columns
    iterate_columns = variables.drop('const', 1)
    vars_size = iterate_columns.size
    r2_adj = []
    var_model = {}

    for k in range(0, vars_size): #Iteramos k veces, siendo k la cantidad de variables

        r2 = {} #Acumulamos los valores de R2 para cada variable en los modelos que armaremos

        i = 1 #Indice para insertar las variables temporalmente removidas en su lugar

        for var in iterate_columns: #Iteramos sobre todas las variables disponibles para la ronda

            variables = variables.drop(var, 1) #Eliminamos la variable para probar

            var_explicativa = var_explicativas[variables] #Fijamos que variables seran las que definiran nuestro modelo

            variables = variables.insert(i, var) #Insertamos nuevamente la variable que sacamos en el lugar que estaba para la prox ronda

            model = sm.OLS(var_objetivo, var_explicativa) #Entrenamos el modelo
            regr = model.fit()

            r2[var] = regr.rsquared_adj #Almacenamos el R2 para cada set de variables que se prueba

            i += 1 #Actualizador de indice


        var_max_r2 = max(r2.items(), key=operator.itemgetter(1))[0] #Seleccionamos aquella con mayor R2

        r2_adj.append(r2[var_max_r2]) #Almacenamos el valor de R2

        var_model[k] = var_max_r2 #Almacenamos las variables que describen a ese modelo

        iterate_columns = iterate_columns.drop(var_max_r2, 1) #No itera mas sobre esa variable tampoco

        variables = variables.drop(var_max_r2, 1) #Dejamos de considerar como una variable posible para definir al modelo

    r2_max = r2_adj.index(max(r2_adj)) #Seteamos el valor de la ronda k donde R2 es maximo

    #r2_variation(vars_size, r2_adj, 'Backward Stepwise Selection', 'k', 'R2') #Grafico los R2 ajustados para ver donde se produce el maximo

    #print("\n> R2 Ajustados: ", r2_adj, "\n") #Muestro el arreglo de los R2 ajustado para cada variable

    variables = var_explicativas.columns
    i = 0
    while i <= r2_max:
        variables = variables.drop(var_model[i], 1)
        i += 1

    variab = []

    for h in range(len(variables)):
        variab.append(variables[h])

    #print("> Variables seleccionadas: ", variab, "\n") #Muestro el arreglo de las variables seleccionadas

    return r2_adj, variab



def backward_stepwise_selection_pvalues(dataframe, target):

    var_explicativas = dataframe.drop(target, axis=1)
    var_explicativas = sm.add_constant(var_explicativas)
    var_objetivo = dataframe[[target]]

    variables = var_explicativas.columns
    iterate_columns = variables.drop('const', 1)
    vars_size = iterate_columns.size
    r2_adj = []
    vars_out = []

    for k in range(0, vars_size):

        var_explicativa = var_explicativas[variables]

        model = sm.OLS(var_objetivo, var_explicativa) #Entrenamos el modelo
        regr = model.fit()

        r2_adj.append(regr.rsquared_adj) #Almacenamos el valor de R2

        p_values = regr.pvalues.drop('const') #Obtenemos los p-values sin considerar la constante
        pvalue_index = p_values.argmax()
        pvalue_var = p_values.keys()[pvalue_index]

        variables = variables.drop(pvalue_var) #Eliminamos la variable cuyo p-value es el mas grande

        vars_out.append(pvalue_var) #Almacenamos la variable que queda fuera en cada ronda

    r2_max = r2_adj.index(max(r2_adj)) #Seteamos el valor de la ronda k donde R2 es maximo

    #r2_variation(vars_size, r2_adj, 'Backward Stepwise Selection with p-values', 'k', 'R2') #Grafico los R2 ajustados para ver donde se produce el maximo

    #print("\n> R2 Ajustados: ", r2_adj, "\n") #Muestro el arreglo de los R2 ajustado para cada variable

    variables = var_explicativas.columns
    i = 0
    while i <= r2_max:
        variables = variables.drop(vars_out[i], 1)
        i += 1

    variab = []

    for h in range(len(variables)):
        variab.append(variables[h])

    #print("> Variables seleccionadas: ", variab, "\n") #Muestro el arreglo de las variables seleccionadas

    return r2_adj, variab



def r2_variation(vars_size, r2_adj, title, x_label, y_label): #Funcion para graficar

    # Grafiquemos la solucion
    # Los valores para señalar cada iteracion
    x = np.arange(vars_size)

    # El punto donde R2 es maximo
    r2_max_x = r2_adj.index(max(r2_adj))
    r2_max_y = max(r2_adj)

    # Titulo de nuestro grafico y nombre de los ejes
    plt.title(title, loc='left')
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Grafiquemos la recta con la variacion del R2 en cada vuelta
    plt.plot(x, r2_adj)

    # Grafiquemos dispersion de puntos de X vs R2 ajustado
    plt.scatter(x, r2_adj)

    # Grafiquemos el punto donde R2 es maximo y añadamos su valor en el grafico
    plt.scatter(r2_max_x, r2_max_y, marker='*', s=100, color='red')
    plt.text(r2_max_x * (1 + 0.01), r2_max_y * (0.97), round(r2_max_y, 3), fontsize=12)
    plt.text(r2_max_x * (1 + 0.01), r2_max_y * (0.95), 'k = ' + str(r2_max_x), fontsize=12)
    plt.show()



def create_model(dataframe, target, mode):

    if (mode == 'FSS'):

        r2_adj = (forward_stepwise_selection(dataframe, target))[0]

        var_model = (forward_stepwise_selection(dataframe, target))[1]

    elif (mode == 'BSS'):

        r2_adj = (backward_stepwise_selection(dataframe, target))[0]

        var_model = (backward_stepwise_selection(dataframe, target))[1]

    elif (mode == 'BSSpvalues'):

        r2_adj = (backward_stepwise_selection_pvalues(dataframe, target))[0]

        var_model = (backward_stepwise_selection_pvalues(dataframe, target))[1]

    var_explicativas = dataframe.drop(target, axis=1)
    var_explicativas = sm.add_constant(var_explicativas)

    var_explicativa = var_explicativas[var_model]

    var_objetivo = dataframe[[target]]

    model = sm.OLS(var_objetivo, var_explicativa)
    regr = model.fit()
    print("\n", regr.summary())

    return regr



def main():

    df = create_dataframe('insurance.csv') #Creo el dataframe

    df = add_dummies(df, ['sex', 'smoker', 'region']) #Hago el cambio de las variables Cualitativas --> Cuantitativas

    print("\n", df.head())

    print("\n> --- Forward Stepwise Selection ---")
    modelFSS = create_model(df, 'charges', 'FSS')
    chargesFSS = modelFSS.predict([[1, 1, 25, 26.3, 0, 1, 0]])

    print("\n> --- Backward Stepwise Selection ---")
    modelBSS = create_model(df, 'charges', 'BSS')
    chargesBSS = modelBSS.predict([[1, 25, 26.3, 0, 1, 0, 0]])

    print("\n> --- Backward Stepwise Selection with p-values ---")
    modelBSSPvalues = create_model(df, 'charges', 'BSSpvalues')
    chargesBSSPvalues = modelBSSPvalues.predict([[1, 25, 26.3, 1, 1, 0, 0, 0]])

    resultado = {"FSS": [int(chargesFSS), modelFSS.rsquared_adj], "BSS": [int(chargesBSS), modelBSS.rsquared_adj], "BSS with p-values": [int(chargesBSSPvalues), modelBSSPvalues.rsquared_adj]}

    print("\n\n> Representacion de los modelos junto con sus resultados: { Modelo: [charges, r2_adj_Modelo] }\n\n -->", resultado)


if __name__ == '__main__':
    main()