
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


def transformarDataset(dataset): #Vamos a clasificar con 1 si 'Life expectancy' >=72 o sino con 0 

    columnaParaModificar = dataset['Life expectancy']

    listaDeIndices = getIndicesDataset(dataset)


    for posicion in listaDeIndices:

        if (columnaParaModificar[posicion] >= 72):

            dataset.loc[posicion,['Life expectancy']] = 1

        else:

            dataset.loc[posicion,['Life expectancy']] = 0

def graficosUtiles(dataset):
  
    plt.scatter(list(dataset[' BMI ']), list(dataset['Life expectancy']), c="red")
    plt.xlabel(' BMI ')
    plt.ylabel('Life expectancy')
    plt.title("BMI vs Life Expectancy")
    plt.grid()
    plt.show()
    plt.savefig('graficoUtil_1.png')


    plt.scatter(list(dataset['Adult Mortality']), list(dataset['Life expectancy']), c="red")
    plt.xlabel('Adult Mortality')
    plt.ylabel('Life expectancy')
    plt.title("Adult Mortality vs Life Expectancy")
    plt.grid()
    plt.show()
    plt.savefig('graficoUtil_2.png')     

    plt.scatter(list(dataset['Hepatitis B']), list(dataset['Life expectancy']), c="red")
    plt.xlabel('Hepatitis B')
    plt.ylabel('Life expectancy')
    plt.title("Hepatitis B vs Life Expectancy")
    plt.grid()
    plt.show()
    plt.savefig('graficoUtil_3.png') 


def graficosUtilesSinTransformar(dataset):

    plt.scatter(list(dataset[' BMI ']), list(dataset['Life expectancy']), c="red")
    plt.xlabel(' BMI ')
    plt.ylabel('Life expectancy')
    plt.title("BMI vs Life Expectancy")
    plt.grid()
    plt.show()
    plt.savefig('graficoUtilTransformado_1.png')


    plt.scatter(list(dataset['Adult Mortality']), list(dataset['Life expectancy']), c="red")
    plt.xlabel('Adult Mortality')
    plt.ylabel('Life expectancy')
    plt.title("Adult Mortality vs Life Expectancy")
    plt.grid()
    plt.show()
    plt.savefig('graficoUtilTransformado_2.png')     

    plt.scatter(list(dataset['Hepatitis B']), list(dataset['Life expectancy']), c="red")
    plt.xlabel('Hepatitis B')
    plt.ylabel('Life expectancy')
    plt.title("Hepatitis B vs Life Expectancy")
    plt.grid()
    plt.show()
    plt.savefig('graficoUtilTransformado_3.png')


def main():

    dataset = obtenerDatasetLimpio('Life_Expectancy.csv')
    print(dataset.head())

    graficosUtilesSinTransformar(dataset)

    transformarDataset(dataset)

    print("\n > Muestro el dataset transformado: \n", dataset.head())

    graficosUtiles(dataset)


if __name__ == '__main__':
    main()    