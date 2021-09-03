import numpy as np
import matplotlib.pyplot as plt
"""
Completar las funciones señaladas con la logica
correspondiente segun conceptos vistos en clase 0. 
No remover ni modifcar la constante X
"""
X = np.arange(10)


def y_values(x, mode):
    """
    Given an array of X values, return the y
    values according to the mode parameter.
    mode can be l for a linear function y = m*x + b
    where b = 2 and m = 3 and q for a quadratic function
    y = x^2
    >>> y_values(X, 'l')
    array([ 2,  5,  8, 11, 14, 17, 20, 23, 26, 29])
    >>> y_values(X, 'q')
    array([ 0,  1,  4,  9, 16, 25, 36, 49, 64, 81])
    """

    arregloRetorno = []

    if mode == 'l':

        for i in x:
            arregloRetorno.append(3*i+2)

    if mode == 'q':

        for i in x:
            arregloRetorno.append(i*i)

    return np.array(arregloRetorno)



def plot_x_y(x,y, mode):
    """
    Given x and y values of the same size, use matplot 'plt'
    to plot a specific chart. This function can be solve in one line
    """
    return plt.plot(x,y,mode)


if __name__ == '__main__':
    #a sample result, try as many as you wany
    Y = y_values(X, 'l')
    plot_x_y(X, Y, 'r')
    plt.show()
