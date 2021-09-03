"""
Completar las siguientes funciones definidas de modo que cumplan
con la logica solicitada. El proposito de este ejercicio es recordar
elementos de Python (req. Informatica General). Se asignan tests previamente
definidos (no es necesario agregar nuevos casos).
"""

def list_has_even_size(lst):
    """
    Given a list return True if its size
    is an even number, otherwise False

    >>> list_has_even_size([1, 2, 3])
    False
    >>> list_has_even_size(['a', 'b', 'c', 'd'])
    True
    >>> list_has_even_size([])
    True
    """
    if len(lst) % 2 == 0:
        return True
    else:
        return False


def sum_of_elements(lst):
    """
    Given a list with integer numbers
    return the sum of all the elements.
    >>> sum_of_elements([1, 25, 45, 30])
    101
    >>> sum_of_elements([5, 3, -1])
    7
    >>> sum_of_elements([])
    0
    >>> sum_of_elements([-5, -2])
    -7
    """
    sumador = 0

    for i in lst:

        sumador += i

    return sumador


def remove_elements(array):
    """
    Given a 2d array, return an array
    that contains only the rows that have not
    None as an element
    >>> remove_elements([[1, 2], [3, 4]])
    [[1, 2], [3, 4]]
    >>> remove_elements([[1, None], [2, 3]])
    [[2, 3]]
    >>> remove_elements([[1, None], [1, None]])
    []
    """
    lstRetornar = []

    for sublista in array:
        if (sublista[0] != None) and (sublista[1] != None):
            lstRetornar.append(sublista)

    return lstRetornar

def replace_value(array):
    """
    Given a 2d array, replace every 'x'/'X' character
    with an 'o'/'O' character.
    >>> replace_value([['a', 'x'], ['o', 'b']])
    [['a', 'o'], ['o', 'b']]
    >>> replace_value([['a', 'b'], ['c', 'd']])
    [['a', 'b'], ['c', 'd']]
    >>> replace_value([['X', 'x'], ['xx', 'XX']])
    [['O', 'o'], ['xx', 'XX']]
    """

    for sublista in range(len(array)):
        for elemento in range(len(array[sublista])):
            if array[sublista][elemento] == 'x':
                array[sublista][elemento] = 'o'
            if array[sublista][elemento] == 'X':
                array[sublista][elemento] = 'O'
    return array