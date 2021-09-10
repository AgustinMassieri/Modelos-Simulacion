import pandas as pd
import matplotlib.pyplot as plt


def create_dataframe(filename):
    """
    Given a filename in csv format
    create a panda dataframe and return it
    """
    df = pd.read_csv(filename)
    return df


def plot_scatter(x, y, x_label, y_label):
    """
    Plot a scatter of (x, y) points
    given as a parameter
    """
    plt.scatter(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def plot_line(x, y):
    """
    Plot a line from (x, y) points
    given as a parameter
    """
    plt.plot(x, y)
    plt.show()


def main():
    # For testing purposes, to analyze the data
    df = create_dataframe('Chwirut1.csv')
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', df.columns.size)
    print(df.head())
    # Program to visualize data
    print('Select a variable to plot against Quality')

    plot_scatter(df[['metal_distance']], df[['ultrasonic_response']], 'metal distance', 'ultrasonic response')


if __name__ == '__main__':
    main()
