import pandas as pd
import matplotlib.pyplot as plt


def create_dataframe(filename):
    """
    Given a filename in csv format
    create a panda dataframe and return it
    """
    data = pd.read_csv(filename)

    return data


def plot_scatter(x, y, x_label, y_label):
    """
    Plot a scatter of (x, y) points
    given as a parameter
    """
    plt.scatter(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def main():
    # For testing purposes, to analyze the data
    df = create_dataframe('50_Startups.csv')
    print(df.head())
    # Program to visualize data
    print('Select a variable to plot against Profit')
    print('R&D Spend, Administration, Marketing Spend')
    x_value = str(input("Variable: "))
    if x_value in ['R&D Spend', 'Administration', 'Marketing Spend']:
        plot_scatter(df[[x_value]], df[['Profit']], x_value, "Profit")
    else:
        print("Not a valid variable. Restart the program and retry")


if __name__ == '__main__':
    main()