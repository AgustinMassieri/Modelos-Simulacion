from typing import Counter
import pandas as pd
from sklearn import tree


def bootstrap_sample(df, n, p):
    """
    Create n bootstrap samples which size will be
    p  times the original dataframe df. p between 0 and 1.
    Return an array with the bootstrap samples.
    """
    bootstrap_samples = []

    for iteration in range(n):
        bootstrap_samples.append(df.sample(frac=p,replace=True))

    return bootstrap_samples


def bagging_predict(b_sample, x, target):
    """
    Train as many decision trees as b_samples exists.
    Store its prediction for an observation x and then with
    voting function return either 1 or 0
    """
    predictions = []

    for sample in b_sample:
        Y = sample[[target]]
        X = sample.drop(Y, axis=1)
        aDecisionTree = tree.DecisionTreeClassifier().fit(X,Y)
        predictions += list(aDecisionTree.predict(x))

    return voting(predictions)


def voting(lst):
    """
    Given a list with 0s an 1s return
    the number with higher prevalence
    """
    counter = Counter(lst)

    if counter[0] > counter[1]:
        return 0
    else: 
        return 1

def bagging_score(b_sample, df, target, p = 0.2):
    """
    Return the accuracy of a given bagging model
    """

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

def count_hits(values, df, target):
    """
    Given a dictionary with predictions for certain indexes of
    the original dataset, apply voting function and return the count of the ones
    that are the same to real data.
    """

    hits = 0

    for index in values.keys():
        votingResult = voting(values[index])

        if votingResult == df.loc[index][target]:
            hits += 1

    return hits


def main():
    x = [[500, 0, 1, 40, 1, 5000, 1, 1, 0, 100000]]
    df = pd.read_csv('churn_cleaned.csv')
    # create 100 boostrap samples with 50% original size each one
    b_sample = bootstrap_sample(df, 100, 0.5)
    print(f'Quantity of bootstrap samples: {len(b_sample)}')
    pred = bagging_predict(b_sample, x, 'Exited')
    print(
        "CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, isActiveMember, EstimatedSalary")
    print(x[0])
    if pred == 1:
        print("We predict an EXIT of the client")
    else:
        print("We predict client will REMAIN in the bank")

    print("Accuracy: ", bagging_score(b_sample, df, 'Exited'))

if __name__ == '__main__':
    main()