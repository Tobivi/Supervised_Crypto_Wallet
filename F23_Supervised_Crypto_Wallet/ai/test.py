import numpy as np
import pandas as pd


def attribute_matrix(X):
    a2 = X[:, 3]**2 # Age
    a3 = X[:, 4]**2 # Num Transactions
    a4 = X[:, 2]**2 # Balance
    a5 = X[:, 5]**2 # Knowledge Index

    return np.column_stack((a2, a3, a4, a5))


if __name__ == "__main__":
    DATAPATH = './ai/data'

    df = pd.read_csv(DATAPATH + "/user-data.csv", delimiter=",")
    df_out = pd.read_csv(DATAPATH + "/not-normalized.csv", delimiter=",")

    D = df_out.to_numpy()

    X = attribute_matrix(D)

    max = np.max(X, axis=0)
    min = np.min(X, axis=0)
    print(max, min)