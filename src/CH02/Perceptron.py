#
#       2020.09.28 
#       Rosenblatt's Perceptron
#       Heesung Yang
#
#       env : Local
#       
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame

from numpy.random import multivariate_normal # To multivariate Gaussian Distribution

#
# Parameters
N1 = 20 # Number of Dataset 1
Mu1 = [15, 10] 

N2 = 30 # Number of Dataset 2
Mu2 = [0, 0]

Variances = [15, 30]

def prepare_dataset(variance):
    """
    Variance를 받아서 i.d.d인 covariance matrix를 만들고 
    DataFrame 구성
    """
    cov1 = np.array([[variance, 0],
                     [0, variance]]) # i.d.d 
    cov2 = np.array([[variance, 0],
                     [0, variance]]) # i.d.d

    df1 = DataFrame(multivariate_normal(Mu1, cov1, N1), columns = ['x', 'y'])
    df1['type'] = 1
    df2 = DataFrame(multivariate_normal(Mu2, cov2, N2), columns = ['x', 'y'])
    df2['type'] = -1

    df = pd.concat([df1, df2], ignore_index=True)
    df = df.reindex(np.random.permutation(df.index)).reset_index(drop=True)

    return df

# Perceptron
def run_simulation(variance, data_graph, param_graph):
    train_set = prepare_dataset(variance)
    train_set1 = train_set[train_set['type'] == 1]
    train_set2 = train_set[train_set['type'] == -1]

    ymin, ymax = train_set.y.min()-5, train_set.y.max()+10
    xmin, xmax = train_set.x.min()-5, train_set.x.max()+10

    data_graph.set_ylim([ymin-1, ymax+1])
    data_graph.set_xlim([xmin-1, xmax+1])

    data_graph.scatter(train_set1.x, train_set1.y, marker='o') # +1 : o
    data_graph.scatter(train_set2.x, train_set2.y, marker='x') # -1 : x

    # Init Parameter
    w0 = w1 = w2 = 0.0
    bias = 0.5 * (train_set.x.mean() + train_set.y.mean()) # set to mean value

    # Iteration
    paramhist = DataFrame([[w0, w1, w2]], columns=['w0', 'w1', 'w2'])
    for i in range(30):
        for index, point in train_set.iterrows():
            x, y, type = point.x, point.y, point.type
            if type * (w0*bias + w1*x + w2*y) <= 0:
                w0 += type * 1
                w1 += type * x
                w2 += type * y
        paramhist = paramhist.append(Series([w0, w1, w2], ['w0', 'w1', 'w2']), ignore_index=True)

    err = 0
    for index, point in train_set.iterrows():
        x, y, type = point.x, point.y, point.type
        if type * (w0*bias + w1*x + w2*y) <= 0:
            err += 1
    err_rate = err * 100 / len(train_set)

    linex = np.arange(xmin-5, xmax+5)
    liney = - linex * w1 / w2 - bias * w0 / w2
    label = "ERR %.2f%%" % err_rate
    data_graph.plot(linex, liney, label=label, color='red')
    data_graph.legend(loc=1)
    paramhist.plot(ax=param_graph)
    param_graph.legend(loc=1)

# Main
if __name__ == "__main__":
    fig = plt.figure()

    for c, variance in enumerate(Variances):
        subplots1 = fig.add_subplot(2, 2, c+1)
        subplots2 = fig.add_subplot(2, 2, c+2+1)
        run_simulation(variance, subplots1, subplots2)

    fig.show()
    plt.savefig('perceptron_results.png')