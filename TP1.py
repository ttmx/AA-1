from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KernelDensity
from numpy import loadtxt
import numpy as np
from sklearn import metrics

def standardize(df):
    return (df-df.mean())/df.std()

def load_data(file_name):
    data = loadtxt(file_name,delimiter="\t")
    data[:,:-1] = standardize(data[:,:-1])
    np.random.shuffle(data)
    return data


test = load_data("./TP1_test.tsv")
train = load_data("./TP1_train.tsv")

nb = GaussianNB()
print(test)
f = nb.fit(train[: , :4],train[:,-1])

pred = nb.predict(test[:,:4])
met = metrics.accuracy_score(pred,test[:,-1])
print(met)
