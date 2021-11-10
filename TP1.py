from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KernelDensity
from pandas import read_csv
from sklearn import metrics

test = read_csv("./TP1_test.tsv",sep="\t")
train = read_csv("./TP1_train.tsv",sep="\t")

nb = GaussianNB()

print(test.iloc[:,-1:])
f = nb.fit(train.iloc[: , :4],train.iloc[:,-1:].values.ravel())

pred = nb.predict(test.iloc[:,:4])
met = metrics.accuracy_score(pred,test.iloc[:,-1])
print(met)
