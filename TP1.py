from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KernelDensity
from numpy import loadtxt
from sklearn import metrics

def standardize(df):
    return (df-df.mean())/df.std()

test = loadtxt("./TP1_test.tsv",delimiter="\t")
test[:,:-1] = standardize(test[:,:-1])
train = loadtxt("./TP1_train.tsv",delimiter="\t")
train[:,:-1] = standardize(train[:,:-1])

nb = GaussianNB()

print(test)
f = nb.fit(train[: , :4],train[:,-1:].ravel())

pred = nb.predict(test[:,:4])
met = metrics.accuracy_score(pred,test[:,-1])
print(met)
