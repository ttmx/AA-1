import numpy as np
from KernelDensityNB import KernelDensityNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold


def standardize(df):
    return (df - df.mean()) / df.std()


def load_data(file_name):
    data = np.loadtxt(file_name, delimiter="\t")
    data[:, :-1] = standardize(data[:, :-1])
    np.random.shuffle(data)
    return data


def nb_classifier(nb):
    avg_error = 0
    skf = StratifiedKFold(n_splits=folds)
    for train_i, validation_i in skf.split(train[:, :-1], train[:, -1]):
        train_set = train[train_i]
        validation_set = train[validation_i]
        nb.fit(train_set[:, :-1], train_set[:, -1])
        error = 1 - nb.score(validation_set[:, :-1], validation_set[:, -1])
        avg_error += error
    avg_error /= folds
    return avg_error


if __name__ == '__main__':
    test = load_data("./TP1_test.tsv")
    train = load_data("./TP1_train.tsv")

    folds = 5

    min_error = 1
    best_bandwidth = -1
    for bandwidth in np.linspace(0.02, 6, 300):
        error = nb_classifier(KernelDensityNB(bandwidth))
        if error < min_error:
            min_error = error
            best_bandwidth = bandwidth

    print(f"Kernel Density validation error: {min_error} with bandwidth: {best_bandwidth}")

    print(f"Gaussian validation error: {nb_classifier(GaussianNB())}")

    clf = SVC(kernel='rbf')
