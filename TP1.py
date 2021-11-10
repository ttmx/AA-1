import math
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


def cross_validation(nb):
    avg_error = 0
    skf = StratifiedKFold(n_splits=folds)
    for train_i, validation_i in skf.split(train[:, :-1], train[:, -1]):
        train_set = train[train_i]
        validation_set = train[validation_i]
        avg_error += fit_and_score(nb, train_set, validation_set)
    avg_error /= folds
    return avg_error


def fit_and_score(nb, fit, score):
    nb.fit(fit[:, :-1], fit[:, -1])
    error = 1 - nb.score(score[:, :-1], score[:, -1])
    return error


def approximate_normal_test(test_error):
    sigma = math.sqrt(test.size * test_error * (1 - test_error))
    absolute_margin = 1.96 * sigma  # 0.95 confidence interval
    relative_margin = absolute_margin / test.size
    return test_error, relative_margin


if __name__ == '__main__':
    test = load_data("./TP1_test.tsv")
    train = load_data("./TP1_train.tsv")

    folds = 5

    min_error = 1
    best_bandwidth = -1
    for bandwidth in np.linspace(0.02, 6, 300):
        error = cross_validation(KernelDensityNB(bandwidth))
        if error < min_error:
            min_error = error
            best_bandwidth = bandwidth

    print(f"Kernel Density validation error: {min_error} with bandwidth: {best_bandwidth}")

    print(f"Gaussian validation error: {cross_validation(GaussianNB())}")

    min_error = 1
    best_gamma = -1
    for gamma in np.linspace(0.2, 6, 30):
        error = cross_validation(SVC(gamma=gamma, kernel="rbf"))
        if error < min_error:
            min_error = error
            best_gamma = gamma

    print(f"SVM validation error: {min_error} with gamma: {best_gamma}")

    K_error, K_margin = approximate_normal_test(fit_and_score(KernelDensityNB(best_bandwidth), train, test))
    print(f"Kernel Density test error: {K_error}% +- {K_margin}% with bandwidth: {best_bandwidth}")

    G_error, G_margin = approximate_normal_test(fit_and_score(GaussianNB(), train, test))
    print(f"Gaussian test error: {G_error}% +- {G_margin}%")

    SVM_error, SVM_margin = approximate_normal_test(fit_and_score(SVC(gamma=best_gamma, kernel="rbf"), train, test))
    print(f"SVM test error: {SVM_error}% +- {SVM_margin}% with gamma: {best_gamma}")
