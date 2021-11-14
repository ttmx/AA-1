import math
import operator
import numpy as np
import matplotlib.pyplot as plt
from KernelDensityNB import KernelDensityNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold


def load_data():
    train_data = np.loadtxt("./TP1_train.tsv", delimiter="\t")
    data_ = train_data[:, :-1]
    mean = data_.mean(axis=0)
    std = data_.std(axis=0)
    standardize_and_shuffle(train_data, mean, std)
    test_data = np.loadtxt("./TP1_test.tsv", delimiter="\t")
    standardize_and_shuffle(test_data, mean, std)
    return train_data, test_data


def standardize_and_shuffle(data, mean, std):
    data[:, :-1] = standardize(data[:, :-1], mean, std)
    np.random.shuffle(data)


def standardize(data, mean, std):
    return (data - mean) / std


def cross_validation(nb):
    training_error = 0
    validation_error = 0
    skf = StratifiedKFold(n_splits=folds)
    for train_i, validation_i in skf.split(train[:, :-1], train[:, -1]):
        train_set, validation_set = train[train_i], train[validation_i]

        nb.fit(train_set[:, :-1], train_set[:, -1])
        training_error += error_for(nb, train_set)
        validation_error += error_for(nb, validation_set)

    training_error /= folds
    validation_error /= folds
    return training_error, validation_error


def error_for(nb, score_set):
    return 1 - nb.score(score_set[:, :-1], score_set[:, -1])


def fit_and_score(nb, fit, score_set):
    nb.fit(fit[:, :-1], fit[:, -1])
    return error_for(nb, score_set)


def approximate_normal_test(test_error):
    sigma = math.sqrt(test.size * test_error * (1 - test_error))
    absolute_margin = 1.96 * sigma  # 0.95 confidence interval
    relative_margin = absolute_margin / test.size
    return test_error, relative_margin


def mcnemar_test_with(classifier_1, classifier_2, test_set):
    c1_predictions = classifier_1.predict(test_set[:, :-1])
    c2_predictions = classifier_2.predict(test_set[:, :-1])
    correct_predictions = test_set[:, -1]
    e01 = e10 = 0
    for c1p, c2p, cp in np.array([c1_predictions, c2_predictions, correct_predictions]).transpose():
        if (not np.isclose(c1p, cp)) and np.isclose(c2p, cp):
            e01 += 1
        elif (not np.isclose(c2p, cp)) and np.isclose(c1p, cp):
            e10 += 1

    return mcnemar_test(e01, e10)


def mcnemar_test(e01, e10):
    return (math.fabs(e01 - e10) - 1) ** 2 / (e01 + e10)


def gen_plot(data, filename, x_label, title):
    plt.figure()
    plt.plot(data[:, 0], data[:, 1], label="Training Error")
    plt.plot(data[:, 0], data[:, 2], label="Validation Error")
    plt.title(title)
    plt.xticks(data[:, 0], rotation="vertical")
    plt.xlabel(x_label)
    plt.ylabel("Error (decimal)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


def unpack_errors(packed):
    return [[parameter, t_error, v_error] for parameter, (t_error, v_error) in packed]


if __name__ == '__main__':
    train, test = load_data()

    folds = 5

    k_data = unpack_errors(([bandwidth, cross_validation(KernelDensityNB(bandwidth))]
                            for bandwidth in np.linspace(0.02, 0.6, 30)))
    gen_plot(np.array(k_data), "NB.png", "Bandwidth", "NB with KDE")

    [best_bandwidth, _, min_error] = min(k_data, key=operator.itemgetter(2))
    print(f"Kernel Density validation error: {min_error:.4f} with bandwidth: {best_bandwidth:.2f}")

    print(f"Gaussian validation error: {cross_validation(GaussianNB())[1]:.4f}")

    svm_data = unpack_errors(([gamma, cross_validation(SVC(gamma=gamma, kernel="rbf"))]
                              for gamma in np.linspace(0.2, 6, 30)))
    gen_plot(np.array(svm_data), "SVM.png", "Gamma", "SVM")

    [best_gamma, _, min_error] = min(svm_data, key=operator.itemgetter(2))
    print(f"SVM validation error: {min_error:.4f} with gamma: {best_gamma:.1f}")
    print()

    kernel_density_nb = KernelDensityNB(best_bandwidth)
    K_error, K_margin = approximate_normal_test(fit_and_score(kernel_density_nb, train, test))
    print(f"Kernel Density test error: {K_error:.4f}% +- {K_margin:.4f}% with bandwidth: {best_bandwidth:.2f}")

    gaussian_nb = GaussianNB()
    G_error, G_margin = approximate_normal_test(fit_and_score(gaussian_nb, train, test))
    print(f"Gaussian test error: {G_error:.4f}% +- {G_margin:.4f}%")

    svm = SVC(gamma=best_gamma, kernel="rbf")
    SVM_error, SVM_margin = approximate_normal_test(fit_and_score(svm, train, test))
    print(f"SVM test error: {SVM_error:.4f}% +- {SVM_margin:.4f}% with gamma: {best_gamma:.1f}")
    print()

    k_g = mcnemar_test_with(kernel_density_nb, gaussian_nb, test)
    k_svm = mcnemar_test_with(kernel_density_nb, svm, test)
    g_svm = mcnemar_test_with(gaussian_nb, svm, test)

    print(f"Kernel Density vs Gaussian: {k_g:.2f}")
    print(f"Kernel Density vs SVM: {k_svm:.2f}")
    print(f"Gaussian vs SVM: {g_svm:.2f}")
    print()

    tuned_svm_data = ([gamma, C, cross_validation(SVC(gamma=gamma, C=C, kernel="rbf"))]
                      for gamma in np.linspace(0.2, 6, 30)
                      for C in np.logspace(-3, 3, 7))
    tuned_svm_data = ([bandwidth, C, t_error, v_error] for bandwidth, C, (t_error, v_error) in tuned_svm_data)

    [best_gamma, best_C, _, min_error] = min(tuned_svm_data, key=operator.itemgetter(3))
    tuned_svm = SVC(gamma=best_gamma, C=best_C, kernel="rbf")
    SVM_error, SVM_margin = approximate_normal_test(fit_and_score(tuned_svm, train, test))
    print(f"SVM test error: {SVM_error:.4f}% +- {SVM_margin:.4f}% with gamma: {best_gamma:.1f} and C: {best_C}")
    print()

    svm_tuned_svm = mcnemar_test_with(svm, tuned_svm, test)
    print(f"SVM vs Tuned SVM: {svm_tuned_svm:.2f}")
