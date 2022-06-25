from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import matplotlib.pyplot as plt
import numpy as np

from algorithms.umda import UMDA
from algorithms.bmda import BMDA

max_features = 0
current_classifier = None
current_data = None

def is_solution_valid(bitstring) -> bool:
    counter = 0
    for i in bitstring:
        if i == 1: counter += 1

    return counter <= max_features

datasets = [
    ("Random", make_classification(n_samples=100, n_features=20, n_informative=17, n_redundant=1, n_repeated=2, n_classes=2)),
    ("Breast cancer", datasets.load_breast_cancer(return_X_y=True)),
    ("Wine", datasets.load_wine(return_X_y=True))
]

classifiers = [
    ("KNeighborsClassifier", KNeighborsClassifier(15)),
    ("SVC", SVC()),
    ("DecisionTreeClassifier", DecisionTreeClassifier(max_depth=5)),
    ("RandomForestClassifier", RandomForestClassifier(max_depth=5, max_features=20)),
    ("AdaBoostClassifier", AdaBoostClassifier()),
    ("GaussianNB", GaussianNB()),
    ("QuadraticDiscriminantAnalysis", QuadraticDiscriminantAnalysis()),
]

def plot_fitness(axs, no_fs_score, umda_fs_score, bmda_fs_score):
    labels = ["No FS", "UMDA", "BMDA"]
    colors = ["salmon", "yellow", "blue"]
    scores = [no_fs_score, umda_fs_score, bmda_fs_score]
    bar = axs.bar(labels, scores, color=colors, width=0.8, align='center')
    axs.bar_label(bar, fmt="%.4f", label_type="center")

def plot_features(axs, original_count, umda_fs_count, bmda_fs_count):
    labels = ["No FS", "UMDA", "BMDA"]
    colors = ["salmon", "yellow", "blue"]
    scores = [original_count, umda_fs_count, bmda_fs_count]
    bar = axs.bar(labels, scores, color=colors, width=0.8, align='center')
    axs.bar_label(bar, label_type="center")

def classify(features: list[int]) -> float:
    X, y = current_data[0][:], current_data[1]
    i = 0
    indices_to_delete = []
    while i < len(features):
        if features[i] == 0:
            indices_to_delete.append(i)
        i += 1

    X = np.delete(X, indices_to_delete, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    current_classifier.fit(X_train, y_train)
    return current_classifier.score(X_test, y_test)

umda = UMDA(
    classify,
    is_solution_valid,
    5,
    50,
    5,
    25
)

bmda = BMDA(
    classify,
    is_solution_valid,
    5,
    50,
    5,
    25
)

def main():
    global current_classifier
    global current_data
    global max_features

    fig_fitness, axs_fitness = plt.subplots(len(datasets), len(classifiers), sharey=True)
    fig_features, axs_features = plt.subplots(len(datasets), len(classifiers), sharey=True)

    fig_fitness.suptitle('Comparison of classification scores before and after FS', fontsize=16)
    fig_features.suptitle('Comparison of number of features before and after FS', fontsize=16)

    i = 0
    for ds in datasets:
        name, data = ds
        print(f"Dataset: {name}")
        X = data[0]
        j = 0
        axs_fitness[i][0].set_ylabel(name)
        axs_features[i][0].set_ylabel(name)
        for cl_name, classifier in classifiers:
            if i == 0:
                axs_fitness[i][j].set_title(cl_name)
                axs_features[i][j].set_title(cl_name)
                
            current_classifier = classifier
            current_data = data 
            no_fs_score = classify([1 for _ in range(X.shape[1])])

            max_features = X.shape[1] / 2
            umda_best = umda.calculate(X.shape[1])[-1]
            umda_fs_score = umda_best.fitness
            umda_feature_count = umda_best.bitstring.count(1) # Calculating number of features

            bmda_best = umda.calculate(X.shape[1])[-1]
            bmda_fs_score = bmda_best.fitness
            bmda_feature_count = bmda_best.bitstring.count(1) # Calculating number of features

            plot_fitness(axs_fitness[i][j], no_fs_score, umda_fs_score, bmda_fs_score)
            plot_features(axs_features[i][j], X.shape[1], umda_feature_count, bmda_feature_count)
            print(f"\t{cl_name}:{' ' * (30 - len(cl_name))}NO FS: {no_fs_score}, UMDA FS: {umda_fs_score}, BMDA FS: {bmda_fs_score}")
            j += 1
        i += 1
        print()

    plt.show()

if __name__ == "__main__":
    main()