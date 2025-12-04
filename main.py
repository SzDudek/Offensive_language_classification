import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from imblearn.over_sampling import RandomOverSampler


def plot_confusion_matrix(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.xticks([0.5, 1.5], ['0', '1'])
    plt.yticks([0.5, 1.5], ['0', '1'], rotation=0)
    plt.title('Confusion Matrix', fontsize=16)
    plt.savefig(name)


if __name__ == '__main__':
    data = pd.read_csv("HateSpeechDataset.csv")
    data = data[data["Label"].isin(['0','1'])]

    X = data["Content"]
    y = data["Label"].astype(int)

    y = y.astype(int)

    print(y.value_counts())
    print(y.unique())

    classifiers = {
        "kNN": KNeighborsClassifier(),
        "GNB": GaussianNB(),
        "LR": LogisticRegression(max_iter=200)
    }

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_dense = X_train.toarray()
    X_test_dense = X_test.toarray()

    for clf_name, clf in classifiers.items():

        clf_nb = clone(clf)

        if clf_name == "GNB":
            clf_nb.fit(X_train_dense, y_train)
            y_pred_nb = clf_nb.predict(X_test_dense)
        else:
            clf_nb.fit(X_train, y_train)
            y_pred_nb = clf_nb.predict(X_test)


        report_nb = classification_report(y_test, y_pred_nb)
        plot_confusion_matrix(y_test, y_pred_nb, "confusion_mtx_{}_nb.png".format(clf_name))

        with open("{}_non_balanced.txt".format(clf_name), "w") as f:
            f.write(report_nb)

        # sm = SMOTE()
        ros = RandomOverSampler()

        X_train_bal, y_train_bal = ros.fit_resample(X_train, y_train)

        clf_bal = clone(clf)

        if clf_name == "GNB":
            # GNB requires dense
            X_train_bal_dense = X_train_bal.toarray()
            clf_bal.fit(X_train_bal_dense, y_train_bal)
            y_pred_bal = clf_bal.predict(X_test_dense)
        else:
            clf_bal.fit(X_train_bal, y_train_bal)
            y_pred_bal = clf_bal.predict(X_test)

        report_bal = classification_report(y_test, y_pred_bal)
        plot_confusion_matrix(y_test, y_pred_bal, "confusion_mtx_{}_ROS.png".format(clf_name))

        with open("{}_SMOTE.txt".format(clf_name), "w") as f:
            f.write(report_bal)
