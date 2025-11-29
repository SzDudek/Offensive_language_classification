import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE


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
    X = data["Content"]
    y = data["Label"]

    print(y.value_counts())
    print(y.unique())

    y = y[y.isin(['0', '1'])]
    X = X[y.index]

    y = y.astype(int)

    print(y.value_counts())
    print(y.unique())

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42, stratify=y
    )

    clf_non_balanced = LogisticRegression(max_iter=100)
    clf_non_balanced.fit(X_train, y_train)
    y_pred_non_balanced = clf_non_balanced.predict(X_test)
    report_non_balanced = classification_report(y_test, y_pred_non_balanced)
    plot_confusion_matrix(y_test, y_pred_non_balanced, "confusion_mtx_LR_nb.png")

    with open("LR_non_balanced.txt", "w") as f:
        f.write(report_non_balanced)

    sm = SMOTE()
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

    clf_balanced = LogisticRegression(max_iter=100)
    clf_balanced.fit(X_train_bal, y_train_bal)
    y_pred_balanced = clf_balanced.predict(X_test)
    report_balanced = classification_report(y_test, y_pred_balanced)
    plot_confusion_matrix(y_test, y_pred_balanced, "confusion_mtx_LR_SMOTE.png")

    with open("LR_SMOTE.txt", "w") as f:
        f.write(report_balanced)

    print("\n=== WITHOUT BALANCING ===")
    print(report_non_balanced)
    print("\n=== WITH SMOTE BALANCING ===")
    print(report_balanced)
