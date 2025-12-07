import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from imblearn.over_sampling import RandomOverSampler
from scipy import sparse


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


def run_rskf(X, y, n_splits=5, n_repeats=2):
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2)
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(rskf.split(X, y)):
        print(f"\n---- Fold {fold + 1} ----")

        X_tr = X[train_idx]
        y_tr = y.iloc[train_idx]

        X_val = X[val_idx]
        y_val = y.iloc[val_idx]

        clf = LinearSVC()

        print("training...")
        clf.fit(X_tr, y_tr)

        print("validating...")
        y_val_pred = clf.predict(X_val)

        score = balanced_accuracy_score(y_val, y_val_pred)
        fold_scores.append(score)

        print(f"Fold score: {score:.4f}")
    return fold_scores


if __name__ == '__main__':
    X_bal = sparse.load_npz("balanced_X.npz")
    y_bal = pd.read_csv("balanced_y.csv", header=None, dtype=int)[0]

    balanced_scores = run_rskf(X_bal, y_bal)

    data = pd.read_csv("HateSpeechDataset.csv")
    data = data[data["Label"].isin(['0','1'])]

    X = data["Content"].reset_index(drop=True)
    y = data["Label"].astype(int).reset_index(drop=True)
    y = y.astype(int)

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    imbalanced_scores = run_rskf(X_vec, y)
    print("\n============================")
    print(f"Imbalanced dataset mean CV balanced accuracy: {sum(imbalanced_scores) / len(imbalanced_scores):.4f}\n")
    print(f"Balanced dataset mean CV balanced accuracy: {sum(balanced_scores) / len(balanced_scores):.4f}")
    print("============================\n")
