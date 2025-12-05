import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from scipy import sparse

import joblib

data = pd.read_csv("HateSpeechDataset.csv")
data = data[data["Label"].isin(['0','1'])]

X = data["Content"]
y = data["Label"].astype(int)

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

ros = RandomOverSampler()
X_train_bal, y_train_bal = ros.fit_resample(X_train, y_train)

sparse.save_npz("balanced_X_train.npz", X_train_bal)
pd.Series(y_train_bal).to_csv("balanced_y_train.csv", index=False)

sparse.save_npz("X_test.npz", X_test)
y_test.to_csv("y_test.csv", index=False)

joblib.dump(vectorizer, "tfidf_vectorizer.pkl")