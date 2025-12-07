import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import RandomOverSampler
from scipy import sparse
import joblib

data = pd.read_csv("HateSpeechDataset.csv")
data = data[data["Label"].isin(['0', '1'])]

X = data["Content"]
y = data["Label"].astype(int)

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

ros = RandomOverSampler()
X_bal, y_bal = ros.fit_resample(X_vec, y)

print(X_bal)

sparse.save_npz("balanced_X.npz", X_bal)
pd.Series(y_bal).to_csv("balanced_y.csv", index=False, header=False)

joblib.dump(vectorizer, "tfidf_vectorizer.pkl")