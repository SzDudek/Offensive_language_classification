import gensim
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import TruncatedSVD
import tensorflow_text as text
import tensorflow as tf
import tensorflow_hub as hub

tfhub_handle_preprocess = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
tfhub_handle_encoder = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"

bert_preprocess = hub.KerasLayer(tfhub_handle_preprocess)
bert_encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=False)

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

def bert_encode(texts, batch_size=32):
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = bert_preprocess(batch)
        outputs = bert_encoder(inputs)

        # CLS token embedding
        cls_embeddings = outputs["pooled_output"]
        embeddings.append(cls_embeddings.numpy())

    return np.vstack(embeddings)

def tokenize_texts(texts):
    return [gensim.utils.simple_preprocess(text) for text in texts]


def doc_vector(tokens, model, vec_size):
    vectors = [model.wv[w] for w in tokens if w in model.wv]
    if not vectors:
        return np.zeros(vec_size, dtype=float)
    return np.mean(vectors, axis=0)


def build_w2v_fatures(X_tr, X_val, vec_size=100, window_size=5, min_count=2):
    X_tr = tokenize_texts(X_tr)
    X_val = tokenize_texts(X_val)
    w2v_model = Word2Vec(
        sentences=X_tr,
        vector_size=vec_size,
        window=window_size,
        min_count=min_count
    )

    X_tr = np.vstack([
        doc_vector(tokens, w2v_model, vec_size)
        for tokens in X_tr
    ])

    X_val = np.vstack([
        doc_vector(tokens, w2v_model, vec_size)
        for tokens in X_val
    ])

    return X_tr, X_val


def run_rskf(X, y, n_splits=5, n_repeats=1, sampling='', classifier='svc', rep='tfidf'):
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(rskf.split(X, y)):
        print(f"\n---- Fold {fold + 1} ----")

        X_tr = X[train_idx]
        y_tr = y.iloc[train_idx]

        if sampling != '':
            print("Balancing data...")
        if sampling == 'ros':
            ros = RandomOverSampler()
            X_tr, y_tr = ros.fit_resample(X_tr, y_tr)
        elif sampling == 'rus':
            rus = RandomUnderSampler()
            X_tr, y_tr = rus.fit_resample(X_tr, y_tr)
        elif sampling == 'smote':
            smote = SMOTE()
            X_tr, y_tr = smote.fit_resample(X_tr, y_tr)
        elif sampling != '':
            raise Exception(f"Unknown sampling technique: {sampling}")

        if classifier == 'svc':
            clf = LinearSVC()
        elif classifier == 'lr':
            clf = LogisticRegression(max_iter=1000)
        elif classifier == 'sgd':
            if sampling == '':
                clf = SGDClassifier(class_weight={0: 1, 1: 5})
            else:
                clf = SGDClassifier()
        else:
            raise Exception(f"Unknown classifier {classifier}")

        X_val = X[val_idx]
        y_val = y.iloc[val_idx]

        if rep == 'tfidf':
            print("Reducing features by SVD...")
            svd = TruncatedSVD(n_components=200)
            X_tr = svd.fit_transform(X_tr)
            X_val = svd.transform(X_val)

        elif rep == 'word2vec':
            print("Generating vectors with word2vec...")
            X_tr, X_val = build_w2v_fatures(X_tr, X_val)

        elif rep == 'bert':
            print("Encoding texts with BERT...")
            preprocess_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
            encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=False)

            X_tr_pre = preprocess_layer(X_tr)
            X_val_pre = preprocess_layer(X_val)

            X_tr_enc = encoder(X_tr_pre)
            X_val_enc = encoder(X_val_pre)

            X_tr, X_val = X_tr_enc['pooled_output'].numpy(), X_val_enc['pooled_output'].numpy()

            # X_tr = bert_encode(list(X_tr))
            # X_val = bert_encode(list(X_val))
        else:
            raise Exception(f"Unknown representation: {rep}")

        print("Training classifier...")
        clf.fit(X_tr, y_tr)

        print("Predicting labels...")
        y_val_pred = clf.predict(X_val)
        score = balanced_accuracy_score(y_val, y_val_pred)
        fold_scores.append(score)

        print(f"Fold score: {score:.4f}")

        if fold == 0:
            plot_confusion_matrix(y_val, y_val_pred, f"{classifier}_{sampling}.png")

    return fold_scores


if __name__ == '__main__':
    configurations = [
        # {"classifier": "svc", "sampling": ""},
        # {"classifier": "svc", "sampling": "ros"},
        # {"classifier": "svc", "sampling": "rus"},
        # {"classifier": "svc", "sampling": "smote"},
        #
        # {"classifier": "lr", "sampling": ""},
        # {"classifier": "lr", "sampling": "ros"},
        # {"classifier": "lr", "sampling": "rus"},
        # {"classifier": "lr", "sampling": "smote"},
        #
        # {"classifier": "sgd", "sampling": ""},
        # {"classifier": "sgd", "sampling": "ros"},
        # {"classifier": "sgd", "sampling": "rus"},
        # {"classifier": "sgd", "sampling": "smote"},

        {"classifier": "svc", "sampling": "", "representation": "word2vec"},
        {"classifier": "svc", "sampling": "ros", "representation": "word2vec"},
        {"classifier": "svc", "sampling": "rus", "representation": "word2vec"},
        {"classifier": "svc", "sampling": "smote", "representation": "word2vec"},

        {"classifier": "lr", "sampling": "", "representation": "word2vec"},
        {"classifier": "lr", "sampling": "ros", "representation": "word2vec"},
        {"classifier": "lr", "sampling": "rus", "representation": "word2vec"},
        {"classifier": "lr", "sampling": "smote", "representation": "word2vec"},

        {"classifier": "sgd", "sampling": "", "representation": "word2vec"},
        {"classifier": "sgd", "sampling": "ros", "representation": "word2vec"},
        {"classifier": "sgd", "sampling": "rus", "representation": "word2vec"},
        {"classifier": "sgd", "sampling": "smote", "representation": "word2vec"},

        # {"classifier": "svc", "sampling": "", "representation": "bert"},
        # {"classifier": "svc", "sampling": "ros", "representation": "bert"},
        # {"classifier": "svc", "sampling": "rus", "representation": "bert"},
        # {"classifier": "svc", "sampling": "smote", "representation": "bert"},
        #
        # {"classifier": "lr", "sampling": "", "representation": "bert"},
        # {"classifier": "lr", "sampling": "ros", "representation": "bert"},
        # {"classifier": "lr", "sampling": "rus", "representation": "bert"},
        # {"classifier": "lr", "sampling": "smote", "representation": "bert"},
        #
        # {"classifier": "sgd", "sampling": "", "representation": "bert"},
        # {"classifier": "sgd", "sampling": "ros", "representation": "bert"},
        # {"classifier": "sgd", "sampling": "rus", "representation": "bert"},
        # {"classifier": "sgd", "sampling": "smote", "representation": "bert"}
    ]

    data = pd.read_csv("HateSpeechDataset.csv")
    data = data[data["Label"].isin(['0','1'])]

    X = data["Content"].reset_index(drop=True)
    y = data["Label"].astype(int).reset_index(drop=True)

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    # f = open("output.csv", "w")
    # f.write("clf,sampling,BAC\n")/
    for config in configurations:
        rep = config.get("representation", "tfidf")

        X_input = X if rep == 'tfidf' else X_vec

        balanced_scores = run_rskf(X_input, y, sampling=config["sampling"], classifier=config["classifier"], rep=rep)
        print(f"twoj stary {config}")
        # print(f"Mean balanced accuracy for {config["classifier"]} with {config["sampling"]}: {sum(balanced_scores)/len(balanced_scores):.4f}")
        # f.write(f"{config["classifier"]},{config["sampling"]},{sum(balanced_scores)/len(balanced_scores):.4f}\n")

    # f.close()
