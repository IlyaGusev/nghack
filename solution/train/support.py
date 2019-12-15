# coding: utf-8

import argparse
import joblib
import math
import numpy as np
import pandas as pd
import pickle

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline


_CLASS_TO_INDEX = {
    'negative': 0,
    'neutral': 1,
    'positive': 2
}


def preprocess(data):
    data['class'] = data['label'].apply(lambda label: _CLASS_TO_INDEX[label])
    data['text'].fillna('', inplace=True)

    return data


def train(train_data_path, output_path, val_data_path=None):
    train_data = pd.read_csv(train_data_path, index_col=0)
    if val_data_path:
        val_data = pd.read_csv(val_data_path, index_col=0)
    else:
        val_data = None

    train_data = preprocess(train_data)
    if val_data is not None:
        val_data = preprocess(val_data)

    model = Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=(1, 2))),
        ('model', LogisticRegression(C=10))
    ])

    model.fit(train_data['text'], train_data['class'])

    if val_data is not None:
        val_preds = model.predict(val_data['text'])
        print('F1-score = {:.2%}'.format(f1_score(val_data['class'], val_preds, average='macro')))

    joblib.dump(model, output_path, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='data/support_split_train.csv')
    parser.add_argument('--valid', default='data/support_split_val.csv')
    parser.add_argument('--output', default='models/support.pkl')

    args = parser.parse_args()

    train(args.train, args.output, args.valid)


if __name__ == "__main__":
    main()
