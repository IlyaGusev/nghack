import joblib
import pandas as pd
import scipy
import sys


_CLASSES = [
    'negative',
    'neutral',
    'positive'
]


def load_model(path='models/support.pkl'):
    return joblib.load(path)


def preprocess(data):
    data['text'].fillna('', inplace=True)
    return data


def predict(input_csv_path, output_csv_path):
    model, char_vectorizer, word_vectorizer = load_model()

    df_test = pd.read_csv(input_csv_path, index_col='id')
    df_test = preprocess(df_test)

    X_chars = char_vectorizer.transform(df_test['text'])
    X_words = word_vectorizer.transform(df_test['text'])

    X = scipy.sparse.hstack([X_chars, X_words])

    predictions = model.predict(X)
    df_test['label'] = [_CLASSES[class_index] for class_index in predictions]

    df_test.to_csv(output_csv_path)


if __name__ == '__main__':
    predict(sys.argv[1], sys.argv[2])
