import sys
import fasttext
import pandas
import pickle
import scipy

import numpy as np


TO_LABEL = {
    '__label__0': 'FAQ - интернет',
    '__label__1': 'FAQ - тарифы и услуги',
    '__label__2': 'SIM-карта и номер',
    '__label__3': 'Баланс',
    '__label__4': 'Личный кабинет',
    '__label__5': 'Мобильные услуги',
    '__label__6': 'Мобильный интернет',
    '__label__7': 'Оплата',
    '__label__8': 'Роуминг',
    '__label__9': 'Устройства',
    '__label__10': 'запрос обратной связи',
    '__label__11': 'мобильная связь - зона обслуживания',
    '__label__12': 'мобильная связь - тарифы',
    '__label__13': 'тарифы - подбор'
}


def to_label(ft):
    return TO_LABEL.get(ft)


if __name__ == '__main__':
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    df_test = pandas.read_csv(input_csv, index_col='id')

    ft_model_path = "models/intent.ftz"
    ft_model = fasttext.load_model(ft_model_path)

    tf_model = pickle.load(open("models/intent_tfidf.bin", "rb"))
    tf_char_vectorizer = pickle.load(open("models/char_vectorizer.bin", "rb"))
    tf_word_vectorizer = pickle.load(open("models/word_vectorizer.bin", "rb"))

    def process_text(text):
        text = str(text).strip().lower()
        text = text.replace('\n', ' ')
        text = text.strip("“ ”‘ ’«»\"'?!.;: ")
        return text

    df_test['text'].fillna('', inplace=True)
    texts = [process_text(text) for text in df_test["text"].tolist()]

    # ft
    ft_preds_raw = ft_model.predict(texts, k=32)
    ft_preds = list()
    for labels, probs in zip(ft_preds_raw[0], ft_preds_raw[1]):
        d = sorted(list(zip(labels, probs)), key=lambda x: int(x[0].replace('__label__', '')))
        (labels, probs) = zip(*d)
        ft_preds.append(probs)

    # tf
    X_val_chars = tf_char_vectorizer.transform(df_test['text'].tolist())
    X_val_words = tf_word_vectorizer.transform(df_test['text'].tolist())

    X_val = scipy.sparse.hstack([X_val_chars, X_val_words])
    tf_preds = tf_model.predict_proba(X_val)

    # combine
    preds = list()
    for pr1, pr2 in zip(tf_preds, ft_preds):
        preds.append(f'__label__{np.argmax(((pr1 + pr2) / 2))}')

    labels = [to_label(label) for label in preds]
    df_test['label'] = labels

    df_test.to_csv(output_csv)
