# coding: utf-8

import pandas as pd
import pymorphy2

from tqdm import tqdm
from razdel import tokenize


_HARD_PRONOUNS = [
    ('ввиду', 'в виду'),
    ('вместо', 'в место'),
    ('вследствие', 'в следствии'),
    ('вследствие', 'в следствие'),
    ('навстречу', 'на встречу'),
    ('наподобие', 'на подобие'),
    ('наподобие', 'на подобии'),
    ('насчёт', 'на счёт'),
    ('насчет', 'на счет'),
    ("вслед", "в след"),
    ("в виде", "ввиде"),
    ("в течение", "в течении"),
    ("в продолжение", "в продолжении"),
    ("в заключение", "в заключении"),
    ("в завершение", "в завершение"),
    ("в отличие от", "в отличии от"),
    ("в сравнении с", "в сравнение с"),
    ("в связи с", "в связе с"),
    ("по окончании", "по окончание"),
    ("по прибытии", "по прибытие")
]


def generate_errors(text):
    text = text.lower()
    for from_text, to_text in _HARD_PRONOUNS:
        if from_text in text:
            yield text.replace(from_text, to_text).capitalize()
        if to_text in text:
            yield text.replace(to_text, from_text).capitalize()


def main():
    with open('data/opencorpora.txt') as f:
        correct_sentences = [line.strip() for line in f]

    data = []
    for correct_sentence in tqdm(correct_sentences):
        for error_sentence in generate_errors(correct_sentence):
            data.append((correct_sentence, error_sentence, 'в течение'))

    data = pd.DataFrame(data, columns=['correct_sentence', 'sentence_with_a_mistake', 'mistake_type'])
    data.to_csv('data/grammar_pron.csv')


if __name__ == "__main__":
    main()
