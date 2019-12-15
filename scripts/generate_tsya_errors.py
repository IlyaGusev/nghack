# coding: utf-8

import pandas as pd

from tqdm import tqdm
from razdel import tokenize


def generate_errors(text):
    for token in tokenize(text):
        symbol_index = 0
        token_text = token.text
        if token_text.endswith('тся'):
            yield (text[:token.start]
                   + token_text[:-3] + 'ться'
                   + text[token.stop:])
        if token_text.endswith('ться'):
            yield (text[:token.start]
                   + token_text[:-4] + 'тся'
                   + text[token.stop:])


def main():
    with open('data/opencorpora.txt') as f:
        correct_sentences = [line.strip() for line in f]

    data = []
    for correct_sentence in tqdm(correct_sentences):
        for error_sentence in generate_errors(correct_sentence):
            data.append((correct_sentence, error_sentence, 'тся-ться'))

    data = pd.DataFrame(data, columns=['correct_sentence', 'sentence_with_a_mistake', 'mistake_type'])
    data.to_csv('data/grammar_tsya.csv')


if __name__ == "__main__":
    main()
