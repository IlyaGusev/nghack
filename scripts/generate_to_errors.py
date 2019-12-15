# coding: utf-8

import pandas as pd

from tqdm import tqdm
from razdel import tokenize


def generate_errors(text):
    tokens = list(tokenize(text))
    pos = 0
    while pos < len(tokens):
        token = tokens[pos]
        token_text = token.text
        if token_text.lower() in {'тоже', 'также'}:
            yield (text[:token.start]
                   + token_text[:-2] + ' же'
                   + text[token.stop:])
            pos += 1
        elif token_text.lower() == 'чтоб':
            yield (text[:token.start]
                   + token_text[:-1] + ' б'
                   + text[token.stop:])
            pos += 1
        elif token_text.lower() == 'чтобы':
            yield (text[:token.start]
                   + token_text[:-2] + ' бы'
                   + text[token.stop:])
            pos += 1
        elif pos + 1 < len(tokens) and token_text.lower() in {'то', 'так'} and tokens[pos + 1].text == 'же':
            yield (text[:token.start]
                   + token_text + 'же'
                   + text[tokens[pos + 1].stop:])
            pos += 2
        elif pos + 1 < len(tokens) and token_text.lower() == 'что' and tokens[pos + 1].text in {'б', 'бы'}:
            yield (text[:token.start]
                   + token_text + tokens[pos + 1].text
                   + text[tokens[pos + 1].stop:])
            pos += 2
        else:
            pos += 1


def main():
    with open('data/opencorpora.txt') as f:
        correct_sentences = [line.strip() for line in f]

    data = []
    for correct_sentence in tqdm(correct_sentences):
        for error_sentence in generate_errors(correct_sentence):
            data.append((correct_sentence, error_sentence, 'тоже, чтоб'))

    data = pd.DataFrame(data, columns=['correct_sentence', 'sentence_with_a_mistake', 'mistake_type'])
    data.to_csv('data/grammar_to.csv')


if __name__ == "__main__":
    main()
