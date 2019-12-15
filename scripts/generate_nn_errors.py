# coding: utf-8

import pandas as pd
import pymorphy2

from tqdm import tqdm
from razdel import tokenize


_MORPH = pymorphy2.MorphAnalyzer()
_ALLOWED_POS_TAGS = {'ADJF', 'ADJS', 'PRTF', 'PRTS', 'ADVB'}


def _is_dictionary_word(token):
    return any([parse.is_known for parse in _MORPH.parse(token)])


def _should_be_distorted(token):
    parses = _MORPH.parse(token)
    for parse in parses:
        if any(tag in parse.tag for tag in _ALLOWED_POS_TAGS):
            return True
    return False


def _is_good_pair(token, distorted_token):
    return (
        _is_dictionary_word(distorted_token)
        and _should_be_distorted(token)
        and _should_be_distorted(distorted_token)
    )


def generate_nn_errors(text):
    for token in tokenize(text):
        symbol_index = 0
        token_text = token.text
        while symbol_index < len(token_text):
            if token_text[symbol_index] != 'н':
                symbol_index += 1
            else:
                if symbol_index + 1 < len(token_text) and token_text[symbol_index + 1] == 'н':
                    error_token_text = token_text[:symbol_index] + token_text[symbol_index + 1:]
                    if _is_good_pair(token_text, error_token_text):
                        yield (text[:token.start]
                               + error_token_text
                               + text[token.stop:])
                    symbol_index += 2
                else:
                    error_token_text = token_text[:symbol_index] + 'н' + token_text[symbol_index:]
                    if _is_good_pair(token_text, error_token_text):
                        yield (text[:token.start]
                               + error_token_text
                               + text[token.stop:])
                    symbol_index += 1


def main():
    with open('data/opencorpora.txt') as f:
        correct_sentences = [line.strip() for line in f]

    data = []
    for correct_sentence in tqdm(correct_sentences):
        for error_sentence in generate_nn_errors(correct_sentence):
            data.append((correct_sentence, error_sentence, 'н-нн'))

    data = pd.DataFrame(data, columns=['correct_sentence', 'sentence_with_a_mistake', 'mistake_type'])
    data.to_csv('data/grammar_nn.csv')


if __name__ == "__main__":
    main()
