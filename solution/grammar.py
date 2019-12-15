import sys
import argparse
import pandas
from razdel import tokenize

tokens_fixes = {
    "бес": "без",
    "вши": "ваши",
    "веером": "вечером",
    "длинны": "длины",
    "длинна": "длина",
}

substrings_fixes = {
    "белее или менее": "более или менее",
    "белее чем скромные": "более чем скромные",
    "без везти": "без вести",
    "в пошлом веке": "в прошлом веке",
    "в течении года": "в течение года",
    "несколько не изменился": "нисколько не изменился",
    "не кто не может": "никто не может",
    "ни кому": "никому",
    "одно и тоже": "одно и то же",
    "как то так же": "как-то так же",
    "бес толку": "бестолку",
}

def fix_mistakes(input_csv, output_csv):
    df_test = pandas.read_csv(input_csv, index_col='id')
    original_sentences = df_test['sentence_with_a_mistake'].tolist()

    # Substring fixes
    for i, sentence in enumerate(original_sentences):
        for key, value in substrings_fixes.items():
            if key in sentence or key.capitalize() in sentence:
                original_sentences[i] = sentence.replace(key, value)
                original_sentences[i] = original_sentences[i].replace(key.capitalize(), value.capitalize())

    # Tokens fixes
    tokenized_sentences = [(sentence, list(tokenize(sentence))) for sentence in original_sentences]
    fixed_sentences = []
    for sentence, tokens in tokenized_sentences:
        fixed_sentence = ""
        offset = 0
        for i, token in enumerate(tokens):
            tokens[i].start += offset
            tokens[i].stop += offset
            token_text = token.text
            fixed_token_text = tokens_fixes.get(token_text, None)
            if fixed_token_text is not None:
                tokens[i].text = fixed_token_text
                offset += len(fixed_token_text) - len(token_text)
        fixed_sentence = sentence
        for token in tokens:
            fixed_sentence = fixed_sentence[:token.start] + token.text + fixed_sentence[token.stop:]
        fixed_sentences.append(fixed_sentence)

    df_test['correct_sentence'] = fixed_sentences
    df_test.to_csv(output_csv)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_csv', help="path to input file")
    parser.add_argument('output_csv', help="path to output file")
    args = parser.parse_args()
    fix_mistakes(**vars(args))

