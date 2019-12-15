import sys
import argparse
import pandas
import fasttext
from sentencepiece import SentencePieceProcessor as sp_processor
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

    tsya_model_path = "models/tsya_predictor.bin"
    bpe_model_path = "models/grammar_bpe.model"
    tsya_predictor = fasttext.load_model(tsya_model_path)
    bpe_model = sp_processor()
    bpe_model.load(bpe_model_path)

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

    for i, sentence in enumerate(fixed_sentences):
        tsya_count = sentence.count("тся")
        tsjya_count = sentence.count("ться")
        if tsya_count + tsjya_count != 1:
            continue
        processed_sentence = " ".join(bpe_model.EncodeAsPieces(sentence))
        tsya_predictions = tsya_predictor.predict(processed_sentence)
        tsya_proba = float(tsya_predictions[1][0])
        tsya_label = int(tsya_predictions[0][0][-1])
        if tsya_label == 0 and tsya_proba > 0.9 and tsya_count == 1:
            fixed_sentences[i] = sentence.replace("тся", "ться")
        elif tsya_label == 0 and tsya_proba > 0.9 and tsjya_count == 1:
            fixed_sentences[i] = sentence.replace("ться", "тся")

    df_test['correct_sentence'] = fixed_sentences
    df_test.to_csv(output_csv)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_csv', help="path to input file")
    parser.add_argument('output_csv', help="path to output file")
    args = parser.parse_args()
    fix_mistakes(**vars(args))

