import sys
import argparse
import pandas
import fasttext
import pymorphy2
from sentencepiece import SentencePieceProcessor as sp_processor
from razdel import tokenize
from razdel.substring import Substring


from grammar_ngram_lm import load_grams, make_ngram_correction, make_hypotheses_neni

grams = load_grams()



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
    # "в течении года": "в течение года",
    "несколько не изменился": "нисколько не изменился",
    "не кто не может": "никто не может",
    "ни кому": "никому",
    "одно и тоже": "одно и то же",
    "как то так же": "как-то так же",
    "бес толку": "бестолку"
}

_ALLOWED_POS_TAGS_FOR_TAKI = {'ADVB', 'VERB', 'INFN', 'PRCL'}
_ALLOWED_POS_TAGS_FOR_TO = {'ADVB', 'NPRO'}

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

_MORPH = pymorphy2.MorphAnalyzer()


def _fix_dictionary(original_sentences):
    for i, sentence in enumerate(original_sentences):
        for key, value in substrings_fixes.items():
            if key in sentence or key.capitalize() in sentence:
                original_sentences[i] = sentence.replace(key, value)
                original_sentences[i] = original_sentences[i].replace(key.capitalize(), value.capitalize())

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

    return fixed_sentences


def _fix_tsya(fixed_sentences,
              tsya_border=0.65,
              tsya_model_path="models/tsya_predictor.bin",
              bpe_model_path="models/grammar_bpe.model"):
    tsya_predictor = fasttext.load_model(tsya_model_path)
    bpe_model = sp_processor()
    bpe_model.load(bpe_model_path)

    for i, sentence in enumerate(fixed_sentences):
        tsya_count = sentence.count("тся")
        tsjya_count = sentence.count("ться")
        if tsya_count + tsjya_count != 1:
            continue
        processed_sentence = " ".join(bpe_model.EncodeAsPieces(sentence.lower()))
        tsya_predictions = tsya_predictor.predict(processed_sentence)
        tsya_proba = float(tsya_predictions[1][0])
        tsya_label = int(tsya_predictions[0][0][-1])
        if tsya_label == 0 and tsya_proba > tsya_border and tsya_count >= 1 and tsjya_count == 0:
            fixed_sentences[i] = sentence.replace("тся", "ться")
        elif tsya_label == 0 and tsya_proba > tsya_border and tsjya_count >= 1 and tsya_count == 0:
            fixed_sentences[i] = sentence.replace("ться", "тся")


def _fix_nn(fixed_sentences,
            nn_border=0.65,
            nn_model_path="models/nn_predictor.bin",
            bpe_model_path="models/opencorpora_bpe.model"):
    nn_predictor = fasttext.load_model(nn_model_path)
    bpe_model = sp_processor()
    bpe_model.load(bpe_model_path)

    for i, sentence in enumerate(fixed_sentences):
        nn_count = sentence.count("нн")
        if nn_count != 1:
            continue
        processed_sentence = " ".join(bpe_model.EncodeAsPieces(sentence.lower()))
        nn_predictions = nn_predictor.predict(processed_sentence)
        nn_proba = float(nn_predictions[1][0])
        nn_label = int(nn_predictions[0][0][-1])
        if nn_label == 0 and nn_proba > nn_border and nn_count == 1:
            fixed_sentences[i] = sentence.replace("нн", "н")

def _fix_merge_to(fixed_sentences,
                  border=0.65,
                  model_path="models/to_merge_predictor.bin",
                  bpe_model_path="models/opencorpora_bpe.model"):
    predictor = fasttext.load_model(model_path)
    bpe_model = sp_processor()
    bpe_model.load(bpe_model_path)

    for i, sentence in enumerate(fixed_sentences):
        text = sentence
        chto_bi_count = text.count("что бы")
        to_je_count = text.count("то же")
        tak_je_count = text.count("так же")
        if chto_bi_count + to_je_count + tak_je_count != 1:
            continue
        processed_sentence = " ".join(bpe_model.EncodeAsPieces(sentence.lower()))
        predictions = predictor.predict(processed_sentence)
        proba = float(predictions[1][0])
        label = int(predictions[0][0][-1])
        if label == 1 or proba < border:
            continue
        if chto_bi_count == 1:
            fixed_sentences[i] = sentence.replace("что бы", "чтобы")
        elif to_je_count == 1:
            fixed_sentences[i] = sentence.replace("то же", "тоже")
        elif tak_je_count == 1:
            fixed_sentences[i] = sentence.replace("так же", "также")


def _fix_izza_on_text(text):
    tokens = list(tokenize(text))
    result_tokens = []
    i = 0
    while i < len(tokens) - 1:
        if tokens[i].text.lower() == 'из' and (tokens[i+1].text.lower() == 'за' or tokens[i+1].text.lower() == 'под'):
            result_tokens.append(
                Substring(tokens[i].start, tokens[i+1].stop, tokens[i].text + '-' + tokens[i+1].text)
            )
            i += 2
        else:
            result_tokens.append(tokens[i])
            i += 1

    fixed_sentence = text
    for token in result_tokens:
        fixed_sentence = fixed_sentence[:token.start] + token.text + fixed_sentence[token.stop:]

    return fixed_sentence


def _fix_izza(fixed_sentences):
    return [_fix_izza_on_text(text) for text in fixed_sentences]


def _is_good_for_particle(prev_token, allowed_pos_tags):
    for parse in _MORPH.parse(prev_token):
        if any(tag in parse.tag for tag in allowed_pos_tags):
            return True
    return False


def _fix_particles_on_text(text):
    tokens = list(tokenize(text))
    result_text = ''
    prev_end = 0
    for i, token in enumerate(tokens):
        if token.text not in {'то', 'либо', 'нибудь', 'таки'}:
            result_text += text[prev_end: token.start] + token.text
        elif token.text == 'таки':
            if i > 0 and _is_good_for_particle(tokens[i - 1].text, _ALLOWED_POS_TAGS_FOR_TAKI):
                result_text += '-' + token.text
            else:
                result_text += text[prev_end: token.start] + token.text
        else:
            if i > 0 and _is_good_for_particle(tokens[i - 1].text, _ALLOWED_POS_TAGS_FOR_TO):
                result_text += '-' + token.text
            else:
                result_text += text[prev_end: token.start] + token.text

        prev_end = token.stop

    if tokens:
        result_text += text[tokens[-1].stop:]

    return result_text


def _fix_particles(fixed_sentences):
    return [_fix_particles_on_text(text) for text in fixed_sentences]


def _fix_pronouns(fixed_sentences,
                  nn_border=0.65,
                  nn_model_path="models/pronoun_model.bin",
                  bpe_model_path="models/opencorpora_bpe.model"):
    nn_predictor = fasttext.load_model(nn_model_path)
    bpe_model = sp_processor()
    bpe_model.load(bpe_model_path)

    for i, sentence in enumerate(fixed_sentences):
        for from_text, to_text in _HARD_PRONOUNS:
            if (from_text in sentence or to_text in sentence
                or from_text.capitalize() in sentence
                or to_text.capitalize() in sentence
            ):
                processed_sentence = " ".join(bpe_model.EncodeAsPieces(sentence.lower()))
                nn_predictions = nn_predictor.predict(processed_sentence)
                nn_proba = float(nn_predictions[1][0])
                nn_label = int(nn_predictions[0][0][-1])
                if nn_label == 0 and nn_proba > nn_border:
                    if from_text in sentence:
                        fixed_sentences[i] = sentence.replace(from_text, to_text)
                    elif from_text.capitalize() in sentence:
                        fixed_sentences[i] = sentence.replace(from_text.capitalize(), to_text)
                    elif to_text in sentence:
                        fixed_sentences[i] = sentence.replace(to_text, from_text)
                    else:
                        fixed_sentences[i] = sentence.replace(to_text.capitalize(), from_text)


def _fix_neni(sentences):
    return [
        make_ngram_correction(
            text=s,
            hypo_makers=[make_hypotheses_neni],
            grams=grams,
        )
        for s in sentences
    ]



def fix_mistakes(input_csv, output_csv):
    df_test = pandas.read_csv(input_csv, index_col='id')
    original_sentences = df_test['sentence_with_a_mistake'].tolist()

    fixed_sentences = _fix_dictionary(original_sentences)
    _fix_tsya(fixed_sentences)
    _fix_nn(fixed_sentences)
    fixed_sentences = _fix_izza(fixed_sentences)
    fixed_sentences = _fix_particles(fixed_sentences)
    _fix_merge_to(fixed_sentences)
    fixed_sentences = _fix_neni(fixed_sentences)
    _fix_pronouns(fixed_sentences)

    df_test['correct_sentence'] = fixed_sentences
    df_test.to_csv(output_csv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_csv', help="path to input file")
    parser.add_argument('output_csv', help="path to output file")
    args = parser.parse_args()
    fix_mistakes(**vars(args))
