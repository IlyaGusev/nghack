import sys
import argparse
import pandas


from grammar_ngram_lm import load_grams, make_ngram_correction, make_hypotheses_neni

grams = load_grams()

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

    fixed_sentences = _fix_neni(original_sentences)

    df_test['correct_sentence'] = fixed_sentences
    df_test.to_csv(output_csv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_csv', help="path to input file")
    parser.add_argument('output_csv', help="path to output file")
    args = parser.parse_args()
    fix_mistakes(**vars(args))
