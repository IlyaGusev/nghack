import pandas as pd
import pickle
import textdistance
import numpy as np
from nltk import wordpunct_tokenize


def make_hypotheses_neni(tokens, result=None, max_changes=3):
    if max_changes <= 0:
        return
    mc = max_changes - 1
    tokens = tuple(tokens)
    if result is None:
        result = {tuple(tokens)}
    elif tokens in result:
        return
    result.add(tokens)
    for i, token in enumerate(tokens):
        if token == 'не' or token == 'ни':
            make_hypotheses_neni(tuple(tokens[:i] + ('ни',) + tokens[(i+1):]), result=result, max_changes=mc)
            make_hypotheses_neni(tuple(tokens[:i] + ('не',) + tokens[(i+1):]), result=result, max_changes=mc)
            if i + 1 < len(tokens) and len(tokens[i+1]) >= 2:  # exclude 'нив'
                make_hypotheses_neni(tuple(tokens[:i] + ('ни' + tokens[i+1],) + tokens[(i+2):]), result=result, max_changes=mc)
                make_hypotheses_neni(tuple(tokens[:i] + ('не' + tokens[i+1],) + tokens[(i+2):]), result=result, max_changes=mc)
        elif token.startswith('не') or token.startswith('ни'):
             make_hypotheses_neni(tuple(tokens[:i] + ('не', token[2:]) + tokens[(i+1):]), result=result, max_changes=mc)
             make_hypotheses_neni(tuple(tokens[:i] + ('ни', token[2:]) + tokens[(i+1):]), result=result, max_changes=mc)
    return result


def make_hypotheses_what(tokens):
    pos = 0
    while pos < len(tokens):
        token = tokens[pos]
        if token.lower() in {'тоже', 'также'}:
            yield (tokens[:pos]
                   + [token[:-2]] + ['же']
                   + tokens[pos + 1:])
            pos += 1
        elif token.lower() == 'чтоб':
            yield (tokens[:pos]
                   + [token[:-1]] + ['б']
                   + tokens[pos + 1:])
            pos += 1
        elif token.lower() == 'чтобы':
            yield (tokens[:pos]
                   + [token[:-2]] + ['бы']
                   + tokens[pos + 1:])
            pos += 1
        elif pos + 1 < len(tokens) and token.lower() in {'то', 'так'} and tokens[pos + 1] == 'же':
            yield (tokens[:pos]
                   + [token + 'же']
                   + tokens[pos + 2:])
            pos += 2
        elif pos + 1 < len(tokens) and token.lower() == 'что' and tokens[pos + 1] in {'б', 'бы'}:
            yield (tokens[:pos]
                   + [token + tokens[pos + 1]]
                   + tokens[pos + 2:])
            pos += 2
        else:
            pos += 1


NO_COUNT_NUM = 1e-10
NO_COUNT_DEN = 1.0

ORDER_WEIGHTS = [None, 0.001, 0.01, 100, 1000, 10000]


def get_log_proba(toks, grams, order=3, prnt=False):
    log_proba = 1.0
    ps = []
    for i, word in enumerate(toks):
        num = 1e-20
        den = 1e-10
        for n in range(1, order+1):
            if i+1 >= n:
                num += ORDER_WEIGHTS[n] * grams[n].get('_'.join(toks[(i-n+1):(i+1)]), NO_COUNT_NUM)
                den += ORDER_WEIGHTS[n] * grams[n-1].get('_'.join(toks[(i-n+1):(i)]), NO_COUNT_DEN)
        p = num / den
        if prnt:
            print(num, den, '_'.join(toks[(i-1+1):(i+1)]), p)
        ps.append(np.log(p))
        log_proba += np.log(p)
    if prnt:
        print(ps)
    return log_proba


def rank_hypotheses(toks, hypotheses, grams, leven_penalty=0.2, order=5):
    scores = []
    for hypo in hypotheses:
        scores.append([' '.join(hypo), get_log_proba(hypo, grams=grams, order=order), len(hypo)])

    d = pd.DataFrame(scores)
    d.columns = ['text', 'lm', 'n']
    d['leven'] = d.text.apply(lambda x: textdistance.levenshtein(x, ' '.join(toks)))
    d['penalty'] = np.log(leven_penalty) * d.leven
    d['score'] = d.lm + d.penalty
    d.sort_values('score', ascending=False, inplace=True)
    return d


def denormalize(orig, original_tokens, filtered_tokens, new_tokens):
    if tuple(new_tokens) == tuple(filtered_tokens):
        # если замен не было, возвращем исходный текст, чтобы нечаянно его не попортить
        return orig
    result = ' '
    i0 = 0
    i1 = 0
    while i0 < len(original_tokens):
        # print('"{}  {} {}" "{}" '.format(result, i0, i1, original_tokens[i0]))
        prev_is_punct = result[-1] in '.?-:!'
        if i0 < len(original_tokens) and i1 < len(new_tokens) and original_tokens[i0].lower() == new_tokens[i1].lower():
            if not prev_is_punct:
                result = result + ' '
            result = result + original_tokens[i0]
            i0 += 1
            i1 += 1
        elif not original_tokens[i0].isalpha():
            if original_tokens[i0].isnumeric():
                if not prev_is_punct:
                    result = result + ' '
            result = result + original_tokens[i0]
            i0 += 1
        elif i0 < len(original_tokens) and i1 < len(new_tokens):
            # print('change "{}" -> "{}"'.format(original_tokens[i0], new_tokens[i1]))
            if abs(len(original_tokens[i0]) - len(new_tokens[i1])) <= 1: # probably, the same token, but with correction
                tok = new_tokens[i1]
                if original_tokens[i0][0].isupper():
                    tok = tok.capitalize()
                if not prev_is_punct:
                    result = result + ' '
                    result = result + tok
                i0 += 1
                i1 += 1
            elif len(original_tokens[i0]) > len(new_tokens[i1]):
                tok = new_tokens[i1] + ' ' + new_tokens[i1+1]
                if original_tokens[i0][0].isupper():
                    tok = tok.capitalize()
                if not prev_is_punct:
                    result = result + ' '
                    result = result + tok
                i0 += 1
                i1 += 2
            else:
                tok = new_tokens[i1]
                if original_tokens[i0][0].isupper():
                    tok = tok.capitalize()
                if not prev_is_punct:
                    result = result + ' '
                    result = result + tok
                i0 += 2
                i1 += 1
        else:
            #raise ValueError('{} is wrong with i0={}, i1={}'.format(result, i0, i1))
            # we don't know what to do and decide to do nothing
            return orig
    result = result.strip()
    while orig.endswith(' '):
        orig = orig[:(-1)]
        result = result + ' '
    return result


def make_ngram_correction(text, hypo_makers, grams):
    original_tokens = wordpunct_tokenize(text)
    filtered_tokens = [t.lower() for t in original_tokens if t.isalpha()]
    hypos = [filtered_tokens]
    for maker in hypo_makers:
        hypos.extend(maker(filtered_tokens))
    d = rank_hypotheses(filtered_tokens, hypos, grams=grams, leven_penalty=0.15, order=3, )
    new_text = d.text.iloc[0]
    result = denormalize(text, original_tokens, filtered_tokens, new_text.split())
    return result


def load_grams(path='models/gf3.pkl'):
    with open(path, 'rb') as f:
        gf3 = pickle.load(f)
    return gf3


if __name__ == '__main__':
    gf3 = load_grams()
