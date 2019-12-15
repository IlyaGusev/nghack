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
        scores.append([' '.join(hypo), get_log_proba(hypo, order=order), len(hypo)])

    d = pd.DataFrame(scores)
    d.columns = ['text', 'lm', 'n']
    d['leven'] = d.text.apply(lambda x: textdistance.levenshtein(x, ' '.join(toks)))
    d['penalty'] = np.log(leven_penalty) * d.leven
    d['score'] = d.lm + d.penalty
    d.sort_values('score', ascending=False, inplace=True)
    return d


def denormalize(orig, toks0, toks):
    result = ' '
    i0 = 0
    i1 = 0
    while i0 < len(toks0):
        #print('"{}  {} {}" "{}" '.format(result, i0, i1, toks0[i0]))
        prev_is_punct = result[-1] in '.?-:!'
        if i0 < len(toks0) and i1 < len(toks) and toks0[i0].lower() == toks[i1].lower():
            if not prev_is_punct:
                result = result + ' '
            result = result + toks0[i0]
            i0 += 1
            i1 += 1
        elif not toks0[i0].isalpha():
            if toks0[i0].isnumeric():
                if not prev_is_punct:
                    result = result + ' '
            result = result + toks0[i0]
            i0 += 1
        elif i0 < len(toks0) and i1 < len(toks):
            #print('change "{}" -> "{}"'.format(toks0[i0], toks[i1]))
            if abs(len(toks0[i0]) - len(toks[i1])) <= 1: # probably, the same token, but with correction
                tok = toks[i1]
                if toks0[i0][0].isupper():
                    tok = tok.capitalize()
                if not prev_is_punct:
                    result = result + ' '
                    result = result + tok
                i0 += 1
                i1 += 1
            elif len(toks0[i0]) > len(toks[i1]):
                tok = toks[i1] + ' ' + toks[i1+1]
                if toks0[i0][0].isupper():
                    tok = tok.capitalize()
                if not prev_is_punct:
                    result = result + ' '
                    result = result + tok
                i0 += 1
                i1 += 2
            else:
                tok = toks[i1]
                if toks0[i0][0].isupper():
                    tok = tok.capitalize()
                if not prev_is_punct:
                    result = result + ' '
                    result = result + tok
                i0 += 2
                i1 += 1
        else:
            raise ValueError('{} is wrong with i0={}, i1={}'.format(result, i0, i1))
    result = result.strip()
    while orig.endswith(' '):
        orig = orig[:(-1)]
        result = result + ' '
    return result


def make_ngram_correction(text, hypo_makers, grams):
    toks0 = wordpunct_tokenize(text)
    toks = [t.lower() for t in toks0 if t.isalpha()]
    hypos = [toks]
    for maker in hypo_makers:
        hypos.extend(maker(toks))
    d = rank_hypotheses(toks, hypos, grams=grams, leven_penalty=0.15, order=3, )
    new_text = d.text.iloc[0]
    result = denormalize(text, toks0, new_text.split())
    return result


def load_grams(path='models/gf3.pkl'):
    with open(path, 'rb') as f:
        gf3 = pickle.load(f)
    return gf3


if __name__ == '__main__':
    gf3 = load_grams()
