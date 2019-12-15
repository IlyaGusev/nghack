import pandas as pd
import pickle
import textdistance
import numpy as np
from nltk import wordpunct_tokenize


def make_hypotheses_neni(tokens, result=None, max_changes=2):
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


def rank_hypotheses(toks, hypotheses, leven_penalty=0.2, order=5):
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


if __name__ == '__main__':
    with open('models/gf3.pkl', 'rb') as f:
        gf3 = pickle.load(f)
