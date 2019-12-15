import argparse
import fasttext
import pandas


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input.csv')
    parser.add_argument('output', help='output.csv')
    parser.add_argument('--model', help='fastText model', default='models/intent.ftz')

    return parser.parse_args()


TO_LABEL = {
    '__label__0': 'FAQ - интернет',
    '__label__1': 'FAQ - тарифы и услуги',
    '__label__2': 'SIM-карта и номер',
    '__label__3': 'Баланс',
    '__label__4': 'Личный кабинет',
    '__label__5': 'Мобильные услуги',
    '__label__6': 'Мобильный интернет',
    '__label__7': 'Оплата',
    '__label__8': 'Роуминг',
    '__label__9': 'Устройства',
    '__label__10': 'запрос обратной связи',
    '__label__11': 'мобильная связь - зона обслуживания',
    '__label__12': 'мобильная связь - тарифы',
    '__label__13': 'тарифы - подбор'
}


def to_label(ft):
    return TO_LABEL.get(ft)


if __name__ == '__main__':
    args = parse_args()

    df_test = pandas.read_csv(args.input, index_col='id')

    model = fasttext.load_model(args.model)
    preds = model.predict(list(df_test.text.values))

    df_test['label'] = list(map(lambda x: to_label(x[0]), preds[0]))
    df_test.to_csv(args.output)
