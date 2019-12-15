import sys
import fasttext
import pandas


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
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    df_test = pandas.read_csv(input_csv, index_col='id')

    model_path = "models/intent.ftz"
    model = fasttext.load_model(model_path)
    texts = df_test["text"].tolist()
    texts = [str(text).replace("\n", " ") for text in texts]
    preds = model.predict(texts)

    labels = [to_label(label[0]) for label in preds[0]]
    df_test['label'] = labels
    df_test.to_csv(output_csv)
