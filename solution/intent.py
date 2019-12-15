import argparse
import fasttext
import pandas


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input.csv')
    parser.add_argument('output', help='output.csv')
    parser.add_argument('--labels', help='fastText labels', required=True)
    parser.add_argument('--model', help='fastText model', required=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    to_label = pandas.read_csv(args.labels).to_dict('records')
    to_label = {row['fasttext_label']:row['label'] for row in to_label}

    df_test = pandas.read_csv(args.input, index_col='id')

    model = fasttext.load_model(args.model)
    preds = model.predict(list(df_test.text.values))

    df_test['label'] = list(map(lambda x: to_label[x[0]], preds[0]))
    df_test.to_csv(args.output)
