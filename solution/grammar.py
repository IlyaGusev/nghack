import sys
import pandas


if __name__ == '__main__':
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    
    df_test = pandas.read_csv(input_csv, index_col='id')
    df_test['correct_sentence'] = df_test['sentence_with_a_mistake']
    df_test.to_csv(output_csv)