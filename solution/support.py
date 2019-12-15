import sys
import pandas


if __name__ == '__main__':
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    
    df_test = pandas.read_csv(input_csv, index_col='id')
    df_test['label'] = 'positive' # the most popular label
    df_test.to_csv(output_csv)