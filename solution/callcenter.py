import sys
import pandas as pd
import numpy as np
import catboost


DUR_RU = 'Длительность разговора с оператором, сек'
DUR_EN = 'oper_duration'

RU_COLS = [
    'Время начала вызова', 'Время окончания вызова', 'Время постановки в очередь',
    'Время переключения на оператора', 'Время окончания разговора с оператором',
]
EN_COLS = ['call_start_time', 'call_end_time', 'queue_time', 'oper_start_time', 'oper_end_time']

zero_time = pd.to_datetime('00:00:00')

SEC_PER_DAY = 60*60*24


def extract_features(data):
    times = data[RU_COLS].apply(pd.to_datetime)
    times.columns = EN_COLS

    abs_times = times.apply(lambda x: (x - zero_time).dt.total_seconds()).fillna(-9999)
    abs_times.columns = [c + '_abs' for c in EN_COLS]

    day = abs_times / SEC_PER_DAY * 2 * np.pi
    hour = (abs_times % (24 * 60)) / (24 * 60) * 2 * np.pi
    minute = (abs_times % 60) / 60 * 2 * np.pi
    day_sines = np.sin(day)
    day_cosines = np.cos(day)
    hour_sines = np.sin(hour)
    hour_cosines = np.cos(hour)
    minute_sines = np.sin(minute)
    minute_cosines = np.cos(minute)

    day_sines.columns = ['day_sin__' + c for c in EN_COLS]
    day_cosines.columns = ['day_cos__' + c for c in EN_COLS]
    hour_sines.columns = ['hour_sin__' + c for c in EN_COLS]
    hour_cosines.columns = ['hour_cos__' + c for c in EN_COLS]
    minute_sines.columns = ['minute_sin__' + c for c in EN_COLS]
    minute_cosines.columns = ['minute_cos__' + c for c in EN_COLS]

    null_times = times.isnull().astype(int)
    null_times.columns = [c + "_miss" for c in EN_COLS]

    diffs = pd.DataFrame(index=times.index)
    for i, c1 in enumerate(EN_COLS):
        for j, c2 in enumerate(EN_COLS[(i + 1):]):
            diffs['delta_{}_{}'.format(c1, c2)] = (times[c2] - times[c1]).dt.total_seconds().fillna(-9999)
    x = pd.concat(
        [abs_times, day_sines, day_cosines, hour_sines, hour_cosines, minute_sines, minute_cosines, null_times, diffs],
        axis=1)
    x[DUR_EN] = data[DUR_RU].fillna(-9999)
    x[DUR_EN + '_miss'] = data[DUR_RU].isnull().astype(int)

    devia = x['delta_oper_start_time_oper_end_time'] - x[DUR_EN]
    devia[x['oper_duration_miss'] == 1] = 0
    devia[x['delta_oper_start_time_oper_end_time'] < 0] = 0

    x['oper_time_deviation'] = devia

    return x


if __name__ == '__main__':
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    
    df_test = pd.read_csv(input_csv, index_col='id')

    model = catboost.CatBoostClassifier()
    model.load_model('../models/call_center_catboost.model')

    x_test = extract_features(df_test)
    prediction = (model.predict_proba(x_test)[:, 1] > 0.42).astype(int)

    df_test['Метка'] = prediction
    df_test.to_csv(output_csv)
