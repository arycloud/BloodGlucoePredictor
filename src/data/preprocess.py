import os
import pandas as pd


def preprocess_data():
    base_path = os.path.join(os.getcwd())
    gl_dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')
    hr_dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')
    gl = pd.read_csv(base_path + "data/blood-glucose-data.csv", parse_dates=['point_timestamp'],
                     date_parser=gl_dateparse,
                     index_col='point_timestamp')
    gl['glucose_value'] = gl['point_value(mg/dL)']
    gl['ts'] = gl.index
    del gl['point_value(mg/dL)']
    hr = pd.read_csv(base_path + "data/heart-rate-data.csv", parse_dates=['point_timestamp'], date_parser=hr_dateparse,
                     index_col='point_timestamp')
    hr['hr_rate'] = hr['point_value']
    hr['ts'] = hr.index
    del hr['point_value']

    date_rng = pd.date_range(start=gl.index.min(), end=gl.index.max(), freq='min')
    ts_df = pd.DataFrame(date_rng.tolist(), columns=['ts'])

    gl['ts'] = gl.index.map(lambda x: x.strftime('%Y-%m-%d %H:%M'))
    hr['ts'] = hr.index.map(lambda x: x.strftime('%Y-%m-%d %H:%M'))
    ts_df['ts'] = ts_df['ts'].map(lambda x: x.strftime('%Y-%m-%d %H:%M'))

    hr_filled = pd.merge(ts_df, hr, on='ts', how='outer')
    hr_filled.fillna({'hr_rate': hr['hr_rate'].mean()}, inplace=True)
    hr_filled['hr_rate_ma'] = hr_filled['hr_rate'].ewm(span=5, adjust=True).mean()

    glucose_hr_merged_df = pd.merge(gl, hr_filled, on=['ts'], how='inner')
    glucose_hr_merged_df.index = pd.DatetimeIndex(glucose_hr_merged_df['ts'])

    df = glucose_hr_merged_df[['hr_rate_ma', 'ts', 'glucose_value']].to_csv(base_path + "/preprocessed_data2.csv",
                                                                            sep=",", index=False,
                                                                            header=['heart_rate_value', 'timestamp',
                                                                                    'glucose_level_value'])
    return True
