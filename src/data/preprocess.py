import os
import pandas as pd


def preprocess_data():
    BASE_DIR = os.path.join(os.getcwd())
    blood_glucose_filepath = os.path.join(BASE_DIR, 'data/blood-glucose-data.csv')
    heart_rate_filepath = os.path.join(BASE_DIR, 'data/heart-rate-data.csv')
    blood_glucose_dataset = pd.read_csv(blood_glucose_filepath, parse_dates=['point_timestamp'],
                                        index_col='point_timestamp')
    blood_glucose_dataset_parsed = blood_glucose_dataset.asfreq(freq='1Min', method='ffill')
    heart_rate_dataset = pd.read_csv(heart_rate_filepath, parse_dates=['point_timestamp'],
                                     index_col='point_timestamp')

    blood_glucose_dataset_parsed['point_timestamp_yyyymmdd_hh_mm'] = blood_glucose_dataset_parsed.index.map(
        lambda x: x.strftime('%Y-%m-%d %H:%M'))

    heart_rate_dataset['point_timestamp_yyyymmdd_hh_mm'] = heart_rate_dataset.index.map(
        lambda x: x.strftime('%Y-%m-%d %H:%M'))
    bg_hr_dataset = pd.merge(blood_glucose_dataset_parsed, heart_rate_dataset,
                             on=['point_timestamp_yyyymmdd_hh_mm'], how='outer', indicator=True)
    bg_hr_dataset.index = pd.DatetimeIndex(bg_hr_dataset['point_timestamp_yyyymmdd_hh_mm'])
    bg_hr_dataset.index.name = 'point_timestamp_idx'
    bg_hr_dataset.loc[(bg_hr_dataset._merge == 'left_only'), 'point_value'] = heart_rate_dataset.point_value.mean()
    bg_hr_dataset = bg_hr_dataset.drop(columns=['timezone_offset_x', 'point_timestamp_yyyymmdd_hh_mm',
                                                'timezone_offset_y', '_merge'])
    print(bg_hr_dataset.head())
    # bg_hr_dataset = bg_hr_dataset['point_value(mg/dL)', 'point_value']
    bg_hr_dataset.columns = ['glucose_level_value', 'heart_rate_value']
    bg_hr_dataset.to_csv("data/preprocessed_data.csv", sep=",")

    return True
