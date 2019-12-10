import os
import pandas as pd
from keras.layers import Dense, Dropout, LeakyReLU
from keras.layers import LSTM
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from sklearn.preprocessing import MinMaxScaler
from src.visualization import clark_error_analysis


def generate_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def build_parse_dataset(file_path):
    # print(file_path)
    dataframe = pd.read_csv(file_path, parse_dates=['timestamp'],
                            index_col='timestamp')

    values = dataframe[['heart_rate_value', 'glucose_level_value']].values
    return values


def scale_reframed_dataset(dataset):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(dataset)
    reframed = generate_supervised(scaled, 1, 1)
    values = reframed.values
    return values, scaler


def split_train_test(dataset, train_size, test_size):
    # print(f'train size: {train_size}')
    # print(f'test size: {test_size}')
    # print(f'ds shape: {dataset.shape}')
    train = dataset[:train_size, :]
    test = dataset[test_size:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    # print(train_X.shape, train_y.shape, test_X.shape)
    return train, test, train_X, train_y, test_X, test_y


def build_lstm_model(input_neurons, loss_func, optimizer):
    BASE_DIR = os.path.join(os.getcwd())
    parsed_dataset = build_parse_dataset(os.path.join(BASE_DIR, 'data/preprocessed_data2.csv'))
    eval_ds = parsed_dataset[-12:, :]
    from_this = parsed_dataset[-12, :]

    parsed_dataset = parsed_dataset[:-12, :]

    dataset_values, scaler = scale_reframed_dataset(parsed_dataset)
    train_size = int(len(dataset_values) * 0.27)
    test_size = len(dataset_values) - train_size
    train, test, train_X, train_y, test_X, test_y = split_train_test(dataset_values, train_size, test_size)

    # print(f'train size: {train.shape} & test size {test.shape}')
    # print(f'train x: {train_X.shape} & train y {train_y.shape}')
    model = Sequential()
    # input_neurons = 50
    model.add(LSTM(input_neurons, input_shape=(train_X.shape[1], train_X.shape[2]),
                   activation='elu'))
    model.add(LeakyReLU(alpha=0.5))
    model.add(Dense(1,))
    optimizer = Adam(learning_rate=0.0020)
    model.compile(loss=loss_func, optimizer=optimizer)
    return model, train, test, train_X, train_y, test_X, test_y, scaler, eval_ds, from_this
