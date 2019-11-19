import numpy as np
from sklearn.metrics import mean_squared_error

from src.visualization import clark_error_analysis


def next_hour_predictions(model, scaler, test_X, test_y):
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # invert scaling for forecast
    inv_yhat = np.concatenate((test_X[:60], yhat[:60]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat[:, -2:])
    inv_yhat = inv_yhat[:, -1]

    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_X[:60], test_y[:60]), axis=1)
    inv_y = scaler.inverse_transform(inv_y[:, -2:])
    inv_y = inv_y[:, -1]
    inv_yhat_12 = inv_yhat[4::5]
    inv_y_12 = inv_y[4::5]
    rmse = np.sqrt(mean_squared_error(inv_y_12, inv_yhat_12))
    print('RMSE: %.3f' % rmse)
    plt, zone = clark_error_analysis.clarke_error_grid(inv_y_12, inv_yhat_12, 'Next Hour Predictions')
    print(f'Zone is {zone}')
    plt.show()
    plt.savefig('next_hour_predictions.png')


def make_prediction(model, scaler, test_X, test_y):
    new_test_X = test_X
    new_test_y = test_y
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # invert scaling for forecast
    inv_yhat = np.concatenate((test_X, yhat), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat[:, -2:])
    inv_yhat = inv_yhat[:, -1]

    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_X, test_y), axis=1)
    inv_y = scaler.inverse_transform(inv_y[:, -2:])
    inv_y = inv_y[:, -1]
    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)

    plt, zone = clark_error_analysis.clarke_error_grid(inv_y, inv_yhat, 'Testing')
    print(f'Zone is {zone}')
    plt.show()
    plt.savefig('test_predictions.png')
    next_hour_predictions(model, scaler, new_test_X, new_test_y)
