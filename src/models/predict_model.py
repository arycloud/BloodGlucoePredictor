import numpy as np
from sklearn.metrics import mean_squared_error
from src.models.build_model import generate_supervised
from src.visualization import clark_error_analysis, grid_analysis


def next_hour_predictions(heart_rate, values_at_t_minus1, model, scaler, next_n, timestep=5):
    preds = []
    print(heart_rate.shape)
    print(values_at_t_minus1.shape)
    for t in range(next_n):
        values_at_t = np.array([heart_rate[t], 0.0]).reshape(1, 2)
        values = np.concatenate((values_at_t_minus1, values_at_t), axis=0)
        values_scaled = scaler.transform(values)
        x = generate_supervised(values_scaled, 1, 1).values[:, :3]
        x = x.reshape(1, 1, x.shape[1])

        yhat = model.predict(x)
        inv_y = np.concatenate((values_at_t, yhat), axis=1)
        inv_y = scaler.inverse_transform(inv_y[:, -2:])
        inv_y = inv_y[:, -1][0]
        preds.append(inv_y)
        values_at_t_minus1 = np.array([heart_rate[t], inv_y]).reshape((1, 2))
    return [glucose_level for iter, glucose_level in enumerate(preds) if iter % timestep == 0]


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
    # plt.savefig('test_predictions.png')
    # print(new_test_X[-1])
    acc = grid_analysis.zone_accuracy(inv_y, inv_yhat, detailed=True)
    # print(acc)
    print('Parkes:')
    for val in acc:
        print(100 * val)
