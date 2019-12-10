import numpy as np
from src.data import preprocess
from src.models import build_model, train_model, predict_model
from src.visualization import model_history_plot, clark_error_analysis, grid_analysis

if __name__ == '__main__':
    # preprocess the data
    # preprocess.preprocess_data()
    # model parameters
    NO_OF_NEURONS = 50
    LOSS_FUNCTION = 'mean_squared_error'
    OPTIMIZER = 'adam'
    model, train, test, train_X, train_y, test_X, test_y, scaler, eval_ds, from_this = build_model.build_lstm_model(
        NO_OF_NEURONS, LOSS_FUNCTION, OPTIMIZER
    )
    model_history = train_model.fit_model(model, test, train_X, train_y)
    model_history_plot.plot_history(model_history)
    predict_model.make_prediction(model, scaler, test_X, test_y)
    # print(eval_ds.shape)
    # print(eval_ds[:, 1])
    # print(eval_ds[2:])
    # preds, ref_vals = predict_model.next_hour_predictions(eval_ds[:, 0],
    #                                             np.array([from_this[1], from_this[0]]).reshape((1, 2)),
    #                                             model, scaler, 60, 5)
    pred_interval = 1
    preds = predict_model.next_hour_predictions(eval_ds[:, 0], np.array([from_this[1], from_this[0]]).reshape((1, 2)),
                                                model, scaler, 12, pred_interval)
    # print("Next hour predictions: ", preds)
    actules = [g for iter, g in enumerate(eval_ds[:, 1]) if iter % pred_interval == 0]
    # print(actules)
    plt, zone = clark_error_analysis.clarke_error_grid(actules, preds, 'Validation')
    plt.show()
    # print("Ref values: ", ref_vals)
    # x = list(x)
    # plt, zone = clark_error_analysis.clarke_error_grid(ref_vals, preds, 'Next Hour Predictions')
    # plt.show()
    print(zone)
    # print(type(actules))
    # print(type(preds))
    # acc = grid_analysis.zone_accuracy(actules, preds)
    # print('Parkes:')
    # print(acc)

    model.save("models/trained_model.h5")
