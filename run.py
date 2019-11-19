from src.data import preprocess
from src.models import build_model, train_model, predict_model
from src.visualization import model_history_plot

if __name__ == '__main__':
    # preprocess the data
    preprocess.preprocess_data()
    # model parameters
    NO_OF_NEURONS = 50
    LOSS_FUNCTION = 'mean_squared_error'
    OPTIMIZER = 'adam'
    model, train, test, train_X, train_y, test_X, test_y, scaler = build_model.build_lstm_model(
        NO_OF_NEURONS, LOSS_FUNCTION, OPTIMIZER
    )
    model_history = train_model.fit_model(model, test, train_X, train_y)
    model_history_plot.plot_history(model_history)
    predict_model.make_prediction(model, scaler, test_X, test_y)
    model.save("models/trained_model.h5")
