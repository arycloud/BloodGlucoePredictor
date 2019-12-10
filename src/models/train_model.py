def fit_model(model, test, train_X, train_y):
    test_X, test_y = test[:, :-1], test[:, -1]
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    history = model.fit(
        train_X,
        train_y,
        epochs=150, batch_size=80,
        validation_data=(test_X, test_y),
        verbose=2, shuffle=False)
    print(history)
    return history
