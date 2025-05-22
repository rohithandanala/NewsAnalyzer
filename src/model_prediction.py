
def predict_model(model, x_test):
    prediction = model.predict(x_test)
    return prediction