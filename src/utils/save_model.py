from joblib import dump

def save_trained_model(model, model_name):
    dump(model, f'Models/{model_name}.joblib')
    print(f"Succesfully saved model to Models/{model_name}.joblib")