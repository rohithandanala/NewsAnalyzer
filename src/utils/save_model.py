from joblib import dump
import os

def save_trained_model(model, vectorizer, model_name, save_path):

    #creating directory to save model and vectorizer
    os.makedirs(f"{save_path}/{model_name}", exist_ok=True)
    model_path = f"{save_path}{model_name}/model.joblib"
    vectorizer_path = f"{save_path}{model_name}/vectorizer.joblib"

    dump(model, model_path)
    print(f"saved model at {model_path}")
    dump(vectorizer, vectorizer_path)
    print(f"saved vectorizer at {vectorizer_path}")

    
    print(f"Succesfully saved model to Models/{model_name}.joblib")

    return {"model": model_path,
                "vectorizer": vectorizer_path}