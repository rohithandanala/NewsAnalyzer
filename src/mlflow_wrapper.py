import mlflow
import mlflow.pyfunc

class TextClassifierWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import joblib
        self.model = joblib.load(context.artifacts["model"])
        self.vectorizer = joblib.load(context.artifacts["vectorizer"])

    def predict(self, context, model_input):
        import pandas as pd
        if isinstance(model_input, str):
            model_input = pd.DataFrame([model_input], columns=["text"])
        elif isinstance(model_input, list):
            model_input = pd.DataFrame(model_input, columns=["text"])
        text_transformed = self.vectorizer.transform(model_input["text"])
        return self.model.predict(text_transformed)