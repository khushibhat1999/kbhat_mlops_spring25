from metaflow import FlowSpec, step, Parameter, retry, timeout, catch, conda
import pandas as pd
import mlflow
import joblib
import os

class ScoringFlow(FlowSpec):
    input_data_path = Parameter("input_data_path", help="Path to CSV with new data")
    model_name = Parameter("model_name", default="RandomForest")

    @conda(libraries={"pandas": "1.5.3", "mlflow": "2.1.1"})
    @retry(times=2)
    @timeout(seconds=60)
    @catch(var="error")
    @step
    def start(self):
        print(f"scoring flow started for model: {self.model_name}")
        self.next(self.ingest_data)

    @conda(libraries={"pandas": "1.5.3"})
    @retry(times=2)
    @timeout(seconds=120)
    @catch(var="error")
    @step
    def ingest_data(self):
        try:
            self.new_data = pd.read_csv(self.input_data_path)
            print(f"new data shape: {self.new_data.shape}")
        except Exception as e:
            print(f"failed to read input data: {e}")
            raise e
        self.next(self.transform_data)

    @conda(libraries={"pandas": "1.5.3", "scikit-learn": "1.2.2"})
    @retry(times=2)
    @timeout(seconds=180)
    @catch(var="error")
    @step
    def transform_data(self):
        try:
            from dataprocessing import preprocess_data
            X, _ = preprocess_data(self.new_data)  # ignore y if it exists
            self.X_new = X
            print(f"transformed new data: {self.X_new.shape}")
        except Exception as e:
            print(f"data transformation failed: {e}")
            raise e
        self.next(self.load_model)

    @conda(libraries={"mlflow": "2.1.1", "scikit-learn": "1.2.2"})
    @retry(times=2)
    @timeout(seconds=120)
    @catch(var="error")
    @step
    def load_model(self):
        try:
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            model_uri = f"models:/{self.model_name}/latest"
            print(f"loading model from MLflow: {model_uri}")
            self.model = mlflow.sklearn.load_model(model_uri)
        except Exception as e:
            print(f"model load failed: {e}")
            raise e
        self.next(self.predict)

    @conda(libraries={"scikit-learn": "1.2.2"})
    @retry(times=2)
    @timeout(seconds=120)
    @catch(var="error")
    @step
    def predict(self):
        try:
            self.preds = self.model.predict(self.X_new)
            print(f"predictions made on {len(self.preds)} records")
        except Exception as e:
            print(f"prediction failed: {e}")
            raise e
        self.next(self.output_results)

    @conda(libraries={"pandas": "1.5.3"})
    @retry(times=2)
    @timeout(seconds=90)
    @catch(var="error")
    @step
    def output_results(self):
        try:
            output_df = self.X_new.copy()
            output_df["prediction"] = self.preds
            output_path = "predictions.csv"
            output_df.to_csv(output_path, index=False)
            print(f"predictions saved to {output_path}")
        except Exception as e:
            print(f"failed to write predictions: {e}")
            raise e
        self.next(self.end)

    @step
    def end(self):
        print("scoring flow complete!!!")

if __name__ == '__main__':
    ScoringFlow()