from metaflow import FlowSpec, step, Parameter
import pandas as pd
import mlflow
import joblib
import os

class ScoringFlow(FlowSpec):
    input_data_path = Parameter("input_data_path", help="Path to CSV with new data")
    model_name = Parameter("model_name", default="RandomForest")

    @step
    def start(self):
        print(f"scoring flow started for model: {self.model_name}")
        self.next(self.ingest_data)

    @step
    def ingest_data(self):
        self.new_data = pd.read_csv(self.input_data_path)
        print(f"new data shape: {self.new_data.shape}")
        self.next(self.transform_data)

    @step
    def transform_data(self):
        from dataprocessing import preprocess_data
        X, _ = preprocess_data(self.new_data)  # ignore y if it exists
        self.X_new = X
        print(f"transformed new data: {self.X_new.shape}")
        self.next(self.load_model)

    @step
    def load_model(self):
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        model_uri = f"models:/{self.model_name}/latest"
        print(f"loading model from MLflow: {model_uri}")
        self.model = mlflow.sklearn.load_model(model_uri)
        self.next(self.predict)

    @step
    def predict(self):
        self.preds = self.model.predict(self.X_new)
        print(f"predictions made on {len(self.preds)} records")
        self.next(self.output_results)

    @step
    def output_results(self):
        output_df = self.X_new.copy()
        output_df["prediction"] = self.preds
        output_path = "predictions.csv"
        output_df.to_csv(output_path, index=False)
        print(f"predictions saved to {output_path}")
        self.next(self.end)

    @step
    def end(self):
        print("scoring flow complete!!!")

if __name__ == '__main__':
    ScoringFlow()
