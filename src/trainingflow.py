from metaflow import FlowSpec, step, Parameter
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os

class TrainingFlow(FlowSpec):
    random_seed = Parameter("random_seed", default=42)
    test_size = Parameter("test_size", default=0.2)
    ridge_alpha = Parameter("ridge_alpha", default=1.0)
    rf_n_estimators = Parameter("rf_n_estimators", default=100)

    @step
    def start(self):
        # import dataprocessing as dataprocessing
        # self.preprocess_data = dataprocessing.preprocess_data
        self.next(self.ingest_data)

    @step
    def ingest_data(self):
        self.data = pd.read_csv("~/Downloads/mlops/data/songs_normalize.csv")  
        print(f"Data shape: {self.data.shape}")
        self.next(self.transform_data)

    @step
    def transform_data(self):
        from dataprocessing import preprocess_data
        self.X, self.y = preprocess_data(self.data)
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_seed
        )
        self.next(self.train_model)

    @step
    def train_model(self):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_squared_error

        rf = RandomForestRegressor(n_estimators=self.rf_n_estimators, random_state=self.random_seed)
        ridge = Ridge(alpha=self.ridge_alpha)

        rf.fit(self.X_train, self.y_train)
        ridge.fit(self.X_train, self.y_train)

        rf_preds = rf.predict(self.X_val)
        ridge_preds = ridge.predict(self.X_val)

        self.models = {
            "RandomForest": {"model": rf, "rmse": mean_squared_error(self.y_val, rf_preds)},
            "Ridge": {"model": ridge, "rmse": mean_squared_error(self.y_val, ridge_preds)}
        }
        self.next(self.choose_best_model)

    @step
    def choose_best_model(self):
        self.best_model_name = min(self.models, key=lambda k: self.models[k]["rmse"])
        self.best_model = self.models[self.best_model_name]["model"]
        self.best_rmse = self.models[self.best_model_name]["rmse"]
        print(f"best model: {self.best_model_name} with RMSE {self.best_rmse:.4f}")
        self.next(self.register_model)

    @step
    def register_model(self):
        mlflow.set_tracking_uri("http://127.0.0.1:5000")  
        mlflow.set_experiment("LocalModelTraining")

        with mlflow.start_run(run_name=f"{self.best_model_name}_run") as run:
            mlflow.log_param("model_type", self.best_model_name)
            mlflow.log_param("random_seed", self.random_seed)
            mlflow.log_param("test_size", self.test_size)
            mlflow.log_param("ridge_alpha", self.ridge_alpha)
            mlflow.log_param("rf_n_estimators", self.rf_n_estimators)
            mlflow.log_metric("rmse", self.best_rmse)

            model_dir = f"models/{self.best_model_name}.pkl"
            os.makedirs("models", exist_ok=True)
            joblib.dump(self.best_model, model_dir)

            mlflow.sklearn.log_model(self.best_model, "model")

            model_uri = f"runs:/{run.info.run_id}/model"
            mlflow.register_model(model_uri, self.best_model_name)
            
            print(f"registered {self.best_model_name} to MLflow with RMSE {self.best_rmse:.4f}")
        self.next(self.end)

    @step
    def end(self):
        print("flow complete!!!")

if __name__ == '__main__':
    TrainingFlow()
