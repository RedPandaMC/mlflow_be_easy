"""Trains and Tests A Model"""

import datetime

import mlflow
import mlflow.sklearn
import optuna
from optuna.pruners import WilcoxonPruner
from optuna.samplers import TPESampler
from prefect import flow, task
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

mlflow.set_tracking_uri("sqlite:///mlflow_db.sqlite3")


def optimize_model(data_frame):

    data = data_frame.drop(columns=["target"])
    target = data_frame["target"]

    sample_size = 100
    sampled_data = data.sample(n=sample_size, random_state=42)
    sampled_target = target[sampled_data.index]

    train_x, valid_x, train_y, valid_y = train_test_split(
        sampled_data, sampled_target, test_size=0.33
    )

    def objective(trial):
        param = {
            "max_depth": trial.suggest_int("max_depth", 1, 4),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 0, 50),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        }

        model = XGBClassifier(**param, use_label_encoder=False, eval_metric="mlogloss")

        with mlflow.start_run(nested=True):
            mlflow.log_params(param)
            model.fit(train_x, train_y)

            preds = model.predict(valid_x)
            accuracy = accuracy_score(valid_y, preds)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(model, "model")

        return accuracy

    pruner = WilcoxonPruner(p_threshold=0.15)
    sampler = TPESampler()
    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///optuna_db.sqlite3",
        study_name=f"mlops_exam_exercise-{datetime.datetime.now()}",
        pruner=pruner,
        sampler=sampler,
    )
    study.optimize(objective, n_trials=20)

    return study.best_trial


def train_and_log_best_model(data_frame, best_trial):
    with mlflow.start_run():
        data = data_frame.drop(columns=["target"])
        target = data_frame["target"]

        train_x, test_x, train_y, test_y = train_test_split(
            data, target, test_size=0.3, random_state=42
        )

        model = XGBClassifier(
            **best_trial.params, use_label_encoder=False, eval_metric="mlogloss"
        )
        model.fit(train_x, train_y)

        preds = model.predict(test_x)
        accuracy = accuracy_score(test_y, preds)

        mlflow.log_params(best_trial.params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")

        run_id = mlflow.active_run().info.run_id
        mlflow.register_model(f"runs:/{run_id}/model", "Model")

    return model, accuracy


def create_model(data_frame):
    best_trial = optimize_model(data_frame)
    _, accuracy = train_and_log_best_model(data_frame, best_trial)
    print(f"Best model accuracy: {accuracy}")
