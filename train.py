import pathlib
from dataclasses import dataclass

import dvc.api
import fire
import pandas as pd
import skops.io as sio
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from sklearn.tree import DecisionTreeClassifier
from spoty_gp.transformers import get_preprocessor


@dataclass
class ModelParams:
    random_state: int
    max_depth: int
    min_samples_leaf: int


@dataclass
class MlFlowAddress:
    host: str
    port: str


@dataclass
class BaseConfig:
    params: ModelParams
    mlflow_address: MlFlowAddress


# cs = ConfigStore.instance()
# cs.store(name="base_config", node=BaseConfig)


# если через fire указали брать датасет из локальной репы, но его нет, завершить с ошибкой
# если через fire указали брать датасет из ремоута - берём из ремоута
# по умолчанию пробовать взять датасет из локальной репы

# TODO: добавить проверку на наличие последней (!) версии датасета в локальной репе (fire case 1)


# @hydra.main(version_base=None, config_path="spotify-song-genre-prediction/conf", config_name="config")
# def _main(cfg: BaseConfig) -> None:
# 	# загрузка датасета из dvc
# 	print(cfg.params.random_state)

# 	# for df_name in ["train", "test"]:
# 	# 	with dvc.api.open(f"data/train_test/{df_name}.csv") as f:
# 	# 		df = pd.read_csv(f)
# 	# 		df.to_csv("data//train.csv", index=False)


# 	# препроцессинг

# 	# сохранение модели и метрик модели


def resolve_data_location(search_location, verbose):
    data_dir = pathlib.Path("data/train_test/")
    if search_location == "local":
        if verbose:
            print("Searching for a local copy of data...")
        is_data_present = not bool(
            {"train.csv", "test.csv"}.difference({i.name for i in data_dir.iterdir()})
        )  # true if both train.csv and test.csv are in the data_dir
        if is_data_present:
            if verbose:
                print("Found local copy of data, using it for training dataset.")
            with open(data_dir / "train.csv") as train_f:
                train_df = pd.read_csv(train_f, index_col="index")
            return train_df
        else:
            # TODO: добавить флаг strict_location={True, False}
            # TODO: реализовать перевод поиска в remote, если стоит флаг strict_location=False
            raise Exception(
                "Some of the data {'train.csv', 'test.csv'} is not found locally. Try running with a search=remote option."
            )
    elif search_location == "remote":
        train_path = str(data_dir / "train.csv")
        test_path = str(data_dir / "test.csv")

        if verbose:
            print("Searching for data in the remote...")
        with dvc.api.open(train_path) as train_f, dvc.api.open(test_path) as test_f:
            if verbose:
                print("Collecting training data...")
            train_df = pd.read_csv(train_f, index_col="index")
            if verbose:
                print("Collecting testing data...")
            test_df = pd.read_csv(test_f, index_col="index")

            if verbose:
                print("Saving data to a local storage...")
        train_df.to_csv(train_path, index_label="index", index=False)
        test_df.to_csv(test_path, index_label="index", index=False)
        return train_df
    else:
        raise Exception("Parameter 'search' should be in {'local', 'remote'}")


def get_trained(X, y, use_old_hyperparams, verbose, cfg):
    if use_old_hyperparams:
        if verbose:
            print("Training the model with existing hyperparameters...")
        model = DecisionTreeClassifier(
            random_state=cfg.params.random_state,
            max_depth=cfg.params.max_depth,
            min_samples_leaf=cfg.params.min_samples_leaf,
        )
    else:
        raise Exception(
            f"Case use_old_hyperparams={use_old_hyperparams} is not implemented. Use use_old_hyperparams=True insead"
        )
    model.fit(X, y)
    return model


def log_current_run(X, y, model, uri):
    import mlflow
    from mlflow.models import infer_signature

    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    mlflow.set_experiment("Training with optimal hyperparams")

    # Start an MLflow run
    with mlflow.start_run(run_name="Run with optimal hyperparams"):
        # Log the hyperparameters
        mlflow.log_params(model.get_params())

        # Log the loss metric
        mlflow.log_metric("accuracy", model.score(X, y))

        # Infer the model signature
        signature = infer_signature(X, model.predict(X))

        # Log the model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="models/clf.skops",
            signature=signature,
            input_example=X,
            registered_model_name="optimal-hyperparams",
        )


def main(
    search="local",
    verbose=True,
    use_old_hyperparams=True,
    logging=False,
    log_uri="from_config",
    host=None,
    port=None,
) -> None:
    """
    Epstein didn't kill himself
    """

    # init hydra config
    if verbose:
        print("Initializing hydra config...")
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=BaseConfig)
    initialize(version_base="1.3", config_path="spoty_gp/conf")
    cfg = compose(config_name="config")
    # load data
    # TODO: rewrite as a dataloaders module instead of a function
    train_df = resolve_data_location(search, verbose)

    # preprocessing and splitting
    X, y = train_df.drop("genre", axis=1), train_df["genre"].to_numpy()
    data_preproc = get_preprocessor()
    data_preproc.fit(X)
    X = data_preproc.transform(X)

    # training
    model = get_trained(X, y, use_old_hyperparams, verbose, cfg)

    # saving the model
    if verbose:
        print("Saving the model...")
    with open("models/dt_clf.skops", "wb") as model_f:
        sio.dump(model, model_f)

    # logging
    if logging:
        if verbose:
            print("Logging the run...")

        if log_uri == "from_config":
            host = cfg.mlflow_address.host
            port = cfg.mlflow_address.port
        elif log_uri == "custom":
            pass
        else:
            raise Exception(
                "Parameter `log_uri` should be from {'from_config', 'custom'}"
            )

        log_current_run(X, y, model=model, uri=f"http://{host}:{port}")


if __name__ == "__main__":
    fire.Fire(main)
