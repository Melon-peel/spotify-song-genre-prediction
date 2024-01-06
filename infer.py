import pathlib

import pandas as pd
import skops.io as sio
from spoty_gp.transformers import get_preprocessor


def main(
    data_dir="data/train_test/",
    data_from="local",
    model_path="models/dt_clf.skops",
    model_from="local",
    verbose=True,
):
    train_path = str(pathlib.Path(data_dir) / "train.csv")
    test_path = str(pathlib.Path(data_dir) / "test.csv")

    # load data
    if data_from == "local":
        if verbose:
            print("Loading the data for inference from the local storage...")
        test_df = pd.read_csv(test_path, index_col="index")

        if verbose:
            print("Loading the data for preprocessing from the local storage...")
        train_df = pd.read_csv(train_path, index_col="index")

    else:
        raise Exception("Param data_from only supports values from {'local', }")

    # load model
    if model_from == "local":
        if verbose:
            print("Loading the model from the local storage...")
        model = sio.load(model_path)
    else:
        raise Exception("Param model_from only supports values from {'local', }")

    # preprocess data
    if verbose:
        print("Preprocessing the data for inference...")
    X_train = train_df.drop("genre", axis=1)
    data_preproc = get_preprocessor()
    data_preproc.fit(X_train)

    X_test = test_df.drop("genre", axis=1)
    X_test = data_preproc.transform(X_test)
    y_test = test_df["genre"].to_numpy()

    # inference
    if verbose:
        print("Making inference from the model...")
    predictions = model.predict(X_test)

    # saving results
    if verbose:
        print("Saving predictions to predictions.csv")

    results_df = pd.DataFrame.from_dict(
        {"genre actual": y_test, "genre predicted": predictions}
    )
    results_df.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()
