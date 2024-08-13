from pathlib import Path
import shutil
import os

import click
from sklearn.model_selection import train_test_split

from rbm_robust.pipelines.lstm_pipeline import MagPipeline
from rbm_robust.pipelines.preprocessing_pipeline import run_d02, run_d05, run_d02_Mag, run_d05_Mag
from rbm_robust.pipelines.radarcadia_pipeline import RadarcadiaPipeline
from rbm_robust.pipelines.scoring_pipeline import (
    mag_training_and_testing_pipeline,
    d02_training_and_testing_pipeline,
    d05_training_and_testing_pipeline,
)
from rbm_robust.pipelines.unetPipeline import D02Pipeline


@click.group()
def cli():
    pass


@click.command()
@click.option("--epochs", default=50, help="Number of epochs to train the model")
@click.option("--learning_rate", default=0.001, help="Learning rate for the model")
@click.option("--datasource", default="radarcadia", help="The datasource to use")
@click.option("--breathing_type", default="all", help="Type of breathing to use")
@click.option("--label_type", default="guassian", help="Type of labels to use. Possible values are ecg and gaussian")
@click.option("--log", default=False, help="Whether to use log transformed data")
@click.option("--dual_channel", default=False, help="Whether to use log transformed data and not log transformed data")
@click.option("--wavelet", default="morl", help="Type of wavelet to use: morl, gaus5")
@click.option("--identity", default=False, help="Whether to use identity check")
@click.option("--loss", default="bce", help="The used loss function. Valid values are bce and mse")
@click.option("--diff", default=False, help="Whether to use the first derivative of the radar signal")
@click.option("--mag", default=False, help="Whether to use the magnitude of the radar signal")
@click.option("--combined", default=False, help="Whether to use the magnitude of the radar signal")
@click.option("--training_path", default=None, help="Path to the training data")
@click.option("--testing_path", default=None, help="Path to the testing data")
def train(
    epochs: int,
    learning_rate: float,
    datasource: str,
    breathing_type: str,
    label_type: str,
    log: bool,
    dual_channel: bool,
    wavelet: str,
    identity: bool,
    loss: str,
    diff: bool,
    mag: bool,
    combined: bool,
    training_path: str,
    testing_path: str,
):
    if mag:
        ml_time_power(
            epochs=epochs,
            learning_rate=learning_rate,
            log=log,
            wavelet=wavelet,
            loss=loss,
            training_path=training_path,
            testing_path=testing_path,
        )
    elif datasource == "radarcadia" or combined:
        ml_radarcadia(
            epochs=epochs,
            learning_rate=learning_rate,
            breathing_type=breathing_type,
            label_type=label_type,
            log=log,
            dual_channel=dual_channel,
            wavelet=wavelet,
            identity=identity,
            loss=loss,
            training_path=training_path,
            testing_path=testing_path,
        )
    elif datasource == "d02":
        ml_d02(
            epochs=epochs,
            learning_rate=learning_rate,
            breathing_type=breathing_type,
            label_type=label_type,
            log=log,
            dual_channel=dual_channel,
            wavelet=wavelet,
            identity=identity,
            loss=loss,
            diff=diff,
            training_path=training_path,
            testing_path=testing_path,
        )
    else:
        raise ValueError("Datasource not found")


@click.command()
@click.option("--base_path", required=True, help="Base path for the data")
@click.option("--target_path", required=True, help="Target path for the processed data")
@click.option("--cwt", required=False, default=False, help="Whether to transform the power of the radar using CWT")
def preprocess_d02(base_path: str, target_path: str, cwt: bool = True):
    preprocessing_d02(base_path, target_path, cwt)


@click.command()
@click.option("--base_path", required=True, help="Base path for the data")
@click.option("--target_path", required=True, help="Target path for the processed data")
@click.option("--cwt", required=False, help="Whether to transform the power of the radar using CWT")
def preprocess_d05(base_path: str, target_path: str, cwt: bool = True):
    preprocessing_d05(base_path, target_path, cwt)


def ml_time_power(
    learning_rate: float,
    epochs: int,
    log: bool,
    wavelet: str,
    loss: str,
    training_path: str,
    testing_path: str,
):
    possible_subjects = [path.name for path in Path(training_path).iterdir() if path.is_dir()]
    testing_subjects = [path.name for path in Path(testing_path).iterdir() if path.is_dir()]

    use_ecg_labels = False

    training_subjects, validation_subjects = train_test_split(possible_subjects, test_size=0.2, random_state=42)
    pipeline = MagPipeline(
        learning_rate=learning_rate,
        data_path=training_path,
        epochs=epochs,
        training_subjects=training_subjects,
        validation_subjects=validation_subjects,
        testing_subjects=testing_subjects,
        ecg_labels=use_ecg_labels,
        log_transform=log,
        wavelet_type=wavelet,
        loss=loss,
        testing_path=Path(testing_path),
    )
    mag_training_and_testing_pipeline(pipeline=pipeline, testing_path=Path(testing_path))


def ml_d02(
    learning_rate: float,
    epochs: int,
    breathing_type: str,
    label_type: str,
    log: bool,
    dual_channel: bool,
    wavelet: str,
    identity: bool,
    loss: str,
    diff: bool,
    training_path: str,
    testing_path: str,
):
    possible_subjects = [path.name for path in Path(training_path).iterdir() if path.is_dir()]
    testing_subjects = [path.name for path in Path(testing_path).iterdir() if path.is_dir()]

    possible_subjects = [subject for subject in possible_subjects if "VP" not in subject]
    testing_subjects = [subject for subject in testing_subjects if "VP" not in subject]

    use_ecg_labels = label_type == "ecg"

    training_subjects, validation_subjects = train_test_split(possible_subjects, test_size=0.2, random_state=42)
    pipeline = D02Pipeline(
        learning_rate=learning_rate,
        data_path=training_path,
        epochs=epochs,
        training_subjects=training_subjects,
        validation_subjects=validation_subjects,
        testing_subjects=testing_subjects,
        ecg_labels=use_ecg_labels,
        log_transform=log,
        wavelet_type=wavelet,
        loss=loss,
        testing_path=Path(testing_path),
        diff=diff,
        dual_channel=dual_channel,
    )
    d02_training_and_testing_pipeline(pipeline=pipeline, testing_path=Path(testing_path))


def ml_radarcadia(
    learning_rate: float,
    epochs: int,
    breathing_type: str,
    label_type: str,
    log: bool,
    dual_channel: bool,
    wavelet: str,
    identity: bool,
    loss: str,
    training_path: str,
    testing_path: str,
):
    testing_path = Path(testing_path)
    possible_subjects = [path.name for path in Path(training_path).iterdir() if path.is_dir()]
    testing_subjects = [path.name for path in Path(testing_path).iterdir() if path.is_dir()]

    use_ecg_labels = label_type == "ecg"

    training_subjects, validation_subjects = train_test_split(possible_subjects, test_size=0.2, random_state=42)
    pipeline = RadarcadiaPipeline(
        learning_rate=learning_rate,
        data_path=training_path,
        epochs=epochs,
        training_subjects=training_subjects,
        validation_subjects=validation_subjects,
        testing_subjects=testing_subjects,
        breathing_type=breathing_type,
        ecg_labels=use_ecg_labels,
        log_transform=log,
        dual_channel=dual_channel,
        wavelet_type=wavelet,
        identity=identity,
        loss=loss,
        testing_path=testing_path,
    )
    d05_training_and_testing_pipeline(pipeline=pipeline, testing_path=testing_path)


def preprocessing_magnitude(dataset: str):
    if dataset == "d02":
        base_path = Path("/home/vault/empkins/tpD/D03/Data/MA_Simon_Meske/Data_D02/data_per_subject")
        target_path = Path(os.getenv("WORK", "/work")) / "DataD02Mag"
        run_d02_Mag(base_path, target_path)
    elif dataset == "d05":
        base_path = Path("/home/vault/empkins/tpD/D03/Data/MA_Simon_Meske/2023_radarcardia_study")
        target_path = Path(os.getenv("WORK", "/work")) / "DataRadarcadiaMag"
        run_d05_Mag(base_path, target_path)
    else:
        raise ValueError("Dataset not found")


def preprocessing_d02(base_path: str, target_path: str, cwt: bool = True):
    base_path = Path(base_path)
    target_path = Path(target_path)
    if cwt:
        run_d02(base_path, target_path)
    else:
        run_d02_Mag(base_path, target_path)
    move_training_data(base_path, target_path, "d02")


def preprocessing_d05(base_path: str, target_path: str, cwt: bool = True):
    base_path = Path(base_path)
    target_path = Path(target_path)
    if cwt:
        run_d05(base_path, target_path)
    else:
        run_d05_Mag(base_path, target_path)
    move_training_data(base_path, target_path, "d05")


def move_training_data(source_path: str, target_path: str, dataset: str):
    if dataset == "d02":
        subjects = [
            "130",
            "268",
            "338",
            "173",
            "242",
            "273",
            "008",
            "241",
            "198",
            "439",
            "272",
            "143",
            "199",
            "249",
            "140",
            "230",
            "111",
            "155",
            "213",
            "203",
            "310",
            "300",
        ]
    elif dataset == "d05":
        subjects = ["VP_01", "VP_15", "VP_11", "VP_03", "VP_18"]
    else:
        raise ValueError("Dataset not found")
    for subject in subjects:
        source_subject_path = source_path / subject
        if not source_subject_path.exists():
            print(f"Source path {source_subject_path} does not exist")
            continue
        print(f"Moving {source_subject_path} to {target_path}")
        shutil.move(source_subject_path, target_path)


if __name__ == "__main__":
    cli.add_command(train, "train")
    cli.add_command(preprocess_d02, "preprocess_d02")
    cli.add_command(preprocess_d05, "preprocess_d05")
    cli()
