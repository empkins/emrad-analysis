import pathlib
import tarfile
from pathlib import Path
import shutil

import click
import numpy as np
from sklearn.model_selection import train_test_split

from rbm_robust.pipelines.cnnLstmPipeline import D02PipelineImproved, PreTrainedPipeline
from rbm_robust.pipelines.preprocessing_pipeline import run_d02, run_radarcadia, run_d02_Mag, run_radarcadia_Mag
from rbm_robust.pipelines.radarcadia_pipeline import RadarcadiaPipeline
from rbm_robust.pipelines.time_power_pipeline import MagPipeline
import os

from rbm_robust.validation.instantenous_heart_rate import ScoreCalculator
from rbm_robust.validation.scoring_pipeline import (
    training_and_testing_pipeline,
    d02_training_and_testing_pipeline,
    pretrained_training_and_testing_pipeline,
    mag_training_and_testing_pipeline,
)


@click.command()
@click.option("--model_path", default=None, help="Path to the model already trained")
@click.option("--epochs", default=50, help="Number of epochs to train the model")
@click.option("--learning_rate", default=0.001, help="Learning rate for the model")
@click.option("--image_based", default=False, help="Whether the model is image based")
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
def main(
    model_path,
    epochs,
    learning_rate,
    image_based,
    datasource,
    breathing_type,
    label_type,
    log,
    dual_channel,
    wavelet,
    identity,
    loss,
    diff,
):
    if model_path is not None and os.path.exists(model_path):
        ml_already_trained(
            model_path=model_path,
            image_based=image_based,
            datasource=datasource,
            label_type=label_type,
            log=log,
            wavelet=wavelet,
            dual_channel=dual_channel,
        )
    elif datasource == "radarcadia":
        ml_radarcadia(
            epochs=epochs,
            learning_rate=learning_rate,
            image_based=image_based,
            breathing_type=breathing_type,
            label_type=label_type,
            log=log,
            dual_channel=dual_channel,
            wavelet=wavelet,
            identity=identity,
            loss=loss,
        )
    elif datasource == "d02":
        ml_d02(
            epochs=epochs,
            learning_rate=learning_rate,
            image_based=image_based,
            breathing_type=breathing_type,
            label_type=label_type,
            log=log,
            dual_channel=dual_channel,
            wavelet=wavelet,
            identity=identity,
            loss=loss,
            diff=diff,
        )
    elif datasource == "magnitude":
        ml_time_power(
            epochs=epochs,
            learning_rate=learning_rate,
            image_based=image_based,
            log=log,
            wavelet=wavelet,
            loss=loss,
        )
    else:
        raise ValueError("Datasource not found")


def ml_time_power(
    learning_rate: float = 0.001,
    epochs: int = 50,
    image_based: bool = False,
    log: bool = False,
    wavelet: str = "morl",
    loss: str = "bce",
    dataset_type: str = "d02",
):
    # path = "/Users/simonmeske/Desktop/TestOrdner/Time_power"
    # testing_path = "/Users/simonmeske/Desktop/TestOrdner/Time_power"
    path = os.getenv("TMPDIR") + "/DataD02/DataD02"
    testing_path = os.getenv("WORK") + "/TestDataD02"
    if dataset_type == "radarcadia":
        path = os.getenv("TMPDIR") + "/Data/DataRadarcadiaMag"
        testing_path = os.getenv("WORK") + "/TestDataRadarcadiaMag"
    # Get Training and Testing Subjects
    data_path = Path(path)
    testing_path = Path(testing_path)
    possible_subjects = [path.name for path in data_path.iterdir() if path.is_dir()]
    testing_subjects = [path.name for path in Path(testing_path).iterdir() if path.is_dir()]

    use_ecg_labels = False

    # Split Data
    training_subjects, validation_subjects = train_test_split(possible_subjects, test_size=0.2, random_state=42)
    pipeline = MagPipeline(
        learning_rate=learning_rate,
        data_path=path,
        epochs=epochs,
        training_subjects=training_subjects,
        validation_subjects=validation_subjects,
        testing_subjects=testing_subjects,
        image_based=image_based,
        ecg_labels=use_ecg_labels,
        log_transform=log,
        wavelet_type=wavelet,
        loss=loss,
        testing_path=testing_path,
    )
    mag_training_and_testing_pipeline(pipeline=pipeline, testing_path=path, image_based=image_based)


def ml_already_trained(model_path, image_based, datasource, label_type, log, wavelet, dual_channel=False):
    if datasource == "radarcadia":
        testing_path = os.getenv("HPCVAULT") + "/TestDataRadarcadia"
        # testing_path = "/Users/simonmeske/Desktop/Masterarbeit/RadarcadiaTestData"
    else:
        testing_path = os.getenv("WORK") + "/TestDataD02"
        # testing_path = "/Users/simonmeske/Desktop/Masterarbeit/TestDataD02"
    testing_path = Path(testing_path)
    testing_subjects = [path.name for path in Path(testing_path).iterdir() if path.is_dir()]

    # Pipeline
    pipeline = PreTrainedPipeline(
        wavelet_type=wavelet,
        image_based=image_based,
        model_path=model_path,
        log_transform=log,
        ecg_labels=label_type == "ecg",
        dual_channel=dual_channel,
        testing_subjects=testing_subjects,
        testing_path=testing_path,
    )
    # Predicting and scoring
    pretrained_training_and_testing_pipeline(pipeline=pipeline, testing_path=testing_path, image_based=image_based)


def ml_d02(
    learning_rate: float = 0.001,
    epochs: int = 50,
    image_based: bool = False,
    breathing_type: str = "all",
    label_type: str = "guassian",
    log: bool = False,
    dual_channel: bool = False,
    wavelet: str = "morl",
    identity: bool = False,
    loss: str = "bce",
    diff: bool = False,
):
    # path = "/Users/simonmeske/Desktop/Masterarbeit/DataD02"
    # testing_path = "/Users/simonmeske/Desktop/Masterarbeit/TestDataD02"
    path = os.getenv("TMPDIR") + "/DataD02"
    testing_path = os.getenv("WORK") + "/TestDataD02"
    # Get Training and Testing Subjects
    data_path = Path(path)
    testing_path = Path(testing_path)
    possible_subjects = [path.name for path in data_path.iterdir() if path.is_dir()]
    testing_subjects = [path.name for path in Path(testing_path).iterdir() if path.is_dir()]

    use_ecg_labels = label_type == "ecg"

    # Split Data
    training_subjects, validation_subjects = train_test_split(possible_subjects, test_size=0.2, random_state=42)
    pipeline = D02PipelineImproved(
        learning_rate=learning_rate,
        data_path=path,
        epochs=epochs,
        training_subjects=training_subjects,
        validation_subjects=validation_subjects,
        testing_subjects=testing_subjects,
        image_based=image_based,
        ecg_labels=use_ecg_labels,
        log_transform=log,
        wavelet_type=wavelet,
        loss=loss,
        testing_path=testing_path,
        diff=diff,
        dual_channel=dual_channel,
    )
    d02_training_and_testing_pipeline(pipeline=pipeline, testing_path=path, image_based=image_based)


def ml_radarcadia(
    learning_rate: float = 0.001,
    epochs: int = 50,
    image_based: bool = False,
    breathing_type: str = "all",
    label_type: str = "guassian",
    log: bool = False,
    dual_channel: bool = False,
    wavelet: str = "morl",
    identity: bool = False,
    loss: str = "bce",
):
    print("Starting")
    # path = "/Users/simonmeske/Desktop/TestOrdner/data_per_subject"
    # path = "/Users/simonmeske/Desktop/Masterarbeit/Radarcadia/Processed_Files"
    # testing_path = "/Users/simonmeske/Desktop/Masterarbeit/RadarcadiaTestData"
    path = os.getenv("TMPDIR") + "/Data/DataRadarcadia"
    testing_path = os.getenv("HPCVAULT") + "/TestDataRadarcadia"
    # Get Training and Testing Subjects
    data_path = Path(path)
    testing_path = Path(testing_path)
    possible_subjects = [path.name for path in data_path.iterdir() if path.is_dir()]
    testing_subjects = [path.name for path in Path(testing_path).iterdir() if path.is_dir()]

    use_ecg_labels = label_type == "ecg"

    # Split Data
    training_subjects, validation_subjects = train_test_split(possible_subjects, test_size=0.2, random_state=42)
    pipeline = RadarcadiaPipeline(
        learning_rate=learning_rate,
        data_path=path,
        epochs=epochs,
        training_subjects=training_subjects,
        validation_subjects=validation_subjects,
        testing_subjects=testing_subjects,
        breathing_type=breathing_type,
        image_based=image_based,
        ecg_labels=use_ecg_labels,
        log_transform=log,
        dual_channel=dual_channel,
        wavelet_type=wavelet,
        identity=identity,
        loss=loss,
        testing_path=testing_path,
    )
    training_and_testing_pipeline(pipeline=pipeline, testing_path=testing_path, image_based=image_based)


def preprocessing_magnitude(dataset: str = "d02"):
    if dataset == "d02":
        base_path = Path("/home/vault/empkins/tpD/D03/Data/MA_Simon_Meske/Data_D02/data_per_subject")
        target_path = os.getenv("WORK") + "/DataD02"
        # base_path = Path("/Users/simonmeske/Desktop/Masterarbeit/ArrayLengthTest")
        # target_path = "/Users/simonmeske/Desktop/TestOrdner/data_per_subject"
        run_d02_Mag(base_path, target_path, process_inputs=True, process_labels=True, process_images=False)
    else:
        base_path = Path("/home/vault/empkins/tpD/D03/Data/MA_Simon_Meske/2023_radarcardia_study")
        target_path = os.getenv("WORK") + "/DataRadarcadiaMag"
        # base_path = Path("/Users/simonmeske/Desktop/Masterarbeit/Radarcadia")
        # target_path = "/Users/simonmeske/Desktop/Masterarbeit/Radarcadia/Processed_Files"
        run_radarcadia_Mag(base_path, target_path, process_inputs=True, process_labels=True, process_images=False)


def preprocessing():
    base_path = Path("/home/vault/empkins/tpD/D03/Data/MA_Simon_Meske/Data_D02/data_per_subject")
    target_path = os.getenv("WORK") + "/DataD02"
    # base_path = Path("/Users/simonmeske/Desktop/Masterarbeit/ArrayLengthTest")
    # target_path = "/Users/simonmeske/Desktop/TestOrdner/data_per_subject"
    run_d02(base_path, target_path, process_inputs=True, process_labels=True, process_images=False)


def preprocessing_radarcadia():
    # base_path = Path("/home/vault/empkins/tpD/D03/Data/MA_Simon_Meske/2023_radarcardia_study")
    # target_path = os.getenv("WORK") + "/DataRadarcadiaOverlap"
    base_path = Path("/Users/simonmeske/Desktop/Masterarbeit/Radarcadia")
    target_path = "/Users/simonmeske/Desktop/Masterarbeit/Radarcadia/Processed_Files"
    run_radarcadia(base_path, target_path)


def pretrained(base_path: str):
    base_path_for_models = Path(base_path)
    for model in base_path_for_models.glob("*.keras"):
        if "radarcadia" not in model.name:
            continue
        if "mse" in model.name:
            continue
        args = _get_args_from_model_name(model.name)
        print(args)
        if args["label_type"] == "ecg":
            continue
        if args["dual_channel"]:
            continue
        if args["wavelet"] == "mexh":
            continue
        if args["wavelet"] == "shan1-1":
            continue
        ml_already_trained(
            model_path=str(model),
            image_based=args["image_based"],
            datasource="d02",
            label_type=args["label_type"],
            log=args["log"],
            wavelet=args["wavelet"],
            dual_channel=args["dual_channel"],
        )


def fix_and_normalize_diff():
    base_paths = [os.getenv("WORK") + "/DataD02", os.getenv("WORK") + "/TestDataD02"]
    for base in base_paths:
        base_path = pathlib.Path(base)
        for subject_path in base_path.iterdir():
            for phase_path in subject_path.iterdir():
                if not phase_path.is_dir():
                    continue
                for input_folder in phase_path.iterdir():
                    if "diff" not in input_folder.name:
                        continue
                    for input_file in input_folder.iterdir():
                        if not input_file.is_file():
                            continue
                        if "png" in input_file.name:
                            continue
                        try:
                            input_data = np.load(input_file)
                            if input_data.ndim == 2:
                                # Diff Data is 2D and needs to be 3D
                                # First normalize the data
                                if input_data.shape != (256, 1000):
                                    print(f"Shape is {input_data.shape} for file {input_file}")
                                    zero_pad = np.zeros((256, 1000))
                                    zero_pad[: input_data.shape[0], : input_data.shape[1]] = input_data
                                    input_data = zero_pad
                                numerator = input_data - np.min(input_data)
                                denominator = np.max(input_data) - np.min(input_data)
                                if denominator == 0:
                                    input_data = np.zeros((256, 1000))
                                else:
                                    input_data = numerator / denominator
                                input_data = input_data.reshape(input_data.shape[0], input_data.shape[1], 1)
                                np.save(input_file, input_data)
                        except Exception as e:
                            print(f"Error in file {input_file} with error {e}")
                            continue


def input_loading():
    test_validation_subjects = [
        "231",
        "134",
        "284",
        "199",
        "269",
        "114",
        "196",
        "175",
        "071",
        "310",
        "094",
        "559",
        "287",
        "142",
        "288",
        "038",
        "251",
        "232",
        "173",
        "159",
        "117",
        "259",
        "156",
        "052",
        "198",
        "261",
        "263",
        "141",
        "121",
        "308",
        "139",
        "270",
        "216",
        "100",
        "137",
    ]
    base_path = os.getenv("WORK") + "/Data"
    target_path = os.getenv("TMPDIR") + "/Data"
    for subject in test_validation_subjects:
        subject_path = Path(base_path) / subject
        target_subject_path = Path(target_path) / subject
        print(f"Copied {subject} to {target_subject_path} from {subject_path}")
        shutil.copytree(subject_path, target_subject_path)


def check_for_empty_arrays():
    # base_path = "/Users/simonmeske/Desktop/TestOrdner/data_per_subject"
    base_path = os.getenv("WORK") + "/Data"
    base_path = pathlib.Path(base_path)
    for subject_path in base_path.iterdir():
        if not subject_path.is_dir():
            continue
        for phase_path in subject_path.iterdir():
            if not phase_path.is_dir():
                continue
            input_path = phase_path / "inputs"
            label_path = phase_path / "labels_gaussian"
            if not input_path.exists() or not label_path.exists():
                continue
            label_files = sorted(path.name for path in label_path.iterdir() if path.is_file())
            for label_file in label_files:
                label = np.load(label_path / label_file)
                if np.all(label == 0):
                    label_to_remove = label_path / label_file
                    input_to_remove = input_path / label_file
                    if label_to_remove.exists() and input_to_remove.exists():
                        label_to_remove.unlink()
                        input_to_remove.unlink()
                    print(f"Empty array in {label_file} at {label_path}")


def move_training_data():
    paths = [
        (os.getenv("WORK") + "/DataD02", os.getenv("WORK") + "/TestDataD02Mag"),
        (os.getenv("WORK") + "/DataRadarcadiaMag", os.getenv("WORK") + "/TestDataRadarcadiaMag"),
    ]
    for path_tuple in paths:
        source_path = path_tuple[0]
        target_path = path_tuple[1]
        # source_path = os.getenv("WORK") + "/DataD02"
        # target_path = os.getenv("WORK") + "/TestDataD02Mag"
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
        for subject in subjects:
            source_subject_path = Path(source_path) / subject
            target_subject_path = Path(target_path)
            if not source_subject_path.exists():
                print(f"Source path {source_subject_path} does not exist")
                continue
            print(f"Moving {source_subject_path} to {target_subject_path}")
            shutil.move(source_subject_path, target_subject_path)


def rename_folders():
    base_path = os.getenv("TMPDIR") + "/Data"
    base_path = pathlib.Path(base_path)
    for subject_path in base_path.iterdir():
        if not subject_path.is_dir():
            continue
        for phase_path in subject_path.iterdir():
            if not phase_path.is_dir():
                continue
            for dirs in phase_path.iterdir():
                if not dirs.is_dir():
                    continue
                if "labels" not in dirs.name or "labels_gaussian" in dirs.name:
                    continue
                new_name = dirs.name.replace("labels", "labels_gaussian")
                new_path = phase_path / new_name
                print(f"Oldname is {dirs} new name is {new_path}")
                dirs.rename(new_path)
                return 0


def check_testing_and_training_paths():
    training_path = "/home/woody/iwso/iwso116h/DataRef"
    testing_path = "/home/woody/iwso/iwso116h/TestDataRef"
    training_path = pathlib.Path(training_path)
    testing_path = pathlib.Path(testing_path)
    training_subjects = [path.name for path in training_path.iterdir() if path.is_dir()]
    testing_subjects = [path.name for path in testing_path.iterdir() if path.is_dir()]
    print(f"Training subjects: {training_subjects}")
    print(f"Testing subjects: {testing_subjects}")
    phase_path = Path("/home/woody/iwso/iwso116h/TestData/004/ei_1")
    prediction_path = phase_path
    prediction_path = Path(
        str(prediction_path).replace("TestDataRef", "Predictions/predictions_mse_0001_25_epochs_ref")
    )
    print(prediction_path)


def scoring():
    base_path = os.getenv("HOME") + "/altPreprocessing/emrad-analysis/Models"
    base_path_for_models = Path(base_path)
    print(base_path_for_models)
    model_files = base_path_for_models.glob("*.keras")
    models = [model for model in model_files if "20240630" in model.name or "20240701" in model.name]
    print(models)
    for model in models:
        print(model.stem)
        args = _get_args_from_model_name(model.name)
        print(args)
        if args["label_type"] == "ecg":
            continue
        if not args["dual_channel"]:
            continue
        ml_already_trained(
            model_path=str(model),
            image_based=args["image_based"],
            datasource=args["datasource"],
            label_type=args["label_type"],
            log=args["log"],
            wavelet=args["wavelet"],
            dual_channel=args["dual_channel"],
        )


def untar_file(tar_path, extract_path):
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=extract_path)
    print(f"Extracted tar file to {extract_path}")


def calculate_scores():
    prediction_path = Path("/home/vault/iwso/iwso116h/Predictions")
    label_path = Path("/home/vault/iwso/iwso116h/TestDataRadarcadia")
    label_folder_name = "labels_gaussian"

    for prediction_tar in prediction_path.iterdir():
        if not prediction_tar.is_file():
            continue
        if "tar" in prediction_tar.suffix:
            prediction_folder_name = prediction_tar.stem
            # Untar
            untar_file(prediction_tar, prediction_path)
        # Calculate Scores
        prominences = range(0.1, 0.35, 0.05)
        for prominence in prominences:
            score_calculator = ScoreCalculator(
                prediction_path=prediction_path,
                label_path=label_path,
                overlap=int(0.4),
                fs=200,
                label_suffix=label_folder_name,
                prominence=prominence,
            )

            if os.getenv("WORK") is None:
                save_path = Path("/Users/simonmeske/Desktop/Masterarbeit")
            else:
                save_path = Path(os.getenv("WORK"))

            scores = score_calculator.calculate_scores()
            # Save the scores as a csv file
            score_path = save_path / "Scores"
            if not score_path.exists():
                score_path.mkdir(parents=True)
            scores.to_csv(score_path / f"scores_{prediction_folder_name}_{prominence}.csv")
            print(f"Scores: {scores}")


def _get_args_from_model_name(model_name: str):
    args = {}
    if "morl" in model_name:
        args["wavelet"] = "morl"
    elif "gaus1" in model_name:
        args["wavelet"] = "gaus1"
    elif "mexh" in model_name:
        args["wavelet"] = "mexh"
    elif "shan1-1" in model_name:
        args["wavelet"] = "shan1-1"

    if "image" in model_name:
        args["image_based"] = True
    else:
        args["image_based"] = False

    if "ecg" in model_name:
        args["label_type"] = "ecg"
    else:
        args["label_type"] = "gaussian"

    if "log" in model_name:
        args["log"] = True
    else:
        args["log"] = False

    if "dual" in model_name:
        args["dual_channel"] = True
    else:
        args["dual_channel"] = False

    if "radarcadia" in model_name:
        args["datasource"] = "radarcadia"
    else:
        args["datasource"] = "d02"

    return args


def score(prediction_path: str, prediction_folder_name: str):
    test_data_folder_name = Path(prediction_path).name
    label_folder_name = "labels_gaussian"
    test_path = Path(prediction_path)

    label_path = test_path
    prediction_path = Path(str(label_path).replace(test_data_folder_name, f"Predictions/{prediction_folder_name}"))

    prominences = range(0.1, 0.4, 0.05)
    for prominence in prominences:
        score_calculator = ScoreCalculator(
            prediction_path=prediction_path,
            label_path=label_path,
            overlap=int(0.4),
            fs=200,
            label_suffix=label_folder_name,
            prominence=prominence,
        )

        if os.getenv("WORK") is None:
            save_path = Path("/Users/simonmeske/Desktop/Masterarbeit")
        else:
            save_path = Path(os.getenv("WORK"))

        scores = score_calculator.calculate_scores()
        # Save the scores as a csv file
        score_path = save_path / "Scores"
        if not score_path.exists():
            score_path.mkdir(parents=True)
        scores.to_csv(score_path / f"scores_{prediction_folder_name}_prominence_{prominence}.csv")

        print(f"Scores: {scores}")
    return scores


def collect_and_score_arrays_d02():
    prediction_base_path = Path("/home/woody/iwso/iwso116h/Predictions")
    label_base_path = Path("/home/woody/iwso/iwso116h/TestDataD02")
    label_folder_name = "labels_gaussian"
    _collect_and_score(prediction_base_path, label_base_path, label_folder_name, "d02")


def _collect_and_score(prediction_base_path, label_base_path, label_folder_name, dataset: str = None):
    for prediction_folder in prediction_base_path.iterdir():
        if dataset not in prediction_folder.name:
            continue
        if not prediction_folder.is_dir():
            continue
        print(f"A Collecting and scoring {prediction_folder.name}")
        if "_40_" not in str(prediction_folder.name):
            continue
        if "image" in str(prediction_folder.name):
            continue
        print(f"B Collecting and scoring {prediction_folder.name}")
        prominences = [round(i, 2) for i in np.arange(0.05, 0.36, 0.05)]
        for prominence in prominences:
            score_calculator = ScoreCalculator(
                prediction_path=prediction_folder,
                label_path=label_base_path,
                overlap=int(0.4),
                fs=200,
                label_suffix=label_folder_name,
                prominence=prominence,
            )

            if os.getenv("WORK") is None:
                save_path = Path("/Users/simonmeske/Desktop/Masterarbeit")
            else:
                save_path = Path(os.getenv("WORK"))

            scores = score_calculator.calculate_scores()
            # Save the scores as a csv file
            score_path = save_path / "Scores"
            if not score_path.exists():
                score_path.mkdir(parents=True)
            scores.to_csv(score_path / f"scores_{prediction_folder.name}_prominence_{prominence}.csv")

            print(f"Scores: {scores}")


def collect_and_score_arrays_radarcadia():
    prediction_base_path = Path("/home/vault/iwso/iwso116h/Predictions")
    label_base_path = Path("/home/vault/iwso/iwso116h/TestDataRadarcadia")
    label_folder_name = "labels_gaussian"
    _collect_and_score(prediction_base_path, label_base_path, label_folder_name, "radarcadia")


if __name__ == "__main__":
    # collect_and_score_arrays_d02()
    # collect_and_score_arrays_radarcadia()
    # main()
    # scoring()
    # pretrained(os.getenv("HOME") + "/emrad-analysis/Models")
    # pretrained(os.getenv("HOME") + "/altPreprocessing/emrad-analysis/Models")
    # preprocessing_magnitude(dataset="d02")
    move_training_data()
    # preprocessing()
    # preprocessing_radarcadia()
