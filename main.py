import gc
import pathlib
from itertools import zip_longest
from pathlib import Path
import shutil

import click
import numpy as np
from sklearn.model_selection import train_test_split

from rbm_robust.data_loading.datasets import D02Dataset
from rbm_robust.models.cnn import CNN
from rbm_robust.pipelines.cnnLstmPipeline import D02PipelineImproved
from rbm_robust.pipelines.preprocessing_pipeline import run_d02, run_radarcadia
from rbm_robust.pipelines.radarcadia_pipeline import RadarcadiaPipeline
from rbm_robust.pipelines.waveletPipeline import WaveletPipeline
from rbm_robust.validation.identityScoring import identityScoring
import os

from rbm_robust.validation.scoring_pipeline import training_and_testing_pipeline, d02_training_and_testing_pipeline
from rbm_robust.validation.wavelet_scoring import waveletPipelineScoring


@click.command()
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
def main(
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
    if datasource == "radarcadia":
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
    else:
        raise ValueError("Datasource not found")


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
    path = os.getenv("TMPDIR") + "/DataD02/DataD02"
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
    )
    d02_training_and_testing_pipeline(pipeline=pipeline, testing_path=path, image_based=image_based)

    # path = os.getenv("TMPDIR") + "/Data"
    # testing_path = os.getenv("WORK") + "/TestData"
    # data_path = Path(path)
    # testing_path = Path(testing_path)
    # possible_subjects = [path.name for path in data_path.iterdir() if path.is_dir()]
    # testing_subjects = [path.name for path in Path(testing_path).iterdir() if path.is_dir()]
    # training_subjects, validation_subjects = train_test_split(possible_subjects, test_size=0.2, random_state=42)
    # pipeline = D02Pipeline(
    #     learning_rate=learning_rate,
    #     data_path=path,
    #     testing_path=testing_path,
    #     epochs=epochs,
    #     training_subjects=training_subjects,
    #     validation_subjects=validation_subjects,
    #     testing_subjects=testing_subjects,
    #     breathing_type="all",
    #     image_based=image_based,
    # )
    # training_and_testing_pipeline(pipeline=pipeline, testing_path=path, image_based=image_based)


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


def wavelet_training(model_path: str = None, remaining_epochs: int = 0):
    path = os.getenv("TMPDIR") + "/Data"
    # path = "/Users/simonmeske/Desktop/Masterarbeit/TestSubjects"
    dataset = D02Dataset(path)
    wavelet_pipeline = WaveletPipeline()
    waveletPipelineScoring(
        pipeline=wavelet_pipeline,
        dataset=dataset,
        training_and_validation_path=path,
        model_path=model_path,
        remaining_epochs=remaining_epochs,
    )


def alt():
    cnn = CNN()
    cnn.self_optimize("/Users/simonmeske/PycharmProjects/emrad-analysis/tests/DataImg")


def preprocessing():
    base_path = Path("/home/vault/empkins/tpD/D03/Data/MA_Simon_Meske/Data_D02/data_per_subject")
    target_path = os.getenv("WORK") + "/DataD02"
    # base_path = Path("/Users/simonmeske/Desktop/TestOrdner/data_per_subject")
    # target_path = "/Users/simonmeske/Desktop/TestOrdner/data_per_subject"
    run_d02(base_path, target_path, process_inputs=True, process_labels=True, process_images=False)
    # check_for_empty_arrays()


def preprocessing_radarcadia():
    base_path = Path("/home/vault/empkins/tpD/D03/Data/MA_Simon_Meske/2023_radarcardia_study")
    target_path = os.getenv("HPCVAULT") + "/DataRadarcadia"
    # base_path = Path("/Users/simonmeske/Desktop/Masterarbeit/Radarcadia")
    # target_path = "/Users/simonmeske/Desktop/Masterarbeit/Radarcadia/Processed_Files"
    run_radarcadia(base_path, target_path)
    # check_for_empty_arrays()


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
    source_path = os.getenv("WORK") + "/DataD02"
    target_path = os.getenv("WORK") + "/TestDataD02"
    subjects = (
        "146",
        "257",
        "254",
        "228",
        "201",
        "385",
        "338",
        "212",
        "077",
        "004",
        "120",
        "147",
        "320",
        "417",
        "286",
        "028",
        "292",
        "155",
        "096",
        "140",
        "447",
    )
    for subject in subjects:
        source_subject_path = Path(source_path) / subject
        target_subject_path = Path(target_path)
        if not source_subject_path.exists():
            print(f"Source path {source_subject_path} does not exist")
            continue
        print(f"Moving {source_subject_path} to {target_subject_path}")
        shutil.move(source_subject_path, target_subject_path)
        # for phase in source_subject_path.iterdir():
        #     if not phase.is_dir():
        #         continue
        #     target_phase_path = target_subject_path / phase.name
        #     source_wavelet_path = phase / input_different_wavelet
        #     source_ecg_label_path = phase / labels_ecg
        #     # print(f"Moving {source_wavelet_path} to {target_phase_path} - source wavelet")
        #     # print(f"Moving {source_ecg_label_path} to {target_phase_path} - source ecg")
        #     shutil.move(source_wavelet_path, target_phase_path)
        #     shutil.move(source_ecg_label_path, target_phase_path)


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


def identity_check():
    # path_to_data = "/Users/simonmeske/Desktop/TestOrdner/data_per_subject"
    path_to_data = os.getenv("TMPDIR") + "/Data"
    identityScoring(D02Dataset(path_to_data), path_to_data)


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


def sanity_check():
    base_path = os.getenv("TMPDIR") + "/Data"
    subject_list = [path.name for path in base_path.iterdir() if path.is_dir()]
    base_path = Path(base_path)
    input_paths = []
    label_paths = []
    for subject in subject_list:
        subject_path = base_path / subject
        for phase in subject_path.iterdir():
            if not phase.is_dir():
                continue
            input_path = phase / "inputs"
            label_path = phase / "labels_gaussian"
            if not input_path.exists() or not label_path.exists():
                continue
            input_files = sorted(input_path.glob("*.npy"))
            label_files = sorted(label_path.glob("*.npy"))
            label_filenames = set([label_file.name for label_file in label_files])
            input_filenames = set([input_file.name for input_file in input_files])
            filename_intersection = label_filenames.intersection(input_filenames)
            input_files = [str(input_file) for input_file in input_files if input_file.name in filename_intersection]
            label_files = [str(label_file) for label_file in label_files if label_file.name in filename_intersection]
            input_paths += input_files
            label_paths += label_files
    # Sanity Check
    all_paths = list(zip(input_paths, label_paths))
    for input_path, label_path in all_paths:
        modified_input_path = input_path.replace("inputs", "labels_gaussian")
        if modified_input_path != label_path:
            raise ValueError(f"Input path: {input_path} does not match label path: {label_path}")
        if not Path(input_path).exists():
            raise FileNotFoundError(f"Input path: {input_path} does not exist")
        if not Path(label_path).exists():
            raise FileNotFoundError(f"Label path: {label_path} does not exist")
    print("Sanity Check successful")


def dim_fix():
    base_path = os.getenv("WORK") + "/DataD02"
    base_path = Path(base_path)
    for subject in base_path.iterdir():
        for phase in subject.iterdir():
            if "labels" in phase.name:
                continue
            if not phase.is_dir():
                continue
            # print(f"Subject is {subject}")
            for input_folder in phase.iterdir():
                for input_file in input_folder.iterdir():
                    # print(f"file is {input_file}")
                    if not input_file.is_file():
                        continue
                    input_data = np.load(input_file)
                    # if input_data.ndim == 2:
                    # Diff Data is 2D and needs to be 3D
                    # input_data = input_data.reshape(input_data.shape[0], input_data.shape[1], 1)
                    # np.save(input_file, input_data)
                    if input_data.ndim == 3 and input_data.shape != (256, 1000, 1):
                        print(f"Shape is {input_data.shape}")
                        print(f"File is {input_file}")
                        print(f"Subject is {subject}")
                        print(f"Phase is {phase}")
                        zero_pad = np.zeros((256, 1000, 1))
                        zero_pad[: input_file.shape[0], : input_file.shape[1], :] = input_file
                        input_file = zero_pad
                        # np.save(input_file, zero_pad)


if __name__ == "__main__":
    # devices = tf.config.experimental.list_physical_devices("GPU")
    # tf.config.experimental.set_memory_growth(devices[0], True)
    # input_loading()
    # check_for_empty_arrays()
    # input_loading()
    # args = sys.argv[1:]
    # model_path = None
    # remaining_epochs = 0
    # if len(args) > 0:
    #     if args[0] == "-mp":
    #         model_path = args[1]
    #     if args[2] == "-epochs":
    #         remaining_epochs = int(args[3])
    # main(model_path, remaining_epochs)
    # main()
    dim_fix()
    # preprocessing()
    # move_training_data()
    # preprocessing_radarcadia()
    # get_data_set_radarcadia()
    # move_training_data()
    # wavelet_training(None, 0)
    # check_testing_and_training_paths()
    # identity_check()
    # dataset_path = Path("/Users/simonmeske/Desktop/TestOrdner/data_per_subject")
    # dataset = D02Dataset(dataset_path)
    #
    # input_label_generator = InputAndLabelGenerator()
    # input_label_generator.generate_training_inputs_and_labels(dataset)
