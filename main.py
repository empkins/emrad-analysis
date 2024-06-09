import gc
import pathlib
from itertools import zip_longest
from pathlib import Path
import shutil
import numpy as np

from rbm_robust.data_loading.datasets import D02Dataset
from rbm_robust.models.cnn import CNN
from rbm_robust.pipelines.cnnLstmPipeline import CnnPipeline
from rbm_robust.pipelines.preprocessing_pipeline import run
from rbm_robust.pipelines.waveletPipeline import WaveletPipeline
from rbm_robust.validation.identityScoring import identityScoring
from rbm_robust.validation.scoring import cnnPipelineScoring
import os
import tensorflow as tf

from rbm_robust.validation.wavelet_scoring import waveletPipelineScoring


def main(model_path: str = None, remaining_epochs: int = 0):
    print("Starting")
    # devices = tf.config.experimental.list_physical_devices("GPU")
    # tf.config.experimental.set_memory_growth(devices[0], True)
    # path = os.environ.get("DATA_PATH")
    # path = "/Users/simonmeske/Desktop/TestOrdner/data_per_subject"
    path = os.getenv("TMPDIR") + "/Data"
    print(path)
    # dataset_path = Path(path)
    dataset = D02Dataset(path)
    cnn_pipeline = CnnPipeline()
    cnnPipelineScoring(cnn_pipeline, dataset, path, model_path=model_path, remaining_epochs=remaining_epochs)


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
    target_path = "/home/woody/iwso/iwso116h/DataWavelet"
    # base_path = Path("/Users/simonmeske/Desktop/TestOrdner/data_per_subject")
    # target_path = "/Users/simonmeske/Desktop/TestOrdner/data_per_subject"
    run(base_path, target_path, process_inputs=True, process_labels=True, process_images=False)
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
    # main(None, 0)
    preprocessing()
    # wavelet_training(None, 0)
    # check_testing_and_training_paths()
    # identity_check()
    # dataset_path = Path("/Users/simonmeske/Desktop/TestOrdner/data_per_subject")
    # dataset = D02Dataset(dataset_path)
    #
    # input_label_generator = InputAndLabelGenerator()
    # input_label_generator.generate_training_inputs_and_labels(dataset)
