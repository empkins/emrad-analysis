import gc
import pathlib
from itertools import zip_longest
from pathlib import Path

import numpy as np

from rbm_robust.data_loading.datasets import D02Dataset
from rbm_robust.models.cnn import CNN
from rbm_robust.pipelines.cnnLstmPipeline import CnnPipeline
from rbm_robust.pipelines.preprocessing_pipeline import run
from rbm_robust.validation.identityScoring import identityScoring
from rbm_robust.validation.scoring import cnnPipelineScoring
import os
import tensorflow as tf


def main():
    print("Starting")
    # devices = tf.config.experimental.list_physical_devices("GPU")
    # tf.config.experimental.set_memory_growth(devices[0], True)
    # path = os.environ.get("DATA_PATH")
    # path = "/Users/simonmeske/Desktop/TestOrdner/data_per_subject"
    path = os.getenv("WORK") + "/Data"
    print(path)
    # dataset_path = Path(path)
    dataset = D02Dataset(path)
    cnn_pipeline = CnnPipeline()
    cnnPipelineScoring(cnn_pipeline, dataset, path)


def alt():
    cnn = CNN()
    cnn.self_optimize("/Users/simonmeske/PycharmProjects/emrad-analysis/tests/DataImg")


def preprocessing():
    base_path = Path("/home/vault/empkins/tpD/D03/Data/MA_Simon_Meske/Data_D02/data_per_subject")
    target_path = "/home/woody/iwso/iwso116h/Data"
    # base_path = Path("/Users/simonmeske/Desktop/TestOrdner/data_per_subject")
    # target_path = "/Users/simonmeske/Desktop/TestOrdner/data_per_subject"
    # run(base_path, target_path, process_inputs=False, process_labels=True, process_images=False
    run(base_path, target_path, process_inputs=False, process_labels=True, process_images=False)
    check_for_empty_arrays()


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
            label_path = phase_path / "labels"
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


def input_loading():
    base_path = "/Users/simonmeske/Desktop/TestOrdner/data_per_subject"
    dataset = D02Dataset(base_path)
    subjects = dataset.subjects
    while True:
        batch_generator(base_path, subjects)


def batch_generator(base_path, training_subjects):
    base_path = Path(base_path)
    subjects = [path.name for path in base_path.iterdir() if path.is_dir()]
    if training_subjects is not None:
        subjects = [subject for subject in subjects if subject in training_subjects]
    while True:
        _get_inputs_and_labels_for_subjects(base_path, subjects)


def _get_inputs_and_labels_for_subjects(base_path, subjects, batch_size=8, image_based=False):
    for subject_id in subjects:
        print(f"Subject: {subject_id}")
        subject_path = base_path / subject_id
        phases = [path.name for path in subject_path.iterdir() if path.is_dir()]
        for phase in phases:
            if "ei" not in phase:
                continue
            if phase == "logs" or phase == "raw":
                continue
            phase_path = subject_path / phase
            input_path = phase_path / "inputs"
            label_path = phase_path / "labels"
            if not input_path.exists() or not label_path.exists():
                continue
            input_names = sorted(path.name for path in input_path.iterdir() if path.is_file())
            input_names = list(filter(lambda x: "png" in x if image_based else "npy" in x, input_names))
            groups = grouper(input_names, batch_size)
            for group in groups:
                inputs = np.stack(
                    [
                        np.load(input_path / number) if number is not None else get_input(input_path, None)
                        for number in group
                    ],
                    axis=0,
                )
                inputs = inputs[:, 50:950, :, :]
                labels = np.stack(
                    [
                        np.load(label_path / number) if number is not None else get_labels(input_path, None)
                        for number in group
                    ],
                    axis=0,
                )
                labels = labels[:, 50:950]
                if inputs.shape != (8, 900, 256, 5):
                    print(f"Inputs shape: {inputs.shape}")
                if labels.shape != (8, 900):
                    print(f"Labels shape: {labels.shape}")


def _reduce_input_array(array):
    arr = array[50:950, :, :]
    return arr


def _reduce_label_array(array):
    arr = array[50:950]
    return arr


def grouper(iterable, n):
    iterators = [iter(iterable)] * n
    return zip_longest(*iterators)


def get_input(base_path, imfs, image_based=False):
    if imfs is not None:
        return np.transpose(np.stack([_load_input(base_path / imf) for imf in imfs], axis=0))
    elif not image_based and imfs is None:
        return np.transpose(np.stack([np.zeros((256, 1000)) for _ in range(5)], axis=0))
    elif image_based and imfs is None:
        return np.transpose(np.stack([np.zeros((256, 1000, 3)) for _ in range(5)], axis=0))


def get_labels(base_path, number):
    if number is not None and "." not in number:
        return np.load(base_path / f"{number}.npy")
    elif number is not None and ".npy" in number:
        return np.load(base_path / f"{number}")
    else:
        return np.zeros((1000))


def _load_input(path, image_based=False):
    if path.suffix == ".npy" and not image_based:
        try:
            arr = _pad_array(np.load(path))
        except Exception as e:
            print(f"Exception: {e}")
            arr = np.zeros((256, 1000))
    return arr


def _pad_array(array):
    if array.shape != (256, 1000):
        padded = np.zeros((256, 1000))
        padded[: array.shape[0], : array.shape[1]] = array
        arr = padded
    return arr


def identity_check():
    # path_to_data = "/Users/simonmeske/Desktop/TestOrdner/data_per_subject"
    path_to_data = os.getenv("TMPDIR") + "/Data"
    identityScoring(D02Dataset(path_to_data), path_to_data)


if __name__ == "__main__":
    # devices = tf.config.experimental.list_physical_devices("GPU")
    # tf.config.experimental.set_memory_growth(devices[0], True)
    # input_loading()
    main()
    # preprocessing()
    # identity_check()
    # dataset_path = Path("/Users/simonmeske/Desktop/TestOrdner/data_per_subject")
    # dataset = D02Dataset(dataset_path)
    #
    # input_label_generator = InputAndLabelGenerator()
    # input_label_generator.generate_training_inputs_and_labels(dataset)
