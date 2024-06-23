import os
import pathlib
from multiprocessing import Pool

from rbm_robust.data_loading.datasets import D02Dataset, RadarCardiaStudyDataset

from rbm_robust.pipelines.cnnLstmPipeline import InputAndLabelGenerator


def run_d02(
    data_path: str,
    target_path: str,
    process_labels: bool = True,
    process_inputs: bool = True,
    process_images: bool = False,
):
    # Get the subjects from the data path
    subjects_dataset = D02Dataset(pathlib.Path(data_path))
    subjects = list(subjects_dataset.subjects)
    subjects = [s for s in subjects if int(s) % 2 == 1]
    subsets = [subjects_dataset.get_subset(participant=subject) for subject in subjects]
    num_processes = 4
    print(num_processes)
    with Pool(num_processes) as p:
        p.starmap(
            process_d02_subset,
            [(subset, target_path, process_labels, process_inputs, process_images) for subset in subsets],
        )


def process_d02_subset(
    data_set: D02Dataset,
    target_path: str,
    process_labels: bool = True,
    process_inputs: bool = True,
    process_images: bool = False,
):
    generator = InputAndLabelGenerator()
    try:
        if process_inputs and process_labels:
            generator.generate_training_inputs_and_labels(data_set, target_path, process_images)
        elif process_inputs:
            generator.generate_training_inputs(data_set, target_path, process_images)
        elif process_labels:
            generator.generate_training_labels(data_set, target_path)
    except Exception as e:
        print(f"Error in processing subset {data_set.subjects[0]} with error {e}")


def run_radarcadia(
    data_path: str,
    target_path: str,
):
    # Get the subjects from the data path
    subjects_dataset = RadarCardiaStudyDataset(pathlib.Path(data_path))
    subsets = []
    subjects = list(set(subjects_dataset.index["subject"]))
    for subject in subjects:
        subject_subset = subjects_dataset.get_subset(subject=subject)
        subsets.append(subject_subset)
    num_processes = 4
    print(num_processes)
    with Pool(num_processes) as p:
        p.starmap(
            process_radarcadia_subset,
            [(subset, target_path) for subset in subsets],
        )


def process_radarcadia_subset(
    data_set: RadarCardiaStudyDataset,
    target_path: str,
):
    generator = InputAndLabelGenerator()
    try:
        generator.generate_training_inputs_and_labels_radarcadia(data_set, target_path)
    except Exception as e:
        print(f"Error in processing subset with error {e}")
