import os
import pathlib
from multiprocessing import Pool

from rbm_robust.data_loading.datasets import D02Dataset

from rbm_robust.pipelines.cnnLstmPipeline import InputAndLabelGenerator


def run(
    data_path: str,
    target_path: str,
    process_labels: bool = True,
    process_inputs: bool = True,
    process_images: bool = False,
):
    # Get the subjects from the data path
    subjects_dataset = D02Dataset(pathlib.Path(data_path))
    subsets = [subjects_dataset.get_subset(participant=subject) for subject in subjects_dataset.subjects]
    num_processes = 4
    print(num_processes)
    with Pool(num_processes) as p:
        p.starmap(
            process_subset,
            [(subset, target_path, process_labels, process_inputs, process_images) for subset in subsets],
        )


def process_subset(
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
