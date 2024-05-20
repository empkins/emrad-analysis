from pathlib import Path


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


def identity_check():
    # path_to_data = "/Users/simonmeske/Desktop/TestOrdner/data_per_subject"
    path_to_data = os.getenv("TMPDIR") + "/Data"
    identityScoring(D02Dataset(path_to_data), path_to_data)


if __name__ == "__main__":
    # devices = tf.config.experimental.list_physical_devices("GPU")
    # tf.config.experimental.set_memory_growth(devices[0], True)
    main()
    # preprocessing()
    # identity_check()
    # dataset_path = Path("/Users/simonmeske/Desktop/TestOrdner/data_per_subject")
    # dataset = D02Dataset(dataset_path)
    #
    # input_label_generator = InputAndLabelGenerator()
    # input_label_generator.generate_training_inputs_and_labels(dataset)
