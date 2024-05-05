from pathlib import Path


from rbm_robust.data_loading.datasets import D02Dataset
from rbm_robust.models.cnn import CNN
from rbm_robust.pipelines.cnnLstmPipeline import CnnPipeline
from rbm_robust.pipelines.preprocessing_pipeline import run
from rbm_robust.validation.scoring import cnnPipelineScoring
import os
import tensorflow as tf


def main():
    print("Starting")
    # devices = tf.config.experimental.list_physical_devices("GPU")
    # tf.config.experimental.set_memory_growth(devices[0], True)
    path = os.environ.get("DATA_PATH")
    print(path)
    dataset_path = Path("/Users/simonmeske/Desktop/TestOrdner/data_per_subject")
    dataset = D02Dataset(dataset_path)
    cnn_pipeline = CnnPipeline()
    cnnPipelineScoring(cnn_pipeline, dataset, "/Users/simonmeske/Desktop/TestOrdner/data_per_subject")


def alt():
    cnn = CNN()
    cnn.self_optimize("/Users/simonmeske/PycharmProjects/emrad-analysis/tests/DataImg")


def preprocessing():
    base_path = Path("/home/vault/empkins/tpD/D03/Data/MA_Simon_Meske/Data_D02/data_per_subject")
    target_path = "/home/woody/iwso/iwso116h/Data"
    run(base_path, target_path, process_inputs=True)


if __name__ == "__main__":
    # devices = tf.config.experimental.list_physical_devices("GPU")
    # tf.config.experimental.set_memory_growth(devices[0], True)
    main()
    # preprocessing()
    # dataset_path = Path("/Users/simonmeske/Desktop/TestOrdner/data_per_subject")
    # dataset = D02Dataset(dataset_path)
    #
    # input_label_generator = InputAndLabelGenerator()
    # input_label_generator.generate_training_inputs_and_labels(dataset)
