from sklearn.model_selection import train_test_split

from rbm_robust.data_loading.datasets import D02Dataset
from rbm_robust.models.identityModel import IdentityModel
from rbm_robust.pipelines.cnnLstmPipeline import CnnPipeline
from rbm_robust.pipelines.identityPipeline import IdentityPipeline


def identityScoring(dataset: D02Dataset, path: str = "/home/woody/iwso/iwso116h/Data"):
    train_data, test_data = train_test_split(dataset.subjects, test_size=0.2, random_state=42)
    training_dataset = dataset.get_subset(participant=train_data)
    training_dataset, validation_dataset = train_test_split(training_dataset, test_size=0.2, random_state=42)
    testing_dataset = dataset.get_subset(participant=test_data)

    identityPipeline = IdentityPipeline(IdentityModel())
    cnn_pipeline = CnnPipeline()
    # cnn_pipeline.prepare_data(training_dataset, validation_dataset, testing_dataset, path)

    print("Training started")
    identityPipeline.self_optimize(training_dataset, validation_dataset, path)
    print("Testing started")
    identityPipeline.run(testing_dataset.subjects, path)
