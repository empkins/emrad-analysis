from pathlib import Path

from rbm_robust.data_loading.datasets import D02Dataset, RadarCardiaStudyDataset
from rbm_robust.pipelines.radarcadia_pipeline import RadarcadiaPipeline


def training_and_testing_pipeline(
    pipeline: RadarcadiaPipeline, training_and_validation_path: str, testing_path: str, data_set_type: str = "D02"
):
    pipeline = pipeline.clone()

    # Get the dataset
    training_and_validation_dataset = None
    if data_set_type == "D02":
        training_and_validation_dataset = D02Dataset(Path(training_and_validation_path))
    elif data_set_type == "RadarCardia":
        training_and_validation_dataset = RadarCardiaStudyDataset(Path(training_and_validation_path))
    else:
        raise ValueError(f"Unknown dataset type {data_set_type}")

    # Train the model on the training and validation dataset
    pipeline.self_optimize(training_and_validation_dataset, training_and_validation_path, testing_path)

    # Calculate the predictions for the trained model
    pipeline.run(Path(testing_path))

    # Score the predictions
    pipeline.score(Path(testing_path))
