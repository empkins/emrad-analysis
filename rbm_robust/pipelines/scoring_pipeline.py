import os
import tarfile
from pathlib import Path

from rbm_robust.pipelines.radarcadia_pipeline import RadarcadiaPipeline
from rbm_robust.pipelines.unetPipeline import D02Pipeline


def d05_training_and_testing_pipeline(pipeline: RadarcadiaPipeline, testing_path: Path):
    pipeline = pipeline.clone()
    # Train the model on the training and validation dataset
    pipeline.self_optimize()
    # Calculate the predictions for the trained model
    pipeline.run(testing_path)
    # Score the predictions
    pipeline.score(testing_path)


def d02_training_and_testing_pipeline(pipeline: D02Pipeline, testing_path: Path):
    pipeline = pipeline.clone()

    # Get the dataset
    # Train the model on the training and validation dataset
    pipeline.self_optimize()

    # Calculate the predictions for the trained model
    pipeline.run(testing_path)

    # Score the predictions
    pipeline.score(testing_path)


def mag_training_and_testing_pipeline(pipeline, testing_path: Path):
    pipeline = pipeline.clone()
    # Train the model on the training and validation dataset
    pipeline.self_optimize()
    # Calculate the predictions for the trained model
    pipeline.run(testing_path)
    # Score the predictions
    pipeline.score(testing_path)
