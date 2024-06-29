import os
import tarfile
from pathlib import Path

from rbm_robust.pipelines.cnnLstmPipeline import D02PipelineImproved
from rbm_robust.pipelines.radarcadia_pipeline import RadarcadiaPipeline


def training_and_testing_pipeline(pipeline: RadarcadiaPipeline, testing_path: Path, image_based: bool = False):
    pipeline = pipeline.clone()
    # Train the model on the training and validation dataset
    pipeline.self_optimize()
    # Calculate the predictions for the trained model
    pipeline.run(testing_path, image_based=image_based)
    # Score the predictions
    pipeline.score(testing_path)


def d02_training_and_testing_pipeline(pipeline: D02PipelineImproved, testing_path: Path, image_based: bool = False):
    pipeline = pipeline.clone()

    # Get the dataset
    # Train the model on the training and validation dataset
    pipeline.self_optimize()

    # Calculate the predictions for the trained model
    pipeline.run(testing_path, image_based=image_based)

    # Score the predictions
    pipeline.score(testing_path)
