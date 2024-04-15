from pathlib import Path

from rbm_robust.data_loading.datasets import D02Dataset
from rbm_robust.pipelines.cnnLstmPipeline import CnnPipeline
from rbm_robust.validation.scoring import cnnPipelineScoring


def main():
    print("Starting")
    # dataset_path = Path("/Users/simonmeske/Desktop/TestOrdner/data_per_subject")
    # dataset = D02Dataset(dataset_path)
    # cnn_pipeline = CnnPipeline()
    # cnnPipelineScoring(cnn_pipeline, dataset)


if __name__ == "__main__":
    main()
