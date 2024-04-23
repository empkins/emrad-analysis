from itertools import groupby
from pathlib import Path

import numpy as np
from keras.src.utils import img_to_array

from rbm_robust.data_loading.datasets import D02Dataset
from rbm_robust.pipelines.cnnLstmPipeline import CnnPipeline
from rbm_robust.validation.scoring import cnnPipelineScoring
import os


def main():
    print("Starting")
    path = os.environ.get("DATA_PATH")
    print(path)
    dataset_path = Path(path)
    dataset = D02Dataset(dataset_path)
    cnn_pipeline = CnnPipeline()
    cnnPipelineScoring(cnn_pipeline, dataset)


# /home/vault/empkins/tpD/D03/Data/MA_Simon_Meske/Data_D02/data_per_subject
if __name__ == "__main__":
    main()
