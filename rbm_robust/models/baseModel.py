from pathlib import Path

import keras
import numpy as np
from tpcp import Algorithm


class BaseModel(Algorithm):
    def predict(
        self,
        testing_subjects: list[str] = None,
        data_path: Path = Path("/home/woody/iwso/iwso116h/TestData"),
        input_folder_name: str = "inputs",
        prediction_folder_name: str = "predictions_unnnamed",
    ):
        if self.model_path is not None:
            self._model = keras.models.load_model(self.model_path)

        print("Prediction started")
        if not isinstance(data_path, Path):
            data_path = Path(data_path)
        data_folder_name = data_path.name
        print(data_path)
        input_file_type = "npy" if not self.image_based else "png"
        subjects = [path.name for path in data_path.iterdir() if path.is_dir()]
        if testing_subjects is not None:
            subjects = [subject for subject in subjects if subject in testing_subjects]
        for subject_id in subjects:
            print(subject_id)
            subject_path = data_path / subject_id
            for phase_path in subject_path.iterdir():
                if not phase_path.is_dir():
                    continue
                input_path = phase_path / input_folder_name
                prediction_path = phase_path
                prediction_path = Path(
                    str(prediction_path).replace(data_folder_name, f"Predictions/{prediction_folder_name}")
                )
                prediction_path.mkdir(parents=True, exist_ok=True)
                input_files = sorted(input_path.glob(f"*.{input_file_type}"))
                for input_file in input_files:
                    model_input = self._get_input_array(input_file)
                    model_input = np.array([model_input])
                    pred = self._model.predict(model_input, verbose=0)
                    pred = pred.flatten()
                    filename = input_file.stem + ".npy"
                    save_path = Path(str(prediction_path)) / filename
                    np.save(save_path, pred)
        return self

    def _get_input_array(self, input_file):
        raise NotImplementedError("Subclasses should implement this method")
