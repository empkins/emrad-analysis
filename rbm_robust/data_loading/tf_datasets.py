from typing import Tuple, List, Optional

import tensorflow as tf
import numpy as np
from pathlib import Path


class DatasetFactory:
    @staticmethod
    def read_file(input_path, label_path):
        """
        Reads input and label files.

        :param input_path: Path to the input file.
        :param label_path: Path to the label file.
        :return: Tuple of input and label data.
        """
        input_file = np.load(input_path)
        label_file = np.load(label_path)
        return input_file, label_file

    @staticmethod
    def read_dual_channel_file(input_path, input_log_path, label_path):
        """
        Reads dual channel input and label files.

        :param input_path: Path to the input file.
        :param input_log_path: Path to the input log file.
        :param label_path: Path to the label file.
        :return: Tuple of input and label data.
        """
        input_file = np.load(input_path)
        input_log_file = np.load(input_log_path)
        label_file = np.load(label_path)
        input_file = np.dstack((input_file, input_log_file))
        return input_file, label_file

    def _get_dataset(self, input_paths, label_paths, batch_size=8):
        """
        General method to create a dataset from input and label paths.

        :param input_paths: List of input file paths.
        :param label_paths: List of label file paths.
        :param batch_size: Batch size for the dataset.
        :param read_func: Function to read the files.
        :return: TensorFlow dataset.
        """
        dataset = tf.data.Dataset.from_tensor_slices((input_paths, label_paths))
        dataset = (
            dataset.map(
                lambda input_path, label_path: tf.numpy_function(
                    self.read_file, [input_path, label_path], [tf.float64, tf.float64]
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
            .repeat()
        )
        return dataset

    @staticmethod
    def _get_all_input_and_label_paths(
        base_path, subject_list, training_phase: str = None, time_power=False, labels="Gaussian"
    ):
        """
        General method to create a dataset from input and label paths.

        :param input_paths: List of input file paths.
        :param label_paths: List of label file paths.
        :param batch_size: Batch size for the dataset.
        :param read_func: Function to read the files.
        :return: TensorFlow dataset.
        """
        base_path = Path(base_path)
        input_paths, label_paths = [], []
        input_folder_name = "inputs" if not time_power else "filtered_radar"
        if labels == "Gaussian":
            label_folder_name = "labels_gaussian"
        elif labels == "ECG":
            label_folder_name = "labels_ecg"
        else:
            raise ValueError(f"Labels {labels} not supported")

        for subject in subject_list:
            subject_path = base_path / subject
            for phase in subject_path.iterdir():
                if training_phase and training_phase not in phase.name:
                    continue
                if not phase.is_dir():
                    continue
                input_path = phase / input_folder_name
                label_path = phase / label_folder_name
                if not input_path.exists() or not label_path.exists():
                    continue
                input_files = sorted(input_path.glob("*.npy"))
                label_files = sorted(label_path.glob("*.npy"))
                label_filenames = {label_file.name for label_file in label_files}
                input_filenames = {input_file.name for input_file in input_files}
                filename_intersection = label_filenames & input_filenames
                input_files = [
                    str(input_file) for input_file in input_files if input_file.name in filename_intersection
                ]
                label_files = [
                    str(label_file) for label_file in label_files if label_file.name in filename_intersection
                ]
                input_paths.extend(input_files)
                label_paths.extend(label_files)

        DatasetFactory._sanity_check_file_paths(
            list(zip(input_paths, label_paths)), input_folder_name, label_folder_name
        )
        return input_paths, label_paths

    @staticmethod
    def _sanity_check_file_paths(all_paths: List[Tuple[Path]], input_folder_name: str, label_folder_name: str):
        """
        Sanity check for file paths. Checks if for all input paths there is a corresponding label path.
        Raises exception when this is not the case.

        :param all_paths: List of tuples containing input and label file paths.
        :param input_folder_name: Name of the input folder.
        :param label_folder_name: Name of the label folder.
        """
        for input_path, label_path in all_paths:
            modified_input_path = input_path.replace(input_folder_name, label_folder_name)
            if modified_input_path != label_path:
                raise ValueError(f"Input path: {input_path} does not match label path: {label_path}")
            if not Path(input_path).exists():
                raise FileNotFoundError(f"Input path: {input_path} does not exist")
            if not Path(label_path).exists():
                raise FileNotFoundError(f"Label path: {label_path} does not exist")

    @staticmethod
    def _get_input_and_label_paths(
        base_path: str,
        subject_list: List[str],
        training_phase: Optional[str] = None,
        wavelet_type: Optional[str] = None,
        ecg_labels: bool = False,
        dual_channel: bool = False,
        time_power: bool = False,
    ) -> Tuple[List[str], List[str], Optional[List[str]]]:
        """
        Retrieves all matching input and label file paths based on the provided criteria.

        :param base_path: Base path to the data.
        :param subject_list: List of subjects.
        :param training_phase: Optional training phase to filter.
        :param wavelet_type: Optional type of wavelet.
        :param ecg_labels: Whether to use ECG labels.
        :param dual_channel: Whether to retrieve dual channel paths.
        :param time_power: Whether to use time power data.
        :return: Tuple of lists containing input, label, and optionally input log file paths.
        """
        base_path = Path(base_path)
        input_paths, label_paths, input_log_paths = [], [], []

        input_folder_name, input_log_folder_name = DatasetFactory._get_input_folder_names(wavelet_type, time_power)
        label_folder_name = "labels_ecg" if ecg_labels else "labels_gaussian"

        for subject in subject_list:
            subject_path = base_path / subject
            for phase in subject_path.iterdir():
                if DatasetFactory._should_skip_phase(phase, training_phase):
                    continue
                input_path, label_path = phase / input_folder_name, phase / label_folder_name
                if dual_channel:
                    input_log_path = phase / input_log_folder_name
                    if not input_log_path.exists():
                        continue
                else:
                    input_log_path = None
                if not input_path.exists() or not label_path.exists():
                    continue
                input_files, label_files = DatasetFactory._get_matching_files(
                    input_path, label_path, dual_channel, input_log_path
                )
                input_paths.extend(input_files)
                label_paths.extend(label_files)

        DatasetFactory._sanity_check_file_paths(
            list(zip(input_paths, label_paths)), input_folder_name, label_folder_name
        )
        if dual_channel:
            return input_paths, label_paths, input_log_paths
        return input_paths, label_paths, None

    def _get_wavelet_input_and_label_paths(
        self,
        base_path: str,
        subject_list: List[str],
        training_phase: Optional[str] = None,
        wavelet_type: str = "morl",
        log_transform: bool = False,
        ecg_labels: bool = False,
    ):
        """
        Retrieves all matching wavelet input and label file paths based on the provided criteria.

        :param base_path: Base path to the data.
        :param subject_list: List of subjects.
        :param training_phase: Optional training phase to filter.
        :param wavelet_type: Type of wavelet.
        :param log_transform: Whether to apply log transform.
        :param ecg_labels: Whether to use ECG labels.
        :return: Tuple of lists containing input and label file paths.
        """
        base_path = Path(base_path)
        input_paths, label_paths = [], []

        input_folder_name = DatasetFactory._get_input_folder_name(wavelet_type, log_transform)
        label_folder_name = "labels_ecg" if ecg_labels else "labels_gaussian"

        for subject in subject_list:
            subject_path = base_path / subject
            for phase in subject_path.iterdir():
                if DatasetFactory._should_skip_phase(phase, training_phase):
                    continue
                input_path, label_path = phase / input_folder_name, phase / label_folder_name
                if not input_path.exists() or not label_path.exists():
                    continue
                input_files, label_files = DatasetFactory._get_matching_files(input_path, label_path, False, None)
                input_paths.extend(input_files)
                label_paths.extend(label_files)

        DatasetFactory._sanity_check_file_paths(
            list(zip(input_paths, label_paths)), input_folder_name, label_folder_name
        )
        return input_paths, label_paths

    @staticmethod
    def _get_input_folder_name(wavelet_type: str, log_transform: bool) -> str:
        folder_name = (
            f"inputs_wavelet_array_{wavelet_type}_log" if log_transform else f"inputs_wavelet_array_{wavelet_type}"
        )
        return folder_name

    @staticmethod
    def _should_skip_phase(phase: Path, training_phase: Optional[str]) -> bool:
        return training_phase and training_phase not in phase.name or not phase.is_dir()

    @staticmethod
    def _get_matching_files(input_path: Path, label_path: Path, dual_channel: bool, input_log_path: Optional[Path]):
        input_files = sorted(input_path.glob("*.npy"))
        label_files = sorted(label_path.glob("*.npy"))
        label_filenames = {label_file.stem for label_file in label_files}
        input_filenames = {input_file.stem for input_file in input_files}
        if dual_channel:
            input_log_files = sorted(input_log_path.glob("*.npy"))
            input_log_filenames = {input_log_file.stem for input_log_file in input_log_files}
            filename_intersection = label_filenames & input_filenames & input_log_filenames
            input_log_files = [
                str(input_log_file)
                for input_log_file in input_log_files
                if input_log_file.stem in filename_intersection
            ]
        else:
            filename_intersection = label_filenames & input_filenames
        input_files = [str(input_file) for input_file in input_files if input_file.stem in filename_intersection]
        label_files = [str(label_file) for label_file in label_files if label_file.stem in filename_intersection]
        return input_files, label_files

    def get_dataset_for_subjects(self, base_path, training_subjects, batch_size: int = 8, training_phase=None):
        input_paths, _, label_paths = self._get_input_and_label_paths(
            base_path, training_subjects, training_phase, dual_channel=False
        )
        dataset = self._build_dataset(input_paths, label_paths, batch_size)
        return dataset, int(len(input_paths) / batch_size)

    def get_single_channel_wavelet_dataset_for_subjects(
        self,
        base_path,
        training_subjects,
        batch_size: int = 8,
        training_phase=None,
        wavelet_type="morl",
        log_transform=False,
        ecg_labels=False,
    ):
        input_paths, label_paths = self._get_wavelet_input_and_label_paths(
            base_path=base_path,
            subject_list=training_subjects,
            training_phase=training_phase,
            wavelet_type=wavelet_type,
            log_transform=log_transform,
            ecg_labels=ecg_labels,
        )
        dataset = self._build_wavelet_single_array_dataset(input_paths, label_paths, batch_size)
        return dataset, int(len(input_paths) / batch_size)

    def get_dual_channel_wavelet_dataset_for_subjects(
        self,
        base_path,
        training_subjects,
        batch_size: int = 8,
        training_phase=None,
        wavelet_type="morl",
        ecg_labels=False,
    ):
        input_paths, input_log_paths, label_paths = self._get_dual_channel_wavelet_input_and_label_paths(
            base_path=base_path,
            subject_list=training_subjects,
            training_phase=training_phase,
            wavelet_type=wavelet_type,
            ecg_labels=ecg_labels,
        )
        dataset = self._build_wavelet_dual_array_dataset(
            input_paths=input_paths, input_log_paths=input_log_paths, label_paths=label_paths, batch_size=batch_size
        )
        return dataset, int(len(input_paths) / batch_size)

    def get_time_power_dataset_for_subjects(
        self,
        base_path,
        subjects,
        batch_size: int = 8,
    ):
        input_paths, label_paths = self._get_all_input_and_label_paths(base_path, subjects, time_power=True)
        dataset = self._build_time_power_dataset(input_paths, label_paths, batch_size)
        return dataset, int(len(input_paths) / batch_size)

    def _build_time_power_dataset(self, input_paths, label_paths, batch_size=8):
        def process_path(input_path, label_path):
            input_data, label_data = tf.numpy_function(
                self.read_file, [input_path, label_path], [tf.float64, tf.float64]
            )
            # Set the shape of the tensors explicitly
            input_data = tf.ensure_shape(input_data, [5, 1000])
            label_data = tf.ensure_shape(
                label_data,
                [
                    1000,
                ],
            )
            return input_data, label_data

        dataset = tf.data.Dataset.from_tensor_slices((input_paths, label_paths))
        dataset = (
            dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )
        return dataset

    def get_wavelet_dataset_for_subjects_radarcadia(
        self,
        base_path,
        subjects,
        batch_size: int = 8,
        breathing_type: str = "all",
        wavelet_type="morl",
        ecg_labels=False,
        log_transform=False,
        dual_channel=False,
        identity=False,
    ):
        if dual_channel:
            return self._get_wavelet_dataset_radarcadia_dual_channel(
                base_path=base_path,
                subjects=subjects,
                batch_size=batch_size,
                breathing_type=breathing_type,
                wavelet_type=wavelet_type,
                ecg_labels=ecg_labels,
                identity=identity,
            )
        else:
            return self._get_wavelet_dataset_radarcadia_single_channel(
                base_path=base_path,
                subjects=subjects,
                batch_size=batch_size,
                breathing_type=breathing_type,
                wavelet_type=wavelet_type,
                ecg_labels=ecg_labels,
                log_transform=log_transform,
                identity=identity,
            )

    def _get_wavelet_dataset_radarcadia_dual_channel(
        self,
        base_path,
        subjects,
        batch_size: int = 8,
        breathing_type: str = "all",
        wavelet_type="morl",
        ecg_labels=False,
        identity=False,
    ):
        input_paths, input_log_paths, label_paths = self._get_all_wavelet_input_and_label_paths_radarcadia_dual_channel(
            base_path=base_path,
            subject_list=subjects,
            breathing_type=breathing_type,
            wavelet_type=wavelet_type,
            ecg_labels=ecg_labels,
            identity=identity,
        )
        dataset = self._build_wavelet_dual_array_dataset(
            input_paths=input_paths, label_paths=label_paths, batch_size=batch_size, input_log_paths=input_log_paths
        )
        return dataset, int(len(input_paths) / batch_size)

    def _get_all_wavelet_input_and_label_paths_radarcadia_dual_channel(
        self,
        base_path,
        subject_list,
        breathing_type: str = "all",
        wavelet_type="morl",
        ecg_labels=False,
        identity=False,
    ):
        base_path = Path(base_path)
        input_paths = []
        input_log_paths = []
        label_paths = []
        input_folder_name = self._get_input_folder_name_radarcadia(
            wavelet_type=wavelet_type, log_transform=False, identity=identity
        )
        input_folder_name_log = self._get_input_folder_name_radarcadia(
            wavelet_type=wavelet_type, log_transform=True, identity=identity
        )
        label_folder_name = self._get_label_folder_name_radarcadia(ecg_labels)
        for subject in subject_list:
            subject_path = base_path / subject
            for location in subject_path.iterdir():
                if breathing_type != "all" and breathing_type not in location.name:
                    continue
                if not location.is_dir():
                    continue
                input_path = location / input_folder_name
                input_log_path = location / input_folder_name_log
                label_path = location / label_folder_name
                if not input_path.exists() or not label_path.exists() or not input_log_path.exists():
                    continue
                input_files = sorted(input_path.glob("*.npy"))
                input_log_files = sorted(input_log_path.glob("*.npy"))
                label_files = sorted(label_path.glob("*.npy"))
                label_filenames = set([label_file.stem for label_file in label_files])
                input_filenames = set([input_file.stem for input_file in input_files])
                input_log_filenames = set([input_log_file.stem for input_log_file in input_log_files])
                filename_intersection = label_filenames & input_filenames & input_log_filenames
                input_files = [
                    str(input_file) for input_file in input_files if input_file.stem in filename_intersection
                ]
                label_files = [
                    str(label_file) for label_file in label_files if label_file.stem in filename_intersection
                ]
                input_log_files = [
                    str(input_log_file)
                    for input_log_file in input_log_files
                    if input_log_file.stem in filename_intersection
                ]
                input_paths += input_files
                input_log_paths += input_log_files
                label_paths += label_files
        # Sanity Check
        self._sanity_check_radarcadia(input_paths, label_paths, input_folder_name, label_folder_name)
        return input_paths, input_log_paths, label_paths

    def _get_wavelet_dataset_radarcadia_single_channel(
        self,
        base_path,
        subjects,
        batch_size: int = 8,
        breathing_type: str = "all",
        wavelet_type="morl",
        ecg_labels=False,
        log_transform=False,
        identity=False,
    ):
        input_paths, label_paths = self._get_all_wavelet_input_and_label_paths_radarcadia(
            base_path=base_path,
            subject_list=subjects,
            breathing_type=breathing_type,
            wavelet_type=wavelet_type,
            ecg_labels=ecg_labels,
            log_transform=log_transform,
            identity=identity,
        )
        dataset = self._build_wavelet_single_array_dataset(input_paths, label_paths, batch_size)
        return dataset, int(len(input_paths) / batch_size)

    def _get_all_wavelet_input_and_label_paths_radarcadia(
        self,
        base_path,
        subject_list,
        breathing_type: str = "all",
        wavelet_type="morl",
        ecg_labels=False,
        log_transform=False,
        identity=False,
    ):
        base_path = Path(base_path)
        input_paths = []
        label_paths = []
        input_folder_name = self._get_input_folder_name_radarcadia(
            wavelet_type=wavelet_type, log_transform=log_transform, identity=identity
        )
        label_folder_name = self._get_label_folder_name_radarcadia(ecg_labels)
        for subject in subject_list:
            subject_path = base_path / subject
            for location in subject_path.iterdir():
                if breathing_type != "all" and breathing_type not in location.name:
                    continue
                if not location.is_dir():
                    continue
                input_path = location / input_folder_name
                label_path = location / label_folder_name
                if not input_path.exists() or not label_path.exists():
                    continue
                input_files = sorted(input_path.glob("*.npy"))
                label_files = sorted(label_path.glob("*.npy"))
                label_filenames = set([label_file.stem for label_file in label_files])
                input_filenames = set([input_file.stem for input_file in input_files])
                filename_intersection = label_filenames.intersection(input_filenames)
                input_files = [
                    str(input_file) for input_file in input_files if input_file.stem in filename_intersection
                ]
                label_files = [
                    str(label_file) for label_file in label_files if label_file.stem in filename_intersection
                ]
                input_paths += input_files
                label_paths += label_files
        # Sanity Check
        self._sanity_check_radarcadia(input_paths, label_paths, input_folder_name, label_folder_name)
        return input_paths, label_paths

    def _sanity_check_radarcadia(self, input_paths, label_paths, input_folder_name, label_folder_name):
        all_paths = list(zip(input_paths, label_paths))
        for input_path, label_path in all_paths:
            modified_input_path = input_path.replace(input_folder_name, label_folder_name)
            if modified_input_path != label_path:
                raise ValueError(f"Input path: {input_path} does not match label path: {label_path}")
            if not Path(input_path).exists():
                raise FileNotFoundError(f"Input path: {input_path} does not exist")
            if not Path(label_path).exists():
                raise FileNotFoundError(f"Label path: {label_path} does not exist")

    def _get_input_folder_name_radarcadia(self, wavelet_type="morl", log_transform=False, identity=False):
        input_folder_name = "inputs_wavelet_array_" if not identity else "inputs_identity_array_"
        input_folder_name += f"{wavelet_type}"
        if log_transform:
            input_folder_name += "_log"
        return input_folder_name

    def _get_label_folder_name_radarcadia(self, ecg_labels=False):
        return "labels_gaussian" if not ecg_labels else "labels_ecg"

    def _build_wavelet_dual_array_dataset(self, input_paths, input_log_paths, label_paths, batch_size=8):
        dataset = tf.data.Dataset.from_tensor_slices((input_paths, input_log_paths, label_paths))
        dataset = (
            dataset.map(
                lambda input_path, input_log_path, label_path: tf.numpy_function(
                    self.read_dual_channel_file, [input_path, input_log_path, label_path], [tf.float64, tf.float64]
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )
        return dataset

    def _build_wavelet_single_array_dataset(self, input_paths, label_paths, batch_size=8):
        dataset = tf.data.Dataset.from_tensor_slices((input_paths, label_paths))
        dataset = (
            dataset.map(
                lambda input_path, label_path: tf.numpy_function(
                    self.read_file, [input_path, label_path], [tf.float64, tf.float64]
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )
        return dataset

    def _build_dataset(self, input_paths, label_paths, batch_size=8):
        dataset = tf.data.Dataset.from_tensor_slices((input_paths, label_paths))
        dataset = (
            dataset.map(
                lambda input_path, label_path: tf.numpy_function(
                    self.read_file, [input_path, label_path], [tf.float64, tf.float64]
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
            .repeat()
        )
        return dataset
