from typing import Tuple, List, Optional

import tensorflow as tf
import numpy as np
from pathlib import Path

from keras.src.utils import img_to_array, load_img


class DatasetFactory:
    @staticmethod
    def read_file(input_path, label_path):
        input_file = np.load(input_path)
        label_file = np.load(label_path)
        return input_file, label_file

    @staticmethod
    def read_dual_channel_file(input_path, input_log_path, label_path):
        input_file = np.load(input_path)
        input_log_file = np.load(input_log_path)
        label_file = np.load(label_path)
        input_file = np.dstack((input_file, input_log_file))
        return input_file, label_file

    @staticmethod
    def read_dual_channel_image(input_path, input_log_path, label_path):
        input_file = img_to_array(load_img(input_path, target_size=(256, 1000))) / 255
        input_log_file = img_to_array(load_img(input_log_path, target_size=(256, 1000))) / 255
        label_file = np.load(label_path)
        input_file = np.dstack((input_file, input_log_file))
        return input_file, label_file

    @staticmethod
    def read_single_channel_image_file(input_path, label_path):
        input_file = img_to_array(load_img(input_path, target_size=(256, 1000))) / 255
        label_file = np.load(label_path)
        return input_file, label_file

    def _get_wavelet_dataset(self, input_paths, label_paths, batch_size=8):
        dataset = tf.data.Dataset.from_tensor_slices((input_paths, label_paths))
        dataset = (
            dataset.map(
                lambda input_path, label_path: tf.numpy_function(
                    self.read_single_channel_image_file, [input_path, label_path], [tf.float64, tf.float64]
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
            .repeat()
        )
        return dataset

    def _get_dataset(self, input_paths, label_paths, batch_size=8):
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
    def _get_all_input_and_label_paths(base_path, subject_list, training_phase: str = None, time_power=False):
        base_path = Path(base_path)
        input_paths = []
        label_paths = []
        input_folder_name = "inputs" if not time_power else "filtered_radar"
        for subject in subject_list:
            subject_path = base_path / subject
            print(f"Subject: {subject}")
            print(f"Subject Path: {subject_path}")
            for phase in subject_path.iterdir():
                if training_phase is not None and training_phase not in phase.name:
                    continue
                if not phase.is_dir():
                    continue
                input_path = phase / input_folder_name
                label_path = phase / "labels_gaussian"
                if not input_path.exists() or not label_path.exists():
                    print(f"Input Path: {input_path} or Label path: {label_path} does not exist")
                    continue
                print(f"Input Path: {input_path}")
                print(f"Label Path: {label_path}")
                input_files = sorted(input_path.glob("*.npy"))
                label_files = sorted(label_path.glob("*.npy"))
                print(f"Input Files: {input_files}")
                print(f"Label Files: {label_files}")
                label_filenames = set([label_file.name for label_file in label_files])
                input_filenames = set([input_file.name for input_file in input_files])
                filename_intersection = label_filenames.intersection(input_filenames)
                print(f"Filename Intersection: {filename_intersection}")
                input_files = [
                    str(input_file) for input_file in input_files if input_file.name in filename_intersection
                ]
                label_files = [
                    str(label_file) for label_file in label_files if label_file.name in filename_intersection
                ]
                input_paths += input_files
                label_paths += label_files
        # Sanity Check
        DatasetFactory()._sanity_check_file_paths(
            list(zip(input_paths, label_paths)), input_folder_name, "labels_gaussian", False
        )
        print(f"Input count: {len(input_paths)}")
        print(f"Label count: {len(label_paths)}")
        for input_path, label_path in zip(input_paths, label_paths):
            input_arr = np.load(input_path)
            label_arr = np.load(label_path)
            print(f"Input Shape: {input_arr.shape}")
            print(f"Label Shape: {label_arr.shape}")
        return input_paths, label_paths

    @staticmethod
    def _sanity_check_file_paths(
        all_paths: List[Tuple[Path]], input_folder_name: str, label_folder_name: str, image_based: bool
    ):
        for input_path, label_path in all_paths:
            modified_input_path = input_path.replace(input_folder_name, label_folder_name)
            if modified_input_path != label_path:
                raise ValueError(f"Input path: {input_path} does not match label path: {label_path}")
            if not Path(input_path).exists():
                raise FileNotFoundError(f"Input path: {input_path} does not exist")
            if not Path(label_path).exists():
                raise FileNotFoundError(f"Label path: {label_path} does not exist")

    @staticmethod
    def _get_all_wavelet_input_and_ecg_label_paths(base_path, subject_list, training_phase: str = None):
        base_path = Path(base_path)
        input_paths = []
        label_paths = []
        for subject in subject_list:
            subject_path = base_path / subject
            for phase in subject_path.iterdir():
                if training_phase is not None and training_phase not in phase.name:
                    continue
                if not phase.is_dir():
                    continue
                input_path = phase / "inputs"
                label_path = phase / "labels_ecg"
                if not input_path.exists() or not label_path.exists():
                    continue
                input_files = sorted(input_path.glob("*.png"))
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
        all_paths = list(zip(input_paths, label_paths))
        for input_path, label_path in all_paths:
            modified_input_path = input_path.replace("inputs", "labels_ecg")
            modified_input_path = modified_input_path.replace("png", "npy")
            if modified_input_path != label_path:
                raise ValueError(f"Input path: {input_path} does not match label path: {label_path}")
            if not Path(input_path).exists():
                raise FileNotFoundError(f"Input path: {input_path} does not exist")
            if not Path(label_path).exists():
                raise FileNotFoundError(f"Label path: {label_path} does not exist")
        return input_paths, label_paths

    @staticmethod
    def _get_dual_channel_wavelet_input_and_label_paths(
        base_path,
        subject_list,
        training_phase: str = None,
        wavelet_type: str = "morl",
        ecg_labels: bool = False,
    ):
        input_folder_name = f"inputs_wavelet_array_{wavelet_type}"
        input_log_folder_name = f"inputs_wavelet_array_{wavelet_type}_log"
        label_folder_name = "labels_ecg" if ecg_labels else "labels_gaussian"
        base_path = Path(base_path)
        input_paths = []
        input_log_paths = []
        label_paths = []
        for subject in subject_list:
            subject_path = base_path / subject
            for phase in subject_path.iterdir():
                if training_phase is not None and training_phase not in phase.name:
                    continue
                if not phase.is_dir():
                    continue
                input_path = phase / input_folder_name
                input_log_path = phase / input_log_folder_name
                label_path = phase / label_folder_name
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
                input_log_files = [
                    str(input_log_file)
                    for input_log_file in input_log_files
                    if input_log_file.stem in filename_intersection
                ]
                label_files = [
                    str(label_file) for label_file in label_files if label_file.stem in filename_intersection
                ]
                input_paths += input_files
                input_log_paths += input_log_files
                label_paths += label_files
        # Sanity Check
        all_paths = list(zip(input_paths, input_log_paths, label_paths))
        for input_path, input_log_path, label_path in all_paths:
            modified_input_path = input_path.replace(input_folder_name, label_folder_name)
            if modified_input_path != label_path:
                raise ValueError(f"Input path: {input_path} does not match label path: {label_path}")
            if not Path(input_path).exists():
                raise FileNotFoundError(f"Input path: {input_path} does not exist")
            if not Path(label_path).exists():
                raise FileNotFoundError(f"Label path: {label_path} does not exist")
            if not Path(input_log_path).exists():
                raise FileNotFoundError(f"Input Log path: {input_log_path} does not exist")
        return input_paths, input_log_paths, label_paths

    def _get_single_wavelet_input_and_label_paths(
        self,
        base_path,
        subject_list,
        training_phase: str = None,
        wavelet_type: str = "morl",
        log_transform: bool = False,
        ecg_labels: bool = False,
        diff: bool = False,
        image_based=False,
    ):
        input_folder_name = (
            f"inputs_wavelet_array_{wavelet_type}_log" if log_transform else f"inputs_wavelet_array_{wavelet_type}"
        )
        if image_based:
            input_folder_name = input_folder_name.replace("array", "image")
        if diff:
            input_folder_name = f"inputs_wavelet_array_diff"
        label_folder_name = "labels_ecg" if ecg_labels else "labels_gaussian"
        base_path = Path(base_path)
        input_paths = []
        label_paths = []
        input_datatype = "png" if image_based else "npy"
        for subject in subject_list:
            subject_path = base_path / subject
            for phase in subject_path.iterdir():
                if training_phase is not None and training_phase not in phase.name:
                    continue
                if not phase.is_dir():
                    continue
                input_path = phase / input_folder_name
                label_path = phase / label_folder_name
                if not input_path.exists() or not label_path.exists():
                    continue
                input_files = sorted(input_path.glob(f"*.{input_datatype}"))
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
        self._sanity_check_radarcadia(input_paths, label_paths, input_folder_name, label_folder_name, image_based)
        return input_paths, label_paths

    @staticmethod
    def _get_all_wavelet_input_and_label_paths(base_path, subject_list, training_phase: str = None):
        base_path = Path(base_path)
        input_paths = []
        label_paths = []
        for subject in subject_list:
            subject_path = base_path / subject
            for phase in subject_path.iterdir():
                if training_phase is not None and training_phase not in phase.name:
                    continue
                if not phase.is_dir():
                    continue
                input_path = phase / "inputs"
                label_path = phase / "labels_gaussian"
                if not input_path.exists() or not label_path.exists():
                    continue
                input_files = sorted(input_path.glob("*.png"))
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
        all_paths = list(zip(input_paths, label_paths))
        for input_path, label_path in all_paths:
            modified_input_path = input_path.replace("inputs", "labels_gaussian")
            modified_input_path = modified_input_path.replace("png", "npy")
            if modified_input_path != label_path:
                raise ValueError(f"Input path: {input_path} does not match label path: {label_path}")
            if not Path(input_path).exists():
                raise FileNotFoundError(f"Input path: {input_path} does not exist")
            if not Path(label_path).exists():
                raise FileNotFoundError(f"Label path: {label_path} does not exist")
        return input_paths, label_paths

    def get_dataset_for_subjects(self, base_path, training_subjects, batch_size: int = 8, training_phase=None):
        input_paths, label_paths = self._get_all_input_and_label_paths(base_path, training_subjects, training_phase)
        dataset = self._build_dataset(input_paths, label_paths, batch_size)
        return dataset, int(len(input_paths) / batch_size)

    def get_wavelet_dataset_and_ecg_labels_for_subjects(
        self, base_path, training_subjects, batch_size: int = 8, training_phase=None
    ):
        input_paths, label_paths = self._get_all_wavelet_input_and_ecg_label_paths(
            base_path, training_subjects, training_phase
        )
        dataset = self._build_wavelet_dataset_single_image(input_paths, label_paths, batch_size)
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
        diff=False,
        image_based=False,
    ):
        input_paths, label_paths = self._get_single_wavelet_input_and_label_paths(
            base_path=base_path,
            subject_list=training_subjects,
            training_phase=training_phase,
            wavelet_type=wavelet_type,
            log_transform=log_transform,
            ecg_labels=ecg_labels,
            diff=diff,
            image_based=image_based,
        )
        if not image_based:
            dataset = self._build_wavelet_single_array_dataset(input_paths, label_paths, batch_size)
        else:
            dataset = self._build_wavelet_dataset_single_image(input_paths, label_paths, batch_size)
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
            input_data = tf.ensure_shape(input_data, (1000, 5))  # Example shape for input
            label_data = tf.ensure_shape(label_data, (1000,))  # Example shape for labels
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
        image_based=False,
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
                image_based=image_based,
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
                image_based=image_based,
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
        image_based=False,
        identity=False,
    ):
        input_paths, input_log_paths, label_paths = self._get_all_wavelet_input_and_label_paths_radarcadia_dual_channel(
            base_path=base_path,
            subject_list=subjects,
            breathing_type=breathing_type,
            wavelet_type=wavelet_type,
            ecg_labels=ecg_labels,
            image_based=image_based,
            identity=identity,
        )
        if not image_based:
            dataset = self._build_wavelet_dual_array_dataset(
                input_paths=input_paths, label_paths=label_paths, batch_size=batch_size, input_log_paths=input_log_paths
            )
        else:
            dataset = self._build_wavelet_dual_image_dataset(
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
        image_based=False,
        identity=False,
    ):
        base_path = Path(base_path)
        input_paths = []
        input_log_paths = []
        label_paths = []
        input_folder_name = self._get_input_folder_name_radarcadia(
            wavelet_type=wavelet_type, log_transform=False, image_based=image_based, identity=identity
        )
        input_folder_name_log = self._get_input_folder_name_radarcadia(
            wavelet_type=wavelet_type, log_transform=True, image_based=image_based, identity=identity
        )
        label_folder_name = self._get_label_folder_name_radarcadia(ecg_labels)
        input_datatype = "png" if image_based else "npy"
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
                input_files = sorted(input_path.glob(f"*.{input_datatype}"))
                input_log_files = sorted(input_log_path.glob(f"*.{input_datatype}"))
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
        self._sanity_check_radarcadia(input_paths, label_paths, input_folder_name, label_folder_name, image_based)
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
        image_based=False,
        identity=False,
    ):
        input_paths, label_paths = self._get_all_wavelet_input_and_label_paths_radarcadia(
            base_path=base_path,
            subject_list=subjects,
            breathing_type=breathing_type,
            wavelet_type=wavelet_type,
            ecg_labels=ecg_labels,
            log_transform=log_transform,
            image_based=image_based,
            identity=identity,
        )
        if not image_based:
            dataset = self._build_wavelet_single_array_dataset(input_paths, label_paths, batch_size)
        else:
            dataset = self._build_wavelet_dataset_single_image(input_paths, label_paths, batch_size)
        return dataset, int(len(input_paths) / batch_size)

    def _get_all_wavelet_input_and_label_paths_radarcadia(
        self,
        base_path,
        subject_list,
        breathing_type: str = "all",
        wavelet_type="morl",
        ecg_labels=False,
        log_transform=False,
        image_based=False,
        identity=False,
    ):
        base_path = Path(base_path)
        input_paths = []
        label_paths = []
        input_folder_name = self._get_input_folder_name_radarcadia(
            wavelet_type=wavelet_type, log_transform=log_transform, image_based=image_based, identity=identity
        )
        label_folder_name = self._get_label_folder_name_radarcadia(ecg_labels)
        input_datatype = "png" if image_based else "npy"
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
                input_files = sorted(input_path.glob(f"*.{input_datatype}"))
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
        self._sanity_check_radarcadia(input_paths, label_paths, input_folder_name, label_folder_name, image_based)
        return input_paths, label_paths

    def _sanity_check_radarcadia(
        self, input_paths, label_paths, input_folder_name, label_folder_name, image_based=False
    ):
        input_datatype = ".png" if image_based else ".npy"
        all_paths = list(zip(input_paths, label_paths))
        for input_path, label_path in all_paths:
            modified_input_path = input_path.replace(input_folder_name, label_folder_name)
            modified_input_path = modified_input_path.replace(input_datatype, ".npy")
            if modified_input_path != label_path:
                raise ValueError(f"Input path: {input_path} does not match label path: {label_path}")
            if not Path(input_path).exists():
                raise FileNotFoundError(f"Input path: {input_path} does not exist")
            if not Path(label_path).exists():
                raise FileNotFoundError(f"Label path: {label_path} does not exist")

    def _get_input_folder_name_radarcadia(
        self, wavelet_type="morl", log_transform=False, image_based=False, identity=False
    ):
        input_folder_name = "inputs_wavelet_" if not identity else "inputs_identity_"
        if image_based:
            input_folder_name += "image_"
        else:
            input_folder_name += "array_"
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

    def _build_wavelet_dual_image_dataset(self, input_paths, input_log_paths, label_paths, batch_size=8):
        dataset = tf.data.Dataset.from_tensor_slices((input_paths, input_log_paths, label_paths))
        dataset = (
            dataset.map(
                lambda input_path, input_log_path, label_path: tf.numpy_function(
                    self.read_dual_channel_image,
                    [input_path, input_log_path, label_path],
                    [tf.float32, tf.float64],
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

    def get_wavelet_dataset_for_subjects(self, base_path, training_subjects, batch_size: int = 8, training_phase=None):
        input_paths, label_paths = self._get_all_wavelet_input_and_label_paths(
            base_path, training_subjects, training_phase
        )
        dataset = self._build_wavelet_dataset_single_image(input_paths, label_paths, batch_size)
        return dataset, int(len(input_paths) / batch_size)

    def _build_wavelet_dataset_single_image(self, input_paths, label_paths, batch_size=8):
        dataset = tf.data.Dataset.from_tensor_slices((input_paths, label_paths))
        dataset = (
            dataset.map(
                lambda input_path, label_path: tf.numpy_function(
                    self.read_single_channel_image_file, [input_path, label_path], [tf.float32, tf.float64]
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


class RefactoredDatasetFactory:
    @staticmethod
    def read_file(input_path, label_path, dual_channel=False, time_power=False):
        """
        Reads input and label files. Supports single and dual channel data, and image-based inputs.
        """
        input_file = np.load(input_path)
        if dual_channel:
            input_log_file = np.load(input_path.replace("inputs", "inputs_log"))
            input_file = np.dstack((input_file, input_log_file))
        label_file = np.load(label_path)
        if time_power:
            input_file = tf.ensure_shape(input_file, (1000, 5))
            label_file = tf.ensure_shape(label_file, (1000,))
        return input_file, label_file

    def _get_dataset(self, input_paths, label_paths, batch_size=8, dual_channel=False, time_power=False):
        """
        General method to create a dataset from input and label paths.
        """
        dataset = tf.data.Dataset.from_tensor_slices((input_paths, label_paths))
        dataset = (
            dataset.map(
                lambda input_path, label_path: tf.numpy_function(
                    self.read_file, [input_path, label_path, dual_channel, time_power], [tf.float64, tf.float64]
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
            .repeat()
        )
        return dataset

    @staticmethod
    def _get_all_input_and_label_paths(base_path, subject_list, training_phase: Optional[str] = None, time_power=False):
        """
        Retrieves all matching input and label file paths based on the provided criteria.
        """
        base_path = Path(base_path)
        input_paths, label_paths = [], []
        input_folder_name = "inputs" if not time_power else "filtered_radar"
        for subject in subject_list:
            subject_path = base_path / subject
            for phase in subject_path.iterdir():
                if training_phase and training_phase not in phase.name:
                    continue
                if not phase.is_dir():
                    continue
                input_path, label_path = phase / input_folder_name, phase / "labels_gaussian"
                if not input_path.exists() or not label_path.exists():
                    continue
                input_files, label_files = sorted(input_path.glob("*.npy")), sorted(label_path.glob("*.npy"))
                label_filenames = {label_file.name for label_file in label_files}
                input_filenames = {input_file.name for input_file in input_files}
                filename_intersection = label_filenames.intersection(input_filenames)
                input_paths += [
                    str(input_file) for input_file in input_files if input_file.name in filename_intersection
                ]
                label_paths += [
                    str(label_file) for label_file in label_files if label_file.name in filename_intersection
                ]
        return input_paths, label_paths

    def get_dataset_for_subjects(
        self,
        base_path,
        training_subjects,
        batch_size=8,
        training_phase=None,
        dual_channel=False,
        image_based=False,
        time_power=False,
    ):
        """
        Public method to get a dataset for specified subjects, potentially with dual channel and/or image-based data.
        """
        input_paths, label_paths = self._get_all_input_and_label_paths(
            base_path, training_subjects, training_phase, time_power
        )
        return self._get_dataset(input_paths, label_paths, batch_size, dual_channel, image_based), int(
            len(input_paths) / batch_size
        )
