import concurrent
import gc
from itertools import zip_longest, groupby

import tensorflow as tf
import numpy as np
from pathlib import Path

from keras.src.utils import img_to_array, load_img
from sklearn.preprocessing import MinMaxScaler


class DatasetFactory:
    @staticmethod
    def read_file(input_path, label_path):
        try:
            input_file = np.load(input_path)
            # input_file = np.absolute(input_file)
            # Normalize the input file
            # input_file = np.zeros_like(input_file)
            # counter = input_file - np.min(input_file)
            # divider = np.max(input_file) - np.min(input_file)
            # if divider == 0:
            #     input_file = np.zeros_like(input_file)
            # else:
            #     input_file = counter / divider
            scaler = MinMaxScaler()
            input_file = scaler.transform(input_file)
            label_file = np.load(label_path)
            # Normalize
            return input_file, label_file
        except Exception as e:
            print(f"Exception: {e}")
            return None

    @staticmethod
    def read_dual_channel_file(input_path, input_log_path, label_path):
        try:
            input_file = np.load(input_path)
            input_file = (input_file - np.min(input_file)) / (np.max(input_file) - np.min(input_file))
            input_log_file = np.load(input_log_path)
            input_log_file = (input_log_file - np.min(input_log_file)) / (
                np.max(input_log_file) - np.min(input_log_file)
            )
            label_file = np.load(label_path)
            input_file = np.dstack((input_file, input_log_file))
            return input_file, label_file
        except Exception as e:
            print(f"Exception: {e}")
            return None

    @staticmethod
    def read_dual_channel_image(input_path, input_log_path, label_path):
        try:
            input_file = img_to_array(load_img(input_path, target_size=(256, 1000))) / 255
            input_log_file = img_to_array(load_img(input_log_path, target_size=(256, 1000))) / 255
            label_file = np.load(label_path)
            input_file = np.dstack((input_file, input_log_file))
            return input_file, label_file
        except Exception as e:
            print(f"Exception: {e}")
            return None

    @staticmethod
    def read_image_file(input_path, label_path):
        input_file = img_to_array(load_img(input_path, target_size=(256, 1000))) / 255
        # input_file = np.transpose(input_file, (1, 0, 2))
        label_file = np.load(label_path)
        return input_file, label_file

    def _get_wavelet_dataset(self, input_paths, label_paths, batch_size=8):
        dataset = tf.data.Dataset.from_tensor_slices((input_paths, label_paths))
        dataset = (
            dataset.map(
                lambda input_path, label_path: tf.numpy_function(
                    self.read_image_file, [input_path, label_path], [tf.float64, tf.float64]
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
    def _get_all_input_and_label_paths(base_path, subject_list, training_phase: str = None):
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
                input_files = sorted(input_path.glob("*.npy"))
                label_files = sorted(label_path.glob("*.npy"))
                label_filenames = set([label_file.name for label_file in label_files])
                input_filenames = set([input_file.name for input_file in input_files])
                filename_intersection = label_filenames.intersection(input_filenames)
                input_files = [
                    str(input_file) for input_file in input_files if input_file.name in filename_intersection
                ]
                label_files = [
                    str(label_file) for label_file in label_files if label_file.name in filename_intersection
                ]
                input_paths += input_files
                label_paths += label_files
        # Sanity Check
        all_paths = list(zip(input_paths, label_paths))
        for input_path, label_path in all_paths:
            modified_input_path = input_path.replace("inputs", "labels_gaussian")
            if modified_input_path != label_path:
                raise ValueError(f"Input path: {input_path} does not match label path: {label_path}")
            if not Path(input_path).exists():
                raise FileNotFoundError(f"Input path: {input_path} does not exist")
            if not Path(label_path).exists():
                raise FileNotFoundError(f"Label path: {label_path} does not exist")
        return input_paths, label_paths

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

    @staticmethod
    def _get_single_wavelet_input_and_label_paths(
        base_path,
        subject_list,
        training_phase: str = None,
        wavelet_type: str = "morl",
        log_transform: bool = False,
        ecg_labels: bool = False,
    ):
        input_folder_name = (
            f"inputs_wavelet_array_{wavelet_type}_log" if log_transform else f"inputs_wavelet_array_{wavelet_type}"
        )
        label_folder_name = "labels_ecg" if ecg_labels else "labels_gaussian"
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
                input_path = phase / input_folder_name
                label_path = phase / label_folder_name
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
        all_paths = list(zip(input_paths, label_paths))
        for input_path, label_path in all_paths:
            modified_input_path = input_path.replace(input_folder_name, label_folder_name)
            if modified_input_path != label_path:
                raise ValueError(f"Input path: {input_path} does not match label path: {label_path}")
            if not Path(input_path).exists():
                raise FileNotFoundError(f"Input path: {input_path} does not exist")
            if not Path(label_path).exists():
                raise FileNotFoundError(f"Label path: {label_path} does not exist")
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
    ):
        input_paths, label_paths = self._get_single_wavelet_input_and_label_paths(
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
        input_paths, label_paths = self._get_dual_channel_wavelet_input_and_label_paths(
            base_path=base_path,
            subject_list=training_subjects,
            training_phase=training_phase,
            wavelet_type=wavelet_type,
            ecg_labels=ecg_labels,
        )
        dataset = self._build_wavelet_dual_array_dataset(input_paths, label_paths, batch_size)
        return dataset, int(len(input_paths) / batch_size)

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
                    self.read_image_file, [input_path, label_path], [tf.float32, tf.float64]
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


class DeprecatedDatasetFactory:
    def __init__(self, base_path, batch_size=8, image_based=False, training_subjects=None, validation_subjects=None):
        self.base_path = base_path
        self.batch_size = batch_size
        self.image_based = image_based
        self.training_subjects = training_subjects
        self.validation_subjects = validation_subjects

    def batch_generator(self):
        base_path = Path(self.base_path)
        subjects = [path.name for path in base_path.iterdir() if path.is_dir()]
        if self.training_subjects is not None:
            subjects = [subject for subject in subjects if subject in self.training_subjects]
        while True:
            yield from self._get_inputs_and_labels_for_subjects(base_path, subjects)

    def _read_parallel(self, input_path, label_path, grouped_inputs, group):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._load_in_and_labels, input_path, label_path, grouped_inputs, number)
                for number in group
            ]
            return [fut.result() for fut in futures]

    def _get_inputs_and_labels_for_subjects_improved(self, base_path, subjects):
        for subject_id in subjects:
            subject_path = base_path / subject_id
            phases = [path.name for path in subject_path.iterdir() if path.is_dir()]
            for phase in phases:
                if phase == "logs" or phase == "raw":
                    continue
                phase_path = subject_path / phase
                input_path = phase_path / "inputs"
                label_path = phase_path / "labels_gaussian"
                input_names = sorted(path.name for path in input_path.iterdir() if path.is_file())
                grouped_inputs = {
                    int(k): list(filter(lambda x: "png" in x if self.image_based else "npy" in x, list(g)))
                    for k, g in groupby(input_names, key=lambda s: s.split("_")[0])
                }
                megabatches = self.grouper(grouped_inputs.keys(), 20 * self.batch_size)
                for megabatch in megabatches:
                    if all(v is None for v in megabatch):
                        continue
                    batch = self._read_parallel(input_path, label_path, grouped_inputs, megabatch)
                    for i in range(0, len(batch), self.batch_size):
                        inputs = np.stack([batch[i + j][0] for j in range(self.batch_size)], axis=0)
                        labels = np.stack([batch[i + j][1] for j in range(self.batch_size)], axis=0)
                        np.nan_to_num(inputs, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                        np.nan_to_num(labels, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                        yield inputs, labels
                    del batch
                del megabatches

    def _load_in_and_labels(self, input_path, label_path, grouped_inputs, number):
        in_files = grouped_inputs[number] if number is not None else None
        inputs = self.get_input(input_path, in_files)
        labels = self.get_labels(label_path, number)
        return inputs, labels

    def _get_inputs_and_labels_for_subjects(self, base_path, subjects):
        for subject_id in subjects:
            subject_path = base_path / subject_id
            phases = [path.name for path in subject_path.iterdir() if path.is_dir()]
            for phase in phases:
                if phase == "logs" or phase == "raw":
                    continue
                phase_path = subject_path / phase
                input_path = phase_path / "inputs"
                label_path = phase_path / "labels_gaussian"
                if not input_path.exists() or not label_path.exists():
                    continue
                input_names = sorted(path.name for path in input_path.iterdir() if path.is_file())
                input_names = list(filter(lambda x: "png" in x if self.image_based else "npy" in x, input_names))
                groups = self.grouper(input_names, self.batch_size)
                for group in groups:
                    inputs = np.stack(
                        [
                            np.load(input_path / number) if number is not None else self.get_input(input_path, None)
                            for number in group
                        ],
                        axis=0,
                    )
                    labels = np.stack(
                        [
                            np.load(label_path / number) if number is not None else self.get_labels(input_path, None)
                            for number in group
                        ],
                        axis=0,
                    )
                    yield inputs, labels
                    del inputs, labels
                    gc.collect()

    def _get_inputs_and_labels_for_subjects_grouped(self, base_path, subjects):
        for subject_id in subjects:
            subject_path = base_path / subject_id
            phases = [path.name for path in subject_path.iterdir() if path.is_dir()]
            for phase in phases:
                if phase == "logs" or phase == "raw":
                    continue
                phase_path = subject_path / phase
                input_path = phase_path / "inputs"
                label_path = phase_path / "labels_gaussian"
                if not input_path.exists() or not label_path.exists():
                    continue
                input_names = sorted(path.name for path in input_path.iterdir() if path.is_file())
                grouped_inputs = {
                    int(k): list(filter(lambda x: "png" in x if self.image_based else "npy" in x, list(g)))
                    for k, g in groupby(input_names, key=lambda s: s.split("_")[0])
                }
                inputs_sorted = list(grouped_inputs.keys())
                inputs_sorted.sort()
                groups = self.grouper(inputs_sorted, self.batch_size)
                for group in groups:
                    inputs = np.stack(
                        [
                            self.get_input(input_path, grouped_inputs[number])
                            if number is not None
                            else self.get_input(input_path, None)
                            for number in group
                        ],
                        axis=0,
                    )
                    labels = np.stack([self.get_labels(label_path, number) for number in group], axis=0)
                    yield inputs, labels
                    del inputs, labels
                    gc.collect()

    def grouper(self, iterable, n):
        iterators = [iter(iterable)] * n
        return zip_longest(*iterators)

    def get_input(self, base_path, imfs):
        if imfs is not None:
            return np.transpose(np.stack([self._load_input(base_path / imf) for imf in imfs], axis=0))
        elif not self.image_based and imfs is None:
            return np.transpose(np.stack([np.zeros((256, 1000)) for _ in range(5)], axis=0))
        elif self.image_based and imfs is None:
            return np.transpose(np.stack([np.zeros((256, 1000, 3)) for _ in range(5)], axis=0))

    def get_labels(self, base_path, number):
        if number is not None and "." not in number:
            return np.load(base_path / f"{number}.npy")
        elif number is not None and ".npy" in number:
            return np.load(base_path / f"{number}")
        else:
            return np.zeros((1000))

    def validation_generator(self):
        base_path = Path(self.base_path)
        subjects = [path.name for path in base_path.iterdir() if path.is_dir()]
        if self.validation_subjects is not None:
            subjects = [subject for subject in subjects if subject in self.validation_subjects]
        yield from self._get_inputs_and_labels_for_subjects(base_path, subjects)

    def _load_input(self, path):
        if path.suffix == ".png" and self.image_based:
            arr = img_to_array(load_img(path, target_size=(256, 1000)))
            if arr.shape != (256, 1000, 3):
                padded = np.zeros((256, 1000, 3))
                padded[: arr.shape[0], : arr.shape[1], : arr.shape[2]] = arr
                arr = padded
        elif path.suffix == ".npy" and not self.image_based:
            try:
                arr = self._pad_array(np.load(path))
            except Exception as e:
                print(f"Exception: {e}")
                arr = np.zeros((256, 1000))
        return arr

    def _pad_array(self, array):
        if array.shape != (256, 1000):
            padded = np.zeros((256, 1000))
            padded[: array.shape[0], : array.shape[1]] = array
            arr = padded
        return arr

    def get_training_dataset(self):
        batch_generator = self.batch_generator
        training_dataset = (
            tf.data.Dataset.from_generator(
                batch_generator,
                output_signature=(
                    tf.TensorSpec(shape=(self.batch_size, 1000, 256, 5), dtype=tf.float64),
                    tf.TensorSpec(shape=(self.batch_size, 1000), dtype=tf.float64),
                ),
            )
            .prefetch(tf.data.AUTOTUNE)
            .repeat()
        )
        return training_dataset

    def get_validation_dataset(self):
        validation_generator = self.validation_generator
        validation_dataset = (
            tf.data.Dataset.from_generator(
                validation_generator,
                output_signature=(
                    tf.TensorSpec(shape=(self.batch_size, 1000, 256, 5), dtype=tf.float64),
                    tf.TensorSpec(shape=(self.batch_size, 1000), dtype=tf.float64),
                ),
            )
            .batch(self.batch_size)
            .repeat()
            .prefetch(tf.data.AUTOTUNE)
        )
        return validation_dataset
