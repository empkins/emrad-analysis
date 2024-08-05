import os.path
from pathlib import Path
from typing import Tuple, List

import emrad_toolbox
import numpy
import numpy as np
import pywt
import resampy
from PyEMD import EMD
from matplotlib import pyplot as plt
from scipy.signal import filtfilt, butter
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from tpcp import Algorithm, Parameter, make_action_safe, cf

import pandas as pd


class ButterBandpassFilter(Algorithm):
    _action_methods = "filter"

    # Input Parameters
    high_pass_filter_cutoff_hz: Parameter[float]
    low_pass_filter_cutoff_hz: Parameter[float]
    band_pass_filter_order: Parameter[int]

    # Results
    filtered_signal_: pd.Series

    def __init__(
        self,
        high_pass_filter_cutoff_hz: float = 80,
        low_pass_filter_cutoff_hz: float = 18,
        band_pass_filter_order: int = 4,
    ):
        self.low_pass_filter_cutoff_hz = low_pass_filter_cutoff_hz
        self.high_pass_filter_cutoff_hz = high_pass_filter_cutoff_hz
        self.band_pass_filter_order = band_pass_filter_order

    @make_action_safe
    def filter(self, radar_data: pd.Series, sample_frequency_hz: float):
        """Bandpass filter, filtering the power signal of the radar

        Args:
            radar_data (pd.Series): rad (magnitude/power of complex radar signal)
            sample_frequency_hz (float): For Radar: When aligned already with biopac is 1000Hz, raw data is 1953.125Hz.

        Returns:
            _type_: bandpass-filterd signal
        """
        if isinstance(radar_data, pd.Series):
            radar = radar_data.to_numpy().flatten()
        else:
            radar = radar_data.flatten()

        nyq = 0.5 * sample_frequency_hz
        low = self.low_pass_filter_cutoff_hz / nyq
        high = self.high_pass_filter_cutoff_hz / nyq
        b, a = butter(self.band_pass_filter_order, [low, high], btype="band", analog=False)
        res = filtfilt(b, a, radar, axis=0)
        self.filtered_signal_ = pd.Series(res)
        return self


class Downsampling(Algorithm):
    _action_methods = "downsample"

    # Input Parameters
    target_sample_frequency_hz: Parameter[float]

    # Results
    downsampled_signal_: numpy.ndarray

    def __init__(self, target_sample_frequency_hz: float = 200):
        self.target_sample_frequency_hz = target_sample_frequency_hz

    @make_action_safe
    def downsample(self, signal: pd.Series, sample_frequency_hz: float, fs_in: float = 1000):
        """Downsample the input signal to the target frequency

        Args:
            signal (pd.Series): Input signal to be downsampled
            sample_frequency_hz (float): Sampling frequency of the input signal
            fs_in (float): Input sampling frequency
        Returns:
            _type_: Downsampled signal
        """
        if fs_in <= self.target_sample_frequency_hz:
            self.downsampled_signal_ = signal.to_numpy()
            return self
        if isinstance(signal, pd.Series):
            signal = signal.to_numpy()
        self.downsampled_signal_ = resampy.resample(
            signal, sr_orig=fs_in, sr_new=self.target_sample_frequency_hz, axis=0, parallel=True
        )
        return self


class EmpiricalModeDecomposer(Algorithm):
    _action_methods = "decompose"

    sampling_rate: Parameter[float]

    # Input Parameters
    n_imfs: Parameter[int]

    # Results
    imfs_: numpy.ndarray

    def __init__(self, n_imfs: int = 4, sampling_rate: float = 200):
        self.n_imfs = n_imfs
        self.sampling_rate = sampling_rate

    @make_action_safe
    def decompose(self, signal: numpy.array):
        """Decompose the input signal into IMFs

        Args:
            signal (pd.Series): Input signal to be decomposed

        Returns:
            _type_: Decomposed signal
        """
        # Hier kann manchmal ein leeres array zurückgegeben werden
        emd = EMD()
        self.imfs_ = emd.emd(signal, numpy.arange(len(signal)) / self.sampling_rate, max_imf=self.n_imfs)
        self.imfs_ = self.process_array(self.imfs_)
        # if len(self.imfs_) != 5:
        #     filled_array = numpy.zeros((5, 1000))
        #     filled_array[: self.imfs_.shape[0], : self.imfs_.shape[1]] = self.imfs_
        #     self.imfs_ = filled_array
        return self

    def process_array(self, input_array):
        output_array = np.zeros((5, 1000))
        if input_array.size > 0:
            n = input_array.shape[0]
            for i in range(n - 1):
                output_array[i, :] = input_array[i, :]
            output_array[4, :] = input_array[n - 1, :]
        return output_array


class Segmentation(Algorithm):
    _action_methods = "segment"

    # Input Parameters
    window_size_in_seconds: Parameter[int]
    overlap: Parameter[float]

    # Results
    segmented_signal_: list

    def __init__(self, window_size_in_seconds: int = 5, overlap: float = 0.4):
        self.window_size_in_seconds = window_size_in_seconds
        self.overlap = overlap

    @make_action_safe
    def segment(self, signal: pd.Series, sampling_rate: float):
        step_size = int(self.window_size_in_seconds - self.window_size_in_seconds * self.overlap)
        total_seconds = (signal.index.max() - signal.index.min()).total_seconds()
        step_count = int((total_seconds // step_size) - 1)
        start_time = signal.index[0]
        time_step = signal.index[1] - signal.index[0]
        segments = []
        for j in range(0, step_count):
            end = start_time + pd.Timedelta(seconds=self.window_size_in_seconds)
            # Preprocess the data
            data_segment = signal[start_time:end]
            start_time = start_time + pd.Timedelta(seconds=step_size)
            if len(data_segment) == 0:
                time_diff = pd.Timedelta(seconds=0)
            else:
                time_diff = data_segment.index[-1] - data_segment.index[0]
            # Zero padding
            if len(data_segment) < self.window_size_in_seconds * sampling_rate:
                if isinstance(data_segment, pd.Series):
                    zeros = self.zero_pad_series(data_segment)
                    data_segment = data_segment.append(zeros, ignore_index=True)
                elif isinstance(data_segment, pd.DataFrame) and len(data_segment) > 0:
                    zero_padded = self.zero_pad_df(data_segment)
                    data_segment = data_segment.append(zero_padded, ignore_index=True)
                elif len(data_segment) == 0:
                    print(f"Data segment is empty: {data_segment} in segment {j}")
                    continue
                data_segment.index = pd.date_range(
                    start=data_segment.index[0], periods=len(data_segment), freq=time_step
                )
            segments.append(data_segment)
        self.segmented_signal_ = segments
        return self

    def zero_pad_series(self, data_segment):
        # Get Start Time
        start_time = data_segment.index[0]
        end_time = start_time + pd.Timedelta(seconds=self.window_size_in_seconds)
        index = pd.date_range(start=start_time, end=end_time, freq="1ms")
        df = pd.Series(0, index=index)
        df.update(data_segment)
        return df

    def zero_pad_df(self, data_segment):
        # Get Start Time
        start_time = data_segment.index[0]
        end_time = start_time + pd.Timedelta(seconds=self.window_size_in_seconds)
        index = pd.date_range(start=start_time, end=end_time, freq="1ms")
        df = pd.DataFrame(0, index=index, columns=data_segment.columns)
        df.update(data_segment)
        return df


class Normalizer(Algorithm):
    _action_methods = "normalize"

    # Input Parameters
    method: Parameter[str]

    # Results
    normalized_signal_: pd.Series

    def __init__(self, method: str = "zscore"):
        self.method = method

    @make_action_safe
    def normalize(self, signal: pd.Series):
        """Normalize the input signal

        Args:
            signal (pd.Series): Input signal to be normalized

        Returns:
            _type_: Normalized signal
        """
        self.normalized_signal_ = self.safe_z_score_normalize(signal)
        self.normalized_signal_ = self.safe_min_max_normalize(self.normalized_signal_)
        return self

    def safe_z_score_normalize(self, data):
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return data
        else:
            return (data - mean) / std

    def safe_min_max_normalize(self, data):
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val == min_val:
            return np.zeros_like(data)
        else:
            return (data - min_val) / (max_val - min_val)


class WaveletTransformer(Algorithm):
    _action_methods = ("transform", "transform_diff")

    # Input Parameters
    wavelet_coefficients: Parameter[Tuple[int, int]]
    wavelet_type: Parameter[str]
    sampling_rate: Parameter[float]
    normalizer: Normalizer
    window_size: Parameter[int]
    num_imfs: Parameter[int]
    normalize: Parameter[bool]

    # Results
    transformed_signal_: numpy.array

    def __init__(
        self,
        wavelet_coefficients: Tuple[int, int] = (1, 257),
        wavelet_type="morl",
        sampling_rate: float = 200,
        window_size: int = 5,
        normalizer: Normalizer = cf(Normalizer()),
        num_imfs: int = 5,
        normalize: bool = True,
    ):
        self.wavelet_coefficients = wavelet_coefficients
        self.wavelet_type = wavelet_type
        self.sampling_rate = sampling_rate
        self.normalizer = normalizer
        self.window_size = window_size
        self.num_imfs = num_imfs
        self.normalize = normalize

    @make_action_safe
    def transform_diff(
        self,
        signal: numpy.array,
        subject_id: str,
        phase: str,
        segment: int,
        base_path: str = "Data",
        identity: bool = False,
    ):
        path = self.get_path(
            base_path=base_path, subject_id=subject_id, phase=phase, identity=identity, create_dir=False
        )
        path = path + f"_diff"
        if not os.path.exists(path):
            os.makedirs(path)
        scales = np.geomspace(
            self.wavelet_coefficients[0],
            self.wavelet_coefficients[1],
            num=self.wavelet_coefficients[1] - self.wavelet_coefficients[0],
        )
        coefficients, _ = pywt.cwt(signal, scales, "morl", sampling_period=1 / self.sampling_rate)
        np.save(os.path.join(path, f"{segment}.npy"), coefficients)
        self.transformed_signal_ = []
        return self

    @make_action_safe
    def transform(
        self,
        signal: numpy.array,
        subject_id: str,
        phase: str,
        segment: int,
        base_path: str = "Data",
        img_based: bool = False,
        single_signal: bool = False,
        identity: bool = False,
        diff: bool = False,
    ):
        """Transform the input signal using a wavelet transform

        Args:
            signal (pd.Series): Input signal to be transformed

        Returns:
            _type_: Transformed signal
        """

        if single_signal:
            self._calculate_single_signal(
                signal=signal,
                segment=segment,
                base_path=base_path,
                subject_id=subject_id,
                img_based=img_based,
                phase=phase,
                identity=identity,
            )
            self.transformed_signal_ = []
            return self

        path = self.get_path(base_path, subject_id, phase, identity=identity)
        transformed_signals = []
        for i in range(len(signal)):
            if i > len(signal) and not img_based:
                transformed_signals.append(self._get_empty_array())
                continue
            imf = signal[i]
            scales = np.geomspace(self.wavelet_coefficients[0], self.wavelet_coefficients[1], num=256)
            coefficients, frequencies = pywt.cwt(imf, scales, self.wavelet_type, sampling_period=1 / self.sampling_rate)
            if self.normalize:
                coefficients = self._normalize_and_fix_shape(coefficients)
            if img_based:
                self._save_image(coefficients, frequencies, len(imf), segment, i, path)
            else:
                transformed_signals.append(coefficients)
        if not img_based:
            path = path + f"_morl"
            if not os.path.exists(path):
                print(f"Creating path {path}")
                Path(path).mkdir(parents=True)
            else:
                print(f"Path {path} already exists")
            save_path = os.path.join(path, f"{segment}.npy")
            transformed_signals = np.stack(transformed_signals, axis=2)
            numpy.save(save_path, transformed_signals)
        self.transformed_signal_ = []
        return self

    def _normalize_and_fix_shape(self, coefficients):
        normalizer_clone = self.normalizer.clone()
        if np.iscomplexobj(coefficients):
            coefficients = np.abs(coefficients)
        coefficients_normalized = normalizer_clone.normalize(coefficients).normalized_signal_
        return coefficients_normalized

    def _calculate_single_signal(self, signal, segment, base_path, subject_id, phase, img_based, identity):
        wavelet_types = [
            # "morl",
            # "gaus1",
            "mexh",
            "shan1-1",
        ]
        for wavelet_type in wavelet_types:
            normalizer_clone = self.normalizer.clone()
            path = self.get_path(
                base_path=base_path, subject_id=subject_id, phase=phase, identity=identity, create_dir=False
            )
            scales = np.geomspace(
                self.wavelet_coefficients[0],
                self.wavelet_coefficients[1],
                num=self.wavelet_coefficients[1] - self.wavelet_coefficients[0],
            )
            path = path + f"_{wavelet_type}"
            if not os.path.exists(path):
                os.makedirs(path)
            log_path = path + f"_log"
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            coefficients, frequencies = pywt.cwt(signal, scales, wavelet_type, sampling_period=1 / self.sampling_rate)
            if np.iscomplexobj(coefficients):
                coefficients = np.abs(coefficients)
            coefficients_normalized = normalizer_clone.normalize(coefficients).normalized_signal_
            coefficients_reshaped = coefficients_normalized.reshape(
                coefficients_normalized.shape[0], coefficients_normalized.shape[1], 1
            )
            np.save(os.path.join(path, f"{segment}.npy"), coefficients_reshaped)
            # image_path = path + "_image"
            # self._save_image(coefficients, frequencies, -1, segment, 0, image_path)
            # coefficients_fit_for_log = np.abs(coefficients)
            # constant_value = coefficients_fit_for_log.min() / 2
            # coefficients_fit_for_log += constant_value
            # log_transformed_coefficients = FunctionTransformer(np.log1p, validate=True).fit_transform(
            #     coefficients_fit_for_log
            # )
            # log_transformed_coefficients = normalizer_clone.normalize(log_transformed_coefficients).normalized_signal_
            # log_transformed_coefficients = log_transformed_coefficients.reshape(
            #     log_transformed_coefficients.shape[0], log_transformed_coefficients.shape[1], 1
            # )
            # np.save(os.path.join(log_path, f"{segment}.npy"), log_transformed_coefficients)

    def _normalize(self, coefficients):
        normalizer_clone = self.normalizer.clone()
        normalizer_clone.normalize(pd.Series(coefficients.flatten()))
        shape = (self.wavelet_coefficients[1] - self.wavelet_coefficients[0], self.window_size * self.sampling_rate)
        if self.normalize:
            normalizer_clone.normalize(pd.Series(coefficients.flatten()))
            coefficients = normalizer_clone.normalized_signal_.to_numpy().reshape(coefficients.shape)
        if coefficients.shape != shape:
            zero_padding = numpy.zeros((shape[0], shape[1]))
            zero_padding[: coefficients.shape[0], : coefficients.shape[1]] = coefficients
            coefficients = zero_padding
        np.nan_to_num(coefficients, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        coefficients = coefficients.transpose()
        return coefficients

    def _get_empty_array(self):
        return numpy.zeros(
            (self.wavelet_coefficients[1] - self.wavelet_coefficients[0], self.window_size * self.sampling_rate)
        )

    def _save_array(self, coefficients, segment_nr, imf_nr, normalize, path):
        normalizer_clone = self.normalizer.clone()
        shape = (self.wavelet_coefficients[1] - self.wavelet_coefficients[0], self.window_size * self.sampling_rate)
        if normalize:
            normalizer_clone.normalize(pd.Series(coefficients.flatten()))
            coefficients = normalizer_clone.normalized_signal_.to_numpy().reshape(coefficients.shape)
        if coefficients.shape != shape:
            zero_padding = numpy.zeros((shape[0] - coefficients.shape[0], shape[1]))
            zero_padding[: coefficients.shape[0], : coefficients.shape[1]] = coefficients
            coefficients = zero_padding
        save_path = os.path.join(path, f"{segment_nr}_{imf_nr}.npy")
        numpy.save(save_path, coefficients)

    def _save_image(self, coefficients, frequencies, num_of_imfs, segment_nr, imf_nr, path):
        if not os.path.exists(path):
            os.makedirs(path)
        fig, ax = plt.subplots()
        time = numpy.arange(0, len(coefficients) / self.sampling_rate, 1 / self.sampling_rate)
        ax.imshow(
            numpy.abs(coefficients),
            aspect="auto",
            cmap="jet",
            extent=[time.min(), time.max(), frequencies.min(), frequencies.max()],
        )
        ax.set_xticks([])
        ax.set_yticks([])
        if num_of_imfs != -1:
            plt.savefig(os.path.join(path, f"{segment_nr}_{imf_nr}.png"), bbox_inches="tight", pad_inches=0)
        else:
            plt.savefig(os.path.join(path, f"{segment_nr}.png"), bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    def get_path(self, base_path: str, subject_id: str, phase: str, identity: bool = False, create_dir: bool = True):
        if identity:
            if phase is not None:
                path = f"{base_path}/{subject_id}/{phase}/inputs_identity_array"
            else:
                path = f"{base_path}/{subject_id}/inputs_identity_array"
        else:
            if phase is not None:
                path = f"{base_path}/{subject_id}/{phase}/inputs_wavelet_array"
            else:
                path = f"{base_path}/{subject_id}/inputs_wavelet_array"
        if not os.path.exists(path) and create_dir:
            os.makedirs(path)
        return path


class ImageGenerator(Algorithm):
    _action_methods = "generate"

    # Input Parameters
    coefficients: List
    frequencies: List
    phase: str
    subject_id: str

    # Results
    path_: str

    def generate(self, coefficients: List, frequencies: List, phase: str, subject_id: str, base_path: str):
        """Generate an image from the input coefficients and frequencies

        Args:
            coefficients (List): List of coefficients
            frequencies (List): List of frequencies
            phase (str): Phase of the signal
            subject_id (str): Subject ID

        Returns:
            _type_: Path to the generated image
        """
        emrad_toolbox.plotting.radar_plotting.RadarPlotter()

        return self
