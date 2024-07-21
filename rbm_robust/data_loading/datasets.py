import os
from pathlib import Path
from typing import Optional, Union, List, Tuple, Any, Dict

import numpy as np
from biopsykit.io.biopac import BiopacDataset
import pandas as pd
from empkins_io.sensors.emrad import EmradDataset
from empkins_io.sync import SyncedDataset
from scipy.signal import periodogram, find_peaks
from tpcp import Dataset
import neurokit2 as nk
from functools import lru_cache
from emrad_toolbox.radar_preprocessing.radar import RadarPreprocessor
import pytz
import warnings
import glob
from biopsykit.io.psg import PSGDataset
import itertools
from datetime import datetime, timedelta
from biopsykit.io import load_long_format_csv
from biopsykit.utils.dataframe_handling import multi_xs, wide_to_long
from biopsykit.utils.file_handling import get_subject_dirs
from empkins_io.utils._types import path_t
from rbm_robust.data_loading.base.dataset import BaseDataset
from typing import Dict, Optional, Sequence, Union
from itertools import product


class D02Dataset(Dataset):
    """
    D02Dataset is a subclass of the Dataset class that is specifically designed to handle the D02 dataset.
    It provides methods for loading ECG and radar data, synchronizing the data, and creating an index of participants.
    """

    SAMPLING_RATE_ACQ = 2000
    SAMPLING_RATE_RADAR = 1953.125
    SAMPLING_RATE_DOWNSAMPLED = 1000
    SAMPLING_RATE_DOWNSAMPLED_ML = 200
    _CHANNEL_MAPPING = {
        "ECG": "ecg",
        "SyncSignal": "Sync_Out",
    }
    _PHASE_MAPPING = {
        "lat": {"start": "lat_rating_n", "end": "lat_end"},
        "ei": {"start": "ei_01", "end": "ei_end"},
        "coping": {"start": "coping_trial_1", "end": "coping_end"},
        "training": {
            "start": "training_base-rating_start",
            "end": "training_end",
        },
    }
    EXCLUDE_SUBJECTS = (
        "005",
        "074",
        "044",
        "115",
        "Radar_DB",
        "418",
        "569",
        "535",
        "552",
        "161",
        "537",
        "350",
        "125",  # No Sync Out channel
        "572",  # No acq
        "303",  # Too large DS
        "597",  # No acq file
        "430",  # No acq file
        "512",  # Too large DS
        "158",  # No acq file
        "567",  # No acq file
        "324",  # No acq file
        "131",  # No raw folder
        "202",  # No raw folder
        "540",  # No acq file
        "325",  # Empty
        "565",  # No acq file
        "148",  # No acq file
        "545",  # No acq file
        "568",  # No acq file
        "556",  # No acq file
        "594",  # No acq file
        "502",  # No acq file
        "566",  # No acq file
        "403",
        "516",
        "132",
        "093",
        "123",
        "124",
        "574",
        "536",
        "453",
        "459",
        "318",
        "507",
        "469",
        "460",
        "306",
        "386",
        "276",
        "469",
        "518",
        "467",
        "321",
        "538",
        "438",
        "389",
        "511",
        "553",
    )

    AGAIN = ["245", "144", "249", "198", "207"]

    def __init__(
        self,
        data_path: Path,
        *,
        groupby_cols: Optional[Union[list[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize a D02Dataset instance.

        :param data_path: Path to the directory containing the D02 dataset.
        :param groupby_cols: Columns to group by when creating the index. Default is None.
        :param subset_index: Subset of the index to use. Default is None.
        """
        if not isinstance(data_path, Path):
            data_path = Path(data_path)
        self.data_path = data_path
        self.radar_dataset = None
        self.ecg_dataset = None
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self) -> pd.DataFrame:
        """
        Create an index of participants in the D02 dataset.

        :return: DataFrame containing the index.
        """
        participant_ids = [item.name for item in Path(self.data_path).iterdir() if item.is_dir()]
        participant_ids = [pid for pid in participant_ids if pid not in self.EXCLUDE_SUBJECTS]

        already = [item.name for item in Path("/home/woody/iwso/iwso116h/DataD02").iterdir() if item.is_dir()]
        already = [pid for pid in already if pid not in self.AGAIN]

        participant_ids = [pid for pid in participant_ids if pid not in already]

        df = pd.DataFrame({"participant": participant_ids})
        if df.empty:
            raise ValueError(
                f"The dataset is empty. Are you sure you selected the correct folder? "
                f"Current folder is: {self.data_path}"
            )
        return df

    @property
    def sampling_rate_radar(self):
        """
        Get the sampling rate of the radar data.

        :return: Sampling rate of the radar data.
        """
        return self.SAMPLING_RATE_RADAR

    @property
    def sampling_rate_ecg(self):
        """
        Get the sampling rate of the ECG data.

        :return: Sampling rate of the ECG data.
        """
        return self.SAMPLING_RATE_ACQ

    @property
    def subjects(self) -> list[str]:
        """
        Get a list of unique subjects in the D02 dataset.

        :return: List of unique subjects.
        """
        return self.index["participant"].unique().tolist()

    @property
    def ecg(self) -> pd.DataFrame:
        """
        Load the ECG data for the first subject in the D02 dataset.

        :return: DataFrame containing the ECG data.
        """
        if not (self.is_single(None) or (self.is_single(["participant"]))):
            raise ValueError("Data can only be accessed, when there is just a single participant in the dataset.")

        subject_id = self.subjects[0]
        return self._load_ecg(subject_id)

    @property
    def ecg_clean(self) -> pd.DataFrame:
        """
        Load the ECG data for the first subject in the D02 dataset.

        :return: DataFrame containing the ECG data.
        """
        if not (self.is_single(None) or (self.is_single(["participant"]))):
            raise ValueError("Data can only be accessed, when there is just a single participant in the dataset.")
        subject_id = self.subjects[0]
        ecg_signal = self._load_ecg(subject_id)
        ecg_clean = nk.ecg_clean(
            ecg_signal=ecg_signal["ecg"], sampling_rate=int(self.SAMPLING_RATE_ACQ), method="neurokit"
        )
        return pd.DataFrame(ecg_clean, columns=["ecg_clean"], index=ecg_signal.index)

    @property
    def synced_ecg(self) -> pd.DataFrame:
        """
        Load the synchronized ECG data for the first subject in the D02 dataset.

        :return: DataFrame containing the synchronized ECG data.
        """
        if not (self.is_single(None) or (self.is_single(["participant"]))):
            raise ValueError("Data can only be accessed, when there is just a single participant in the dataset.")
        subject_id = self.subjects[0]
        try:
            synced_data = self._load_synced_data_windowed(subject_id).copy()
        except Exception as _:
            synced_data = self._load_synced_data(subject_id).copy()
        ecg_signal = self.synced_data[["ecg"]]
        ecg_signal["ecg"] = nk.ecg_clean(
            ecg_signal=ecg_signal["ecg"], sampling_rate=int(self.SAMPLING_RATE_ACQ), method="neurokit"
        )
        return ecg_signal[["ecg"]]

    @property
    def filtered_radar(self) -> pd.DataFrame:
        if not (self.is_single(None) or (self.is_single(["participant"]))):
            raise ValueError("Data can only be accessed, when there is just a single participant in the dataset.")
        subject_id = self.subjects[0]
        radar_df = self._load_radar(subject_id)
        filtered_dict = RadarPreprocessor.butterworth_band_pass_filter(
            radar_df["I"], radar_df["Q"], filter_cutoff=(18, 80)
        )
        for key, value in filtered_dict.items():
            radar_df = pd.concat([radar_df, value], axis=1, keys=[key])
        return radar_df

    @property
    def filtered_synced_radar(self) -> pd.DataFrame:
        if not (self.is_single(None) or (self.is_single(["participant"]))):
            raise ValueError("Data can only be accessed, when there is just a single participant in the dataset.")
        radar_df = self.synced_data[["radar_I", "radar_Q"]]
        filtered_dict = RadarPreprocessor.butterworth_band_pass_filter(
            radar_df["radar_I"], radar_df["radar_Q"], filter_cutoff=(18, 80)
        )
        for key, value in filtered_dict.items():
            radar_df = pd.concat([radar_df, value], axis=1, keys=[key])
        return radar_df

    # @property
    # def synced_ecg(self) -> pd.DataFrame:
    #     if not (self.is_single(None) or (self.is_single(["participant"]))):
    #         raise ValueError("Data can only be accessed, when there is just a single participant in the dataset.")
    #     return self.synced_data[["ecg"]]

    @lru_cache(maxsize=1)
    def _load_ecg(self, subject_id: str) -> pd.DataFrame:
        """
        Load the ECG data for a specific subject.

        :param subject_id: ID of the subject.
        :return: DataFrame containing the ECG data.
        """
        subject_path = self.data_path.joinpath(subject_id, "raw")
        acq_path = self._get_only_matching_file_path(subject_path, "acq")
        # Load the ECG data
        return BiopacDataset.from_acq_file(acq_path, channel_mapping=self._CHANNEL_MAPPING).data_as_df(
            index="local_datetime"
        )

    @property
    @lru_cache(maxsize=1)
    def clean_phase_df(self):
        if not (self.is_single(None) or (self.is_single(["participant"]))):
            raise ValueError("Data can only be accessed, when there is just a single participant in the dataset.")
        subject_id = self.subjects[0]
        df = self._log_df(subject_id)
        df = self._clean_phase_df(df)
        return df

    @property
    def phases(self):
        if not (self.is_single(None) or (self.is_single(["participant"]))):
            raise ValueError("Data can only be accessed, when there is just a single participant in the dataset.")
        subject_id = self.subjects[0]
        return self._clean_phase_df(self._log_df(subject_id))

    @property
    @lru_cache(maxsize=1)
    def log(self):
        if not (self.is_single(None) or (self.is_single(["participant"]))):
            raise ValueError("Data can only be accessed, when there is just a single participant in the dataset.")
        subject_id = self.subjects[0]
        return self._log_df(subject_id)

        # try:
        #     subject_id = self.subjects[0]
        #     subject_path = self.data_path.joinpath(subject_id, "logs")
        #
        #     csv_files = glob.glob(os.path.join(subject_path, "*.csv"))
        #
        #     df = pd.DataFrame()
        #     for file in csv_files:
        #         log_df = pd.read_csv(file)
        #         log_df["timestamp"] = log_df["timestamp"].str.replace("\n", "")
        #         log_df["timestamp"] = pd.to_datetime(log_df["timestamp"])
        #         df = pd.concat([df, log_df], axis=1)
        #     return df
        # except Exception as e:
        #     warnings.warn(f"Could not load log data: {e}")
        #     return pd.DataFrame()

    @property
    def synced_radar(self) -> pd.DataFrame:
        if not (self.is_single(None) or (self.is_single(["participant"]))):
            raise ValueError("Data can only be accessed, when there is just a single participant in the dataset.")
        subject_id = self.subjects[0]
        return self._load_synced_radar(subject_id)

    @property
    def radar(self) -> pd.DataFrame:
        """
        Load the radar data for the first subject in the D02 dataset.

        :return: DataFrame containing the radar data.
        """
        if not (self.is_single(None) or (self.is_single(["participant"]))):
            raise ValueError("Data can only be accessed, when there is just a single participant in the dataset.")

        subject_id = self.subjects[0]
        return self._load_radar(subject_id, add_sync_out=True, add_sync_in=True)

    @lru_cache(maxsize=1)
    def _load_radar(self, subject_id: str, add_sync_in: bool = False, add_sync_out: bool = False) -> pd.DataFrame:
        """
        Load the radar data for a specific subject.

        :param subject_id: ID of the subject.
        :param add_sync_in: Whether to add the Sync_In channel to the data. Default is False.
        :param add_sync_out: Whether to add the Sync_Out channel to the data. Default is False.
        :return: DataFrame containing the radar data.
        """
        subject_path = self.data_path.joinpath(subject_id, "raw")
        h5_path = self._get_only_matching_file_path(subject_path, "h5")
        dataset = EmradDataset.from_hd5_file(h5_path, sampling_rate_hz=self.SAMPLING_RATE_RADAR)
        df = dataset.data_as_df(index="local_datetime", add_sync_in=add_sync_in, add_sync_out=add_sync_out)
        df.columns = [val[1] for val in df.columns]
        return df

    @staticmethod
    def _get_only_matching_file_path(path, file_type: str) -> str:
        """
        Get the only file of a specific type in a directory.

        :param path: Path to the directory.
        :param file_type: Type of the file to get.
        :return: Name of the file.
        """
        matching_files = list(path.glob(f"*.{file_type}"))
        if len(matching_files) == 0:
            raise ValueError(f"No {file_type} files found in {path}. Are you sure you selected the correct folder?")
        if len(matching_files) > 1:
            raise ValueError(
                f"Multiple {file_type} files found in {path}. Are you sure you selected the correct folder?"
            )
        return matching_files[0]

    @property
    def synced_filtered_data(self) -> pd.DataFrame:
        if not (self.is_single(None) or (self.is_single(["participant"]))):
            raise ValueError("Data can only be accessed, when there is just a single participant in the dataset.")
        synced_data = self.synced_data
        filtered_dict = RadarPreprocessor.butterworth_band_pass_filter(
            synced_data["radar_I"], synced_data["radar_Q"], filter_cutoff=(18, 80), fs=self.SAMPLING_RATE_DOWNSAMPLED
        )
        synced_data["Magnitude_band_pass"] = filtered_dict["Magnitude_band_pass"]
        synced_data["radar_I"] = filtered_dict["I_band_pass"]
        synced_data["radar_Q"] = filtered_dict["Q_band_pass"]
        synced_data["Magnitude_Filtered_Envelope"] = RadarPreprocessor.envelope(
            average_length=200, magnitude=synced_data["Magnitude_band_pass"]
        )
        return synced_data

    @property
    def synced_data(self) -> pd.DataFrame:
        """
        Load the synchronized ECG and radar data for the first subject in the D02 dataset.

        :return: DataFrame containing the synchronized data.
        """
        if not (self.is_single(None) or (self.is_single(["participant"]))):
            raise ValueError("Data can only be accessed, when there is just a single participant in the dataset.")

        subject_id = self.subjects[0]
        return self._load_synced_data(subject_id)

    @property
    @lru_cache(maxsize=1)
    def load_synced_window(self) -> pd.DataFrame:
        if not (self.is_single(None) or (self.is_single(["participant"]))):
            raise ValueError("Data can only be accessed, when there is just a single participant in the dataset.")

        subject_id = self.subjects[0]
        return self._load_synced_data_windowed(subject_id)

    @lru_cache(maxsize=1)
    def _load_synced_data_windowed(self, subject_id) -> pd.DataFrame:
        ecg_df = self._load_ecg(subject_id)
        # Load the radar data
        radar_df = self._load_radar(subject_id, add_sync_in=True, add_sync_out=False)
        radar_df.rename(columns={"Sync_In": "Sync_Out"}, inplace=True)

        # Resample the data
        synced = SyncedDataset(sync_type="m-sequence")
        # Synchronize the data
        synced.add_dataset("radar", data=radar_df, sync_channel_name="Sync_Out", sampling_rate=self.SAMPLING_RATE_RADAR)
        synced.add_dataset("ecg", data=ecg_df, sync_channel_name="Sync_Out", sampling_rate=self.SAMPLING_RATE_ACQ)
        synced.resample_datasets(fs_out=self.SAMPLING_RATE_DOWNSAMPLED, method="dynamic", wave_frequency=0.2)

        ecg_df = synced.datasets_resampled["ecg_resampled_"].copy()
        radar_df = synced.datasets_resampled["radar_resampled_"].copy()

        result_df = pd.DataFrame()

        start_time = ecg_df.index[0]
        end_time = ecg_df.index[-1]
        window = timedelta(minutes=7)
        while start_time < end_time:
            start = start_time
            stop = start_time + window
            if stop > end_time:
                stop = end_time
            a = start.strftime("%H:%M:%S")
            b = stop.strftime("%H:%M:%S")
            ecg_df_window = ecg_df.between_time(a, b).copy()
            radar_df_window = radar_df.between_time(a, b).copy()
            synced_window = SyncedDataset(sync_type="m-sequence")
            # Synchronize the data
            synced_window.add_dataset(
                "radar",
                data=radar_df_window,
                sync_channel_name="Sync_Out",
                sampling_rate=self.SAMPLING_RATE_DOWNSAMPLED,
            )
            synced_window.add_dataset(
                "ecg", data=ecg_df_window, sync_channel_name="Sync_Out", sampling_rate=self.SAMPLING_RATE_DOWNSAMPLED
            )
            # synced_window.resample_datasets(fs_out=self.SAMPLING_RATE_DOWNSAMPLED, method="dynamic", wave_frequency=0.2)
            synced_window.align_and_cut_m_sequence(
                primary="ecg",
                reset_time_axis=True,
                cut_to_shortest=True,
                sync_params={"sync_region_samples": (0, len(ecg_df_window))},
            )
            dict_shift = synced_window._find_shift(
                primary="ecg_aligned_", sync_params={"sync_region_samples": (-len(ecg_df_window), -1)}
            )
            synced_window.resample_sample_wise(primary="ecg_aligned_", dict_sample_shift=dict_shift)
            df_dict = synced_window.datasets_aligned
            result_df_window = pd.concat(df_dict.values(), axis=1, keys=df_dict.keys())
            result_df_window.columns = [
                "".join(col).replace("aligned_", "") if col[1] != "ecg" else "ecg"
                for col in result_df_window.columns.values
            ]
            result_df = pd.concat([result_df, result_df_window])
            start_time = stop
        return result_df

    @lru_cache(maxsize=1)
    def _load_synced_data(self, subject_id: str) -> pd.DataFrame:
        """
        Load the synchronized ECG and radar data for a specific subject.

        :param subject_id: ID of the subject.
        :return: DataFrame containing the synchronized data.
        """

        # Load the ECG data
        ecg_df = self._load_ecg(subject_id)

        # Load the radar data
        radar_df = self._load_radar(subject_id, add_sync_in=True, add_sync_out=False)
        radar_df.rename(columns={"Sync_In": "Sync_Out"}, inplace=True)

        # Synchronize the data
        synced_dataset = SyncedDataset(sync_type="m-sequence")
        synced_dataset.add_dataset(
            "radar", data=radar_df, sync_channel_name="Sync_Out", sampling_rate=self.SAMPLING_RATE_RADAR
        )
        synced_dataset.add_dataset(
            "ecg", data=ecg_df, sync_channel_name="Sync_Out", sampling_rate=self.SAMPLING_RATE_ACQ
        )
        synced_dataset.resample_datasets(fs_out=self.SAMPLING_RATE_DOWNSAMPLED, method="dynamic", wave_frequency=0.2)
        synced_dataset.align_and_cut_m_sequence(
            primary="ecg",
            reset_time_axis=True,
            cut_to_shortest=True,
            sync_params={"sync_region_samples": (0, 1000000)},
        )
        dict_shift = synced_dataset._find_shift(
            primary="ecg_aligned_", sync_params={"sync_region_samples": (-1000000, -1)}
        )
        synced_dataset.resample_sample_wise(primary="ecg_aligned_", dict_sample_shift=dict_shift)
        df_dict = synced_dataset.datasets_aligned
        result_df = pd.concat(df_dict.values(), axis=1, keys=df_dict.keys())
        print(result_df.columns)
        result_df.columns = [
            "".join(col).replace("aligned_", "") if col[1] != "ecg" else "ecg" for col in result_df.columns.values
        ]
        return result_df

    @lru_cache(maxsize=1)
    def _load_synced_radar(self, subject_id):
        try:
            synced_data = self._load_synced_data_windowed(subject_id).copy()
        except Exception as _:
            synced_data = self._load_synced_data(subject_id).copy()
        df = synced_data.filter(regex="^radar").copy()
        df.columns = [col.replace("radar_", "") for col in df.columns]
        if "Sync_Out" in df.columns:
            df.drop(columns=["Sync_Out"], inplace=True)
        if "Sync_In" in df.columns:
            df.drop(columns=["Sync_In"], inplace=True)
        return df

    def _load_synced_ecg(self, subject_id):
        try:
            synced_data = self._load_synced_data_windowed(subject_id).copy()
        except Exception as _:
            synced_data = self._load_synced_data(subject_id).copy()
        df = synced_data.filter(regex="^ecg")
        df.drop(columns=["ecg_Sync_Out"], inplace=True)
        return df

    def _log_df(self, subject_id: str) -> pd.DataFrame:
        subject_path = self.data_path.joinpath(subject_id, "logs")
        subject_path_cleaned = subject_path.joinpath("cleaned_logs")
        subject_path_cleaned_file = subject_path_cleaned.joinpath("cleaned_logs.csv")
        # if os.path.exists(subject_path_cleaned) and os.path.isfile(subject_path_cleaned_file):
        #     try:
        #         df = pd.read_csv(subject_path_cleaned_file, index_col="index")
        #         df.index = pd.to_datetime(df.index)
        #         return df
        #     except:
        #         return pd.DataFrame()

        csv_files = glob.glob(os.path.join(subject_path, "*.csv"))
        df = pd.DataFrame()
        for file in csv_files:
            log_df = pd.read_csv(file)
            log_df["timestamp"] = log_df["timestamp"].str.replace("\n", "")
            log_df = log_df[log_df["timestamp"].str.contains("\d", regex=True)]
            log_df["timestamp"] = pd.to_datetime(log_df["timestamp"])
            log_df.set_index(log_df["timestamp"], drop=True, inplace=True)
            log_df.drop(["timestamp", "content"], axis=1, inplace=True)
            log_df.index.names = ["index"]
            log_df.rename(index={0: "label"})
            df = pd.concat([df, log_df])
        df = df.sort_index()
        df = df.reset_index()
        df = df.drop_duplicates()
        # If you want to set the index back to its original column
        df.set_index("index", inplace=True)

        # if not os.path.exists(subject_path_cleaned):
        #     os.mkdir(subject_path_cleaned)
        # df.to_csv(subject_path_cleaned_file)
        return df

    def _clean_phase_df(self, df) -> Dict:
        # Check if already saved
        subject_id = self.subjects[0]
        subject_phase_path = self.data_path.joinpath(subject_id, "phases")
        subject_phase_path_file = subject_phase_path.joinpath("phases.csv")
        # if os.path.exists(subject_phase_path) and os.path.isfile(subject_phase_path_file):
        #     df = pd.read_csv(subject_phase_path_file, index_col=0)
        #     phases = df.to_dict(orient="index")
        #     return phases
        phases = {}
        alternative_ends = ["ei", "training", "coping", "lat"]
        start_codes = ["ei_01", "training_base-rating_start", "coping_trial_1", "lat_rating_n"]
        for phase in self._PHASE_MAPPING.keys():
            start_time = self._PHASE_MAPPING[phase]["start"]
            end_time = self._PHASE_MAPPING[phase]["end"]
            matching_rows_start = df[df["label"] == start_time].index.tolist()
            matching_rows_end = df[df["label"] == end_time].index.tolist()
            if len(matching_rows_start) > len(matching_rows_end):
                for start in matching_rows_start:
                    if len(matching_rows_start) == len(matching_rows_end):
                        break
                    fitting_alternative_ends = [end for end in alternative_ends if end != phase]
                    filtered_df = df[df.index > start]
                    mask = filtered_df["label"].apply(
                        lambda x: any(str(x).startswith(value) for value in fitting_alternative_ends)
                    )
                    mask_start = filtered_df["label"].apply(lambda x: any(str(x) == value for value in start_codes))
                    mask = mask | mask_start
                    filtered_df = filtered_df[mask]
                    end_to_add = (
                        df.index[df.index.get_loc(filtered_df.index[0]) - 1] if len(filtered_df) > 0 else df.index[-1]
                    )
                    matching_rows_end.append(end_to_add)
            elif len(matching_rows_start) < len(matching_rows_end):
                raise ValueError(f"Phase {phase} has more end times than start times.")

            if len(matching_rows_start) != len(matching_rows_end):
                raise ValueError(f"Phase {phase} has more end times than start times.")
            matching_rows_start.sort()
            matching_rows_end.sort()
            pairs = list(zip(matching_rows_start, matching_rows_end))
            index = 1
            for pair in pairs:
                if len(pairs) > 1:
                    phase_name = f"{phase}_{index}"
                    index += 1
                else:
                    phase_name = phase
                phases[phase_name] = {"start": pair[0], "end": pair[1]}
        # Save Phase dict as df
        phase_df = pd.DataFrame.from_dict(phases, orient="index")
        # if not os.path.exists(subject_phase_path):
        #     os.mkdir(subject_phase_path)
        # phase_df.to_csv(subject_phase_path_file)
        return phases


class RadarDatasetRaw(Dataset):
    SAMPLING_RATE_PSG: float = 256.0
    SAMPLING_RATE_RADAR: float = 1953.125
    POSITIONS = ["L", "0", "R"]
    POSTURES = ["1", "2", "3", "4", "5", "6"]
    MISSING = []
    EXCLUDE_SUBJECTS = ("02", "03", "16", "17")
    SYNCHRONIZABLE = {"19": 14, "20": 34, "21": 34}
    START_TIMES = {
        "04": "12:27:14",
        "06": "13:19:26",
        # "07": "11:23:53",
        "08": "14:35:07",
        "09": "10:43:05",
        "10": "10:14:21",
        "11": "11:39:40",
        "12": "14:57:30",
        "13": "14:15:50",
        "14": "16:02:00",
        "15": "10:21:35",
        "19": "17:15:28",
        "20": "12:42:39",
        "21": "15:50:38",
    }

    def __init__(
        self,
        data_path: Path,
        exclude_missing: bool = True,
        *,
        groupby_cols: Optional[Union[List[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None,
    ):
        self.data_path = data_path
        self.channels = ["AUX", "ECG II"]
        self.exclude_missing = exclude_missing
        self.PHASES = [
            f"{position} + {posture}" for position, posture in itertools.product(self.POSITIONS, self.POSTURES)
        ]
        self.psg_dataset = None
        self.radardataset = None
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self) -> pd.DataFrame:
        participant_ids = [f.name.split("_")[1] for f in sorted(self.data_path.glob("Subject*/"))]
        if self.exclude_missing:
            participant_ids = [pid for pid in participant_ids if pid not in self.MISSING]
        else:
            print("Incomplete Subjects included, watch out for missing datastreams")

        index = list(itertools.product(participant_ids, self.PHASES))
        index = [x for x in index if x[0] not in self.EXCLUDE_SUBJECTS]
        # set locale to german to get correct date format
        # locale.setlocale(locale.LC_ALL, "de_DE")
        df = pd.DataFrame(
            {
                "Subject": [ind[0] for ind in index],
                "Phase": [ind[1] for ind in index],
            }
        )

        if len(df) == 0:
            raise ValueError(
                "The dataset is empty. Are you sure you selected the correct folder? Current folder is: "
                f"{self.data_path}"
            )
        return df

    @property
    def phase_times(self) -> pd.DataFrame:
        """
        Returns a dataframe which contains
        - one row with subject_id number, phase name and time of the beginning of the phase if there is just a single phase
        - multiple rows with subject_id number, phase names and times of the beginning/ending of the phases if there are multiple phases
        Works only if there is just a single participant in the dataset.
        """
        if not (
            self.is_single(None) or (self.is_single(["Subject"]) and self.groupby("Phase").shape[0] == len(self.PHASES))
        ):
            raise ValueError("Data can only be accessed, when there is just a single participant in the dataset.")

        # Get Subject Number and Phase Names
        phase_names = self.index["Phase"]
        subject_number = self.index["Subject"][0]

        # Get Start and End Times for each phase
        time_pairs = [self.__get_phase_times(phase, subject_number) for phase in phase_names]
        start_dates = [time_pair[0] for time_pair in time_pairs]
        end_dates = [time_pair[1] for time_pair in time_pairs]

        # return dataframe with all phase times
        return pd.DataFrame(
            {
                "Subject": subject_number,  # Works although this is no list and just a single value
                "Phase": phase_names,
                "From": start_dates,
                "To": end_dates,
            }
        )

    @property
    def sampling_rate_radar(self) -> float:
        """Returns sampling rate of the Radar recording in Hz."""
        return self.SAMPLING_RATE_RADAR

    @property
    def sampling_rate_biopac(self) -> float:
        """The sampling rate of the PSG recording in Hz."""
        return self.SAMPLING_RATE_PSG

    @property
    def subjects(self) -> List[str]:
        """Returns a list of all subjects in the dataset."""
        return self.index["Subject"].unique().tolist()

    @property
    def ecg(self) -> pd.DataFrame:
        """Returns a dataframe with the ECG data of a single phase or for one subject."""
        ecg = self.synced_data
        ecg = ecg[("psg_aligned_", "ECG II")]
        return ecg

    @property
    def respiration(self) -> pd.DataFrame:
        """Returns a dataframe with the respiration data of a single phase or for one subject."""
        # load all data
        resp = self.synced_data

        # select only the respiration data
        thorax = resp[("psg_aligned_", "RIP Thora")]
        abdomen = resp[("psg_aligned_", "RIP Abdom")]
        thorax = thorax.rename("thorax respiration")
        abdomen = abdomen.rename("abdomen respiration")
        # concat thorax and abdomen to one dataframe
        resp = pd.concat([thorax, abdomen], axis=1)

        return resp

    @property
    def radar_top(self) -> pd.DataFrame:
        """Returns a dataframe with the radar data of a single phase or for one subject."""

        # load all data
        rad = self.synced_data

        # select only the top radar data
        first_level = ["radar_5_aligned_", "radar_6_aligned_", "radar_7_aligned_"]
        second_level = ["I", "Q"]
        multiindex = pd.MultiIndex.from_product([first_level, second_level])
        rad = rad[multiindex]

        # rename the columns
        rad = rad.rename(
            columns={"radar_5_aligned_": "rad5", "radar_6_aligned_": "rad6", "radar_7_aligned_": "rad7"}, level=0
        )

        return rad

    @property
    def radar_bottom(self) -> pd.DataFrame:
        """Returns a dataframe with the radar data of the four bottom sensors of a single phase or for one subject."""

        # load all data
        rad = self.synced_data

        # select only the radar data
        first_level = ["radar_1_aligned_", "radar_2_aligned_", "radar_3_aligned_", "radar_4_aligned_"]
        second_level = ["I", "Q"]
        multiindex = pd.MultiIndex.from_product([first_level, second_level])
        rad = rad[multiindex]

        # rename the columns
        rad = rad.rename(
            columns={
                "radar_1_aligned_": "rad1",
                "radar_2_aligned_": "rad2",
                "radar_3_aligned_": "rad3",
                "radar_4_aligned_": "rad4",
            },
            level=0,
        )

        return rad

    @property
    def synced_data(self) -> pd.DataFrame:
        """Returns a dataframe with all datastreams of a single phase or for one subject."""
        subject_id = self.subjects[0]
        if subject_id == "20" or subject_id == "21" or subject_id == "19":
            return self._load_special_sync(subject_id, phase=None)
        if self.is_single(["Phase"]):
            phase = self.index["Phase"][0]
            result_df = self._load_sync(subject_id, phase)
        elif self.is_single(["Subject"]) and self.groupby("Phase").shape[0] == len(self.PHASES):
            result_df = self._load_sync(subject_id=subject_id, phase=None)

        else:
            raise ValueError(
                "Data can only be accessed for a single participant or a single phase "
                "of one single participant in the subset"
            )
        return result_df

    @property
    def ecg_unsynced_uncut(self) -> pd.DataFrame:
        """Returns a dataframe with the ECG data of a single phase or for one subject."""
        subject_id = self.subjects[0]
        subject_str = "Subject_{}".format(subject_id)
        edf_file = subject_str + ".edf"
        edf_path = self.data_path / subject_str / edf_file

        # Create PSG Dataframe
        df = PSGDataset.from_edf_file(edf_path, datastreams=self.channels).data_as_df(index="local_datetime")
        df.rename(columns={"AUX": "Sync_Out"}, inplace=True)
        return df

    @property
    def freq_cut(self):
        df = self.ecg_raw
        sync_abs = np.abs(np.ediff1d(df["Sync_Out"]))
        fs = 256
        fft_sync, psd_sync = periodogram(sync_abs, fs=fs, window="hamming")
        psd_sync = SyncedDataset()._normalize_signal(psd_sync)

        idx_peak = find_peaks(psd_sync, height=0.5)[0][0]
        freq_sync = fft_sync[idx_peak]
        return freq_sync

    @property
    def freq_uncut(self):
        df = self.ecg_unsynced_uncut
        sync_abs = np.abs(np.ediff1d(df["Sync_Out"]))
        fs = 256
        fft_sync, psd_sync = periodogram(sync_abs, fs=fs, window="hamming")
        psd_sync = SyncedDataset()._normalize_signal(psd_sync)

        idx_peak = find_peaks(psd_sync, height=0.5)[0][0]
        freq_sync = fft_sync[idx_peak]
        return freq_sync

    @property
    def radar_1_bottom(self):
        """Returns a dataframe with the radar data of the first bottom sensor downsampled to 1000 Hz."""
        subject_id = self.subjects[0]
        subject_str = "Subject_{}".format(subject_id)
        radar_bottom_file = "data_{}-bett.h5".format(subject_id)

        # construct paths to data files
        radar_bottom_path = self.data_path / subject_str / radar_bottom_file

        # Create Radar Datasets
        radar_bottom_dataset = EmradDataset.from_hd5_file(radar_bottom_path)

        # Create Radar Dataframes
        radar_bottom_df = radar_bottom_dataset.data_as_df(index="local_datetime", add_sync_in=True, add_sync_out=True)
        # convert all NAN to zeroes
        radar_bottom_df = radar_bottom_df.fillna(0)

        # convert multicoloum from radar Sync_In and Sync_out to integer instead of float because some channels use float and some use integer
        for i in range(1, 4):
            if ("rad" + str(i), "Sync_In") not in radar_bottom_df.columns:
                continue
            radar_bottom_df[("rad" + str(i), "Sync_In")] = radar_bottom_df[("rad" + str(i), "Sync_In")].astype(int)
            radar_bottom_df[("rad" + str(i), "Sync_Out")] = radar_bottom_df[("rad" + str(i), "Sync_Out")].astype(int)

        radar_df = radar_bottom_df["rad1"]
        return radar_df

    @property
    def radar_1_downsampled(self) -> pd.DataFrame:
        """Returns a dataframe with the radar data of the first bottom sensor downsampled to 1000 Hz."""
        subject_id = self.subjects[0]
        subject_str = "Subject_{}".format(subject_id)
        radar_top_file = "data_{}.h5".format(subject_id)
        radar_bottom_file = "data_{}-bett.h5".format(subject_id)

        # construct paths to data files
        radar_top_path = self.data_path / subject_str / radar_top_file

        # Create Radar Datasets
        radar_top_dataset = EmradDataset.from_hd5_file(radar_top_path)

        # Create Radar Dataframes
        radar_top_df = radar_top_dataset.data_as_df(index="local_datetime", add_sync_in=True, add_sync_out=True)
        # convert all NAN to zeroes
        radar_top_df = radar_top_df.fillna(0)

        # convert multicoloum from radar Sync_In and Sync_out to integer instead of float because some channels use float and some use integer
        for i in range(1, 4):
            radar_top_df[("rad" + str(i), "Sync_In")] = radar_top_df[("rad" + str(i), "Sync_In")].astype(int)
            radar_top_df[("rad" + str(i), "Sync_Out")] = radar_top_df[("rad" + str(i), "Sync_Out")].astype(int)

        radar_df = radar_top_df["rad1"]
        synced_dataset = SyncedDataset(sync_type="m-sequence")
        synced_dataset.add_dataset(
            "radar_1", data=radar_df, sync_channel_name="Sync_In", sampling_rate=self.SAMPLING_RATE_RADAR
        )
        synced_dataset.resample_datasets(fs_out=256, method="dynamic", wave_frequency=250)
        return synced_dataset.datasets_resampled["radar_1_resampled_"]

    @lru_cache(maxsize=1)
    def _load_special_sync(self, subject_id: str, phase: Optional[str]) -> pd.DataFrame:
        subject_str = "Subject_{}".format(subject_id)
        edf_file = subject_str + ".edf"
        radar_top_file = "data_{}.h5".format(subject_id)
        radar_bottom_file = "data_{}-bett.h5".format(subject_id)

        # construct paths to data files
        edf_path = self.data_path / subject_str / edf_file
        radar_top_path = self.data_path / subject_str / radar_top_file
        radar_bottom_path = self.data_path / subject_str / radar_bottom_file

        # Create Radar Datasets
        radar_bottom_dataset = EmradDataset.from_hd5_file(radar_bottom_path)
        radar_top_dataset = EmradDataset.from_hd5_file(radar_top_path)

        # Create Radar Dataframes
        radar_bottom_df = radar_bottom_dataset.data_as_df(index="local_datetime", add_sync_in=True, add_sync_out=True)
        radar_top_df = radar_top_dataset.data_as_df(index="local_datetime", add_sync_in=True, add_sync_out=True)
        # convert all NAN to zeroes
        radar_bottom_df = radar_bottom_df.fillna(0)
        radar_top_df = radar_top_df.fillna(0)

        # Create PSG Dataframe
        df = PSGDataset.from_edf_file(edf_path, datastreams=self.channels).data_as_df(index="local_datetime")
        df.rename(columns={"AUX": "Sync_Out"}, inplace=True)
        change_start_time = subject_id in self.START_TIMES.keys()
        if change_start_time:
            print(f"Before cutting {len(df.index)}")
            df = df.between_time(self.START_TIMES[subject_id], df.index[-1].strftime("%H:%M:%S")).copy()
            print(f"After cutting {len(df.index)}")
        # # convert multicoloum from radar Sync_In and Sync_out to integer instead of float because some channels use float and some use integer
        for i in range(1, 5):
            radar_bottom_df[("rad" + str(i), "Sync_In")] = radar_bottom_df[("rad" + str(i), "Sync_In")].astype(int)
            radar_bottom_df[("rad" + str(i), "Sync_Out")] = radar_bottom_df[("rad" + str(i), "Sync_Out")].astype(int)
            if change_start_time:
                radar_bottom_df = radar_bottom_df.between_time(
                    self.START_TIMES[subject_id], radar_bottom_df.index[-1].strftime("%H:%M:%S")
                ).copy()
        for i in range(1, 4):
            radar_top_df[("rad" + str(i), "Sync_In")] = radar_top_df[("rad" + str(i), "Sync_In")].astype(int)
            radar_top_df[("rad" + str(i), "Sync_Out")] = radar_top_df[("rad" + str(i), "Sync_Out")].astype(int)
            if change_start_time:
                radar_top_df = radar_top_df.between_time(
                    self.START_TIMES[subject_id], radar_top_df.index[-1].strftime("%H:%M:%S")
                ).copy()

        # Create and fill Synced Dataset
        synced_dataset = SyncedDataset(sync_type="m-sequence")
        synced_dataset.add_dataset(
            "radar_1",
            data=radar_bottom_df["rad1"],
            sync_channel_name="Sync_In",
            sampling_rate=self.SAMPLING_RATE_RADAR,
        )
        synced_dataset.add_dataset(
            "radar_2", data=radar_bottom_df["rad2"], sync_channel_name="Sync_In", sampling_rate=self.SAMPLING_RATE_RADAR
        )
        synced_dataset.add_dataset(
            "radar_3", data=radar_bottom_df["rad3"], sync_channel_name="Sync_In", sampling_rate=self.SAMPLING_RATE_RADAR
        )
        synced_dataset.add_dataset(
            "radar_4", data=radar_bottom_df["rad4"], sync_channel_name="Sync_In", sampling_rate=self.SAMPLING_RATE_RADAR
        )
        synced_dataset.add_dataset(
            "radar_5", data=radar_top_df["rad1"], sync_channel_name="Sync_In", sampling_rate=self.SAMPLING_RATE_RADAR
        )
        synced_dataset.add_dataset(
            "radar_6", data=radar_top_df["rad2"], sync_channel_name="Sync_In", sampling_rate=self.SAMPLING_RATE_RADAR
        )
        synced_dataset.add_dataset(
            "radar_7", data=radar_top_df["rad3"], sync_channel_name="Sync_In", sampling_rate=self.SAMPLING_RATE_RADAR
        )
        synced_dataset.add_dataset("psg", data=df, sync_channel_name="Sync_Out", sampling_rate=self.SAMPLING_RATE_PSG)

        # Resample and align datasets

        wave_freq = 14 if subject_id == "19" else 34

        synced_dataset.resample_datasets(fs_out=256, method="dynamic", wave_frequency=wave_freq)
        synced_dataset.align_and_cut_m_sequence(
            primary="psg",
            reset_time_axis=True,
            cut_to_shortest=True,
            sync_params={"sync_region_samples": (0, 100000)},
        )
        dict_shift = synced_dataset._find_shift(
            primary="psg_aligned_", sync_params={"sync_region_samples": (-100000, -1)}
        )
        synced_dataset.resample_sample_wise(primary="psg_aligned_", dict_sample_shift=dict_shift)
        df_dict = synced_dataset.datasets_aligned

        # concat all dataframes to one
        result_df = pd.concat(df_dict.values(), axis=1, keys=df_dict.keys())
        # delete sync coloums
        multiindex = result_df.columns
        keep_columns = ~multiindex.get_level_values(1).str.contains("asd;lkfja;s")
        result_df = result_df.loc[:, keep_columns]

        # Cut df to phase and return
        return self.__cut_df(df=result_df, phase=phase, subject_id=subject_id)

    def _normalize_signal(cls, data: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def _determine_actual_sampling_rate_hack(self, dataset: Dict[str, Any], **kwargs):
        wave_frequency = kwargs.get("wave_frequency", None)
        data = dataset["data"]
        sync_channel = dataset["sync_channel"]
        fs = dataset["sampling_rate"]
        # sync_abs = np.abs(np.ediff1d(data[sync_channel]))
        sync_abs = data[sync_channel]
        print(fs)
        print(len(np.where(sync_abs == 1)[0]))
        fft_sync, psd_sync = periodogram(sync_abs, fs=fs, window="hamming")
        psd_sync = self._normalize_signal(psd_sync)

        # TODO: change to peak with highest amplitude instead of first peak (argmax)
        # idx_peak = find_peaks(psd_sync, height=0.5)[0][0]
        idx_peak = np.argmax(find_peaks(psd_sync, height=0.5)[0])

        print(f"Max Peak at {idx_peak}")
        print(fft_sync[find_peaks(psd_sync, height=0.5)[0]])
        freq_sync = fft_sync[idx_peak]

        print(wave_frequency)
        print(freq_sync)
        fs_measured = (wave_frequency / freq_sync) * fs
        print(f"Measured sampling rate: {fs_measured}")
        print(f"Sync frequency: {freq_sync}")

        return fft_sync, psd_sync

    @lru_cache(maxsize=1)
    def _load_sync(self, subject_id: str, phase: Optional[str]) -> pd.DataFrame:
        """Loads the sync data of a single phase or for one subject."""
        subject_str = "Subject_{}".format(subject_id)
        edf_file = subject_str + ".edf"
        radar_top_file = "data_{}.h5".format(subject_id)
        radar_bottom_file = "data_{}-bett.h5".format(subject_id)

        # construct paths to data files
        edf_path = self.data_path / subject_str / edf_file
        radar_top_path = self.data_path / subject_str / radar_top_file
        radar_bottom_path = self.data_path / subject_str / radar_bottom_file

        # Create Radar Datasets
        radar_bottom_dataset = EmradDataset.from_hd5_file(radar_bottom_path)
        radar_top_dataset = EmradDataset.from_hd5_file(radar_top_path)

        # Create Radar Dataframes
        radar_bottom_df = radar_bottom_dataset.data_as_df(index="local_datetime", add_sync_in=True)
        radar_top_df = radar_top_dataset.data_as_df(index="local_datetime", add_sync_in=True)
        # convert all NAN to zeroes
        radar_bottom_df = radar_bottom_df.fillna(0)
        radar_top_df = radar_top_df.fillna(0)

        # Create PSG Dataframe
        df = PSGDataset.from_edf_file(edf_path, datastreams=self.channels).data_as_df(index="local_datetime")
        df.rename(columns={"AUX": "Sync_Out"}, inplace=True)
        change_start_time = subject_id in self.START_TIMES.keys()
        if change_start_time:
            # df = df.between_time("11:24", "12:06")
            df = df.between_time(self.START_TIMES[subject_id], df.index[-1].strftime("%H:%M:%S")).copy()

        # convert multicoloum from radar Sync_In and Sync_out to integer instead of float because some channels use float and some use integer
        for i in range(1, 5):
            radar_bottom_df[("rad" + str(i), "Sync_In")] = radar_bottom_df[("rad" + str(i), "Sync_In")].astype(int)
            # radar_bottom_df[("rad" + str(i), "Sync_Out")] = radar_bottom_df[("rad" + str(i), "Sync_Out")].astype(int)
            if change_start_time:
                radar_bottom_df = radar_bottom_df.between_time(
                    self.START_TIMES[subject_id], radar_bottom_df.index[-1].strftime("%H:%M:%S")
                ).copy()

        for i in range(1, 4):
            radar_top_df[("rad" + str(i), "Sync_In")] = radar_top_df[("rad" + str(i), "Sync_In")].astype(int)
            # radar_top_df[("rad" + str(i), "Sync_Out")] = radar_top_df[("rad" + str(i), "Sync_Out")].astype(int)
            if change_start_time:
                radar_top_df = radar_top_df.between_time(
                    self.START_TIMES[subject_id], radar_top_df.index[-1].strftime("%H:%M:%S")
                ).copy()

        df = self._cut_to_first_switch(df, "Sync_Out")

        # TODO: Take the first radar data of the bottom sensor and use it as a seperate dataset
        radar_bottom_first = radar_bottom_df["rad1"]

        radar_bottom_first = self._cut_to_first_switch(radar_bottom_first, "Sync_In")
        # Reset Time Axis
        radar_bottom_first.index = radar_bottom_first.index - radar_bottom_first.index[0] + df.index[0]

        # radar_bottom_first.index = pd.to_timedelta(radar_bottom_first.index - radar_bottom_first.index[0])
        # radar_bottom_first.index = df.index[0] + radar_bottom_first.index

        # radar_bottom_df = self._cut_to_first_switch(radar_bottom_df, "Sync_In")
        # radar_bottom_df.index = pd.to_timedelta(radar_bottom_df.index - radar_bottom_df.index[0])
        # radar_bottom_df.index = df.index[0] + radar_bottom_df.index
        # radar_top_df = self._cut_to_first_switch(radar_top_df, "Sync_In")

        # TODO: Erst die Daten cutten auf das Erscheinen der ersten Null und dann bei einem Datensatz den Timeindex
        #   von einem Datensatz zu einem Timedelta Index ändern und dann die Daten zusammenfügen und wieder auf die
        #   gemeinsame Zeitachse resamplen

        # TODO: Hier erst mal Plotten und prüfen ob beide zur selben Zeit anfangen

        # Create and fill Synced Dataset
        synced = SyncedDataset(sync_type="m-sequence")
        synced.add_dataset("psg", data=df, sync_channel_name="Sync_Out", sampling_rate=self.SAMPLING_RATE_PSG)
        synced.add_dataset(
            "radar_1",
            data=radar_bottom_first,
            sync_channel_name="Sync_In",
            sampling_rate=self.SAMPLING_RATE_RADAR,
        )

        # synced.add_dataset(
        #     "radar_2", data=radar_bottom_df["rad2"], sync_channel_name="Sync_In", sampling_rate=self.SAMPLING_RATE_RADAR
        # )
        # synced.add_dataset(
        #     "radar_3", data=radar_bottom_df["rad3"], sync_channel_name="Sync_In", sampling_rate=self.SAMPLING_RATE_RADAR
        # )
        # synced.add_dataset(
        #     "radar_4", data=radar_bottom_df["rad4"], sync_channel_name="Sync_In", sampling_rate=self.SAMPLING_RATE_RADAR
        # )
        # synced.add_dataset(
        #     "radar_5", data=radar_top_df["rad1"], sync_channel_name="Sync_In", sampling_rate=self.SAMPLING_RATE_RADAR
        # )
        # synced.add_dataset(
        #     "radar_6", data=radar_top_df["rad2"], sync_channel_name="Sync_In", sampling_rate=self.SAMPLING_RATE_RADAR
        # )
        # synced.add_dataset(
        #     "radar_7", data=radar_top_df["rad3"], sync_channel_name="Sync_In", sampling_rate=self.SAMPLING_RATE_RADAR
        # )
        # Resample and align datasets
        synced.resample_datasets(fs_out=256, method="dynamic", wave_frequency=250)
        synced.align_and_cut_m_sequence(
            primary="radar_1",
            reset_time_axis=True,
            cut_to_shortest=True,
            sync_params={"sync_region_samples": (0, 100000)},
        )
        dict_shift = synced._find_shift(primary="psg_aligned_", sync_params={"sync_region_samples": (-100000, -1)})
        synced.resample_sample_wise(primary="psg_aligned_", dict_sample_shift=dict_shift)
        df_dict = synced.datasets_aligned

        # concat all dataframes to one
        result_df = pd.concat(df_dict.values(), axis=1, keys=df_dict.keys())
        # delete sync coloums
        multiindex = result_df.columns
        keep_columns = ~multiindex.get_level_values(1).str.contains("asd;lkfja;s")
        result_df = result_df.loc[:, keep_columns]

        # Cut df to phase and return
        return self.__cut_df(df=result_df, phase=phase, subject_id=subject_id)

    def _cut_to_first_switch(self, df: pd.DataFrame, sync_column: str) -> pd.DataFrame:
        """Cuts the dataframe to the first switch of the sync signal."""
        if isinstance(df.columns, pd.MultiIndex):
            # Do the same as in the else branch but for all Multiindex columns that contain the sync_column
            sync_columns = [col for col in df.columns if sync_column in col]
            first_switch = max([(df[sync] == 1).idxmax() for sync in sync_columns])
            # first_switch = max([[df[sync][df[sync]].diff() != 0][0].index[0] for sync in sync_columns])
            return df.loc[first_switch:]
        else:
            sync_signal = df[sync_column]
            if sync_column != "Sync_In":
                sync_signal = 0.5 * (np.sign(sync_signal - np.mean(sync_signal)) + 1)
            first_switch = sync_signal[sync_signal != 0].index[0]
            index_pos = df.index.get_loc(first_switch)
            first_switch = df.index[index_pos - 1]
            return df.loc[first_switch:]

    @property
    def radar_1_cut_and_reset(self):
        df = self.radar_1_bottom
        subject_id = self.subjects[0]
        if subject_id in self.START_TIMES.keys():
            df = df.between_time(self.START_TIMES[subject_id], df.index[-1].strftime("%H:%M:%S"))
        if subject_id == "04":
            df = df.between_time(self.START_TIMES[subject_id], "13:10:50")
        df = self._cut_to_first_switch(df, "Sync_In")
        df.index = df.index - df.index[0] + self.psg_cut.index[0]
        return df

    @property
    @lru_cache(maxsize=1)
    def psg_cut(self):
        subject_id = self.subjects[0]
        df = self.ecg_raw
        if subject_id in self.START_TIMES.keys():
            df = df.between_time(self.START_TIMES[subject_id], df.index[-1].strftime("%H:%M:%S"))
        if subject_id == "04":
            df = df.between_time(self.START_TIMES[subject_id], "13:10:50")
        if subject_id == "10":
            df = df.between_time(self.START_TIMES[subject_id], "11:01:53")
        df = self._cut_to_first_switch(df, "Sync_Out")
        return df

    @property
    def psg_cut_binarized(self):
        df = self.psg_cut.copy()
        df["Sync_Out_Bin"] = 0.5 * (np.sign(df["Sync_Out"] - np.mean(df["Sync_Out"])) + 1)
        return df

    @property
    def psg_cut_resampled(self):
        df = self.psg_cut

        sync = SyncedDataset(sync_type="m-sequence")
        sync.add_dataset(name="psg", data=df, sync_channel_name="Sync_Out", sampling_rate=self.SAMPLING_RATE_PSG)
        sync.resample_datasets(fs_out=100, method="dynamic", wave_frequency=2)

        return sync.datasets_resampled["psg_resampled_"]

    @property
    def radar_cut_reset_resampled(self):
        df = self.radar_1_cut_and_reset
        sync = SyncedDataset(sync_type="m-sequence")
        sync.add_dataset(name="radar_1", data=df, sync_channel_name="Sync_In", sampling_rate=self.SAMPLING_RATE_RADAR)
        sync.resample_datasets(fs_out=100, method="dynamic", wave_frequency=250)
        return sync.datasets_resampled["radar_1_resampled_"]

    @property
    def wave_freq_psg_uncut(self):
        # wave_frequency = kwargs.get("wave_frequency", None)
        data = self.ecg_raw
        sync_channel = "Sync_Out"
        fs = 256
        if "ECG II" in data.columns:
            binarized = 0.5 * (np.sign(data[sync_channel] - np.mean(data[sync_channel])) + 1)
            sync_abs = np.abs(np.ediff1d(binarized))
        else:
            sync_abs = np.abs(np.ediff1d(data[sync_channel]))
        fft_sync, psd_sync = periodogram(sync_abs, fs=fs, window="hamming")
        psd_sync = self._normalize_signal(psd_sync)

        idx_peak = np.argmax(psd_sync)
        freq_sync = fft_sync[idx_peak]
        return freq_sync

    @property
    def wave_freq_psg_cut(self):
        # wave_frequency = kwargs.get("wave_frequency", None)
        data = self.psg_cut
        sync_channel = "Sync_Out"
        fs = 256
        if "ECG II" in data.columns:
            binarized = 0.5 * (np.sign(data[sync_channel] - np.mean(data[sync_channel])) + 1)
            sync_abs = np.abs(np.ediff1d(binarized))
        else:
            sync_abs = np.abs(np.ediff1d(data[sync_channel]))
        fft_sync, psd_sync = periodogram(sync_abs, fs=fs, window="hamming")
        psd_sync = self._normalize_signal(psd_sync)

        idx_peak = np.argmax(psd_sync)
        freq_sync = fft_sync[idx_peak]
        return freq_sync

    @property
    def test_sync(self):
        psg = self.psg_cut
        radar = self.radar_1_cut_and_reset

        # Create and fill Synced Dataset
        synced = SyncedDataset(sync_type="m-sequence")
        synced.add_dataset(name="psg", data=psg, sync_channel_name="Sync_Out", sampling_rate=self.SAMPLING_RATE_PSG)
        synced.add_dataset(
            name="radar_1", data=radar, sync_channel_name="Sync_In", sampling_rate=self.SAMPLING_RATE_RADAR
        )
        # Resample
        synced.resample_datasets(fs_out=256, method="dynamic", wave_frequency=250)
        # Align
        synced.align_and_cut_m_sequence(
            primary="psg",
            reset_time_axis=True,
            cut_to_shortest=True,
        )
        dict_shift = synced._find_shift(primary="psg_aligned_", sync_params={"sync_region_samples": (-100000, -1)})
        synced.resample_sample_wise(primary="psg_aligned_", dict_sample_shift=dict_shift)
        df_dict = synced.datasets_aligned
        result_df = pd.concat(df_dict.values(), axis=1, keys=df_dict.keys())
        multiindex = result_df.columns
        keep_columns = ~multiindex.get_level_values(1).str.contains("asd;lkfja;s")
        result_df = result_df.loc[:, keep_columns]
        return result_df

    @property
    @lru_cache(maxsize=1)
    def radar_raw(self):
        subject_id = self.subjects[0]
        subject_str = "Subject_{}".format(subject_id)
        radar_bottom_file = "data_{}-bett.h5".format(subject_id)
        radar_bottom_path = self.data_path / subject_str / radar_bottom_file
        radar_bottom_dataset = EmradDataset.from_hd5_file(radar_bottom_path)
        df = radar_bottom_dataset.data_as_df(index="local_datetime", add_sync_in=True, add_sync_out=True)
        return df

    @property
    def ecg_sampling_rate(self) -> float:
        """Returns the sampling rate of the ECG data."""
        subject_id = self.subjects[0]
        subject_str = "Subject_{}".format(subject_id)
        edf_file = subject_str + ".edf"
        edf_path = self.data_path / subject_str / edf_file

        # Create PSG Dataframe
        psg = PSGDataset.from_edf_file(edf_path, datastreams=self.channels)
        return psg.sampling_rate

    @property
    def radar_sampling_rate(self) -> float:
        """Returns the sampling rate of the radar data."""
        subject_id = self.subjects[0]
        subject_str = "Subject_{}".format(subject_id)
        radar_bottom_file = "data_{}-bett.h5".format(subject_id)
        radar_top_file = "data_{}.h5".format(subject_id)
        radar_top_path = self.data_path / subject_str / radar_top_file

        radar_bottom_path = self.data_path / subject_str / radar_bottom_file
        radar_bottom_dataset = EmradDataset.from_hd5_file(radar_bottom_path)
        radar_top_dataset = EmradDataset.from_hd5_file(radar_top_path)

        df_bottom = radar_bottom_dataset.data_as_df(index="local_datetime", add_sync_in=True, add_sync_out=True)
        df_top = radar_top_dataset.data_as_df(index="local_datetime", add_sync_in=True, add_sync_out=True)

        total_seconds_bottom = (df_bottom.index[-1] - df_bottom.index[0]).total_seconds()
        freq_bottom = len(df_bottom) / total_seconds_bottom

        total_seconds_top = (df_top.index[-1] - df_top.index[0]).total_seconds()
        freq_top = len(df_top) / total_seconds_top

        nan_count_bottom = df_bottom.isna().sum().sum()
        nan_count_top = df_top.isna().sum().sum()
        # return freq_bottom, freq_top, nan_count_bottom, nan_count_top
        return 1.0

    @property
    @lru_cache(maxsize=1)
    def radar_cut(self) -> pd.DataFrame:
        """Like radar raw but it has been cut"""
        subject_id = self.subjects[0]
        df = self.radar_raw
        if subject_id in self.START_TIMES.keys():
            df = df.between_time(self.START_TIMES[subject_id], df.index[-1].strftime("%H:%M:%S"))
        return df

    @property
    @lru_cache(maxsize=1)
    def ecg_cut(self):
        """Like ecg_raw but it has been cut"""
        subject_id = self.subjects[0]
        df = self.ecg_raw
        if subject_id in self.START_TIMES.keys():
            df = df.between_time(self.START_TIMES[subject_id], df.index[-1].strftime("%H:%M:%S"))
        return df

    @property
    @lru_cache(maxsize=1)
    def ecg_raw(self):
        subject_id = self.subjects[0]
        subject_str = "Subject_{}".format(subject_id)
        edf_file = subject_str + ".edf"
        edf_path = self.data_path / subject_str / edf_file

        # Create PSG Dataframe
        df = PSGDataset.from_edf_file(edf_path, datastreams=self.channels).data_as_df(index="local_datetime")
        df.rename(columns={"AUX": "Sync_Out"}, inplace=True)

        return df

    @property
    def ecg_available_channels(self) -> List[str]:
        subject_id = self.subjects[0]
        subject_str = "Subject_{}".format(subject_id)
        edf_file = subject_str + ".edf"
        edf_path = self.data_path / subject_str / edf_file

        # Create PSG Dataframe
        psg = PSGDataset.from_edf_file(edf_path)
        return psg.channels

    @property
    def ecg_raw_wave_freq(self):
        df = self.ecg_raw
        sync_abs = np.abs(np.ediff1d(df["Sync_Out"]))
        fs = 256
        fft_sync, psd_sync = periodogram(sync_abs, fs=fs, window="hamming")
        psd_sync = SyncedDataset()._normalize_signal(psd_sync)

        idx_peak = find_peaks(psd_sync, height=0.5)[0][0]
        freq_sync = fft_sync[idx_peak]
        return freq_sync

    @property
    def ecg_wave_freq_middle_part(self):
        df = self.ecg_cut
        start = df.index[0] + timedelta(minutes=20)
        end = start + timedelta(minutes=10)

        start_time_str = start.strftime("%H:%M:%S")
        end_time_str = end.strftime("%H:%M:%S")

        df = df.between_time(start_time_str, end_time_str).copy()

        return self._get_wave_freq("Sync_Out", 256, df)

    @property
    def radar_raw_wave_freq(self):
        df = self.radar_raw
        df = df.fillna(0)
        radar_wave_freq = {}
        fs = 1953.125
        for name, sub_df in df.groupby(level=0, axis=1):
            single_df = sub_df.droplevel(0, axis=1)
            radar_wave_freq[name] = self._get_wave_freq("Sync_In", fs, single_df)
        return radar_wave_freq

    @property
    def radar_cut_wave_freq(self):
        df = self.radar_cut
        df = df.fillna(0)
        radar_wave_freq = {}
        fs = 1953.125
        for name, sub_df in df.groupby(level=0, axis=1):
            single_df = sub_df.droplevel(0, axis=1)
            single_df["Sync_In"] = SyncedDataset()._binarize_signal(single_df["Sync_In"])
            radar_wave_freq[name] = self._get_wave_freq("Sync_In", fs, single_df)
        return radar_wave_freq

    @property
    def ecg_cut_wave_freq(self):
        df = self.ecg_cut
        return self._get_wave_freq("Sync_Out", 256, df)

    def _get_wave_freq(self, sync_column: str, fs: float, df: pd.DataFrame):
        sync_abs = np.abs(np.ediff1d(df[sync_column]))
        fft_sync, psd_sync = periodogram(sync_abs, fs=fs, window="hamming")
        psd_sync = SyncedDataset()._normalize_signal(psd_sync)
        idx_peak = find_peaks(psd_sync, height=0.5)[0][0]
        freq_sync = fft_sync[idx_peak]
        return freq_sync

    @staticmethod
    def _reduce_df(df: pd.DataFrame, sync_column: str) -> pd.DataFrame:
        bin_df = SyncedDataset()._binarize_signal(df[sync_column])
        start_index = df.index.get_loc((bin_df == 1).idxmax())
        start_index = df.index[start_index - 1]
        end_index = df.index.get_loc((bin_df[::-1] == 1).idxmax())
        end_index = df.index[end_index + 1]
        return df.loc[start_index:end_index]

    def __get_csv_file(self, path: Path) -> Path:
        """
        returns name of the ATimeLoggerCSV
        could be integrated in get_phase_times
        expects the path to a folder of the Subject as argument where a csv file from the ATimeLogger is located
        """
        # get path of csv file in that folder
        csv_name = glob.glob(f"{path}/*.csv")
        if len(csv_name) == 0:
            raise FileNotFoundError("There is no csv file with phase times in {}".format(path))
        if len(csv_name) > 1:
            warnings.warn("There are multiple csv files in the folder. Using the first one.")
        csv_path = Path(csv_name[0])

        if not csv_path.is_file():
            raise FileNotFoundError("There is no csv file named {}".format(csv_path))

        return csv_path

    def __get_phase_times(self, phase: Optional[str], subject_id: str) -> (datetime, datetime):
        """
        Returns the start and end time of a phase as datetime object
        expects the phase as string and the subject_id as string
        """

        # create dataframe from csv file
        subject_folder = self.data_path / "Subject_{}".format(subject_id)
        csv_path = self.__get_csv_file(subject_folder)
        df_phase_times = pd.read_csv(csv_path, delimiter=",", header=0, skipfooter=5, engine="python")

        # Reverse Dataframe to get Phases in the right order (from first to last)
        df_phase_times = df_phase_times.iloc[::-1].reset_index(drop=True)

        # Get the start and end time of the phase
        phase_number = self.PHASES.index(phase)
        start_time = df_phase_times["From"][phase_number]
        end_time = df_phase_times["To"][phase_number]

        # Convert to datetime object
        format = "%Y-%m-%d %H:%M:%S"
        start_datetime = datetime.strptime(start_time, format)
        end_datetime = datetime.strptime(end_time, format)

        # offset on these times needed.
        offset = self.data_path / "Subject_{}".format(subject_id) / "offset_seconds.txt"
        with open(offset, "r") as f:
            file_content = f.read()
            if file_content == "":
                offset_seconds = 0
            else:
                offset_seconds = int(file_content)

            start_datetime = start_datetime - timedelta(seconds=offset_seconds)
            end_datetime = end_datetime - timedelta(seconds=offset_seconds)
        return start_datetime, end_datetime

    def __cut_df(self, df: pd.DataFrame, phase: str, subject_id: str) -> pd.DataFrame:
        """
        Cut df with time index to only the rows within the phase time interval
        if phase is None than the whole Signal should be returned beginning from the first official Phase from self.PHASES[0].
        """

        # determine offset either 60 if subject ID is 17 or higher or 120 for the rest (recording issues)
        subject_number = int(subject_id)
        offset = 120
        if subject_number >= 17:
            offset = 60
        if phase is None:
            # no phase given, return whole signal -> beginning from first phase
            first_phase = self.PHASES[0]
            beginning = self.__get_phase_times(subject_id=subject_id, phase=first_phase)[0]
            # Make Timezone aware for indexing
            beginning = pytz.FixedOffset(offset).localize(beginning)
            return df

        begin_time, end_time = self.__get_phase_times(subject_id=subject_id, phase=phase)
        # Make Timezone aware for indexing
        begin_time, end_time = pytz.FixedOffset(offset).localize(begin_time), pytz.FixedOffset(offset).localize(
            end_time
        )
        return df.loc[begin_time:end_time]

    def __find_start_end(self, df1: pd.DataFrame, df2: pd.DataFrame) -> (datetime, datetime):
        """
        Returns the start and end time of the intersection of two dataframes
        expects two dataframes with a time index
        """
        start_time = max(df1.index[0], df2.index[0])
        end_time = min(df1.index[-1], df2.index[-1])
        return start_time, end_time


class RadarDatasetRawModified(Dataset):
    SAMPLING_RATE_PSG: float = 256.0
    SAMPLING_RATE_RADAR: float = 1953.125
    POSITIONS = ["L", "0", "R"]
    POSTURES = ["1", "2", "3", "4", "5", "6"]
    MISSING = []

    def __init__(
        self,
        data_path: Path,
        exclude_missing: bool = True,
        *,
        groupby_cols: Optional[Union[List[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None,
    ):
        self.data_path = data_path
        self.channels = ["AUX", "ECG II", "RIP Thora", "RIP Abdom"]
        self.exclude_missing = exclude_missing
        self.PHASES = [
            f"{position} + {posture}" for position, posture in itertools.product(self.POSITIONS, self.POSTURES)
        ]
        self.psg_dataset = None
        self.radardataset = None
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self) -> pd.DataFrame:
        participant_ids = [f.name.split("_")[1] for f in sorted(self.data_path.glob("Subject*/"))]
        if self.exclude_missing:
            participant_ids = [pid for pid in participant_ids if pid not in self.MISSING]
        else:
            print("Incomplete Subjects included, watch out for missing datastreams")

        index = list(itertools.product(participant_ids, self.PHASES))
        # set locale to german to get correct date format
        # locale.setlocale(locale.LC_ALL, "de_DE")
        df = pd.DataFrame(
            {
                "Subject": [ind[0] for ind in index],
                "Phase": [ind[1] for ind in index],
            }
        )

        if len(df) == 0:
            raise ValueError(
                "The dataset is empty. Are you sure you selected the correct folder? Current folder is: "
                f"{self.data_path}"
            )
        return df

    @property
    def phase_times(self) -> pd.DataFrame:
        """
        Returns a dataframe which contains
        - one row with subject_id number, phase name and time of the beginning of the phase if there is just a single phase
        - multiple rows with subject_id number, phase names and times of the beginning/ending of the phases if there are multiple phases
        Works only if there is just a single participant in the dataset.
        """
        if not (
            self.is_single(None) or (self.is_single(["Subject"]) and self.groupby("Phase").shape[0] == len(self.PHASES))
        ):
            raise ValueError("Data can only be accessed, when there is just a single participant in the dataset.")

        # Get Subject Number and Phase Names
        phase_names = self.index["Phase"]
        subject_number = self.index["Subject"][0]

        # Get Start and End Times for each phase
        time_pairs = [self.__get_phase_times(phase, subject_number) for phase in phase_names]
        start_dates = [time_pair[0] for time_pair in time_pairs]
        end_dates = [time_pair[1] for time_pair in time_pairs]

        # return dataframe with all phase times
        return pd.DataFrame(
            {
                "Subject": subject_number,  # Works although this is no list and just a single value
                "Phase": phase_names,
                "From": start_dates,
                "To": end_dates,
            }
        )

    @property
    def sampling_rate_radar(self) -> float:
        """Returns sampling rate of the Radar recording in Hz."""
        return self.SAMPLING_RATE_RADAR

    @property
    def sampling_rate_biopac(self) -> float:
        """The sampling rate of the PSG recording in Hz."""
        return self.SAMPLING_RATE_PSG

    @property
    def subjects(self) -> List[str]:
        """Returns a list of all subjects in the dataset."""
        return self.index["Subject"].unique().tolist()

    @property
    def ecg(self) -> pd.DataFrame:
        """Returns a dataframe with the ECG data of a single phase or for one subject."""
        ecg = self.synced_data
        ecg = ecg[[("psg_aligned_", "ECG II")]]
        ecg.columns = [col[1] for col in ecg.columns.values]
        ecg.columns = ["ecg"]
        return ecg

    @property
    def respiration(self) -> pd.DataFrame:
        """Returns a dataframe with the respiration data of a single phase or for one subject."""
        # load all data
        resp = self.synced_data

        # select only the respiration data
        thorax = resp[("psg_aligned_", "RIP Thora")]
        abdomen = resp[("psg_aligned_", "RIP Abdom")]
        thorax = thorax.rename("thorax respiration")
        abdomen = abdomen.rename("abdomen respiration")
        # concat thorax and abdomen to one dataframe
        resp = pd.concat([thorax, abdomen], axis=1)

        return resp

    @property
    def radar_top(self) -> pd.DataFrame:
        """Returns a dataframe with the radar data of a single phase or for one subject."""

        # load all data
        rad = self.synced_data

        # select only the top radar data
        first_level = ["radar_5_aligned_", "radar_6_aligned_", "radar_7_aligned_"]
        second_level = ["I", "Q"]
        multiindex = pd.MultiIndex.from_product([first_level, second_level])
        rad = rad[multiindex]

        # rename the columns
        rad = rad.rename(
            columns={"radar_5_aligned_": "rad5", "radar_6_aligned_": "rad6", "radar_7_aligned_": "rad7"}, level=0
        )
        rad.columns = ["_".join(col) for col in rad.columns.values]
        return rad

    @property
    def radar_bottom(self) -> pd.DataFrame:
        """Returns a dataframe with the radar data of the four bottom sensors of a single phase or for one subject."""

        # load all data
        rad = self.synced_data

        # select only the radar data
        first_level = ["radar_1_aligned_", "radar_2_aligned_", "radar_3_aligned_", "radar_4_aligned_"]
        second_level = ["I", "Q"]
        multiindex = pd.MultiIndex.from_product([first_level, second_level])
        rad = rad[multiindex]

        # rename the columns
        rad = rad.rename(
            columns={
                "radar_1_aligned_": "rad1",
                "radar_2_aligned_": "rad2",
                "radar_3_aligned_": "rad3",
                "radar_4_aligned_": "rad4",
            },
            level=0,
        )

        return rad

    @property
    def synced_data(self) -> pd.DataFrame:
        """Returns a dataframe with all datastreams of a single phase or for one subject."""
        subject_id = self.subjects[0]
        if self.is_single(["Phase"]):
            phase = self.index["Phase"][0]
            result_df = self._load_sync(subject_id, phase)
        elif self.is_single(["Subject"]) and self.groupby("Phase").shape[0] == len(self.PHASES):
            result_df = self._load_sync(subject_id=subject_id, phase=None)

        else:
            raise ValueError(
                "Data can only be accessed for a single participant or a single phase "
                "of one single participant in the subset"
            )
        return result_df

    @lru_cache(maxsize=1)
    def _load_sync(self, subject_id: str, phase: Optional[str]) -> pd.DataFrame:
        """Loads the sync data of a single phase or for one subject."""
        subject_str = "Subject_{}".format(subject_id)
        edf_file = subject_str + ".edf"
        radar_top_file = "data_{}.h5".format(subject_id)
        radar_bottom_file = "data_{}-bett.h5".format(subject_id)

        # construct paths to data files
        edf_path = self.data_path / subject_str / edf_file
        radar_top_path = self.data_path / subject_str / radar_top_file
        radar_bottom_path = self.data_path / subject_str / radar_bottom_file

        # Create Radar Datasets
        radar_bottom_dataset = EmradDataset.from_hd5_file(radar_bottom_path)
        radar_top_dataset = EmradDataset.from_hd5_file(radar_top_path)

        # Create Radar Dataframes
        radar_bottom_df = radar_bottom_dataset.data_as_df(index="local_datetime", add_sync_in=True, add_sync_out=True)
        radar_top_df = radar_top_dataset.data_as_df(index="local_datetime", add_sync_in=True, add_sync_out=True)
        # convert all NAN to zeroes
        radar_bottom_df = radar_bottom_df.fillna(0)
        radar_top_df = radar_top_df.fillna(0)

        # convert multicoloum from radar Sync_In and Sync_out to integer instead of float because some channels use float and some use integer
        for i in range(1, 5):
            radar_bottom_df[("rad" + str(i), "Sync_In")] = radar_bottom_df[("rad" + str(i), "Sync_In")].astype(int)
            radar_bottom_df[("rad" + str(i), "Sync_Out")] = radar_bottom_df[("rad" + str(i), "Sync_Out")].astype(int)
        for i in range(1, 4):
            radar_top_df[("rad" + str(i), "Sync_In")] = radar_top_df[("rad" + str(i), "Sync_In")].astype(int)
            radar_top_df[("rad" + str(i), "Sync_Out")] = radar_top_df[("rad" + str(i), "Sync_Out")].astype(int)

        # Create PSG Dataframe
        df = PSGDataset.from_edf_file(edf_path, datastreams=self.channels).data_as_df(index="local_datetime")
        df.rename(columns={"AUX": "Sync_Out"}, inplace=True)
        df = RadarDatasetRawModified._reduce_df(df, "Sync_Out")
        # Cut Dataframes between the first switch from 0 to 1 and vice versa at the end

        # Create and fill Synced Dataset
        synced_dataset = SyncedDataset(sync_type="m-sequence")
        synced_dataset.add_dataset(
            "radar_1", data=radar_bottom_df["rad1"], sync_channel_name="Sync_In", sampling_rate=self.SAMPLING_RATE_RADAR
        )
        synced_dataset.add_dataset(
            "radar_2", data=radar_bottom_df["rad2"], sync_channel_name="Sync_In", sampling_rate=self.SAMPLING_RATE_RADAR
        )
        synced_dataset.add_dataset(
            "radar_3", data=radar_bottom_df["rad3"], sync_channel_name="Sync_In", sampling_rate=self.SAMPLING_RATE_RADAR
        )
        # synced_dataset.add_dataset(
        #     "radar_4", data=radar_bottom_df["rad4"], sync_channel_name="Sync_In", sampling_rate=self.SAMPLING_RATE_RADAR
        # )
        # synced_dataset.add_dataset(
        #     "radar_5", data=radar_top_df["rad1"], sync_channel_name="Sync_In", sampling_rate=self.SAMPLING_RATE_RADAR
        # )
        # synced_dataset.add_dataset(
        #     "radar_6", data=radar_top_df["rad2"], sync_channel_name="Sync_In", sampling_rate=self.SAMPLING_RATE_RADAR
        # )
        # synced_dataset.add_dataset(
        #     "radar_7", data=radar_top_df["rad3"], sync_channel_name="Sync_In", sampling_rate=self.SAMPLING_RATE_RADAR
        # )
        synced_dataset.add_dataset("psg", data=df, sync_channel_name="Sync_Out", sampling_rate=self.SAMPLING_RATE_PSG)

        # Resample and align datasets

        # synced_dataset.align_and_cut_m_sequence(
        #     primary="psg", reset_time_axis=True, cut_to_shortest=True, sync_params={"sync_region_samples": (0, 10000)}
        # )
        #

        #
        synced_dataset.resample_datasets(fs_out=self.SAMPLING_RATE_PSG, method="static")
        synced_dataset.align_and_cut_m_sequence(
            primary="psg",
            reset_time_axis=True,
            cut_to_shortest=True,
            sync_params={"sync_region_samples": (0, 100000)},
        )
        dict_shift = synced_dataset._find_shift(
            primary="psg_aligned_", sync_params={"sync_region_samples": (-100000, -1)}
        )
        synced_dataset.resample_sample_wise(primary="psg_aligned_", dict_sample_shift=dict_shift)
        df_dict = synced_dataset.datasets_aligned

        # synced_dataset.resample_datasets(fs_out=self.SAMPLING_RATE_PSG, method="static")
        # synced_dataset.align_and_cut_m_sequence(
        #     primary="psg",
        #     reset_time_axis=True,
        #     cut_to_shortest=True,
        #     sync_params={"sync_region_samples": (0, 1000000), "sampling_rate": self.SAMPLING_RATE_PSG},
        # )
        # df_dict = synced_dataset.datasets_aligned
        # result_df = pd.concat(df_dict.values(), axis=1, keys=df_dict.keys())
        # result_df.columns = [
        #     "".join(col).replace("aligned_", "") if col[1] != "ecg" else "ecg" for col in result_df.columns.values
        # ]

        # concat all dataframes to one
        result_df = pd.concat(df_dict.values(), axis=1, keys=df_dict.keys())
        # delete sync coloums
        multiindex = result_df.columns
        keep_columns = ~multiindex.get_level_values(1).str.contains("asd;lkfja;s")
        result_df = result_df.loc[:, keep_columns]

        # Cut df to phase and return
        return self.__cut_df(df=result_df, phase=phase, subject_id=subject_id)

    @staticmethod
    def _reduce_df(df: pd.DataFrame, sync_column: str) -> pd.DataFrame:
        bin_df = SyncedDataset()._binarize_signal(df[sync_column])
        start_index = df.index.get_loc((bin_df == 1).idxmax())
        start_index = df.index[start_index - 1]
        end_index = df.index.get_loc((bin_df[::-1] == 1).idxmax())
        end_index = df.index[end_index + 1]
        return df.loc[start_index:end_index]

    def __get_csv_file(self, path: Path) -> Path:
        """
        returns name of the ATimeLoggerCSV
        could be integrated in get_phase_times
        expects the path to a folder of the Subject as argument where a csv file from the ATimeLogger is located
        """
        # get path of csv file in that folder
        csv_name = glob.glob(f"{path}/*.csv")
        if len(csv_name) == 0:
            raise FileNotFoundError("There is no csv file with phase times in {}".format(path))
        if len(csv_name) > 1:
            warnings.warn("There are multiple csv files in the folder. Using the first one.")
        csv_path = Path(csv_name[0])

        if not csv_path.is_file():
            raise FileNotFoundError("There is no csv file named {}".format(csv_path))

        return csv_path

    def __get_phase_times(self, phase: Optional[str], subject_id: str) -> (datetime, datetime):
        """
        Returns the start and end time of a phase as datetime object
        expects the phase as string and the subject_id as string
        """

        # create dataframe from csv file
        subject_folder = self.data_path / "Subject_{}".format(subject_id)
        csv_path = self.__get_csv_file(subject_folder)
        df_phase_times = pd.read_csv(csv_path, delimiter=",", header=0, skipfooter=5, engine="python")

        # Reverse Dataframe to get Phases in the right order (from first to last)
        df_phase_times = df_phase_times.iloc[::-1].reset_index(drop=True)

        # Get the start and end time of the phase
        phase_number = self.PHASES.index(phase)
        start_time = df_phase_times["From"][phase_number]
        end_time = df_phase_times["To"][phase_number]

        # Convert to datetime object
        format = "%Y-%m-%d %H:%M:%S"
        start_datetime = datetime.strptime(start_time, format)
        end_datetime = datetime.strptime(end_time, format)

        # offset on these times needed.
        offset = self.data_path / "Subject_{}".format(subject_id) / "offset_seconds.txt"
        with open(offset, "r") as f:
            file_content = f.read()
            if file_content == "":
                offset_seconds = 0
            else:
                offset_seconds = int(file_content)

            start_datetime = start_datetime - timedelta(seconds=offset_seconds)
            end_datetime = end_datetime - timedelta(seconds=offset_seconds)
        return start_datetime, end_datetime

    def __cut_df(self, df: pd.DataFrame, phase: str, subject_id: str) -> pd.DataFrame:
        """
        Cut df with time index to only the rows within the phase time interval
        if phase is None than the whole Signal should be returned beginning from the first official Phase from self.PHASES[0].
        """

        # determine offset either 60 if subject ID is 17 or higher or 120 for the rest (recording issues)
        subject_number = int(subject_id)
        offset = 120
        if subject_number >= 17:
            offset = 60
        if phase is None:
            # no phase given, return whole signal -> beginning from first phase
            first_phase = self.PHASES[0]
            beginning = self.__get_phase_times(subject_id=subject_id, phase=first_phase)[0]
            # Make Timezone aware for indexing
            beginning = pytz.FixedOffset(offset).localize(beginning)
            return df

        begin_time, end_time = self.__get_phase_times(subject_id=subject_id, phase=phase)
        # Make Timezone aware for indexing
        begin_time, end_time = pytz.FixedOffset(offset).localize(begin_time), pytz.FixedOffset(offset).localize(
            end_time
        )
        return df.loc[begin_time:end_time]

    def __find_start_end(self, df1: pd.DataFrame, df2: pd.DataFrame) -> (datetime, datetime):
        """
        Returns the start and end time of the intersection of two dataframes
        expects two dataframes with a time index
        """
        start_time = max(df1.index[0], df2.index[0])
        end_time = min(df1.index[-1], df2.index[-1])
        return start_time, end_time


class RadarCardiaStudyDataset(BaseDataset):
    exclude_ecg_seg_failed: bool
    exclude_ecg_corrupted: bool

    BIOPAC_CHANNEL_MAPPING: Dict[str, str] = {
        "ecg": "ecg",
        "icg": "icg_der",
        "ppg": "ppg",
        "pcg": "pcg",
        "sync": "sync",
    }

    SUBJECTS_ECG_SEG_FAILED: Tuple[str] = ("VP_04",)

    SUBJECT_ECG_CORRUPTED: Tuple[Tuple] = (
        ("VP_17", "baseline", "normal"),
        ("VP_17", "temporalis", "normal"),
    )

    def __init__(
        self,
        base_path: path_t,
        groupby_cols: Optional[Sequence[str]] = None,
        subset_index: Optional[Sequence[str]] = None,
        use_cache: Optional[bool] = False,
        calc_biopac_timelog_shift: Optional[bool] = True,
        trigger_data_extraction: Optional[bool] = False,
        exclude_ecg_seg_failed: Optional[bool] = True,
        exclude_ecg_corrupted: Optional[bool] = True,
    ):
        self.exclude_ecg_seg_failed = exclude_ecg_seg_failed
        self.exclude_ecg_corrupted = exclude_ecg_corrupted

        super().__init__(
            base_path=base_path,
            groupby_cols=groupby_cols,
            subset_index=subset_index,
            use_cache=use_cache,
            calc_biopac_timelog_shift=calc_biopac_timelog_shift,
            trigger_data_extraction=trigger_data_extraction,
        )

    def create_index(self):
        subject_ids = [
            subject_dir.name for subject_dir in get_subject_dirs(self.base_path.joinpath("data_per_subject"), "VP_*")
        ]

        if self.exclude_ecg_seg_failed:
            for subject_id in self.SUBJECTS_ECG_SEG_FAILED:
                if subject_id in subject_ids:
                    subject_ids.remove(subject_id)

        breathing = ["normal", "hold"]
        body_parts_1 = [
            "baseline",
            "temporalis",
            "carotis",
            "brachialis",
            "radialis_prox",
            "radialis_med",
            "radialis_dist",
            "dorsalis_pedis",
            "poplitea",
        ]
        body_parts_2 = ["aorta_prox", "aorta_med", "aorta_dist"]

        index = list(product(body_parts_1, ["normal"]))
        index.extend(list(product(body_parts_2, breathing)))
        index = [(subject, *i) for subject, i in product(subject_ids, index)]

        if self.exclude_ecg_corrupted:
            for subject_id, body_part, breathing in self.SUBJECT_ECG_CORRUPTED:
                if (subject_id, body_part, breathing) in index:
                    index.remove((subject_id, body_part, breathing))

        index = pd.DataFrame(index, columns=["subject", "location", "breathing"])

        return index
