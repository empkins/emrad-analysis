from pathlib import Path
from typing import Optional, Union, List
from biopsykit.io.biopac import BiopacDataset
import pandas as pd
from empkins_io.sensors.emrad import EmradDataset
from empkins_io.sync import SyncedDataset
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


class D02Dataset(Dataset):
    """
    D02Dataset is a subclass of the Dataset class that is specifically designed to handle the D02 dataset.
    It provides methods for loading ECG and radar data, synchronizing the data, and creating an index of participants.
    """

    SAMPLING_RATE_ACQ = 2000
    SAMPLING_RATE_RADAR = 1953.125
    SAMPLING_RATE_DOWNSAMPLED = 1000
    _CHANNEL_MAPPING = {
        "ECG": "ecg",
        "SyncSignal": "sync",
    }

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

    @property
    def synced_ecg(self) -> pd.DataFrame:
        if not (self.is_single(None) or (self.is_single(["participant"]))):
            raise ValueError("Data can only be accessed, when there is just a single participant in the dataset.")
        return self.synced_data[["ecg"]]

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
        return dataset.data_as_df(index="local_datetime", add_sync_in=add_sync_in, add_sync_out=add_sync_out)

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
        radar_df = radar_df.droplevel(0, axis=1)
        # Synchronize the data
        synced_dataset = SyncedDataset(sync_type="m-sequence")
        synced_dataset.add_dataset(
            "radar", data=radar_df, sync_channel_name="Sync_In", sampling_rate=self.SAMPLING_RATE_RADAR
        )
        synced_dataset.add_dataset("ecg", data=ecg_df, sync_channel_name="sync", sampling_rate=self.SAMPLING_RATE_ACQ)
        synced_dataset.resample_datasets(fs_out=self.SAMPLING_RATE_DOWNSAMPLED, method="dynamic", wave_frequency=0.2)
        synced_dataset.align_and_cut_m_sequence(
            primary="radar",
            reset_time_axis=True,
            cut_to_shortest=True,
            sync_params={"sync_region_samples": (0, 1000000), "sampling_rate": self.SAMPLING_RATE_DOWNSAMPLED},
        )
        df_dict = synced_dataset.datasets_aligned
        result_df = pd.concat(df_dict.values(), axis=1, keys=df_dict.keys())
        result_df.columns = [
            "".join(col).replace("aligned_", "") if col[1] != "ecg" else "ecg" for col in result_df.columns.values
        ]
        # cols_to_drop = result_df.columns[result_df.columns.str.contains("sync", case=False)]
        # result_df = result_df.drop(columns=cols_to_drop)
        return result_df


class RadarDatasetRaw(Dataset):
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
        synced_dataset.add_dataset("psg", data=df, sync_channel_name="AUX", sampling_rate=self.SAMPLING_RATE_PSG)

        # Resample and align datasets
        synced_dataset.resample_datasets(fs_out=self.SAMPLING_RATE_PSG, method="static")
        synced_dataset.align_and_cut_m_sequence(
            primary="psg", reset_time_axis=True, cut_to_shortest=True, sync_params={"sync_region_samples": (0, 10000)}
        )
        df_dict = synced_dataset.datasets_aligned

        # concat all dataframes to one
        result_df = pd.concat(df_dict.values(), axis=1, keys=df_dict.keys())
        # delete sync coloums
        multiindex = result_df.columns
        keep_columns = ~multiindex.get_level_values(1).str.contains("asd;lkfja;s")
        result_df = result_df.loc[:, keep_columns]

        # Cut df to phase and return
        return self.__cut_df(df=result_df, phase=phase, subject_id=subject_id)

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
