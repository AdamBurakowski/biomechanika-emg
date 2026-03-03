from typing import Tuple
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def load_data(
        file_path: str | Path
        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads data from a specified file path, clears it, and formats it for use
    in plotting and analysis.

    Parameters
    ----------
    file_path : str | Path
        The path to the file containing the data to be cleared and formatted.

    Returns
    -------
    pd.DataFrame:
        A DataFrame containing the session statistics extracted from the file.
    pd.DataFrame:
        A DataFrame containing the markers extracted from the file.
    pd.DataFrame:
        A DataFrame containing the EMG data extracted from the file.

    """
    file_headers = ["SESSION STATISTICS", "MARKERS", "EMG"]
    statistics = ""
    markers = ""
    data = ""
    current_header = ""

    with open(file_path, "r", encoding="utf-16") as file:
        lines = file.readlines()
    for line in lines:
        if line.strip() in file_headers:
            current_header = line.strip()
            continue
        if current_header == "SESSION STATISTICS":
            statistics += line.rstrip() + "\n"
        elif current_header == "MARKERS":
            markers += line.rstrip() + "\n"
        elif current_header == "EMG":
            data += line.rstrip() + "\n"

    # Convert strings to DataFrames
    statistics_df = pd.DataFrame([row.split("\t")
                                  for row in statistics.strip().split("\n")])
    markers_df = pd.DataFrame([row.split("\t")
                               for row in markers.strip().split("\n")])
    data_df = pd.DataFrame([row.split("\t")
                            for row in data.strip().split("\n")])

    return statistics_df, markers_df, data_df


def preprocess_data(
        data: pd.DataFrame
        ) -> pd.DataFrame:
    """
    Preprocesses the EMG data for analysis and plotting.

    Parameters
    ----------
    data : pd.DataFrame
        The EMG data to be preprocessed.

    Returns
    -------
    pd.DataFrame
        The preprocessed EMG data.
    """
    # move the first row to the header
    data.columns = data.iloc[0]
    data = data[1:]

    # convert values to numeric, ignoring errors
    data = data.replace(',', '.', regex=True).apply(
        pd.to_numeric, errors='coerce')

    return data


def plot_data(
        data_1: pd.Series,
        data_2: pd.Series,
        strings: list[str],
        window_size: int = 25
        ) -> None:
    """
    Plots the EMG data.

    Parameters
    ----------
    data_1 : pd.Series
        The first channel to be plotted.
    data_2 : pd.Series
        The second channel to be plotted.
    strings : list[str]
        A list of strings to be used in the plot title and labels.
    """
    moving_avg_1 = data_1.rolling(window=window_size).mean()
    moving_avg_2 = data_2.rolling(window=window_size).mean()
    fig, axs = plt.subplots(2, 1, figsize=(15, 8))
    axs[0].plot(range(len(data_1)), data_1, label="channel 1")
    axs[0].plot(range(len(data_2)), data_2, label="channel 2")
    axs[1].plot(range(len(moving_avg_1)),
                moving_avg_1, label="channel 1 (moving avg)")
    axs[1].plot(range(len(moving_avg_2)),
                moving_avg_2, label="channel 2 (moving avg)")
    axs[0].set_title(f"{strings[0]} - Raw Data")
    axs[1].set_title(f"{strings[0]} - Moving Average ({window_size})")
    for ax in axs:
        ax.set_xlabel("Time")
        ax.set_ylabel("EMG Signal")
        ax.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    """
    The main function of the script. It loads, preprocesses, and plots the EMG
    data from the specified files.
    """
    # load data
    total_data = pd.DataFrame()
    path = Path(__file__).parent / "data"
    for file in path.glob("*.txt"):
        print(f"Processing file: {file}")
        stats, markers, data = load_data(file)
        data = preprocess_data(data)
        total_data = pd.concat([total_data, data], ignore_index=True)

    # plot data
    plot_data(
        total_data.iloc[:, 1],
        total_data.iloc[:, 2],
        ["Total Data"],
        window_size=15)

if __name__ == "__main__":
    main()
