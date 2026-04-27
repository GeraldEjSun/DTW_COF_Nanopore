"""
Specialized DTW alignment for nanopore traces using 1D current vectors with time reconstruction.

This module performs DTW on current-only traces and then reconstructs aligned time-current
pairs from the original DataFrames. It is optimized for 1D current signals while preserving
the original time axis for plotting and analysis.
"""

import numpy as np
import pandas as pd
from Signal_Processing_and_Alignment import Signal_Processing
from tslearn.metrics import dtw
from tslearn.metrics import dtw_path
from tslearn.preprocessing import TimeSeriesResampler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from typing import List, Optional, Tuple


def dtw_alignment_1d_time(
    time_normalized_data: List[pd.DataFrame],
    resample_length: Optional[int] = None,
    plot_series: bool = False,
    plot_alignment: bool = False,
    show_matrix: bool = False,
    series_standard: Optional[np.ndarray] = None,
    smooth: bool = False,
    constraint: Optional[str] = None,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Aligns nanopore current traces using 1D DTW and preserves the original time axis.

    Differences from dtw_alignment.py:
    - Uses 1D current vectors extracted from DataFrame traces instead of full 2D time-current arrays.
    - Reconstructs aligned time/current pairs after alignment, preserving the original time axis.
    - Returns only resampled aligned current traces and the averaged current waveform.

    Args:
        time_normalized_data (List[pd.DataFrame]): List of normalized nanopore traces.
            Each DataFrame must have time values in the first column and current values in the second.
        resample_length (Optional[int]): Target number of points for resampled aligned traces.
            If None, the longest aligned trace length is used.
        plot_series (bool): If True, plots the averaged current trace after alignment.
        plot_alignment (bool): If True, plots all aligned time-current traces.
        show_matrix (bool): If True, displays the normalized pairwise DTW distance matrix.
        series_standard (Optional[np.ndarray]): Optional reference current trace for alignment.
            If provided, this 1D current array is used instead of selecting the closest trace.
        smooth (bool): If True, applies Gaussian smoothing to the interpolated aligned traces.
        constraint (Optional[str]): Optional DTW constraint type.
            Supported value is 'sakoe_chiba'.

    Returns:
        Tuple[List[np.ndarray], np.ndarray]:
            - List[np.ndarray]: Resampled aligned current traces for each input signal.
            - np.ndarray: Average current trace computed from the aligned series.
    """
    series_current = [series.iloc[:, 1].to_numpy().T for series in time_normalized_data]

    if series_standard is None:
        dtw_matrix = np.zeros((len(series_current), len(series_current)))

        for i in range(len(series_current) - 1):
            for j in range(i + 1, len(series_current)):
                series_1 = series_current[i]
                series_2 = series_current[j]
                distance = dtw(series_1, series_2)
                normalized_distance = distance / (len(series_1) + len(series_2))
                dtw_matrix[i, j] = normalized_distance
                dtw_matrix[j, i] = normalized_distance

        rows_sums = dtw_matrix.sum(axis=1)
        min_index = np.argmin(rows_sums)
        normalized_dtw_matrix = dtw_matrix / np.amax(dtw_matrix)

        if show_matrix:
            sns.heatmap(normalized_dtw_matrix, annot=False, fmt='.2f', cmap='YlGnBu')
            plt.show()
            print(len(rows_sums), min_index)

        current_standard = series_current[min_index]
        standard_time = time_normalized_data[min_index].iloc[:, 0].to_numpy().T
        standard_series = time_normalized_data[min_index].to_numpy()
    else:
        current_standard = np.asarray(series_standard)
        standard_time = np.linspace(0, 1, len(current_standard))
        standard_series = current_standard

    aligned_data: List[np.ndarray] = []

    for series, current in zip(time_normalized_data, series_current):
        if constraint is not None:
            path, _ = dtw_path(
                current,
                current_standard,
                global_constraint='sakoe_chiba',
                sakoe_chiba_radius=20,
            )
        else:
            path, _ = dtw_path(current, current_standard)

        aligned_series = np.zeros((len(path), 2))
        aligned_standard_series = np.zeros((len(path), 2))
        np_series = series.to_numpy()

        for k, (idx1, idx2) in enumerate(path):
            aligned_series[k] = np_series[idx1]
            aligned_standard_series[k] = standard_series[idx2]

        aligned_series_current = [x[1] for x in aligned_series]
        aligned_standard_series_time = [x[0] for x in aligned_standard_series]
        aligned_data.append(np.array([aligned_standard_series_time, aligned_series_current]))

    aligned_data_unique: List[np.ndarray] = []

    for series in aligned_data:
        unique_series_time, idx = np.unique(series[0, :], return_index=True)
        unique_series_current = series[1, :][idx]
        aligned_data_unique.append(np.array([unique_series_time, unique_series_current]))

    max_aligned_len = np.max([series.shape[1] for series in aligned_data_unique])
    resampler = TimeSeriesResampler(sz=max_aligned_len if resample_length is None else resample_length)
    interpolated_aligned_data = [
        resampler.fit_transform(series[-1, :]).ravel()
        for series in aligned_data_unique
    ]

    if smooth:
        interpolated_aligned_data = [gaussian_filter1d(series, sigma=5) for series in interpolated_aligned_data]

    average_current = np.sum(interpolated_aligned_data, axis=0)
    average_current /= len(interpolated_aligned_data)

    if plot_alignment:
        for series in aligned_data_unique:
            plt.plot(series[0, :], series[1, :])
        plt.show()

    if plot_series:
        time = np.linspace(0, 1, max_aligned_len)
        plt.plot(time, average_current, color='blue')
        plt.plot(standard_time, current_standard, color='red')
        plt.show()

    return interpolated_aligned_data, average_current


if __name__ == '__main__':
    current_dir = np.getcwd()
    file_name = '1.xlsx'

    time_normalized_data = Signal_Processing(
        file_name,
        start=-10,
        end=None,
        upper_lim=None,
        lower_lim=1,
        cut_off_frequency=2,
        normalization_method='standard',
    )

    _, _ = dtw_alignment_1d_time(
        time_normalized_data,
        plot_series=True,
        plot_alignment=True,
        show_matrix=True,
        smooth=True,
        constraint=None,
    )
