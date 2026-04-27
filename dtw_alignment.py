import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from Signal_Processing_and_Alignment import Signal_Processing
from tslearn.metrics import dtw
from tslearn.metrics import dtw_path
from tslearn.preprocessing import TimeSeriesResampler


def dtw_alignment(
    time_normalized_data: List[pd.DataFrame],
    signal_index: Optional[List[int]] = None,
    resample_length: Optional[int] = None,
    plot_series: bool = False,
    plot_alignment: bool = False,
    show_matrix: bool = False,
    series_standard: Optional[pd.DataFrame] = None,
    smooth: bool = False,
    constraint: Optional[str] = None,
    sakoe_radius: Optional[float] = 50,
    target_path: Optional[str] = None,
    inverse: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """
    Aligns normalized nanopore translocation traces using Dynamic Time Warping (DTW).

    The function selects a standard reference trace, aligns each input trace to that reference,
    optionally smooths the aligned traces, and computes an average aligned current trace.

    Args:
        time_normalized_data (List[pd.DataFrame]): List of time-normalized translocation traces.
            Each DataFrame must contain two columns: time in the first column and current in the second.
        signal_index (Optional[List[int]]): Optional labels or original indices for each trace.
            When provided, these labels are used in plot legends.
        resample_length (Optional[int]): Number of points to resample all aligned traces to.
            If None, the length of the longest aligned trace is used.
        plot_series (bool): If True, plots the resulting average current waveform.
        plot_alignment (bool): If True, plots all aligned traces after DTW alignment.
        show_matrix (bool): If True, displays a normalized pairwise DTW distance matrix.
        series_standard (Optional[pd.DataFrame]): Reference trace used for alignment.
            When None, the trace closest to all others is selected automatically.
        smooth (bool): If True, applies Gaussian smoothing to each aligned current trace.
        constraint (Optional[str]): DTW constraint type. Supported value is 'sakoe_chiba'.
        method (str): Alignment method. Only 'classic' is supported.
        sakoe_radius (Optional[float]): Radius value for Sakoe-Chiba constraint.
            If None, the supplied value is passed directly to tslearn.
        target_path (Optional[str]): Directory to save plots. If None, plots are displayed instead.
        inverse (bool): If True, flips the aligned current series vertically before returning.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
            - List[np.ndarray]: Resampled aligned current arrays for each trace.
            - List[np.ndarray]: Unique aligned trace arrays, each shaped as [2, n]
              with time and current values.
            - np.ndarray: The average current trace across all aligned series.
    """

    if series_standard is None:
        dtw_matrix = np.zeros((len(time_normalized_data), len(time_normalized_data)))

        for i in range(len(time_normalized_data) - 1):
            for j in range(i + 1, len(time_normalized_data)):
                series_1 = time_normalized_data[i]
                series_2 = time_normalized_data[j]
                distance = dtw(series_1, series_2)
                normalized_distance = distance / (len(series_1) + len(series_2))
                dtw_matrix[i, j] = normalized_distance
                dtw_matrix[j, i] = normalized_distance

        rows_sums = dtw_matrix.sum(axis=1)
        min_index = np.argmin(rows_sums)

        if signal_index is not None:
            print(signal_index[min_index])

        normalized_dtw_matrix = dtw_matrix / np.amax(dtw_matrix)

        if show_matrix:
            sns.heatmap(normalized_dtw_matrix, annot=False, fmt=".2f", cmap="YlGnBu")
            plt.show()
            print(len(rows_sums), min_index)

    aligned_data: List[np.ndarray] = []

    if series_standard is None:
        series_standard = time_normalized_data[min_index]

    np_series_standard = series_standard.to_numpy()
    np_series_standard_time = np_series_standard[:, 0].T
    np_series_standard_current = np_series_standard[:, 1].T

    for series in time_normalized_data:
        path, _ = dtw_path(series, series_standard)

        if constraint is None:
            path, _ = dtw_path(series, series_standard)
        elif constraint == 'sakoe_chiba':
            if sakoe_radius is None:
                path, _ = dtw_path(
                    series,
                    series_standard,
                    global_constraint=constraint,
                    sakoe_chiba_radius=sakoe_radius,
                )
            else:
                path, _ = dtw_path(
                    series,
                    series_standard,
                    global_constraint=constraint,
                    sakoe_chiba_radius=0.3 * series.shape[0],
                )

        aligned_series = np.zeros((len(path), 2))
        aligned_standard_series = np.zeros((len(path), 2))
        np_series = series.to_numpy()

        for k, (idx1, idx2) in enumerate(path):
            aligned_series[k] = np_series[idx1]
            aligned_standard_series[k] = np_series_standard[idx2]

        aligned_series_current = [x[1] for x in aligned_series]
        aligned_standard_series_time = [x[0] for x in aligned_standard_series]
        aligned_data.append(np.array([aligned_standard_series_time, aligned_series_current]))

    aligned_data_unique: List[np.ndarray] = []

    for series in aligned_data:
        unique_series_time, idx = np.unique(series[0, :], return_index=True)
        unique_series_current = series[1, :][idx]

        if smooth:
            unique_series_current = gaussian_filter1d(unique_series_current, sigma=5)

        if not inverse:
            aligned_data_unique.append(np.array([unique_series_time, unique_series_current]))
        else:
            aligned_data_unique.append(np.array([unique_series_time, np.flip(unique_series_current)]))

    if plot_alignment:
        for index, series in enumerate(aligned_data_unique):
            if signal_index is None:
                plt.plot(series[0, :], series[1, :], label=str(index))
            else:
                plt.plot(series[0, :], series[1, :], label=str(signal_index[index]))

        if target_path is not None:
            plt.savefig(os.path.join(target_path, f'radius{sakoe_radius}.png'))

        if target_path is not None:
            plt.close('all')
        else:
            plt.legend(fontsize=4)
            plt.show()

    len_aligned_data = np.array([series.shape[1] for series in aligned_data_unique])
    max_aligned_len = len_aligned_data.max()
    _ = np.argmax(len_aligned_data)

    resampler = TimeSeriesResampler(sz=max_aligned_len if resample_length is None else resample_length)
    interplotation_aligned_data = [
        resampler.fit_transform(series[-1, :]).ravel()
        for series in aligned_data_unique
    ]

    average_current = np.sum(interplotation_aligned_data, axis=0)
    average_current /= len(interplotation_aligned_data)

    if plot_series:
        time = np.linspace(0, 1, max_aligned_len)
        plt.plot(time, average_current, color='blue')
        plt.plot(np_series_standard_time, np_series_standard_current, color='red')
        plt.show()

    return interplotation_aligned_data, aligned_data_unique, average_current


if __name__ == '__main__':
    current_dir = os.getcwd()
    dir_120 = 'data_120'
    dir_final = 'data_final'
    file_name = 'direction_2.xlsx'

    compare_plot_path = 'radius_compare_smooth'
    plot_save_path = os.path.join(current_dir, compare_plot_path)
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)

    time_normalized_data, selected_index = Signal_Processing(
        os.path.join(current_dir, dir_final, file_name),
        upper_lim=2000,
        lower_lim=100,
        cut_off_frequency=5,
        normalization_method='standard',
        smooth=False,
        threshold=60,
    )

    _, _, _ = dtw_alignment(
        time_normalized_data,
        signal_index=selected_index,
        plot_series=True,
        plot_alignment=True,
        show_matrix=True,
        smooth=False,
        constraint='sakoe_chiba',
        sakoe_radius=None,
        target_path=None,
        inverse=False,
    )
