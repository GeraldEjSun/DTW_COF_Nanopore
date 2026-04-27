"""
Specialized DTW alignment for nanopore current traces using 1D arrays.

This module performs DTW directly on current vectors extracted from
nanopore translocation DataFrames. It is optimized for 1D arrays and skips
explicit time-current pair alignment during the cost computation.
"""

import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Signal_Processing_and_Alignment import Signal_Processing
from scipy.ndimage import gaussian_filter1d
from tslearn.metrics import dtw
from tslearn.metrics import dtw_path
from tslearn.preprocessing import TimeSeriesResampler


def dtw_alignment_1d(
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
    Aligns nanopore current traces using 1D DTW on current vectors.

    Differences from dtw_alignment.py:
    - Operates on 1D current arrays instead of full 2D time-current DataFrames.
    - Uses only the current profile for DTW cost computation, which is a lightweight
      optimization for nanopore signal alignment.
    - Produces resampled aligned current traces and an averaged current waveform.

    Args:
        time_normalized_data (List[pd.DataFrame]): List of normalized nanopore traces.
            Each DataFrame should contain time in the first column and current in the second.
        resample_length (Optional[int]): Desired number of points after resampling.
            If None, the length of the longest aligned trace is used.
        plot_series (bool): If True, plot the averaged aligned current trace.
        plot_alignment (bool): If True, plot each aligned current trace.
        show_matrix (bool): If True, display the normalized pairwise DTW distance matrix.
        series_standard (Optional[np.ndarray]): Optional reference current trace for alignment.
            When None, the trace closest to all others is selected.
        smooth (bool): If True, apply Gaussian smoothing to each resampled aligned trace.
        constraint (Optional[str]): Optional DTW constraint type.
            Supported value is 'sakoe_chiba'.

    Returns:
        Tuple[List[np.ndarray], np.ndarray]:
            - List[np.ndarray]: Resampled aligned current traces for each signal.
            - np.ndarray: Average current trace across all aligned signals.
    """
    current_traces = [series.iloc[:, 1].to_numpy().T for series in time_normalized_data]

    if series_standard is None:
        dtw_matrix = np.zeros((len(current_traces), len(current_traces)))

        for i in range(len(current_traces) - 1):
            for j in range(i + 1, len(current_traces)):
                series_1 = current_traces[i]
                series_2 = current_traces[j]
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

        standard_trace = current_traces[min_index]
    else:
        standard_trace = np.asarray(series_standard)

    aligned_data: List[np.ndarray] = []

    for trace in current_traces:
        if constraint is not None:
            path, _ = dtw_path(
                trace,
                standard_trace,
                global_constraint='sakoe_chiba',
                sakoe_chiba_radius=20,
            )
        else:
            path, _ = dtw_path(trace, standard_trace)

        aligned_series = np.zeros((len(path), 1))
        aligned_standard_series = np.zeros((len(path), 1))

        for k, (idx1, idx2) in enumerate(path):
            aligned_series[k] = trace[idx1]
            aligned_standard_series[k] = standard_trace[idx2]

        aligned_data.append(aligned_series)

    aligned_data_unique = [np.copy(series) for series in aligned_data]

    if plot_alignment:
        for series in aligned_data_unique:
            plt.plot(np.linspace(0, 1, len(series)), series)
        plt.show()

    len_aligned_data = np.array([len(series) for series in aligned_data_unique])
    max_aligned_len = len_aligned_data.max()

    resampler = TimeSeriesResampler(sz=max_aligned_len if resample_length is None else resample_length)
    interpolated_aligned_data = [resampler.fit_transform(series.ravel()).ravel() for series in aligned_data]

    if smooth:
        interpolated_aligned_data = [gaussian_filter1d(series, sigma=5) for series in interpolated_aligned_data]

    average_current = np.sum(interpolated_aligned_data, axis=0)
    average_current /= len(interpolated_aligned_data)

    if plot_series:
        time = np.linspace(0, 1, max_aligned_len)
        plt.plot(time, average_current, color='blue')
        plt.plot(np.linspace(0, 1, len(standard_trace)), standard_trace, color='red')
        plt.show()

    return interpolated_aligned_data, average_current


if __name__ == '__main__':
    current_dir = os.getcwd()
    file_name = '1.xlsx'

    time_normalized_data = Signal_Processing(
        file_name,
        start=-10,
        end=None,
        upper_lim=None,
        lower_lim=1,
        cut_off_frequency=2,
        normalization_method='delta',
        smooth=True,
    )

    _, _ = dtw_alignment_1d(
        time_normalized_data,
        plot_series=True,
        plot_alignment=True,
        show_matrix=True,
        smooth=True,
        constraint=None,
    )
