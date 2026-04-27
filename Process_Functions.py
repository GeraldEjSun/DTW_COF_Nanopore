from scipy.interpolate import interp1d
from scipy import signal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
from typing import Optional, Tuple
# Bessel LPF
def Bessel_Filter(data: np.ndarray,
                  sampling_frequency: float = 100,
                  order: int = 4,
                  cut_off_frequency: float = 40) -> np.ndarray:
    """
    Applies a Bessel low-pass filter to the input data.

    This function designs a Bessel low-pass filter with the specified order and cutoff frequency,
    then applies it to the data using forward-backward filtering to avoid phase distortion.

    Args:
        data (np.ndarray): The input signal array to be filtered.
        sampling_frequency (float): The sampling frequency of the data in Hz. Default is 100.
        order (int): The order of the Bessel filter. Default is 4.
        cut_off_frequency (float): The cutoff frequency of the filter in Hz. Default is 40.

    Returns:
        np.ndarray: The filtered signal array.
    """
    normalized_cutoff = cut_off_frequency / (0.5 * sampling_frequency)
    b, a = signal.bessel(order, normalized_cutoff, btype='low', analog=False)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

# downsampling
# data: 2D array (time, current)
def Down_sampling(data: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsamples the input data by the specified factor.

    This function reduces the sampling rate of the data by the given factor using
    decimation, which includes low-pass filtering to prevent aliasing.

    Args:
        data (np.ndarray): The input 2D array with shape (n, 2), where columns are time and current.
        factor (int): The downsampling factor. Must be an integer greater than 1.

    Returns:
        np.ndarray: The downsampled data array with reduced length.
    """
    down_sampled_data = signal.decimate(data, factor)
    return down_sampled_data
 
# interpolation data: 2D array
def interplotation(data: pd.DataFrame, target_length: int) -> pd.DataFrame:
    """
    Interpolates the data to a specified target length using cubic spline interpolation.

    This function creates a cubic spline interpolation of the current values over time
    and resamples it to the target length, maintaining the original time range.

    Args:
        data (pd.DataFrame): Input DataFrame with columns for time and current.
        target_length (int): The desired number of points in the interpolated data.

    Returns:
        pd.DataFrame: A DataFrame with interpolated time and current values.
    """
    inter_func = interp1d(data.iloc[:, 0], data.iloc[:, 1], kind='cubic')  
    time_new = np.linspace(data.iloc[0, 0], data.iloc[-1, 0], target_length)
    current_interpolated = inter_func(time_new)
    return pd.DataFrame(np.array([time_new, current_interpolated]).T)



# data: (n,2) dataframe
def normalized_translocation(data: pd.DataFrame,
                             threshold: float = 450,
                             drop_head: int = 5,
                             drop_tail: int = -5) -> pd.DataFrame:
    """
    Normalizes translocation data by filtering, trimming, and standardizing.

    Filters data below the threshold, removes head and tail points, and applies
    z-score normalization (mean=0, std=1) to the current values.

    Args:
        data (pd.DataFrame): Input DataFrame with time and current columns.
        threshold (float): Current threshold below which data is considered translocation. Default is 450.
        drop_head (int): Number of points to drop from the beginning. Default is 5.
        drop_tail (int): Number of points to drop from the end (negative indexing). Default is -5.

    Returns:
        pd.DataFrame: Normalized translocation region DataFrame.
    """
    translocation_region = data[data.iloc[:, 1] < threshold]
    translocation_region = translocation_region.iloc[drop_head:drop_tail, :]
    translocation_region.iloc[:, 1] = (translocation_region.iloc[:, 1] - translocation_region.iloc[:, 1].mean()) / translocation_region.iloc[:, 1].std()
    return translocation_region


def normalized_delta(data: pd.DataFrame,
                     threshold: float = 450,
                     drop_head: int = 5,
                     drop_tail: int = -5) -> pd.DataFrame:
    """
    Normalizes delta current data relative to baseline.

    Calculates the baseline current from the first 5 points, filters translocation region,
    trims head and tail, and computes the relative change (delta I / I0).

    Args:
        data (pd.DataFrame): Input DataFrame with time and current columns.
        threshold (float): Current threshold below which data is considered translocation. Default is 450.
        drop_head (int): Number of points to drop from the beginning. Default is 5.
        drop_tail (int): Number of points to drop from the end (negative indexing). Default is -5.

    Returns:
        pd.DataFrame: Normalized delta current DataFrame.
    """
    current_0 = data.iloc[0:5, 1].mean()
    translocation_region = data[data.iloc[:, 1] < threshold]
    translocation_region = translocation_region.iloc[drop_head:drop_tail, :]
    translocation_region.iloc[:, 1] = (current_0 - translocation_region.iloc[:, 1]) / current_0
    return translocation_region

# time, current: 1D array

def Draw_Plot(time: np.ndarray,
              current: np.ndarray,
              format: str = "-",
              start: int = 0,
              end: Optional[int] = None,
              x_range: float = 60) -> None:
    """
    Plots the time vs current data with specified formatting.

    Creates a line plot of current over time, with grid and axis labels.

    Args:
        time (np.ndarray): Array of time values.
        current (np.ndarray): Array of current values.
        format (str): Plot line format (e.g., '-', '--'). Default is "-".
        start (int): Starting index for plotting. Default is 0.
        end (Optional[int]): Ending index for plotting. If None, plots to end. Default is None.
        x_range (float): X-axis limit in ms. Default is 60.

    Returns:
        None: Displays the plot.
    """
    plt.plot(time[start:end], current[start:end])
    plt.xlabel('Time/ms')
    plt.ylabel('Current/pA')
    plt.grid(True)
    plt.xlim(0, x_range)
    

def normalized_time_axis(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the time axis using min-max scaling.

    Scales the time values to the range [0, 1] based on the minimum and maximum time values.

    Args:
        data (pd.DataFrame): Input DataFrame with time and current columns.

    Returns:
        pd.DataFrame: DataFrame with normalized time axis.
    """
    normalized_translocation_region = copy.deepcopy(data)
    normalized_translocation_region.iloc[:, 0] = (normalized_translocation_region.iloc[:, 0] - min(normalized_translocation_region.iloc[:, 0])) / (max(normalized_translocation_region.iloc[:, 0]) - min(normalized_translocation_region.iloc[:, 0]))
    return normalized_translocation_region
    