from tslearn.metrics import gamma_soft_dtw, soft_dtw
from tslearn.barycenters import softdtw_barycenter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tslearn.preprocessing import TimeSeriesResampler
import time
from typing import List, Optional, Tuple

def barycenter_plot(data: List[np.ndarray], target_length: int = 500, start: Optional[int] = None, end: Optional[int] = None, gamma: Optional[float] = None, inverse: bool = False, 
                    choose_color: str = 'red', plot: bool = False, max_iter: int = 50, tol: float = 1e-3) -> Tuple[List[np.ndarray], np.ndarray, List[np.ndarray]]:
    """
    Computes the barycenter (average) of time series data using Soft DTW and optionally plots it.

    Inputs:
    - data: List[np.ndarray] - List of time series data arrays to compute barycenter for.
    - target_length: int - Target length for resampling (default: 500).
    - start: Optional[int] - Starting index for data slicing (default: None).
    - end: Optional[int] - Ending index for data slicing (default: None).
    - gamma: Optional[float] - Gamma parameter for Soft DTW (default: None, auto-selected).
    - inverse: bool - Whether to flip the barycenter (default: False).
    - choose_color: str - Color for plotting (default: 'red').
    - plot: bool - Whether to plot the barycenter (default: False).
    - max_iter: int - Maximum iterations for barycenter computation (default: 50).
    - tol: float - Tolerance for convergence (default: 1e-3).

    Outputs:
    - currents_resampled: List[np.ndarray] - List of resampled current data arrays.
    - barycenter: np.ndarray - Computed barycenter array.
    - time_series: List[np.ndarray] - Time series representation of the barycenter.
    """
    resampler=TimeSeriesResampler(sz=target_length)
    currents_resampled=[resampler.fit_transform(series).ravel() for series in data[start:end]]

    if gamma is None:
        gamma_select=[]
        for i in range(5):
            gamma_select.append(gamma_soft_dtw(currents_resampled[:], len(currents_resampled),i ))
        gamma= np.mean(gamma_select)
        print(f'auto_gamma:{gamma:.3f}')
        
    barycenter = softdtw_barycenter(currents_resampled, gamma=gamma,max_iter=max_iter,tol=tol)
    if inverse==True:
        barycenter=np.flip(barycenter)
    else:
        pass
    time=np.linspace(0,1,target_length)
    time_series=[np.array([time,barycenter.ravel()])]   
    
    if plot==True:

        
        plt.rcParams['figure.dpi'] = 100


        plt.plot(time,barycenter, label='SoftDTW Barycenter', color=choose_color, linewidth=2)


        plt.legend()
        plt.title('SoftDTW Barycenter Plot')
        plt.xlabel('Time Index')
        plt.ylabel('Normalized Current')
        


    return currents_resampled,barycenter,time_series
    



if __name__ == '__main__' :
    import os
    current_dir=os.getcwd()
    dir_120='data_120'
    dir_compare='data_compare'
    file_name='direction_1.xlsx'
    file_name_2='direction_2.xlsx'
    from Signal_Processing_and_Alignment import Signal_Processing
    time_normalized_data_1=Signal_Processing(os.path.join(current_dir,dir_compare,file_name),sampling_frequency=100,cut_off_frequency=5, threshold=60
                                             ,start=-14,upper_lim=1e3,lower_lim=1e2,normalization_method='standard')

    time_normalized_current_np_1=[series.to_numpy().T[1,:] for series in time_normalized_data_1] 
    
    length_1=barycenter_plot(time_normalized_current_np_1,choose_color='red',gamma=0.5,plot=True)

   
    
    time_normalized_data_2=Signal_Processing(os.path.join(current_dir,dir_compare,file_name_2),sampling_frequency=100,cut_off_frequency=5,threshold=60
                                             ,start=-14,upper_lim=1e3,lower_lim=1e2,normalization_method='standard')
    start_time=time.time()
    time_normalized_current_np_2=[series.to_numpy().T[1,:] for series in time_normalized_data_2] 
    
    length_2=barycenter_plot(time_normalized_current_np_2,choose_color='blue',gamma=0.5,plot=True)
    plt.show()
    


    
    
    