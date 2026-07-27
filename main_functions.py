import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tslearn.preprocessing import TimeSeriesResampler
from Process_Functions import Bessel_Filter, Down_sampling, normalized_translocation, interplotation, Draw_Plot, normalized_time_axis, normalized_delta
from scipy.ndimage import gaussian_filter1d
import os
from tslearn.metrics import dtw_path, dtw
from typing import List, Optional, Tuple, Union
import seaborn as sns
from tslearn.barycenters import softdtw_barycenter
from tslearn.metrics import gamma_soft_dtw

'''
* **file_name**: Name of the input file. The default format is multi-column time–current data (see the format in `Reference1.xlsx`).

* **start, end**: Specify the range of data to be imported. By default, the entire dataset is used.

* **upper_lim, lower_lim**: Define the acceptable range of data length.

* **sampling_frequency**: Sampling frequency of the imported data. Default is **100 kHz**.

* **cut_off_frequency**: Cutoff frequency, which determines the degree of filtering. Default is **2 kHz**.

* **drop_head, drop_tail**: Number of data points removed from the beginning and end after processing. Since the pore signal separation method may generate very high values at both ends, the default is to remove **5 data points** from each side.

* **smooth**: Whether to apply Gaussian filter smoothing. Default is **False**.

* **normalization_method**: Determines the normalization method:

  * `'standard'`: mean / standard deviation
  * `'delta'`: ΔI / I₀

  Default is `'standard'`.

'''
def Signal_Processing(file_name: str, start: Optional[int] = None, end: Optional[int] = None, upper_lim: Optional[int] = None, lower_lim: int = 1, sampling_frequency: int = 100, cut_off_frequency: int = 2,
                      drop_head: int = 5, drop_tail: int = -5, smooth: bool = False, normalization_method: str = 'standard', threshold: int = 60) -> Tuple[List[pd.DataFrame], List[int]]:
    """
    Processes signal data from a file by filtering, normalizing, and slicing based on specified parameters.

    Inputs:
    - file_name: str - Name of the input file (supports .xlsx or .csv formats).
    - start: Optional[int] - Starting index for data slicing (default: None, uses entire data).
    - end: Optional[int] - Ending index for data slicing (default: None, uses entire data).
    - upper_lim: Optional[int] - Upper limit for data length filtering (default: None, no upper limit).
    - lower_lim: int - Lower limit for data length filtering (default: 1).
    - sampling_frequency: int - Sampling frequency in kHz (default: 100).
    - cut_off_frequency: int - Cutoff frequency for filtering in kHz (default: 2).
    - drop_head: int - Number of points to drop from the beginning (default: 5).
    - drop_tail: int - Number of points to drop from the end (default: -5).
    - smooth: bool - Whether to apply Gaussian smoothing (default: False).
    - normalization_method: str - Normalization method ('standard' or 'delta', default: 'standard').
    - threshold: int - Threshold value for normalization (default: 60).

    Outputs:
    - time_normalized_data: List[pd.DataFrame] - List of processed and normalized time series data.
    - selected_signal_index: List[int] - List of selected signal indices based on length criteria.
    """
    if file_name[-3:]=='lsx':
        original_data=pd.read_excel(file_name)
        
    elif file_name[-3:]=='csv':
        original_data=pd.read_csv(file_name)
    original_data = original_data.filter(regex='^(?!Unnamed)')
    signal_index = [int(col) for n, col in enumerate(original_data.columns) if (n + 1) % 3 == 1]  
    data_cleared=original_data.dropna(axis=1,how='all')
    #slice different serieses
    data_sliced = []
    column_num = len(data_cleared.columns)

    for i in range(0, column_num, 2):
        df_sliced = data_cleared.iloc[:, i:i+2].copy()  
        df_sliced.columns = ['t', 'i' ]

        new_df_sliced = df_sliced.dropna(axis=0, how="all").copy()

        if not new_df_sliced.empty:
            new_df_sliced.loc[:, new_df_sliced.columns[0]] -= new_df_sliced.iloc[0, 0]
        data_sliced.append(new_df_sliced)

        

    processed_data=[]
    for series in data_sliced:
        filtered_current=Bessel_Filter(series['i'],sampling_frequency=sampling_frequency,cut_off_frequency=cut_off_frequency)
        filtered_curve=np.array([series['t'],filtered_current])
        down_sampled_curve=filtered_curve
        filtered_curve_dataframe=pd.DataFrame(down_sampled_curve.T,columns=['t','i'])
        if normalization_method=='standard':
            normalized_data=normalized_translocation(filtered_curve_dataframe,threshold=min(filtered_curve_dataframe.iloc[:,1])+threshold,drop_head=drop_head, drop_tail=drop_tail)
            processed_data.append(normalized_data)
        elif normalization_method=='delta':
            normalized_data=normalized_delta(filtered_curve_dataframe,threshold=min(filtered_curve_dataframe.iloc[:,1])+threshold,drop_head=drop_head, drop_tail=drop_tail)
            processed_data.append(normalized_data)


    if upper_lim is None:
        target_data=processed_data
   
    else:
        target_data=[series for series in processed_data if len(series)<upper_lim and len(series)>lower_lim]
        selected_index=[]
        for index, series in enumerate(processed_data):
            if len(series) < upper_lim and len(series) > lower_lim:
                selected_index.append(index)
        selected_signal_index=[signal_index[i] for i in selected_index]
        
                
    
  
    len_target_data=[len(series) for series in target_data]

    if smooth==True:
        for series in target_data:
            series['i']=gaussian_filter1d(series['i'].values,sigma=3)

    
    time_normalized_data=[normalized_time_axis(series) for series in target_data][start:end]

    
    return time_normalized_data,selected_signal_index




def dtw_alignment(time_normalized_data: List[pd.DataFrame], signal_index: Optional[List[int]] = None, resample_length: Optional[int] = None, plot_series: bool = False,
                  plot_alignment: bool = False, show_matrix: bool = False, series_standard: Optional[Union[pd.DataFrame, np.ndarray]] = None, smooth: bool = False, constraint: Optional[str] = None, method: str = 'classic', sakoe_radius: int = 100,
                  target_path: Optional[str] = None, inverse: bool = False) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Performs Dynamic Time Warping (DTW) alignment on time-normalized data series.

    Inputs:
    - time_normalized_data: List[pd.DataFrame] - List of time-normalized data series to align.
    - signal_index: Optional[List[int]] - Indices of signals for labeling (default: None).
    - resample_length: Optional[int] - Length to resample aligned data to (default: None, uses max length).
    - plot_series: bool - Whether to plot individual series (default: False).
    - plot_alignment: bool - Whether to plot aligned series (default: False).
    - show_matrix: bool - Whether to display DTW distance matrix (default: False).
    - series_standard: Optional[Union[pd.DataFrame, np.ndarray]] - Standard series for alignment (default: None, uses the one with min DTW distance).
    - smooth: bool - Whether to apply smoothing to aligned data (default: False).
    - constraint: Optional[str] - DTW constraint type (e.g., 'sakoe_chiba', default: None).
    - method: str - Alignment method ('classic', default: 'classic').
    - sakoe_radius: int - Radius for Sakoe-Chiba constraint (default: 100).
    - target_path: Optional[str] - Path to save plots (default: None).
    - inverse: bool - Whether to flip the aligned current data (default: False).

    Outputs:
    - aligned_data_unique: List[np.ndarray] - List of aligned data series as numpy arrays.
    - average_current: np.ndarray - Averaged current values after alignment and resampling.
    """
    if method=='classic':
        if series_standard is None:
            
            dtw_matrix=np.zeros((len(time_normalized_data),len(time_normalized_data)))

            for i in range(len(time_normalized_data)-1):
                for j in range (i+1, len(time_normalized_data)):
                    series_1=time_normalized_data[i]
                    series_2=time_normalized_data[j]
                    distance=dtw(series_1,series_2)
                    normalized_distance=distance/(len(series_1)+len(series_2))
                    dtw_matrix[i,j]=normalized_distance
                    dtw_matrix[j,i]=normalized_distance
            rows_sums=dtw_matrix.sum(axis=1)
            min_index=np.argmin(rows_sums)
            if signal_index is not None:
                print(signal_index[min_index])
            normalized_dtw_matrix=dtw_matrix/np.amax(dtw_matrix)
            new_mtx = np.where(normalized_dtw_matrix == 0, 1, normalized_dtw_matrix)
            if show_matrix==True:
                plt.rcParams["figure.dpi"] = 200 
                ax=sns.heatmap(normalized_dtw_matrix, annot=False, fmt=".2f", cmap="YlGnBu",annot_kws={"fontweight": "bold","fontfamily":"Arial"},cbar_kws={
        "aspect": 16.7  
})
                ax.set_xticks(range(0, normalized_dtw_matrix.shape[0], 10))
                ax.set_xticklabels(range(0, normalized_dtw_matrix.shape[0], 10), fontsize=20,fontweight="bold", family="Arial")  


                ax.set_yticks(range(0, normalized_dtw_matrix.shape[0], 10))
                ax.set_yticklabels(range(0, normalized_dtw_matrix.shape[0], 10), fontsize=20,fontweight="bold", family="Arial")
                ax.tick_params(axis='x', labelrotation=0)
                cbar = ax.collections[0].colorbar

                cbar.ax.tick_params(labelsize=20) 
                for label in cbar.ax.get_yticklabels():
                    label.set_fontweight("bold")
                    label.set_family("Arial")
                plt.show()
                print(len(rows_sums),min_index)
                min_row,min_col=np.unravel_index(np.argmin(new_mtx),new_mtx.shape)
                print(f'lowest distance is from{(signal_index[min_row],signal_index[min_col])}')
        else:
            pass

        aligned_data=[]
        if series_standard is None:
            
            series_standard=time_normalized_data[min_index]
            np_series_standard=series_standard.to_numpy()
            
                
        else:
            if isinstance(series_standard,pd.DataFrame):
                series_standard=series_standard
                np_series_standard=series_standard.to_numpy()
            elif isinstance(series_standard,np.ndarray):
                if series_standard.ndim==1:
                    np_series_standard=np.array([np.linspace(0,1,len(series_standard)),series_standard]).T
                    series_standard=pd.DataFrame(np_series_standard,columns=['t','i'])
                elif series_standard.ndim==2:
                    np_series_standard=series_standard
                    series_standard=pd.DataFrame(np_series_standard,columns=['t','i'])
                    


            
            
        for i in range(len(time_normalized_data)):
            
            series=time_normalized_data[i]
            standard_series=np_series_standard
            if constraint is  None:
                path,_=dtw_path(series,series_standard)
                
            elif constraint=="sakoe_chiba":
                if sakoe_radius <=1:
                    path,_=dtw_path(series,series_standard,global_constraint=constraint, sakoe_chiba_radius=series.shape[0]*sakoe_radius)
                else:
                    path,_=dtw_path(series,series_standard,global_constraint=constraint, sakoe_chiba_radius=sakoe_radius)
                
            aligned_series=np.zeros((len(path),2))
            aligned_standard_series=np.zeros((len(path),2))
            for i, (idx1, idx2) in enumerate(path):
                np_series=series.to_numpy()
                aligned_series[i] = np_series[idx1]
                aligned_standard_series[i]=standard_series[idx2]
            #aligned_series_time=[x[0] for x in aligned_series]
            aligned_series_current=[x[1] for x in aligned_series]
            aligned_standard_series_time=[x[0] for x in aligned_standard_series]
            aligned_data.append(np.array([aligned_standard_series_time,aligned_series_current]))
        
        aligned_data_unique=[]
            
        for series in aligned_data:
            unique_series_time,idx=np.unique(series[0,:],return_index=True)
            unique_series_current=series[1,:][idx]
            
            if smooth==True:
                unique_series_current=gaussian_filter1d(unique_series_current,sigma=5)

            if inverse==False:
                aligned_data_unique.append(np.array([unique_series_time,unique_series_current]))
            else:
                aligned_data_unique.append(np.array([unique_series_time,np.flip(unique_series_current)]))  
        

            
        if plot_alignment==True:
            for index, series in enumerate(aligned_data_unique):
                
                if signal_index is None:                   
                        plt.plot(series[0,:], series[1,:],label=str(index))                       
                else:                          
                        plt.plot(series[0,:],
                         series[1,:],label=str(signal_index[index]))
            
     
                
                if target_path is not None:
                    plt.savefig(os.path.join(target_path,'radius'+str(sakoe_radius)+'.png'))
            if target_path is not None:
                plt.close('all')
            else:
                plt.legend(fontsize=4)
                plt.plot(series_standard.iloc[:,0],series_standard.iloc[:,1])
                plt.show()
        len_aligned_data=np.array([series.shape[1] for series in aligned_data_unique])
        max_aligned_len=len_aligned_data.max()
        max_len_index=np.argmax(len_aligned_data)


        resampler=TimeSeriesResampler(sz=max_aligned_len if resample_length is None else resample_length)
        
        interplotation_aligned_data=[resampler.fit_transform(series[-1,:]).ravel() for series in aligned_data_unique]

        average_current=np.sum(interplotation_aligned_data,axis=0)
        average_current/=len(interplotation_aligned_data)
    
            
        return aligned_data_unique, average_current
    
    
    
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
    if target_length is None:
        target_length=max([series.shape[0] for series in data])
        resampler=TimeSeriesResampler(sz=target_length)
    else:  
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

    return currents_resampled,barycenter,time_series


def comparison_plot(series1: Optional[np.ndarray] = None, series2: Optional[np.ndarray] = None, flip: bool = False, export: bool = False, file_name: str = '', save_path: str = 'data export') -> None:
    """
    Plots and optionally exports a comparison between two series.

    Inputs:
    - series1: Optional[np.ndarray] - First series to plot (default: None).
    - series2: Optional[np.ndarray] - Second series to plot (default: None).
    - flip: bool - Whether to flip the second series (default: False).
    - export: bool - Whether to export the plot (default: False).
    - file_name: str - File name for export (default: '').
    - save_path: str - Path to save the exported plot (default: 'data export').

    Outputs:
    - None
    """
    time_1=np.linspace(0,1, len(series1))  
    time_2=np.linspace(0,1, len(series2))    
    plt.plot(time_1,series1,color='red',label='direction_1')
    
    if flip==True:
        plt.plot(time_2, np.flip(series2),color='blue',label='direction_2')
    else:
        
        plt.plot(time_2,series2,color='blue')
    plt.show()
    
    
def export_data(data: List[Union[np.ndarray, pd.DataFrame]], save_path: str = '', file_name: str = '1', signal_index: Optional[List[int]] = None) -> None:
    """
    Exports processed data to an Excel file.

    Inputs:
    - data: List[Union[np.ndarray, pd.DataFrame]] - List of data series to export.
    - save_path: str - Directory path to save the file (default: '').
    - file_name: str - Name of the output file without extension (default: '1').
    - signal_index: Optional[List[int]] - Indices for naming columns (default: None).

    Outputs:
    - None
    """
    data_processed=[]
    if signal_index==None:
        for i in range(len(data)):
            series_index=pd.DataFrame([], columns=[str(i+1)])
            data_processed.append(series_index)
            if isinstance(data[0],np.ndarray):
                print(data[0].ndim==1)
                if data[0].ndim==1:
                    data_processed.append(pd.DataFrame({'t':np.linspace(0,1,len(data[i])),'i':data[i]}))
                elif data[0].ndim==2:
                    data_processed.append(pd.DataFrame(data[i].T,columns=['t','i']))
                else:
                    print('TypeError, Please check data type')
            elif isinstance(data[0],pd.DataFrame):
                data_processed.append(data[i])
            else:
                print('TypeError, Please check data type')
            
    else:
        for i,ind in enumerate(signal_index):
            series_index=pd.DataFrame([], columns=[str(ind)])
            data_processed.append(series_index)
            
            if isinstance(data[0],np.ndarray):
                if data[0].ndim==1:
                    data_processed.append(pd.DataFrame({'t':np.linspace(0,1,len(data[i])),'i':data[i]}))
                elif data[0].ndim==2:
                    data_processed.append(pd.DataFrame(data[i].T,columns=['t','i']))
                else:
                    print('TypeError, Please check data type')
            elif isinstance(data[0],pd.DataFrame):
                data_processed.append(data[i])
            else:
                    print('TypeError, Please check data type')

            
            
    

    max_length = max([df.shape[0] for df in data_processed])


    for i, df in enumerate(data_processed):
        if df.shape[0] < max_length:
            num_missing = max_length - df.shape[0]
            missing_df = pd.DataFrame(index=range(num_missing), columns=df.columns)
            data_processed[i] = pd.concat([df, missing_df], ignore_index=True)


    combined_df = pd.concat(data_processed, axis=1)


    cur_dir=os.getcwd()
    abs_save_path=os.path.join(cur_dir,save_path)
    if not os.path.exists(abs_save_path):
        os.makedirs(abs_save_path)
    combined_df.to_excel(os.path.join(abs_save_path,file_name+'.xlsx'), index=False)
    
    
def ddtw_processing(signal_avg_1: Union[List[float], np.ndarray, pd.Series], signal_avg_2: Union[List[float], np.ndarray, pd.Series], signal_weight: float = 1.0,
                    derivative_weight: float = 1.6, extrema_weight: float = 7.4, prominence_value:float = 0.05) -> Tuple[object, object, dict]:
    """
    Performs peak-valley enhanced derivative-DTW alignment between two average signals and plots the results.

    Inputs:
    - signal_avg_1: Union[List[float], np.ndarray, pd.Series] - Reference average signal current values.
    - signal_avg_2: Union[List[float], np.ndarray, pd.Series] - Target average signal current values to align.
    - signal_weight: float - Weight applied to the original signal feature (default: 1.0).
    - derivative_weight: float - Weight applied to the first-order derivative feature (default: 1.6).
    - extrema_weight: float - Weight applied to peak and valley marker features (default: 7.4).
    - prominence_value: float - Minimum prominence for peak and valley detection (default: 0.05).

    Outputs:
    - alignment_figure: object - Plot comparing signal_avg_1 with the DTW-aligned signal_avg_2.
    - mapping_figure: object - Plot showing peak and valley mappings before warping.
    - mse_values: dict - Dictionary containing DTW distance, raw MSE values, and linearly corrected MSE values.
    """
    from scipy.signal import resample, find_peaks
    from sklearn.metrics import mean_squared_error

    smooth_sigma=0
    radius_value=100
    prominence_value=prominence_value

    def zscore(y: Union[List[float], np.ndarray, pd.Series]) -> np.ndarray:
        y=np.asarray(y,dtype=float)
        return (y-np.mean(y))/(np.std(y)+1e-8)

    def fill_nan_by_interp(y: Union[List[float], np.ndarray, pd.Series]) -> np.ndarray:
        y=np.asarray(y,dtype=float)
        x=np.arange(len(y))
        good=np.isfinite(y)

        if good.sum()==0:
            raise ValueError("Warped signal consists entirely of NaNs. DTW path reconstruction failed.")

        if good.sum()<len(y):
            y[~good]=np.interp(x[~good],x[good],y[good])

        return y

    def linear_correct_mse(template_signal: Union[List[float], np.ndarray, pd.Series], target_signal: Union[List[float], np.ndarray, pd.Series]) -> Tuple[float, float, float, float, np.ndarray]:
        template_signal=np.asarray(template_signal,dtype=float)
        target_signal=np.asarray(target_signal,dtype=float)
        mse_raw=mean_squared_error(template_signal,target_signal)

        A=np.vstack([target_signal,np.ones_like(target_signal)]).T
        a,b=np.linalg.lstsq(A,template_signal,rcond=None)[0]
        target_signal_corrected=a*target_signal+b
        mse_corrected=mean_squared_error(template_signal,target_signal_corrected)

        return mse_raw,mse_corrected,a,b,target_signal_corrected

    signal_avg_1=fill_nan_by_interp(signal_avg_1).ravel()
    signal_avg_2=fill_nan_by_interp(signal_avg_2).ravel()

    if len(signal_avg_1)==0 or len(signal_avg_2)==0:
        raise ValueError("signal_avg_1 and signal_avg_2 must contain at least one value.")

    time_avg_1=np.linspace(0,1,len(signal_avg_1))
    signal_avg_2_resampled=resample(signal_avg_2,len(signal_avg_1))

    # Use smoothed curves to find peaks/valleys and calculate DTW features.
    if smooth_sigma>0:
        signal_1_used=gaussian_filter1d(signal_avg_1,sigma=smooth_sigma)
        signal_2_used=gaussian_filter1d(signal_avg_2_resampled,sigma=smooth_sigma)
    else:
        signal_1_used=signal_avg_1.copy()
        signal_2_used=signal_avg_2_resampled.copy()

    # Peaks and valleys.
    peaks_1,_=find_peaks(signal_1_used,prominence=prominence_value)
    peaks_2,_=find_peaks(signal_2_used,prominence=prominence_value)
    valleys_1,_=find_peaks(-signal_1_used,prominence=prominence_value)
    valleys_2,_=find_peaks(-signal_2_used,prominence=prominence_value)

    # Peaks = +1, valleys = -1.
    extrema_1=np.zeros_like(signal_1_used)
    extrema_2=np.zeros_like(signal_2_used)
    extrema_1[peaks_1]=1
    extrema_1[valleys_1]=-1
    extrema_2[peaks_2]=1
    extrema_2[valleys_2]=-1
    extrema_1=gaussian_filter1d(extrema_1,sigma=2)
    extrema_2=gaussian_filter1d(extrema_2,sigma=2)

    # Construct peak-valley enhanced derivative-DTW features.
    feature_1=np.column_stack([
        signal_weight*zscore(signal_1_used),
        derivative_weight*zscore(np.gradient(signal_1_used)),
        extrema_weight*zscore(extrema_1)
    ])

    feature_2=np.column_stack([
        signal_weight*zscore(signal_2_used),
        derivative_weight*zscore(np.gradient(signal_2_used)),
        extrema_weight*zscore(extrema_2)
    ])

    dpath,dtw_distance=dtw_path(feature_2,feature_1,global_constraint="sakoe_chiba",sakoe_chiba_radius=radius_value)

    # Reconstruct signal_avg_2 based on the DTW path.
    warped_signal_2=np.full_like(signal_avg_1,np.nan,dtype=float)
    for signal_1_index in range(len(signal_avg_1)):
        matched_indexes=[idx1 for idx1,idx2 in dpath if idx2==signal_1_index]
        if matched_indexes:
            warped_signal_2[signal_1_index]=np.mean(signal_avg_2_resampled[matched_indexes])

    warped_signal_2=fill_nan_by_interp(warped_signal_2)

    mse_before=mean_squared_error(signal_avg_1,signal_avg_2_resampled)
    mse_after=mean_squared_error(signal_avg_1,warped_signal_2)

    mse_before_raw,mse_before_corrected,a_before,b_before,signal_avg_2_resampled_corrected=linear_correct_mse(signal_avg_1,signal_avg_2_resampled)
    mse_after_raw,mse_after_corrected,a_after,b_after,warped_signal_2_corrected=linear_correct_mse(signal_avg_1,warped_signal_2)

    # Plot 1: DTW aligned curves.
    alignment_figure,alignment_axis=plt.subplots(figsize=(10,5))
    alignment_axis.plot(time_avg_1,signal_avg_1,color="black",linewidth=2,label="Average Signal 1")
    alignment_axis.plot(time_avg_1,warped_signal_2,color="red",linestyle="--",alpha=0.85,label=f"Average Signal 2 after DTW, Radius={radius_value}")
    alignment_axis.legend()
    alignment_axis.set_title("Peak-Valley Enhanced Derivative-DTW Alignment")
    alignment_axis.set_xlabel("Original Time of Average Signal 1")
    alignment_axis.set_ylabel("Signal Current")
    alignment_figure.tight_layout()
    plt.show()

    # Establish signal_avg_1 index -> signal_avg_2 index mapping.
    mapping_1_to_2={}
    for idx1,idx2 in dpath:
        if idx2 not in mapping_1_to_2:
            mapping_1_to_2[idx2]=[]
        mapping_1_to_2[idx2].append(idx1)

    # Plot 2: peak/valley mappings on original curves.
    mapping_figure,mapping_axis=plt.subplots(figsize=(10,5))
    mapping_axis.plot(time_avg_1,signal_avg_1,color="black",linewidth=2,label="Average Signal 1")
    mapping_axis.plot(time_avg_1,signal_avg_2_resampled,color="red",linestyle="--",alpha=0.8,label="Average Signal 2 before warping")
    mapping_axis.scatter(time_avg_1[peaks_1],signal_avg_1[peaks_1],color="black",s=55,marker="o",label="Signal 1 peaks")
    mapping_axis.scatter(time_avg_1[valleys_1],signal_avg_1[valleys_1],color="black",s=55,marker="v",label="Signal 1 valleys")
    mapping_axis.scatter(time_avg_1[peaks_2],signal_avg_2_resampled[peaks_2],color="red",s=55,marker="o",label="Signal 2 peaks")
    mapping_axis.scatter(time_avg_1[valleys_2],signal_avg_2_resampled[valleys_2],color="red",s=55,marker="v",label="Signal 2 valleys")

    extrema_points_1=np.sort(np.concatenate([peaks_1,valleys_1]))
    for point_1 in extrema_points_1:
        matched_2_list=mapping_1_to_2.get(point_1,[])
        if len(matched_2_list)==0:
            continue

        point_2=int(np.median(matched_2_list))
        mapping_axis.plot([time_avg_1[point_1],time_avg_1[point_2]],[signal_avg_1[point_1],signal_avg_2_resampled[point_2]],color="gray",linestyle=":",alpha=0.7)

    mapping_axis.legend(fontsize=8)
    mapping_axis.set_title(f"Peak-Valley Mapping Based on Enhanced DTW, Radius={radius_value}")
    mapping_axis.set_xlabel("Normalized Time")
    mapping_axis.set_ylabel("Signal Current")
    mapping_figure.tight_layout()
    plt.show()

    mse_values={
        "dtw_distance":float(dtw_distance),
        "mse_before":float(mse_before),
        "mse_after":float(mse_after),
        "mse_before_raw":float(mse_before_raw),
        "mse_before_corrected":float(mse_before_corrected),
        "mse_after_raw":float(mse_after_raw),
        "mse_after_corrected":float(mse_after_corrected),
        "a_before":float(a_before),
        "b_before":float(b_before),
        "a_after":float(a_after),
        "b_after":float(b_after)
    }

    return alignment_figure,mapping_figure,mse_values    