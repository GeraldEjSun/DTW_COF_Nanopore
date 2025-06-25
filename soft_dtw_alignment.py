import numpy as np
import pandas as pd
from Signal_Processing_and_Alignment import Signal_Processing
from tslearn.metrics import soft_dtw_alignment,soft_dtw
from tslearn.preprocessing import TimeSeriesResampler
import matplotlib.pyplot as plt
import seaborn as sns
from Process_Functions import interplotation
from tslearn.preprocessing import TimeSeriesResampler
from scipy.ndimage import gaussian_filter1d


def soft_dtw_alignmen_method(time_normalized_data,resample_length=None,plot_series=False,plot_alignment=False,show_matrix=False, series_standard=None,smooth=False):
    if series_standard is None:
        
        dtw_matrix=np.zeros((len(time_normalized_data),len(time_normalized_data)))

        for i in range(len(time_normalized_data)-1):
            for j in range (i+1, len(time_normalized_data)):
                series_1=time_normalized_data[i]
                series_2=time_normalized_data[j]
                distance=soft_dtw(series_1,series_2)
                normalized_distance=distance/(len(series_1)+len(series_2))
                dtw_matrix[i,j]=normalized_distance
                dtw_matrix[j,i]=normalized_distance
        rows_sums=dtw_matrix.sum(axis=1)
        min_index=np.argmin(rows_sums)
        normalized_dtw_matrix=dtw_matrix/np.amax(dtw_matrix)
        if show_matrix==True:
            sns.heatmap(normalized_dtw_matrix, annot=False, fmt=".2f", cmap="YlGnBu")
            plt.show()
            print(len(rows_sums),min_index)
    else:
        pass

    aligned_data=[]
    if series_standard is None:
        series_standard=time_normalized_data[min_index]
        np_series_standard=series_standard.to_numpy()
        np_series_standard_time=np_series_standard[:,0].T
        np_series_standard_current=np_series_standard[:,1].T
    else:
        series_standard=series_standard
        np_series_standard=series_standard.to_numpy()
        np_series_standard_time=np_series_standard[:,0].T
        np_series_standard_current=np_series_standard[:,1].T
        
        
    for i in range(len(time_normalized_data)-1):
        
        series=time_normalized_data[i]
        standard_series=np_series_standard
        path,distance=soft_dtw_alignment(series,series_standard)
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

        
    if plot_alignment==True:
        for series in aligned_data:
            plt.plot(series[0,:], series[1,:])
        plt.show()

    #align_data_after_dtw-align with the longeset data
    aligned_data_unique=[]
    for series in aligned_data:
        unique_series_time,idx=np.unique(series[0,:],return_index=True)
        unique_series_current=series[1,:][idx]
        aligned_data_unique.append(np.array([unique_series_time,unique_series_current]))
        
    len_aligned_data=np.array([series.shape[1] for series in aligned_data_unique])
    max_aligned_len=len_aligned_data.max()
    max_len_index=np.argmax(len_aligned_data)


    resampler=TimeSeriesResampler(sz=max_aligned_len if resample_length is None else resample_length)
    
    interplotation_aligned_data=[resampler.fit_transform(series[-1,:]).ravel() for series in aligned_data_unique]
    if smooth==True:
        interplotation_aligned_data=[gaussian_filter1d(series,sigma=5) for series in interplotation_aligned_data]
    else:
        pass
    average_current=np.sum(interplotation_aligned_data,axis=0)
    average_current/=len(interplotation_aligned_data)
    
    if plot_series==True:
        time=np.linspace(0,1,max_aligned_len)
        plt.plot(time,average_current,color='blue')
        plt.plot(np_series_standard_time,np_series_standard_current,color='red')
        plt.show()
        
    return interplotation_aligned_data,average_current


#Soft_DTW barycenter calculation

if __name__=="__main__":

    time_normalized_data=Signal_Processing('1.xlsx',start=0,end=20,upper_lim=700,lower_lim=200,cut_off_frequency=2)
    aligned_series, average_signal=soft_dtw_alignmen_method(time_normalized_data,plot_series=True,plot_alignment=True,show_matrix=False,smooth=True)
    
    

    
    

    
        
        

        


            
        
    
        
        



