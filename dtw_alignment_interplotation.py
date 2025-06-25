import numpy as np
import pandas as pd
from Signal_Processing_and_Alignment import Signal_Processing
from tslearn.metrics import dtw_path, dtw
from tslearn.preprocessing import TimeSeriesResampler
import matplotlib.pyplot as plt
import seaborn as sns



def dtw_alignment_interplotation(time_normalized_data,resample_length=None,plot_series=False,plot_alignment=False,show_matrix=False,constraint=None):
    dtw_matrix=np.zeros((len(time_normalized_data),len(time_normalized_data)))
    data_length=[series.shape[0] for series in time_normalized_data]
    
    series_resampler=TimeSeriesResampler(sz=max(data_length))
    resampled_current=[series_resampler.fit_transform(series.iloc[:,1].to_numpy().T).ravel() for series in time_normalized_data]
    resampled_time=[np.linspace(0,1,max(data_length)) for i in range(len(time_normalized_data))]
    resampled_data=[]
    for idx in range(0,len(time_normalized_data)):
        resampled_data.append(pd.DataFrame({'t':resampled_time[idx],'i':resampled_current[idx]}))
    
    for i in range(len(resampled_data)-1):
        for j in range (i+1, len(resampled_data)):
            series_1=resampled_data[i]
            series_2=resampled_data[j]
            distance=dtw(series_1,series_2)
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

    aligned_data=[]
    series_standard=resampled_data[min_index]
    np_series_standard=series_standard.to_numpy()
    np_series_standard_time=np_series_standard[:,0].T
    np_series_standard_current=np_series_standard[:,1].T
    for i in range(len(resampled_data)-1):
        if i != min_index:
            series=resampled_data[i]
            standard_series=np_series_standard
            if constraint is not None:
                path,_=dtw_path(series,series_standard,global_constraint="sakoe_chiba", sakoe_chiba_radius=20)
            else:
                path,_=dtw_path(series,series_standard)
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
        else:
            aligned_data.append(np.array([np_series_standard_time,np_series_standard_current]))

    aligned_data_unique=[]
    for series in aligned_data:
        unique_series_time,idx=np.unique(series[0,:],return_index=True)
        unique_series_current=series[1,:][idx]
        aligned_data_unique.append(np.array([unique_series_time,unique_series_current]))      
    if plot_alignment==True:
        for series in aligned_data_unique:
            plt.plot(series[0,:], series[1,:])
        plt.show()

    #align_data_after_dtw-align with the longeset data
 
        
    len_aligned_data=np.array([series.shape[1] for series in aligned_data_unique])
    max_aligned_len=len_aligned_data.max()
    max_len_index=np.argmax(len_aligned_data)
    
    len_aligned_data=np.array([series.shape[1] for series in aligned_data_unique])
    max_aligned_len=len_aligned_data.max()
    max_len_index=np.argmax(len_aligned_data)


    resampler=TimeSeriesResampler(sz=max_aligned_len if resample_length is None else resample_length)
    
    interplotation_aligned_data=[resampler.fit_transform(series[-1,:]).ravel() for series in aligned_data_unique]
    average_current=np.sum(interplotation_aligned_data,axis=0)
    average_current/=len(interplotation_aligned_data)
    
    if plot_series==True:
        time=np.linspace(0,1,max_aligned_len)
        plt.plot(time,average_current,color='blue')
        plt.plot(np_series_standard_time,np_series_standard_current,color='red')
        plt.show()
        
    return aligned_data_unique,average_current


#Soft_DTW barycenter calculation

if __name__=="__main__":
    import os
    current_dir=os.getcwd()
    dir_120='data_120'
    dir_compare='data_compare'
    file_name='1.xlsx'
    time_normalized_data=Signal_Processing('1.xlsx',start=-10,end=None,upper_lim=None,lower_lim=1,cut_off_frequency=2,normalization_method='standard')
    aligned_series=dtw_alignment_interplotation(time_normalized_data,plot_series=True,plot_alignment=True,show_matrix=True)
    
    

    
        
        

        


            
        
    
        
        



