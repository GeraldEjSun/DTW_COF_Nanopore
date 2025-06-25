import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dtw_alignment import dtw_alignment
from soft_dtw_barycenter import barycenter_plot
from Signal_Processing_and_Alignment import Signal_Processing
from dtw_alignment_interplotation import dtw_alignment_interplotation
from tslearn.preprocessing import TimeSeriesResampler
from scipy.ndimage import gaussian_filter1d



#for series in aligned_data_1:
    #plt.plot(series[0,:], series[1,:])
#for series in aligned_data_2:
    #plt.plot(series[0,:], series[1,:])
def comparison_plot(series1=None,series2=None,flip=False,export=False,file_name='',save_path='data export'):
    time_1=np.linspace(0,1, len(series1))  
    time_2=np.linspace(0,1, len(series2))    
    plt.plot(time_1,series1,color='red',label='direction_1')
    
    if flip==True:
        plt.plot(time_2, np.flip(series2),color='blue',label='direction_2')
    else:
        
        plt.plot(time_2,series2,color='blue')
    plt.show()
    if export==True:
        cur_dir=os.getcwd()
        file_name=file_name
        export_df=pd.DataFrame({'time_1':time_1,'direction_1':series1,'time_2':time_2,'direction_2':series2})
        export_df.to_excel(os.path.join(cur_dir,save_path,file_name+'.xlsx'), index=False)
    
def moving_average(series, window_size=5):
    return np.convolve(series, np.ones(window_size)/window_size, mode='valid')
    


if __name__=='__main__':
    import os
    current_dir=os.getcwd()
    dir_120='data_120'
    dir_final='data_final'
    signal_1='direction_1.xlsx'
    signal_2='direction_2.xlsx'
    #data processing
    drop_head=25
    drop_tail=-drop_head
    alignment_type='barycenter'
    direction_1_processed,direction_1_index=Signal_Processing(os.path.join(current_dir,dir_120,signal_1),sampling_frequency=100,
                                            cut_off_frequency=5,threshold=60,normalization_method='standard',upper_lim=700,lower_lim=200,drop_head=drop_head,drop_tail=drop_tail)

    direction_2_processed,direction_2_index=Signal_Processing(os.path.join(current_dir,dir_120,signal_2),sampling_frequency=100,
                                            cut_off_frequency=5,threshold=60,upper_lim=700,lower_lim=200,drop_head=drop_head,drop_tail=drop_tail)
    direction_1_processed_current=[series.to_numpy().T[1,:] for series in direction_1_processed]
    direction_2_processed_current=[series.to_numpy().T[1,:] for series in direction_2_processed]
    #selected_index_1=pd.DataFrame({'direction_1_index':direction_1_index,'direction_2_index':direction_2_index})
    #selected_index_2=pd.DataFrame({'direction_1_index':direction_1_index,'direction_2_index':direction_2_index}) 

    #nearest series alignment
    if alignment_type=='standard':
        aligned_data_1,_,avergage_signal_1=dtw_alignment(direction_1_processed,smooth=True)
        aligned_data_2,_,avergage_signal_2=dtw_alignment(direction_2_processed,smooth=True)
        comparison_plot(series1=avergage_signal_1,series2=avergage_signal_2,flip=False)

    #barycenter calculation
    elif alignment_type=='barycenter':
        direction_1_resampled,direction_1_barycenter,_=barycenter_plot([gaussian_filter1d(series,sigma=5) for series in direction_1_processed_current],
                                                                       gamma=0.5,max_iter=500)
        direction_2_resampled,direction_2_barycenter,_=barycenter_plot([gaussian_filter1d(series,sigma=5) for series in direction_2_processed_current],
                                                                       gamma=0.5,max_iter=500)
        time_1=np.linspace(0,1,len(direction_1_barycenter))
        time_2=np.linspace(0,1,len(direction_2_barycenter))
        direction_1_standard_series=pd.DataFrame({'t':time_1,'i':direction_1_barycenter.ravel()})
        direction_2_standard_series=pd.DataFrame({'t':time_2,'i':direction_2_barycenter.ravel()})
        aligned_data_1,_,avergage_signal_1=dtw_alignment(direction_1_processed,series_standard= direction_1_standard_series)
        aligned_data_2,_, avergage_signal_2=dtw_alignment(direction_2_processed,series_standard= direction_2_standard_series)
        comparison_plot(series1=gaussian_filter1d(avergage_signal_1,sigma=5) ,series2=gaussian_filter1d(avergage_signal_2,sigma=2),flip=True,export=True,file_name='data150_barycenter_700_200_gamma0.5_head25',save_path=os.path.join('data export','barycenter'))
        
    elif alignment_type=='average':
        size_1=max([series.shape[0] for series in direction_1_processed_current])
        size_2=max([series.shape[0] for series in direction_2_processed_current])
        resampler_1=TimeSeriesResampler(sz=size_1)
        resampler_2=TimeSeriesResampler(sz=size_2)       
        aligned_direction_1_processed_current=np.array([resampler_1.fit_transform(series).ravel() for series in direction_1_processed_current])
        aligned_direction_2_processed_current=np.array([resampler_2.fit_transform(series).ravel() for series in direction_2_processed_current])      
        avergage_signal_1=np.mean(aligned_direction_1_processed_current,axis=0)
        avergage_signal_2=np.mean(aligned_direction_2_processed_current,axis=0)
        comparison_plot(series1=avergage_signal_1,series2=avergage_signal_2,flip=True)
        
    else:
        print('TypeError, choose a valid type')
    

    
