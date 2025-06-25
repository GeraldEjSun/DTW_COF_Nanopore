import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tslearn.preprocessing import TimeSeriesResampler
from Process_Functions import Bessel_Filter, Down_sampling, normalized_translocation, interplotation, Draw_Plot, normalized_time_axis, normalized_delta
from scipy.ndimage import gaussian_filter1d
import os
'''
file_name:导入文件的名称 默认为多列时间-电流数据（参考1.xlsx中的格式）
start end 指定导入数据的范围 默认全部
upper lim lower lim 决定数据长度范围

sampling frequency:导入数据的采样频率 默认100(KHz)
cut_off_frequency:截止频率，决定滤波的程度默认2(KHz)
drop head drop_tail 去掉处理后数据的前后几个数据点（由于采用的分离过孔信号方法，前后会产生很高的数据点）默认去掉5个数据点
smooth:是否使用高斯滤波器平滑数据，默认否
normalization_method 决定是-均值/标准差‘standard’还是‘delta’（deltaI/I_0,默认为standard）
'''
def Signal_Processing(file_name,start=None, end=None, upper_lim=None,lower_lim=1,sampling_frequency=100,cut_off_frequency=2,
                      drop_head=5,drop_tail=-5,smooth=False, normalization_method='standard',threshold=60):
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
        # 删除所有值为空的行
        new_df_sliced = df_sliced.dropna(axis=0, how="all").copy()
        # 确保 new_df_sliced 不是空的再进行操作
        if not new_df_sliced.empty:
            new_df_sliced.loc[:, new_df_sliced.columns[0]] -= new_df_sliced.iloc[0, 0]
        data_sliced.append(new_df_sliced)

        
    #process the data
    #set parameters
    processed_data=[]
    for series in data_sliced:
        filtered_current=Bessel_Filter(series['i'],sampling_frequency=sampling_frequency,cut_off_frequency=cut_off_frequency)
        filtered_curve=np.array([series['t'],filtered_current])
        if len(filtered_curve[0])>3000:
            down_sampled_curve=Down_sampling(filtered_curve, factor=3)
        else:
            down_sampled_curve=filtered_curve
        filtered_curve_dataframe=pd.DataFrame(down_sampled_curve.T,columns=['t','i'])
        if normalization_method=='standard':
            normalized_data=normalized_translocation(filtered_curve_dataframe,threshold=min(filtered_curve_dataframe.iloc[:,1])+threshold,drop_head=drop_head, drop_tail=drop_tail)
            processed_data.append(normalized_data)
        elif normalization_method=='delta':
            normalized_data=normalized_delta(filtered_curve_dataframe,threshold=min(filtered_curve_dataframe.iloc[:,1])+threshold,drop_head=drop_head, drop_tail=drop_tail)
            processed_data.append(normalized_data)

    #select data for further alignment
    if upper_lim is None:
        target_data=processed_data
   
    else:
        target_data=[series for series in processed_data if len(series)<upper_lim and len(series)>lower_lim]
        selected_index=[]
        for index, series in enumerate(processed_data):
            if len(series) < upper_lim and len(series) > lower_lim:
                selected_index.append(index)
        selected_signal_index=[signal_index[i] for i in selected_index]
        
                
    
    #get the length of each selected signal and 
    len_target_data=[len(series) for series in target_data]
    #max_len=max(len_target_data)
    #max_index=[len(series) for series in target_data].index(max_len) 

    #using time_axis normalization method to normallize the time 
    if smooth==True:
        for series in target_data:
            series['i']=gaussian_filter1d(series['i'].values,sigma=3)

    
    time_normalized_data=[normalized_time_axis(series) for series in target_data][start:end]

    
    return time_normalized_data,selected_signal_index


if __name__=='__main__':
    current_dir=os.getcwd()
    dir_120='data_120'
    dir_final='data_final'
    dir_compare='data_compare'
    file_name='direction_2.xlsx'
    
    time_normalized_data,selected_signal_index=Signal_Processing(os.path.join(current_dir,dir_final,file_name),upper_lim=1000,lower_lim=200,
                                           cut_off_frequency=5,normalization_method='standard',smooth=False,threshold=60)
    print(selected_signal_index)
    resampler=TimeSeriesResampler(sz=2000)
    for series in time_normalized_data[:]:
        current=resampler.fit_transform(series.iloc[:,1].to_numpy().T).ravel()
        time=np.linspace(0,1,current.shape[0])
        plt.plot(time,current)  
    
    plt.show()
    
    
    