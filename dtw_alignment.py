import numpy as np
import pandas as pd
from Signal_Processing_and_Alignment import Signal_Processing
from tslearn.metrics import dtw_path, dtw
from tslearn.preprocessing import TimeSeriesResampler
import matplotlib.pyplot as plt
import seaborn as sns
from Process_Functions import interplotation
from scipy.ndimage import gaussian_filter1d
import os 

'''
time_mormalized_data:Signal_Processing_and_Alignment处理完之后的数据，格式为一个列表中的多个DataFrame,每个DataFrame两列（一列时间一列电流）
resample_length:指定最后平均序列的长度，默认不指定，取最长序列的长度
plot_series:是否画出最后平均之后的图，默认否
plot_alignment:是否画出对齐的图，默认否
show_matrix:是否画出序列直接DTW累计距离的矩阵，默认否
series_standard:用于被对齐的序列，可以指定为barycenter序列，默认为从所有序列中筛选出的离所有序列最近的序列
method：指定使用的DTW对齐方式
constraint：指定是否使用限制范围的对齐方式，默认否。可选值：'sakoe_chiba',限制对齐路径的范围
sakoe_radius:限制范围调整，越小限制范围越大，越接近线性对齐；越大越接近无限制的DTW
target_path:指定将画出的图存在哪里，默认为无，不保存图片而是直接呈现图片
inverse:是否将对齐过程的图给颠倒
'''
def dtw_alignment(time_normalized_data,signal_index=None,resample_length=None,plot_series=False,
                  plot_alignment=False,show_matrix=False, series_standard=None,smooth=False,constraint=None,method='classic',sakoe_radius=50,
                  target_path=None, inverse=False):
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
            
            
        for i in range(len(time_normalized_data)):
            
            series=time_normalized_data[i]
            standard_series=np_series_standard
            path,_=dtw_path(series,series_standard)
            if constraint is  None:
                path,_=dtw_path(series,series_standard)
                
            elif constraint=="sakoe_chiba":
                if sakoe_radius==None:
                    path,_=dtw_path(series,series_standard,global_constraint=constraint, sakoe_chiba_radius=sakoe_radius)
                
                else:
                    path,_=dtw_path(series,series_standard,global_constraint=constraint, sakoe_chiba_radius=0.3*series.shape[0])
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
                        plt.plot(series[0,:], series[1,:],label=str(signal_index[index]))

     
                
                if target_path is not None:
                    plt.savefig(os.path.join(target_path,'radius'+str(sakoe_radius)+'.png'))
            if target_path is not None:
                plt.close('all')
            else:
                plt.legend(fontsize=4)
                plt.show()

        #align_data_after_dtw-align with the longeset data

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
            
        return interplotation_aligned_data,aligned_data_unique,average_current
    


#Soft_DTW barycenter calculation

if __name__=="__main__":

    current_dir=os.getcwd()
    dir_120='data_120'
    dir_final='data_final'
    file_name='direction_2.xlsx'
    
    compare_plot_path='radius_compare_smooth'
    plot_save_path=os.path.join(current_dir,compare_plot_path)
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    target_path=os.path.join(current_dir,compare_plot_path)
    
    time_normalized_data,selected_index=Signal_Processing(os.path.join(current_dir,dir_final,file_name),upper_lim=2000,lower_lim=100,
                                           cut_off_frequency=5,normalization_method='standard',smooth=False,threshold=60)
   
    _,_, _=dtw_alignment(time_normalized_data,signal_index=selected_index,plot_series=True,
                                                plot_alignment=True,show_matrix=True,smooth=False,constraint="sakoe_chiba",
                                                sakoe_radius=None,target_path=None,inverse=False)

    
    
    

    
    

    
        
        

        


            
        
    
        
        



