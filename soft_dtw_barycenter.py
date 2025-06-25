from tslearn.metrics import gamma_soft_dtw, soft_dtw
from tslearn.barycenters import softdtw_barycenter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tslearn.preprocessing import TimeSeriesResampler
import time
#target_length制定重新采样后的电流长度
#start, end制定选取序列的范围
#gamma设定soft_dtw的超参数
#inverse制定是否将得到的序列颠倒（用语比较相反方向过孔）
#choose color指定画出重心序列的颜色
#plot指定在这个程序中是否画图（在别的程序中引用这个程序中的函数的输出值的时候不需要画图）
#max_iter指定在得到重心序列中的最大迭代次数（太多计算时间增加，太少得到的不准确）
#tol指定收敛范围,作用类似max_iter
def barycenter_plot(data,target_length=500, start=None,end=None,gamma=None,inverse=False, 
                    choose_color='red',plot=False,max_iter=50,tol=1e-3):
    resampler=TimeSeriesResampler(sz=target_length)
    currents_resampled=[resampler.fit_transform(series).ravel() for series in data[start:end]]
# 调用 softdtw_barycenter 函数计算重心
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
    # 步骤 4: 可视化结果
    # 设置图片清晰度
        
        plt.rcParams['figure.dpi'] = 100

    # 绘制原始时间序列
        #for current in currents_resampled:
        #    plt.plot(time,current, c='k',alpha=0.5,linestyle='-')

    # 绘制计算得到的重心
        plt.plot(time,barycenter, label='SoftDTW Barycenter', color=choose_color, linewidth=2)

    # 添加图例和标题
        plt.legend()
        plt.title('SoftDTW Barycenter Plot')
        plt.xlabel('Time Index')
        plt.ylabel('Normalized Current')
        

# 显示图形
    #plt.show()
    
    #currents_resampled是重新调整到相同长度之后的电流,为一个列表中的多个一维数组
    
    #barycenter为得到的电流的重心序列，为一维数组
    #time_series为barycenter加上与之匹配的时间之后的二维数组
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
    


    
    
    