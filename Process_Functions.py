from scipy.interpolate import interp1d
from scipy import signal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
#Bessel LPF

def Bessel_Filter(data,sampling_frequency=100,order=4,cut_off_frequency=40):
    normalized_cutoff=cut_off_frequency/(0.5*sampling_frequency)
    b,a=signal.bessel(order, normalized_cutoff, btype='low', analog=False)
    filtered_data=signal.filtfilt(b,a,data)
    return filtered_data

#downsampling
# data:2D array(time, current) 
def Down_sampling(data, factor):
    down_sampled_data=signal.decimate(data,factor)
    return down_sampled_data
 
#interplotation data:2D array   
def interplotation(data,target_length):
    inter_func = interp1d(data.iloc[:,0], data.iloc[:,1], kind='cubic')  # 使用三次样条插值
    time_new=np.linspace(data.iloc[0,0],data.iloc[-1,0],target_length)
    current_interpolated = inter_func(time_new)
    return pd.DataFrame(np.array([time_new, current_interpolated]).T)



#data: (n,2) dataframe
'''threshold设置数据大小的最大值，超过threshold的数据会被筛掉

'''
def normalized_translocation(data,threshold=450,drop_head=5,drop_tail=-5):
    translocation_region=data[data.iloc[:,1]<threshold]
    translocation_region=translocation_region.iloc[drop_head:drop_tail,:]
    translocation_region.iloc[:,1]=(translocation_region.iloc[:,1]-translocation_region.iloc[:,1].mean())/translocation_region.iloc[:,1].std()
    return translocation_region


def normalized_delta(data,threshold=450,drop_head=5,drop_tail=-5):
    current_0=data.iloc[0:5,1].mean()
    translocation_region=data[data.iloc[:,1]<threshold]
    translocation_region=translocation_region.iloc[drop_head:drop_tail,:]
    translocation_region.iloc[:,1]=(current_0-translocation_region.iloc[:,1])/current_0
    return translocation_region

#time, current:1D array

#画图
def Draw_Plot(time,current,format="-", start=0, end=None,x_range=60):
    plt.plot(time[start:end],current[start:end])
    plt.xlabel('Time/ms')
    plt.ylabel('Current/pA')
    plt.grid=True
    plt.xlim(0,x_range)
    
#时间轴归一化：Min - Max归一化
def normalized_time_axis(data):
    
    normalized_translocation_region=copy.deepcopy(data)
    normalized_translocation_region.iloc[:,0]=(normalized_translocation_region.iloc[:,0]-min(normalized_translocation_region.iloc[:,0]))/(max(normalized_translocation_region.iloc[:,0])-min(normalized_translocation_region.iloc[:,0]))
    return normalized_translocation_region
    