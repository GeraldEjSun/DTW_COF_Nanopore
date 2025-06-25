import pandas as pd
import os
from dtw_alignment import dtw_alignment
from Signal_Processing_and_Alignment import Signal_Processing

'''
输出文件到excel
存储路径和文件名
'''

def export_data(data,save_path='',file_name='1'):
    data_processed=[]
    for i in range(len(data)):
        series_index=pd.DataFrame([], columns=[str(i+1)])
        data_processed.append(series_index)
        if data[0].shape[0]==1:
            data_processed.append(pd.DataFrame(data[0].T,columns='i'))
        elif data[0].shape[0]==2:
            data_processed.append(pd.DataFrame(data[0].T,columns=['t','i']))
            
            
    
#merge all the data 
    max_length = max([df.shape[0] for df in data_processed])

    # 填充每个 DataFrame 至最大行数
    for i, df in enumerate(data_processed):
        if df.shape[0] < max_length:
            num_missing = max_length - df.shape[0]
            missing_df = pd.DataFrame(index=range(num_missing), columns=df.columns)
            data_processed[i] = pd.concat([df, missing_df], ignore_index=True)

    # 水平拼接所有 DataFrame
    combined_df = pd.concat(data_processed, axis=1)

    # 导出为 Excel 文件
    cur_dir=os.getcwd()
    abs_save_path=os.path.join(cur_dir,save_path)
    if not os.path.exists(abs_save_path):
        os.makedirs(abs_save_path)
    combined_df.to_excel(os.path.join(abs_save_path,file_name+'.xlsx'), index=False)
    
    


if __name__=='__main__':
    current_dir=os.getcwd()
    dir_120='data_120'
    dir_compare='data_compare'
    file_name='1.xlsx'
    
    time_normalized_data=Signal_Processing(os.path.join(current_dir,dir_120,file_name),start=None,end=None,
                                           upper_lim=None,lower_lim=1,cut_off_frequency=2,normalization_method='standard',smooth=True)
    save_dir='data export'
    data_type='dtw_alignment'
    
    
    _, series,_=dtw_alignment(time_normalized_data,plot_series=False,
                                                plot_alignment=False,show_matrix=False,smooth=True,constraint="sakoe_chiba",
                                                sakoe_radius=100,target_path=None,inverse=True)
        #export_data(series,file_name='dtw_alignment'+'radius'+str(radius),save_path=os.path.join(save_dir+data_type))
        
    
