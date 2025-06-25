import pandas as pd
import numpy as np
from Process_Functions import normalized_time_axis
# 清理数据，删除所有值均为空的列
file_name='direction_1.xlsx'
original_data=pd.read_csv(file_name)
data_cleared = original_data.dropna(axis=1, how='all')
# 切片不同的序列
data_sliced = []
column_num = len(data_cleared.columns)

for i in range(0, column_num, 2):
    df_sliced = data_cleared.iloc[:, i:i+2].copy() 
    df_sliced.columns = ['t' + str(i // 2 + 1), 'i' + str(i // 2 + 1)]
    
    # 删除所有值为空的行
    new_df_sliced = df_sliced.dropna(axis=0, how="all").copy()
    
    # 确保 new_df_sliced 不是空的再进行操作
    if not new_df_sliced.empty:
        new_df_sliced.loc[:, new_df_sliced.columns[0]] -= new_df_sliced.iloc[0, 0]
    
    
    
def normalized_translocation(data,threshold=450,drop_head=5,drop_tail=-5):
    translocation_region=data[data.iloc[:,1]<threshold]
    translocation_region=translocation_region.iloc[drop_head:drop_tail,:]
    translocation_region.iloc[:,0]=translocation_region.iloc[:,0]-translocation_region.iloc[0,0]
    return translocation_region



translocation_data=[]
for series in data_sliced:
    translocation_series=normalized_translocation(series,threshold=min(series.iloc[:,1])+60,drop_head=5)
    translocation_data.append(translocation_series)

data_processed=[]
for i in range(len(translocation_data)):
    series_index=pd.DataFrame([], columns=[str(i+1)])
    data_processed.append(series_index)
    data_processed.append(normalized_time_axis(translocation_data[i]))
    
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
combined_df.to_excel('data_120_try.xlsx', index=False)
