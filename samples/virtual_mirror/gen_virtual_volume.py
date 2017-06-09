#coding=utf-8
from __future__ import division
import pandas as pd
import numpy as np
import pandas as pd 
from pandas import DataFrame
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

# def time_mirror(series , mirror_axis_time):
# 	# 对称轴对应的秒数
# 	mirror_axis = 3600 * mirror_axis_time[0] + 60 * mirror_axis_time[1] + mirror_axis_time[2]
# 	# 原始时间对应的秒数
# 	tot_sec = 3600 * series.dt.hour + 60 * series.dt.minute + series.dt.second
# 	# 对称前后时间差
# 	delta_sec = 2 * (tot_sec - mirror_axis)
# 	# 对称后时间
# 	mirrir_series = series - pd.to_timedelta(delta_sec,unit='s')

# 	return mirrir_series


# volume = pd.read_csv('../../dataset/dataSets/training/volume(table 6)_training.csv')
# volume['time'] = pd.to_datetime(volume['time'])
# volume_mirror = pd.DataFrame()

# # 21~22的镜像 模拟6~7
# t1 = pd.datetime(2014,1,1,21).time()  #21:30
# t2 = pd.datetime(2014,1,1,22).time()  #22:30

# volume_proto1 = volume[(volume['time'].dt.time>t1) & (volume['time'].dt.time<=t2)]
# volume_mirror1 = volume_proto1.drop(['time'],axis=1)
# volume_mirror1['time'] = time_mirror(volume_proto1['time'] , (14,0,0) ).values
# volume_mirror = pd.concat([volume_mirror , volume_mirror1],axis=0,ignore_index=True)

# # # 18~19的镜像 模拟7~8
# t1 = pd.datetime(2014,1,1,18).time()  #11:00
# t2 = pd.datetime(2014,1,1,19).time()  #12:00

# volume_proto1 = volume[(volume['time'].dt.time>t1) & (volume['time'].dt.time<=t2)]
# volume_mirror1 = volume_proto1.drop(['time'],axis=1)
# volume_mirror1['time'] = time_mirror(volume_proto1['time'] , (13,0,0) ).values
# volume_mirror = pd.concat([volume_mirror , volume_mirror1],axis=0,ignore_index=True)

# # # 16~17的镜像 模拟8~9
# t1 = pd.datetime(2014,1,1,16).time()  #16:00
# t2 = pd.datetime(2014,1,1,17).time()  #17:00

# volume_proto1 = volume[(volume['time'].dt.time>t1) & (volume['time'].dt.time<=t2)]
# volume_mirror1 = volume_proto1.drop(['time'],axis=1)
# volume_mirror1['time'] = time_mirror(volume_proto1['time'] , (12,30,0) ).values
# volume_mirror = pd.concat([volume_mirror , volume_mirror1],axis=0,ignore_index=True)
# # # 13~14的镜像 模拟9~10
# t1 = pd.datetime(2014,1,1,9).time()  #9:00
# t2 = pd.datetime(2014,1,1,10).time()  #10:00

# volume_proto1 = volume[(volume['time'].dt.time>t1) & (volume['time'].dt.time<=t2)]
# volume_mirror1 = volume_proto1.drop(['time'],axis=1)
# volume_mirror1['time'] = time_mirror(volume_proto1['time'] , (9,30,0) ).values
# volume_mirror = pd.concat([volume_mirror , volume_mirror1],axis=0,ignore_index=True)
# print volume_mirror.info()

# volume_mirror['hour'] = volume_mirror['time'].dt.hour
# avgtime = volume_mirror[['has_etc','hour']].groupby(['hour']).count()
# avgtime.plot(kind='bar')
# plt.show()

# volume_mirror.to_csv('../../dataset/dataSets/training/mirrorvolume_[6_8).csv',index=False)


###################################

#################################
# 添加噪声

volume = pd.read_csv('../../dataset/dataSets/training/volume(table 6)_training.csv')
volume['time'] = pd.to_datetime(volume['time'])

print volume.info()

#-------travel time 噪声
np.random.seed(17)  #17
noise = 90*np.random.randn(volume['time'].shape[0])  #90
# noise = np.random.uniform(0,300,volume['time'].shape[0])  #90
noise = pd.to_timedelta(noise,unit='s')

volume['time'] = volume['time'] + noise
volume.to_csv('../../dataset/dataSets/training/noise_time_volume2.csv',index=False)
