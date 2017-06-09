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


# traj = pd.read_csv('../../dataset/dataSets/training/trajectories(table 5)_training.csv')
# traj['starting_time'] = pd.to_datetime(traj['starting_time'])
# traj_mirror = pd.DataFrame()

# # 12~13的镜像 模拟6~7
# t1 = pd.datetime(2014,1,1,12).time()  #21:30
# t2 = pd.datetime(2014,1,1,13).time()  #22:30

# traj_proto1 = traj[(traj['starting_time'].dt.time>t1) & (traj['starting_time'].dt.time<=t2)]
# traj_mirror1 = traj_proto1.drop(['starting_time'],axis=1)
# traj_mirror1['starting_time'] = time_mirror(traj_proto1['starting_time'] , (9,30,0) ).values
# traj_mirror = pd.concat([traj_mirror , traj_mirror1],axis=0,ignore_index=True)

# # # 11~12的镜像 模拟7~8
# t1 = pd.datetime(2014,1,1,11).time()  #11:00
# t2 = pd.datetime(2014,1,1,12).time()  #12:00

# traj_proto1 = traj[(traj['starting_time'].dt.time>t1) & (traj['starting_time'].dt.time<=t2)]
# traj_mirror1 = traj_proto1.drop(['starting_time'],axis=1)
# traj_mirror1['starting_time'] = time_mirror(traj_proto1['starting_time'] , (9,30,0) ).values
# traj_mirror = pd.concat([traj_mirror , traj_mirror1],axis=0,ignore_index=True)

# # # 10~11的镜像 模拟8~9
# t1 = pd.datetime(2014,1,1,10).time()  #16:00
# t2 = pd.datetime(2014,1,1,11).time()  #17:00

# traj_proto1 = traj[(traj['starting_time'].dt.time>t1) & (traj['starting_time'].dt.time<=t2)]
# traj_mirror1 = traj_proto1.drop(['starting_time'],axis=1)
# traj_mirror1['starting_time'] = time_mirror(traj_proto1['starting_time'] , (9,30,0) ).values
# traj_mirror = pd.concat([traj_mirror , traj_mirror1],axis=0,ignore_index=True)
# # # 9~10的镜像 模拟9~10
# t1 = pd.datetime(2014,1,1,9).time()  #9:00
# t2 = pd.datetime(2014,1,1,10).time()  #10:00

# traj_proto1 = traj[(traj['starting_time'].dt.time>t1) & (traj['starting_time'].dt.time<=t2)]
# traj_mirror1 = traj_proto1.drop(['starting_time'],axis=1)
# traj_mirror1['starting_time'] = time_mirror(traj_proto1['starting_time'] , (9,30,0) ).values
# traj_mirror = pd.concat([traj_mirror , traj_mirror1],axis=0,ignore_index=True)
# print traj_mirror.info()

# # traj_mirror['hour'] = traj_mirror['starting_time'].dt.hour
# # avgtime = traj_mirror[['travel_time','hour']].groupby(['hour']).mean()
# # avgtime.plot(kind='bar')
# # plt.show()

# traj_mirror.to_csv('../../dataset/dataSets/training/mirrortraj_[6_8).csv',index=False)


###################################


#################################
# 添加噪声

traj = pd.read_csv('../../dataset/dataSets/training/trajectories(table 5)_training.csv')
traj['starting_time'] = pd.to_datetime(traj['starting_time'])

print traj.info()

# #-------travel time 噪声
# np.random.seed(17)
# traj['travel_time'] = traj['travel_time'] + 3*np.random.randn(traj['travel_time'].shape[0])
# traj.to_csv('../../dataset/dataSets/training/noise_travtime_traj.csv',index=False)

np.random.seed(17)  #17
noise = 90*np.random.randn(traj['starting_time'].shape[0])  #90
noise = pd.to_timedelta(noise,unit='s')
traj['starting_time'] = traj['starting_time'] + noise
traj.to_csv('../../dataset/dataSets/training/noise_starttime_traj.csv',index=False)
