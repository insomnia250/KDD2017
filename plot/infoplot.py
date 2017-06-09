#coding=utf-8
from __future__ import division
import pandas as pd
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from travel_time import cal_avg_travel_time
from tollgate_volume import cal_tollgate_volume

# traj = pd.read_csv('../dataset/dataSets/training/trajectories(table 5)_training.csv')
# traj['starting_time'] = pd.to_datetime(traj['starting_time'])
# traj['hour'] = traj['starting_time'].dt.hour

# print traj.head()
# print traj['starting_time'].min()
# print traj['starting_time'].max()
# # 每天8-10点平均通过时间
# sumtime_by_hour1 = []
# sumtime_by_hour2 = []
# for i,date in enumerate(pd.date_range('20160719','20161017',freq='D')):
# 	# if i==2:break

# 	starttime = date + pd.to_timedelta('8 h')
# 	endtime = date + pd.to_timedelta('10 h')
# 	avgtime1 = cal_avg_travel_time(starttime , endtime ,traj)
# 	# plt.plot(avgtime1['t1'] , avgtime1['A_2'])

# 	sumtime_by_hour1.append(avgtime1['A_2'].sum())

# 	#-----------------
# 	starttime = date + pd.to_timedelta('6 h')
# 	endtime = date + pd.to_timedelta('8 h')
# 	avgtime2 = cal_avg_travel_time(starttime , endtime ,traj)
# 	# plt.plot(avgtime2['t1'] , avgtime2['A_2'])

# 	sumtime_by_hour2.append(avgtime2['A_2'].sum())

# # plt.figure()
# plt.plot(pd.date_range('20160719','20161017',freq='D') , sumtime_by_hour1,'go-')
# plt.plot(pd.date_range('20160719','20161017',freq='D') , sumtime_by_hour2,'bo-')
# plt.figure()
# plt.plot(pd.date_range('20160719','20161017',freq='D') , np.array(sumtime_by_hour2)-np.array(sumtime_by_hour1),'go--')
# plt.figure()
# plt.plot(pd.date_range('20160719','20161017',freq='D') , np.array(sumtime_by_hour2)/np.array(sumtime_by_hour1),'bo--')
# plt.show()



# ######################################################
# volume = pd.read_csv('../dataset/dataSets/training/volume(table 6)_training.csv')
# volume['time'] = pd.to_datetime(volume['time'])
# volume['hour'] = volume['time'].dt.hour

# print volume.head()
# print volume['time'].min()
# print volume['time'].max()
# # 每天8-10点平均通过时间
# sumvolume_by_hour1 = []
# sumvolume_by_hour2 = []
# for i,date in enumerate(pd.date_range('20160919','20161017',freq='D')):
# 	# if i==1:break

# 	starttime = date + pd.to_timedelta('15 h')
# 	endtime = date + pd.to_timedelta('17 h')
# 	volume1 = cal_tollgate_volume(starttime , endtime ,volume)
	
# 	# plt.plot(volume1['t1'] , volume1['A_2'])

# 	sumvolume_by_hour1.append(volume1['1_0'].sum())

# 	#-----------------
# 	starttime = date + pd.to_timedelta('13 h')
# 	endtime = date + pd.to_timedelta('15 h')
# 	volume2 = cal_tollgate_volume(starttime , endtime ,volume)
# 	# plt.plot(volume2['t1'] , volume2['A_2'])

# 	sumvolume_by_hour2.append(volume2['1_0'].sum())

# # plt.figure()
# plt.plot(pd.date_range('20160919','20161017',freq='D') , sumvolume_by_hour1,'go-')
# plt.plot(pd.date_range('20160919','20161017',freq='D') , sumvolume_by_hour2,'bo-')
# plt.figure()
# plt.plot(pd.date_range('20160919','20161017',freq='D') , np.array(sumvolume_by_hour2)-np.array(sumvolume_by_hour1),'go--')
# plt.figure()
# plt.plot(pd.date_range('20160919','20161017',freq='D') , np.array(sumvolume_by_hour2)/np.array(sumvolume_by_hour1),'bo--')
# plt.show()



# # 记录个数按时间段分布

# traj = pd.read_csv('../dataset/dataSets/training/trajectories(table 5)_training.csv')
# traj['starting_time'] = pd.to_datetime(traj['starting_time'])
# traj['hour'] = traj['starting_time'].dt.hour

# sns.distplot(list(traj['hour']), kde = True, hist_kws={'alpha': 0.7},bins=24)
# plt.show()

# 平均时间按24小时统计
traj = pd.read_csv('../dataset/dataSets/training/noise_travtime_traj.csv')
traj['starting_time'] = pd.to_datetime(traj['starting_time'])
traj['hour'] = traj['starting_time'].dt.hour
avgtime = traj[['travel_time','hour']].groupby(['hour']).mean()
print avgtime
avgtime.plot(kind='bar')
plt.show()



# # 目标值分布
# samples = pd.read_csv('../samples/results/travel_time_winextra.csv')
# samples.fillna(0,inplace=True)
# print samples.info()
# sns.distplot(list(samples['A_2']), kde = False, hist_kws={'alpha': 0.7},bins=100)
# sns.distplot(list(samples['A_3']), kde = False, hist_kws={'alpha': 0.7},bins=100)

# plt.figure()
# sns.distplot(list(samples['B_1']), kde = False, hist_kws={'alpha': 0.7},bins=100)
# sns.distplot(list(samples['B_3']), kde = False, hist_kws={'alpha': 0.7},bins=100)

# plt.figure()
# sns.distplot(list(samples['C_1']), kde = False, hist_kws={'alpha': 0.7},bins=100)
# sns.distplot(list(samples['C_3']), kde = False, hist_kws={'alpha': 0.7},bins=100)

# plt.show()