#coding=utf-8
from __future__ import division
import pandas as pd
import numpy as np
import pandas as pd 
from pandas import DataFrame
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt



#--------------------------------------
# # # # 分不同的路线统计 训练集
from extract_samples_traincv import *
traj = pd.read_csv('../dataset/dataSets/training/trajectories(table 5)_training.csv')
traj['starting_time'] = pd.to_datetime(traj['starting_time'])
traj['ending_time'] = traj['starting_time'] + pd.to_timedelta(traj['travel_time'],unit = 's')

traj = merge_linkwise(traj)

startdate = pd.datetime(2016, 7, 19)
enddate = pd.datetime(2016, 10, 17)
for col in ['A_2','A_3','B_1','B_3','C_1','C_3']:
	iid = col.split('_')[0]
	tid = int(col.split('_')[1])

	samples = get_avgtime_sample(traj,iid,tid,startdate,enddate,col,'train','online_results')
	samples['chokevolume'] = samples['involume']-samples['outvolume']
	samples['sign_choke'] = np.sign(samples['chokevolume'])
	samples['rate_choke'] = samples['chokevolume']/samples['involume'].replace(0,1)
	samples.to_csv('./online_results/avgtime/{0}/train_set.csv'.format(col),index=False)

#线下测试集
startdate = pd.datetime(2016, 10, 11)
enddate = pd.datetime(2016, 10, 17)
for col in ['A_2','A_3','B_1','B_3','C_1','C_3']:
	iid = col.split('_')[0]
	tid = int(col.split('_')[1])

	samples = get_avgtime_sample(traj,iid,tid,startdate,enddate,col,'test','online_results')
	samples['chokevolume'] = samples['involume']-samples['outvolume']
	samples['sign_choke'] = np.sign(samples['chokevolume'])
	samples['rate_choke'] = samples['chokevolume']/samples['involume'].replace(0,1)
	samples.to_csv('./online_results/avgtime/{0}/offlinetest_set.csv'.format(col),index=False)

#线上测试集
traj_test = pd.read_csv('../dataset/dataSets/testing_phase1/trajectories(table 5)_test1.csv')
traj_test['starting_time'] = pd.to_datetime(traj_test['starting_time'])
traj_test['ending_time'] = traj_test['starting_time'] + pd.to_timedelta(traj_test['travel_time'],unit = 's')

traj_test = merge_linkwise(traj_test)

startdate = pd.datetime(2016, 10, 18)
enddate = pd.datetime(2016, 10, 24)
for col in ['A_2','A_3','B_1','B_3','C_1','C_3']:
	iid = col.split('_')[0]
	tid = int(col.split('_')[1])

	samples = get_avgtime_sample(traj_test,iid,tid,startdate,enddate,col,'test','online_results')
	samples['chokevolume'] = samples['involume']-samples['outvolume']
	samples['sign_choke'] = np.sign(samples['chokevolume'])
	samples['rate_choke'] = samples['chokevolume']/samples['involume'].replace(0,1)
	samples.to_csv('./online_results/avgtime/{0}/onlinetest_set.csv'.format(col),index=False)
#########################################################################











#－－－－－－－－－－－－－－ＶＯＬＵＭＥ－－－－－－－－－－－－－－－－－
##########################################################################

###--------------------------------
# 分不同的路线统计 

volumedata = pd.read_csv('../dataset/dataSets/training/volume(table 6)_training.csv')
volumedata['time'] = pd.to_datetime(volumedata['time'])
#训练集
startdate = pd.datetime(2016, 9, 19)
enddate = pd.datetime(2016, 10, 17)

for col in ['1_0','1_1','2_0','3_0','3_1']:
	tid = int(col.split('_')[0])
	direct = int(col.split('_')[1])

	samples = get_volume_sample(volumedata,tid,direct,startdate,enddate,col,'train','online_results')
	print samples.info()
	samples.to_csv('./online_results/volume/{0}/train_set.csv'.format(col),index=False)

# #线下测试集
volumedata = pd.read_csv('../dataset/dataSets/training/volume(table 6)_training.csv')
volumedata['time'] = pd.to_datetime(volumedata['time'])
startdate = pd.datetime(2016, 10, 11)
enddate = pd.datetime(2016, 10, 17)
for col in ['1_0','1_1','2_0','3_0','3_1']:
	tid = int(col.split('_')[0])
	direct = int(col.split('_')[1])

	samples = get_volume_sample(volumedata,tid,direct,startdate,enddate,col,'test','online_results')
	print samples.info()
	samples.to_csv('./online_results/volume/{0}/offlinetest_set.csv'.format(col),index=False)

# 线上测试集
volumedata_test = pd.read_csv('../dataset/dataSets/testing_phase1/volume(table 6)_test1.csv')
volumedata_test['time'] = pd.to_datetime(volumedata_test['time'])

startdate = pd.datetime(2016, 10, 18)
enddate = pd.datetime(2016, 10, 24)
for col in ['1_0','1_1','2_0','3_0','3_1']:
	tid = int(col.split('_')[0])
	direct = int(col.split('_')[1])

	samples = get_volume_sample(volumedata_test,tid,direct,startdate,enddate,col,'test','online_results')
	samples.to_csv('./online_results/volume/{0}/onlinetest_set.csv'.format(col),index=False)

# # 虚拟样本 训练集
volumedata = pd.read_csv('../dataset/dataSets/training/noise_time_volume.csv')
volumedata['time'] = pd.to_datetime(volumedata['time'])

startdate = pd.datetime(2016, 9, 19)
enddate = pd.datetime(2016, 10, 17)
for col in ['1_0','1_1','2_0','3_0','3_1']:
	tid = int(col.split('_')[0])
	direct = int(col.split('_')[1])

	samples = get_volume_sample(volumedata,tid,direct,startdate,enddate,col,'virtual','online_results')
	print samples.info()
	samples.to_csv('./online_results/volume/{0}/virtual_train_set.csv'.format(col),index=False)
