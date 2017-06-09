#coding=utf-8
from __future__ import division
import pandas as pd
import numpy as np
import pandas as pd 
from pandas import DataFrame
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt


# 从T1-T2滑窗 提窗统计量
def cal_win_avgtime(traj,iid,tid,T1,T2,win_dt='15min'):
	traj = traj[(traj['intersection_id']==iid) & (traj['tollgate_id']==tid)]
	min_traveltime = traj['travel_time'].quantile(0.5)
	tgavgtime = [[] for x in range(6)] ; tgwinnum = [[] for x in range(6)]
	ftavgtime = [[] for x in range(6)] ; ftwinnum = [[] for x in range(6)]
	winstart = [[] for x in range(6)] ;

	#添加特征统计
	involume = [];outvolume = [];
	involume_win = [[] for x in range(6)] ;
	lastcar_mean = [];lastcar_mid=[];
	lastlink_time = [];firstlink_time=[]
	llt_over_len = []
	llt_over_width = []
	flt_over_len = []
	flt_over_width = []
	for t1 in pd.date_range(T1,T2,freq=win_dt):
		t2 = t1 + pd.to_timedelta('2 h')
		
		# 该2小时 6 个窗的目标值
		for win_num in range(6):
			win_t1 = t1 + win_num*pd.to_timedelta('20 m')
			win_t2 = win_t1 + pd.to_timedelta('20 m')
			
			tgavgtime[win_num].append(traj[(traj['starting_time']>=win_t1) & \
					(traj['starting_time']<win_t2)]['travel_time'].mean() )
			tgwinnum[win_num].append( (win_t1.hour*60+win_t1.minute) ) #该窗编号，win_t1的分钟数
			winstart[win_num].append(win_t1)

		# 前2小时 ftavgtime 窗统计
		for win_num in range(6):
			win_t1 = t1-pd.to_timedelta('2 h') + win_num*pd.to_timedelta('20 m')
			win_t2 = win_t1 + pd.to_timedelta('20 m')
			
			ftavgtime[win_num].append(traj[(traj['starting_time']>=win_t1) & \
					(traj['starting_time']<win_t2)]['travel_time'].mean() )
			ftwinnum[win_num].append( (win_t1.hour*60+win_t1.minute) ) #该窗编号，win_t1的分钟数

		# 前2小时 involume 窗统计			
			involume_win[win_num].append(traj[(traj['starting_time']>=win_t1) & \
					(traj['starting_time']<win_t2)]['travel_time'].count() )


		# 最后min_traveltime秒 进出车流量
		ftt1 = t1-pd.to_timedelta(min_traveltime,unit = 's')
		ftt2 = t1
		involume.append(traj[(traj['starting_time']>=ftt1) & \
				(traj['starting_time']<ftt2)]['travel_time'].count() )
		outvolume.append(traj[(traj['ending_time']>=ftt1) & \
				(traj['ending_time']<ftt2)]['travel_time'].count() )

		#最后10辆 的平均通过时间
		t0 = t1-pd.to_timedelta('2 h')
		ftdata = traj[(traj['ending_time']>=t0) & (traj['ending_time']<t1)].\
				sort_values(by=['ending_time'],ascending=False)
		ncar = min(ftdata.shape[0],10)
		lastcar_mean.append( ftdata.iloc[0:ncar]['travel_time'].mean() )
		lastcar_mid.append( ftdata.iloc[0:ncar]['travel_time'].median() )
		
		#目标前20min 最后一段link的平均通过时间，route总长*时间/link长
		# lastcar 除以linkwise
		t0 = t1-pd.to_timedelta('20 min')
		ftdata = traj[(traj['ending_time']>=t0) & (traj['ending_time']<t1)]
		lastlink_time.append( ftdata['lastlink_time'].mean() )
		firstlink_time.append( ftdata['firstlink_time'].mean() )
		llt_over_len.append( ftdata['llt_over_len'].mean() )
		llt_over_width.append( ftdata['llt_over_width'].mean() )
		flt_over_len.append( ftdata['flt_over_len'].mean() )
		flt_over_width.append( ftdata['flt_over_width'].mean() )

	result = pd.DataFrame()
	for win_num in range(6):
		result['tgavgtime'+str(win_num)] = tgavgtime[win_num]

	for win_num in range(6):
		result['tgwinnum'+str(win_num)] = tgwinnum[win_num]

	for win_num in range(6):
		result['ftavgtime'+str(win_num)] = ftavgtime[win_num]

	for win_num in range(6):
		result['ftwinnum'+str(win_num)] = ftwinnum[win_num]

	for win_num in range(6):
		result['winstart'+str(win_num)] = winstart[win_num]

	result['involume'] = involume
	result['outvolume'] = outvolume
	result['lastcar_mean'] = lastcar_mean
	result['lastcar_mid'] = lastcar_mid

	result['lastlink_time'] = lastlink_time
	result['firstlink_time'] = firstlink_time
	result['llt_over_len'] = llt_over_len
	result['llt_over_width'] = llt_over_width
	result['flt_over_len'] = flt_over_len
	result['flt_over_width'] = flt_over_width

	for win_num in range(6):
		result['involume_win'+str(win_num)] = involume_win[win_num]

	return result


# 添加default特征
def add_default_tfeature(tgdata , hisdata):

	# 按窗编号merge 样本目标的默认值
	for win_num in range(6):
		temp = hisdata[['tgavgtime'+str(win_num),'tgwinnum'+str(win_num)]].groupby('tgwinnum'+str(win_num)).mean()
		tgdata = pd.merge(tgdata,temp,how='left',left_on='tgwinnum'+str(win_num),right_index=True,suffixes=('','_default'))

	# 按窗编号merge 样本ft统计的默认值
	for win_num in range(6):
		temp = hisdata[['ftavgtime'+str(win_num),'ftwinnum'+str(win_num)]].groupby('ftwinnum'+str(win_num)).mean()
		tgdata = pd.merge(tgdata,temp,how='left',left_on='ftwinnum'+str(win_num),right_index=True,suffixes=('','_default'))
	# 按weekday编号merge
	for win_num in range(6):
		temp = hisdata[['tgavgtime'+str(win_num),'weekday']].groupby(['weekday']).mean()
		tgdata = pd.merge(tgdata,temp,how='left',left_on=['weekday'],right_index=True,suffixes=('','_weekday_default'))
	
	# 按目标窗编号merge lastcar 的default
	temp = hisdata[['lastcar_mean','tgwinnum1']].groupby('tgwinnum1').mean()
	tgdata = pd.merge(tgdata,temp,how='left',left_on=['tgwinnum1'],right_index=True,suffixes=('','_default'))
	temp = hisdata[['lastcar_mid','tgwinnum1']].groupby('tgwinnum1').median()
	tgdata = pd.merge(tgdata,temp,how='left',left_on=['tgwinnum1'],right_index=True,suffixes=('','_default'))
	return tgdata
	

# 展开窗口 即单模型
def expand_avgtime_samples(samples):
	expanded_samples = pd.DataFrame()
	for win_num in range(6):
		temp = pd.DataFrame()
		temp['tgavgtime'] =  samples['tgavgtime'+str(win_num)]
		temp['tgavgtime_default'] =  samples['tgavgtime{0}_default'.format(win_num)]
		temp['tgavgtime_weekday_default'] =  samples['tgavgtime{0}_weekday_default'.format(win_num)]
		temp['tgwinnum'] =  samples['tgwinnum'+str(win_num)]
		temp['winstart'] =  samples['winstart'+str(win_num)]
		temp['win_num'] =  win_num
		temp=pd.concat([temp,samples.ix[:,6:]],axis=1)
		expanded_samples = pd.concat([expanded_samples,temp],axis=0,ignore_index=True)

	for win_num in range(6):
		expanded_samples.drop(['tgwinnum'+str(win_num)],axis=1,inplace=True)
		expanded_samples.drop(['winstart'+str(win_num)],axis=1,inplace=True)
		expanded_samples.drop(['tgavgtime{0}_default'.format(win_num)],axis=1,inplace=True)
		expanded_samples.drop(['tgavgtime{0}_weekday_default'.format(win_num)],axis=1,inplace=True)
	return expanded_samples

# 提样本
def get_avgtime_sample(traj,iid,tid,startdate,enddate,col,settype='train',buff='results'):
	samples = DataFrame()

	for date in pd.date_range(startdate , enddate , freq='D'):
		# if pd.datetime(2016,10,1)<=date<=pd.datetime(2016,10,7):continue
		
		if settype=='train':
			T1 = date + pd.to_timedelta('7h40m')    # 7:40-18:50
			T2 = date + pd.to_timedelta('18h50m')

			day_samples = cal_win_avgtime(traj,iid,tid,T1,T2,'10min')
			day_samples['weekday'] = date.weekday()
			print date,len(day_samples),'#windows'
			samples = pd.concat([samples,day_samples],axis=0,ignore_index=True)
			

		elif settype=='test':
			T1 = date + pd.to_timedelta('8h')
			T2 = date + pd.to_timedelta('8h')
			day_samples = cal_win_avgtime(traj,iid,tid,T1,T2,'10min')
			day_samples['weekday'] = date.weekday()
			samples = pd.concat([samples,day_samples],axis=0,ignore_index=True)

			T1 = date + pd.to_timedelta('17h')
			T2 = date + pd.to_timedelta('17h')
			day_samples = cal_win_avgtime(traj,iid,tid,T1,T2,'10min')
			day_samples['weekday'] = date.weekday()
			print len(day_samples)
			samples = pd.concat([samples,day_samples],axis=0,ignore_index=True)

		elif settype=='virtual':
			T1 = date + pd.to_timedelta('8h')
			T2 = date + pd.to_timedelta('8h')
			day_samples = cal_win_avgtime(traj,iid,tid,T1,T2,'10min')
			day_samples['weekday'] = date.weekday()
			samples = pd.concat([samples,day_samples],axis=0,ignore_index=True)

			T1 = date + pd.to_timedelta('17h')
			T2 = date + pd.to_timedelta('17h')
			day_samples = cal_win_avgtime(traj,iid,tid,T1,T2,'10min')
			day_samples['weekday'] = date.weekday()
			print len(day_samples)
			samples = pd.concat([samples,day_samples],axis=0,ignore_index=True)


	samples.fillna(samples.mean(0),inplace=True)
	#----------------------
	if settype == 'train':
		samples.to_csv('./{0}/avgtime/{1}/trainsamples_raw.csv'.format(buff,col),index=False)
		samples = add_default_tfeature(samples , samples)
	elif settype == 'test':
		trainsamples_raw = pd.read_csv('./{0}/avgtime/{1}/trainsamples_raw.csv'.format(buff,col))
		samples = add_default_tfeature(samples , trainsamples_raw)
	elif settype == 'virtual':
		trainsamples_raw = pd.read_csv('./{0}/avgtime/{1}/trainsamples_raw.csv'.format(buff,col))
		samples = add_default_tfeature(samples , trainsamples_raw)
	#----------------------------------------------
	print samples.info()
	# 多目标值展开，为单目标
	samples = expand_avgtime_samples(samples)

	#---------------------
	# 添加差分特征
	ftwinlen = len(samples.filter(regex='ftwinnum').columns)
	for i in range(ftwinlen-1):
		samples['dif_avgtime'+str(i)] = samples['ftavgtime'+str(i+1)] - samples['ftavgtime'+str(i)]


	# 添加乘法默认值特征
	ftwinlen = len(samples.filter(regex='ftwinnum').columns)
	for i in range(ftwinlen):
		samples['rdefault'+str(i)] = samples['tgavgtime_default']/samples['ftavgtime{0}_default'.format(i)].replace(0,1)
		samples['ftavgtime_pro_rdefault'+str(i)] = samples['rdefault'+str(i)]*samples['ftavgtime'+str(i)]

		samples['ddefault'+str(i)] = samples['tgavgtime_default']-samples['ftavgtime{0}_default'.format(i)]
		samples['ftavgtime_plus_ddefault'+str(i)] = samples['ddefault'+str(i)]+samples['ftavgtime'+str(i)]

	return samples


def merge_linkwise(traj):
	linkwise_ft = pd.read_csv('../features/linkwise/avgtime_linkwise_ft.csv')
	traj = pd.merge(traj, linkwise_ft,how='left',on=['intersection_id','tollgate_id'])

	# 最后一段link的通过时间
	traj['lastlink_time'] = (traj['travel_seq'].str.split(';').str[-1].str.split('#').str[-1]).astype(float)
	traj['firstlink_time'] = (traj['travel_seq'].str.split(';').str[0].str.split('#').str[-1]).astype(float)
	traj['llt_over_len'] = traj['lastlink_time']/traj['lastlink_len']
	traj['llt_over_width'] = traj['lastlink_time']/traj['lastlink_width']
	traj['flt_over_len'] = traj['firstlink_time']/traj['firstlink_len']
	traj['flt_over_width'] = traj['firstlink_time']/traj['firstlink_width']
	traj.drop(['length_tot','linknum','R','lastlink_len','lastlink_width','lastlink_R',
		'firstlink_len','firstlink_width','firstlink_R'],axis=1,inplace=True)
	return traj
#--------------------------------------
# # 分不同的路线统计 训练集
# traj = pd.read_csv('../dataset/dataSets/training/trajectories(table 5)_training.csv')
# traj['starting_time'] = pd.to_datetime(traj['starting_time'])
# traj['ending_time'] = traj['starting_time'] + pd.to_timedelta(traj['travel_time'],unit = 's')

# traj = merge_linkwise(traj)

# startdate = pd.datetime(2016, 7, 19)
# enddate = pd.datetime(2016, 10, 10)
# for col in ['A_2','A_3','B_1','B_3','C_1','C_3']:
# 	iid = col.split('_')[0]
# 	tid = int(col.split('_')[1])

# 	samples = get_avgtime_sample(traj,iid,tid,startdate,enddate,col,'train','results')
# 	samples['chokevolume'] = samples['involume']-samples['outvolume']
# 	samples['sign_choke'] = np.sign(samples['chokevolume'])
# 	samples['rate_choke'] = samples['chokevolume']/samples['involume'].replace(0,1)
# 	samples.to_csv('./results/avgtime/{0}/train_set.csv'.format(col),index=False)

# #线下测试集
# startdate = pd.datetime(2016, 10, 11)
# enddate = pd.datetime(2016, 10, 17)
# for col in ['A_2','A_3','B_1','B_3','C_1','C_3']:
# 	iid = col.split('_')[0]
# 	tid = int(col.split('_')[1])

# 	samples = get_avgtime_sample(traj,iid,tid,startdate,enddate,col,'test','results')
# 	samples['chokevolume'] = samples['involume']-samples['outvolume']
# 	samples['sign_choke'] = np.sign(samples['chokevolume'])
# 	samples['rate_choke'] = samples['chokevolume']/samples['involume'].replace(0,1)
# 	samples.to_csv('./results/avgtime/{0}/offlinetest_set.csv'.format(col),index=False)

# #线上测试集
# traj_test = pd.read_csv('../dataset/dataSets/testing_phase1/trajectories(table 5)_test1.csv')
# traj_test['starting_time'] = pd.to_datetime(traj_test['starting_time'])
# traj_test['ending_time'] = traj_test['starting_time'] + pd.to_timedelta(traj_test['travel_time'],unit = 's')

# traj_test = merge_linkwise(traj_test)

# startdate = pd.datetime(2016, 10, 18)
# enddate = pd.datetime(2016, 10, 24)
# for col in ['A_2','A_3','B_1','B_3','C_1','C_3']:
# 	iid = col.split('_')[0]
# 	tid = int(col.split('_')[1])

# 	samples = get_avgtime_sample(traj_test,iid,tid,startdate,enddate,col,'test','results')
# 	samples['chokevolume'] = samples['involume']-samples['outvolume']
# 	samples['sign_choke'] = np.sign(samples['chokevolume'])
# 	samples['rate_choke'] = samples['chokevolume']/samples['involume'].replace(0,1)
# 	samples.to_csv('./results/avgtime/{0}/onlinetest_set.csv'.format(col),index=False)


# # 虚拟样本 训练集
traj = pd.read_csv('../dataset/dataSets/training/noise_starttime_traj.csv')
traj['starting_time'] = pd.to_datetime(traj['starting_time'])
traj['ending_time'] = traj['starting_time'] + pd.to_timedelta(traj['travel_time'],unit = 's')

traj = merge_linkwise(traj)

startdate = pd.datetime(2016, 7, 19)
enddate = pd.datetime(2016, 10, 10)
for col in ['A_2','A_3','B_1','B_3','C_1','C_3']:
	iid = col.split('_')[0]
	tid = int(col.split('_')[1])

	samples = get_avgtime_sample(traj,iid,tid,startdate,enddate,col,'virtual','results')
	samples['chokevolume'] = samples['involume']-samples['outvolume']
	samples['sign_choke'] = np.sign(samples['chokevolume'])
	samples['rate_choke'] = samples['chokevolume']/samples['involume'].replace(0,1)
	samples.to_csv('./results/avgtime/{0}/virtual_train_set.csv'.format(col),index=False)
##########################################################################











#－－－－－－－－－－－－－－ＶＯＬＵＭＥ－－－－－－－－－－－－－－－－－
##########################################################################

# 从T1-T2滑窗 提窗统计量
def cal_win_volume(volumedata,tid,direct,T1,T2,win_dt='15min'):
	volumedata = volumedata[(volumedata['tollgate_id']==tid) & (volumedata['direction']==direct)]

	tgvolume = [[] for x in range(6)] ; tgwinnum = [[] for x in range(6)]
	ftvolume = [[] for x in range(6)] ; ftwinnum = [[] for x in range(6)]
	winstart = [[] for x in range(6)]
	for t1 in pd.date_range(T1,T2,freq=win_dt):
		t2 = t1 + pd.to_timedelta('2 h')
		
		# 该2小时 6 个窗的目标值
		for win_num in range(6):
			win_t1 = t1 + win_num*pd.to_timedelta('20 m')
			win_t2 = win_t1 + pd.to_timedelta('20 m')
			
			tgvolume[win_num].append(volumedata[(volumedata['time']>=win_t1) & \
					(volumedata['time']<win_t2)]['time'].count() )
			tgwinnum[win_num].append( (win_t1.hour*60+win_t1.minute) ) #该窗编号，win_t1的分钟数
			winstart[win_num].append(win_t1)

		# 前2小时 tgvolume 窗统计
		for win_num in range(6):
			win_t1 = t1-pd.to_timedelta('2 h') + win_num*pd.to_timedelta('20 m')
			win_t2 = win_t1 + pd.to_timedelta('20 m')
			
			ftvolume[win_num].append(volumedata[(volumedata['time']>=win_t1) & \
					(volumedata['time']<win_t2)]['time'].count() )
			ftwinnum[win_num].append( (win_t1.hour*60+win_t1.minute) ) #该窗编号，win_t1的分钟数
		# 
	result = pd.DataFrame()
	for win_num in range(6):
		result['tgvolume'+str(win_num)] = tgvolume[win_num]

	for win_num in range(6):
		result['tgwinnum'+str(win_num)] = tgwinnum[win_num]

	for win_num in range(6):
		result['ftvolume'+str(win_num)] = ftvolume[win_num]

	for win_num in range(6):
		result['ftwinnum'+str(win_num)] = ftwinnum[win_num]

	for win_num in range(6):
		result['winstart'+str(win_num)] = winstart[win_num]	
	return result

# 添加default特征
def add_default_vfeature(tgdata , hisdata):

	# 按窗编号merge 样本目标的默认值
	for win_num in range(6):
		temp = hisdata[['tgvolume'+str(win_num),'tgwinnum'+str(win_num)]].groupby('tgwinnum'+str(win_num)).mean()
		tgdata = pd.merge(tgdata,temp,how='left',left_on='tgwinnum'+str(win_num),right_index=True,suffixes=('','_default'))

	# 按窗编号merge 样本ft统计的默认值
	for win_num in range(6):
		temp = hisdata[['ftvolume'+str(win_num),'ftwinnum'+str(win_num)]].groupby('ftwinnum'+str(win_num)).mean()
		tgdata = pd.merge(tgdata,temp,how='left',left_on='ftwinnum'+str(win_num),right_index=True,suffixes=('','_default'))
	# 按weekday编号merge
	for win_num in range(6):
		temp = hisdata[['tgvolume'+str(win_num),'weekday']].groupby(['weekday']).mean()
		tgdata = pd.merge(tgdata,temp,how='left',left_on=['weekday'],right_index=True,suffixes=('','_weekday_default'))
	return tgdata

# 展开窗口 即单模型
def expand_volume_samples(samples):
	expanded_samples = pd.DataFrame()
	for win_num in range(6):
		temp = pd.DataFrame()
		temp['tgvolume'] =  samples['tgvolume'+str(win_num)]
		temp['tgvolume_default'] =  samples['tgvolume{0}_default'.format(win_num)]
		temp['tgvolume_weekday_default'] =  samples['tgvolume{0}_weekday_default'.format(win_num)]
		temp['tgwinnum'] =  samples['tgwinnum'+str(win_num)]
		temp['winstart'] =  samples['winstart'+str(win_num)]
		temp['win_num'] =  win_num
		temp=pd.concat([temp,samples.ix[:,6:]],axis=1)
		expanded_samples = pd.concat([expanded_samples,temp],axis=0,ignore_index=True)

	for win_num in range(6):
		expanded_samples.drop(['tgwinnum'+str(win_num)],axis=1,inplace=True)
		expanded_samples.drop(['winstart'+str(win_num)],axis=1,inplace=True)
		expanded_samples.drop(['tgvolume{0}_default'.format(win_num)],axis=1,inplace=True)
		expanded_samples.drop(['tgvolume{0}_weekday_default'.format(win_num)],axis=1,inplace=True)
	return expanded_samples

def get_volume_sample(volumedata,tid,direct,startdate,enddate,col,settype='train',buff='resluts'):
	samples = DataFrame()

	for date in pd.date_range(startdate , enddate , freq='D'):
		if pd.datetime(2016,10,1)<=date<=pd.datetime(2016,10,7):continue

		if settype=='train':
			T1 = date + pd.to_timedelta('5h20m')   #7:40
			T2 = date + pd.to_timedelta('21h10m')  #18:50

			day_samples = cal_win_volume(volumedata,tid,direct,T1,T2,'10min')
			day_samples['weekday'] = date.weekday()
			print date,len(day_samples)
			samples = pd.concat([samples,day_samples],axis=0,ignore_index=True)
		elif settype=='test':
			T1 = date + pd.to_timedelta('8h')
			T2 = date + pd.to_timedelta('8h')
			day_samples = cal_win_volume(volumedata,tid,direct,T1,T2,'10min')
			day_samples['weekday'] = date.weekday()
			samples = pd.concat([samples,day_samples],axis=0,ignore_index=True)

			T1 = date + pd.to_timedelta('17h')
			T2 = date + pd.to_timedelta('17h')
			day_samples = cal_win_volume(volumedata,tid,direct,T1,T2,'10min')
			day_samples['weekday'] = date.weekday()
			print len(day_samples)
			samples = pd.concat([samples,day_samples],axis=0,ignore_index=True)
		elif settype=='virtual':
			T1 = date + pd.to_timedelta('8h')
			T2 = date + pd.to_timedelta('8h')
			day_samples = cal_win_volume(volumedata,tid,direct,T1,T2,'10min')
			day_samples['weekday'] = date.weekday()
			samples = pd.concat([samples,day_samples],axis=0,ignore_index=True)

			T1 = date + pd.to_timedelta('17h')
			T2 = date + pd.to_timedelta('17h')
			day_samples = cal_win_volume(volumedata,tid,direct,T1,T2,'10min')
			day_samples['weekday'] = date.weekday()
			print len(day_samples)
			samples = pd.concat([samples,day_samples],axis=0,ignore_index=True)

	samples.fillna(samples.mean(0),inplace=True)


	#----------------------
	if settype == 'train':
		samples.to_csv('./{0}/volume/{1}/trainsamples_raw.csv'.format(buff,col),index=False)
		samples = add_default_vfeature(samples , samples)
	elif settype == 'test':
		trainsamples_raw = pd.read_csv('./{0}/volume/{1}/trainsamples_raw.csv'.format(buff,col))
		samples = add_default_vfeature(samples , trainsamples_raw)
	elif settype == 'virtual':
		trainsamples_raw = pd.read_csv('./{0}/volume/{1}/trainsamples_raw.csv'.format(buff,col))
		samples = add_default_vfeature(samples , trainsamples_raw)
	#----------------------------------------------
	# 多目标值展开，为单目标
	samples = expand_volume_samples(samples)

	#---------------------
	# 添加差分特征
	ftwinlen = len(samples.filter(regex='ftwinnum').columns)
	for i in range(ftwinlen-1):
		samples['dif_volume'+str(i)] = samples['ftvolume'+str(i+1)] - samples['ftvolume'+str(i)]


	# 添加比值默认值特征
	ftwinlen = len(samples.filter(regex='ftwinnum').columns)
	for i in range(ftwinlen):
		samples['rdefault'+str(i)] = samples['tgvolume_default']/samples['ftvolume{0}_default'.format(i)].replace(0,1)
		samples['ftvolume_pro_rdefault'+str(i)] = samples['rdefault'+str(i)]*samples['ftvolume'+str(i)]


		samples['ddefault'+str(i)] = samples['tgvolume_default']-samples['ftvolume{0}_default'.format(i)]
		samples['ftvolume_plus_ddefault'+str(i)] = samples['ddefault'+str(i)]+samples['ftvolume'+str(i)]

	return samples

# #--------------------------------
# # 分不同的路线统计 

# volumedata = pd.read_csv('../dataset/dataSets/training/volume(table 6)_training.csv')
# volumedata['time'] = pd.to_datetime(volumedata['time'])
# #训练集
# startdate = pd.datetime(2016, 9, 19)
# enddate = pd.datetime(2016, 10, 10)

# for col in ['1_0','1_1','2_0','3_0','3_1']:
# 	tid = int(col.split('_')[0])
# 	direct = int(col.split('_')[1])

# 	samples = get_volume_sample(volumedata,tid,direct,startdate,enddate,col,'train','results')
# 	print samples.info()
# 	samples.to_csv('./results/volume/{0}/train_set.csv'.format(col),index=False)

# # #线下测试集
# volumedata = pd.read_csv('../dataset/dataSets/training/volume(table 6)_training.csv')
# volumedata['time'] = pd.to_datetime(volumedata['time'])
# startdate = pd.datetime(2016, 10, 11)
# enddate = pd.datetime(2016, 10, 17)
# for col in ['1_0','1_1','2_0','3_0','3_1']:
# 	tid = int(col.split('_')[0])
# 	direct = int(col.split('_')[1])

# 	samples = get_volume_sample(volumedata,tid,direct,startdate,enddate,col,'test','results')
# 	print samples.info()
# 	samples.to_csv('./results/volume/{0}/offlinetest_set.csv'.format(col),index=False)

# # 线上测试集
# volumedata_test = pd.read_csv('../dataset/dataSets/testing_phase1/volume(table 6)_test1.csv')
# volumedata_test['time'] = pd.to_datetime(volumedata_test['time'])

# startdate = pd.datetime(2016, 10, 18)
# enddate = pd.datetime(2016, 10, 24)
# for col in ['1_0','1_1','2_0','3_0','3_1']:
# 	tid = int(col.split('_')[0])
# 	direct = int(col.split('_')[1])

# 	samples = get_volume_sample(volumedata_test,tid,direct,startdate,enddate,col,'test','results')
# 	samples.to_csv('./results/volume/{0}/onlinetest_set.csv'.format(col),index=False)

# # # 虚拟样本 训练集
# volumedata = pd.read_csv('../dataset/dataSets/training/noise_time_volume.csv')
# volumedata['time'] = pd.to_datetime(volumedata['time'])

# startdate = pd.datetime(2016, 9, 19)
# enddate = pd.datetime(2016, 10, 10)
# for col in ['1_0','1_1','2_0','3_0','3_1']:
# 	tid = int(col.split('_')[0])
# 	direct = int(col.split('_')[1])

# 	samples = get_volume_sample(volumedata,tid,direct,startdate,enddate,col,'virtual','results')
# 	print samples.info()
# 	samples.to_csv('./results/volume/{0}/virtual_train_set.csv'.format(col),index=False)