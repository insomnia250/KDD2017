#coding=utf-8
from __future__ import division
import numpy as np 
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns 

from travel_time import cal_avg_travel_time, fillna_by_fbmean
from tollgate_volume import cal_tollgate_volume

# # #########################################################
# #  avg travel time

# traj = pd.read_csv('../dataset/dataSets/training/trajectories(table 5)_training.csv')
# traj['starting_time'] = pd.to_datetime(traj['starting_time'])
# traj['hour'] = traj['starting_time'].dt.hour
# # ---------- am --------------
# for i,col in enumerate(['A_2','A_3','B_1','B_3','C_1','C_3']):
# 	# 每天8-10点平均通过时间
# 	intersection_id = col.split('_')[0]
# 	tollgate_id = int(col.split('_')[1])

# 	avgtime_by_date = []
# 	avgtime_by_date1 = []
# 	for date in pd.date_range('20160719','20161017',freq='D'):
# 		# if i==2:break

# 		starttime = date + pd.to_timedelta('8 h')
# 		endtime = date + pd.to_timedelta('10 h')
# 		avgtime = cal_avg_travel_time(starttime , endtime ,traj)
# 		avgtime = fillna_by_fbmean(avgtime)
# 		avgtime.fillna(0,inplace=True)

# 		# 分日期，时窗画
# 		plt.figure(col +'avg travel time by (date , window)',figsize=(12,8))
# 		plt.plot(avgtime['t1'], avgtime[col],'o-')
# 		plt.xlabel('datetime') ; plt.ylabel('avg time')

# 		tot_avg = traj[(traj['intersection_id']==intersection_id) & (traj['tollgate_id']==tollgate_id) & \
# 				(traj['starting_time']>=starttime) & (traj['starting_time']<endtime)]['travel_time'].mean()
# 		avgtime_by_date.append(tot_avg)

# 		#-----------------
# 		starttime1 = date + pd.to_timedelta('6 h')
# 		endtime1 = date + pd.to_timedelta('8 h')

# 		tot_avg1 = traj[(traj['intersection_id']==intersection_id) & (traj['tollgate_id']==tollgate_id) & \
# 				(traj['starting_time']>=starttime1) & (traj['starting_time']<endtime1)]['travel_time'].mean()
# 		avgtime_by_date1.append(tot_avg1)

# 	# 分日期画
# 	plt.figure(col+'avg travel time by (date)',figsize=(12,8))
# 	plt.xlabel('datetime') ; plt.ylabel('avg time')
# 	plt.plot( pd.date_range('20160719','20161017',freq='D') , avgtime_by_date,'o-')	
# 	plt.plot( pd.date_range('20160719','20161017',freq='D') , avgtime_by_date1,'o-'); plt.legend(['8-10','6-8'])

# 	plt.figure(col +'avg travel time by (date , window)')
# 	plt.savefig('./by_time/avgtime result/am(8-10) by (date win)/'+col +' avg travel time by (date , window)'+'.png')
# 	plt.figure(col +'avg travel time by (date)')
# 	plt.savefig('./by_time/avgtime result/am(8-10) by (date)/'+col +' avg travel time by (date)'+'.png')

# plt.close('all')

# # # ---------- pm --------------
# for i,col in enumerate(['A_2','A_3','B_1','B_3','C_1','C_3']):
# 	# 每天8-10点平均通过时间
# 	intersection_id = col.split('_')[0]
# 	tollgate_id = int(col.split('_')[1])

# 	avgtime_by_date = []
# 	avgtime_by_date1 = []
# 	for date in pd.date_range('20160719','20161017',freq='D'):
# 		# if i==2:break

# 		starttime = date + pd.to_timedelta('17 h')
# 		endtime = date + pd.to_timedelta('19 h')
# 		avgtime = cal_avg_travel_time(starttime , endtime ,traj)
# 		avgtime = fillna_by_fbmean(avgtime)
# 		avgtime.fillna(0,inplace=True)

# 		# 分日期，时窗画
# 		plt.figure(col +'avg travel time by (date , window)',figsize=(12,8))
# 		plt.plot(avgtime['t1'], avgtime[col],'o-')
# 		plt.xlabel('datetime') ; plt.ylabel('avg time')

# 		tot_avg = traj[(traj['intersection_id']==intersection_id) & (traj['tollgate_id']==tollgate_id) & \
# 				(traj['starting_time']>=starttime) & (traj['starting_time']<endtime)]['travel_time'].mean()
# 		avgtime_by_date.append(tot_avg)

# 		#-----------------
# 		starttime1 = date + pd.to_timedelta('15 h')
# 		endtime1 = date + pd.to_timedelta('17 h')

# 		tot_avg1 = traj[(traj['intersection_id']==intersection_id) & (traj['tollgate_id']==tollgate_id) & \
# 				(traj['starting_time']>=starttime1) & (traj['starting_time']<endtime1)]['travel_time'].mean()
# 		avgtime_by_date1.append(tot_avg1)

# 	# 分日期画
# 	plt.figure(col+'avg travel time by (date)',figsize=(12,8))
# 	plt.xlabel('datetime') ; plt.ylabel('avg time')
# 	plt.plot( pd.date_range('20160719','20161017',freq='D') , avgtime_by_date,'o-')	
# 	plt.plot( pd.date_range('20160719','20161017',freq='D') , avgtime_by_date1,'o-'); plt.legend(['17-19','15-17'])

# 	plt.figure(col +'avg travel time by (date , window)')
# 	plt.savefig('./by_time/avgtime result/pm(17-19) by (date win)/'+col +' avg travel time by (date , window)'+'.png')
# 	plt.figure(col +'avg travel time by (date)')
# 	plt.savefig('./by_time/avgtime result/pm(17-19) by (date)/'+col +' avg travel time by (date)'+'.png')

# plt.close('all')

# # ---------- 24h --------------
# for i,col in enumerate(['A_2','A_3','B_1','B_3','C_1','C_3']):
# 	# 每天平均通过时间
# 	intersection_id = col.split('_')[0]
# 	tollgate_id = int(col.split('_')[1])

# 	avgtime_by_date = []
# 	for date in pd.date_range('20160719','20161017',freq='D'):
# 		# if i==2:break
# 		tot_avg = traj[(traj['intersection_id']==intersection_id) & (traj['tollgate_id']==tollgate_id) &\
# 				(pd.to_datetime(traj['starting_time'].dt.date)== date)]['travel_time'].mean()
# 		avgtime_by_date.append(tot_avg)

# 	# 分日期画
# 	plt.figure(col+'avg travel time by (date)',figsize=(12,8))
# 	plt.plot( pd.date_range('20160719','20161017',freq='D') , avgtime_by_date,'o-'); plt.legend(['24h avg'])
# 	plt.xlabel('datetime') ; plt.ylabel('avg time')

# 	plt.figure(col +'avg travel time by (date)')
# 	plt.savefig('./by_time/avgtime result/24h by (date)/'+col +' avg travel time by (date)'+'.png')

# plt.close('all')

# plt.show()
###############################################################




















# #########################################################
#  tollgate volume
volume = pd.read_csv('../dataset/dataSets/training/volume(table 6)_training.csv')
volume['time'] = pd.to_datetime(volume['time'])
volume['hour'] = volume['time'].dt.hour

print volume.head()
# ---------- am --------------
for i,col in enumerate(['1_0','1_1','2_0','3_0','3_1']):
	if i==1:break
	# 每天8-10点车流量
	tollgate_id = int(col.split('_')[0])
	direction = int(col.split('_')[1])

	volumecnt_by_date = []
	volumecnt_by_date1 = []
	for date in pd.date_range('20160919','20161017',freq='D'):
		# if i==2:break

		starttime = date + pd.to_timedelta('8 h')
		endtime = date + pd.to_timedelta('10 h')
		volumecnt = cal_tollgate_volume(starttime , endtime ,volume)

		volumecnt.fillna(0,inplace=True)

		# 分日期，时窗画
		plt.figure(col +'tollgate volume by (date , window)',figsize=(12,8))
		plt.plot(volumecnt['t1'], volumecnt[col],'o-')
		plt.xlabel('datetime') ; plt.ylabel('volume')

		tot_vol = volume[(volume['tollgate_id']==tollgate_id) & (volume['direction']==direction) & \
				(volume['time']>=starttime) & (volume['time']<endtime)]['time'].count()
		volumecnt_by_date.append(tot_vol)

		#-----------------
		starttime1 = date + pd.to_timedelta('6 h')
		endtime1 = date + pd.to_timedelta('8 h')

		tot_vol1 = volume[(volume['tollgate_id']==tollgate_id) & (volume['direction']==direction) & \
				(volume['time']>=starttime1) & (volume['time']<endtime1)]['time'].count()
		volumecnt_by_date1.append(tot_vol1)



	# 分日期画
	plt.figure(col+'tollgate volume by (date)',figsize=(12,8))
	plt.xlabel('datetime') ; plt.ylabel('volume')
	plt.plot( pd.date_range('20160919','20161017',freq='D') , volumecnt_by_date,'o-') 
	plt.plot( pd.date_range('20160919','20161017',freq='D') , volumecnt_by_date1,'o-'); plt.legend(['8-10','6-8'])

	# plt.figure(col +'tollgate volume by (date , window)',figsize=(12,8))
	# plt.savefig('./by_time/volumecnt result/am(8-10) by (date win)/'+col +' tollgate volume by (date , window)'+'.png')
	# plt.figure(col +'tollgate volume by (date)',figsize=(12,8))
	# plt.savefig('./by_time/volumecnt result/am(8-10) by (date)/'+col +' tollgate volume by (date)'+'.png')
plt.show()
# plt.close('all')

# # # ---------- pm --------------
# for i,col in enumerate(['1_0','1_1','2_0','3_0','3_1']):
# 	# 每天8-10点车流量
# 	tollgate_id = int(col.split('_')[0])
# 	direction = int(col.split('_')[1])

# 	volumecnt_by_date = []
# 	volumecnt_by_date1 = []
# 	for date in pd.date_range('20160919','20161017',freq='D'):
# 		# if i==2:break

# 		starttime = date + pd.to_timedelta('17 h')
# 		endtime = date + pd.to_timedelta('19 h')
# 		volumecnt = cal_tollgate_volume(starttime , endtime ,volume)

# 		volumecnt.fillna(0,inplace=True)

# 		# 分日期，时窗画
# 		plt.figure(col +'tollgate volume by (date , window)',figsize=(12,8))
# 		plt.plot(volumecnt['t1'], volumecnt[col],'o-')
# 		plt.xlabel('datetime') ; plt.ylabel('volume')

# 		tot_vol = volume[(volume['tollgate_id']==tollgate_id) & (volume['direction']==direction) & \
# 				(volume['time']>=starttime) & (volume['time']<endtime)]['time'].count()
# 		volumecnt_by_date.append(tot_vol)

# 		#-----------------
# 		starttime1 = date + pd.to_timedelta('15 h')
# 		endtime1 = date + pd.to_timedelta('17 h')

# 		tot_vol1 = volume[(volume['tollgate_id']==tollgate_id) & (volume['direction']==direction) & \
# 				(volume['time']>=starttime1) & (volume['time']<endtime1)]['time'].count()
# 		volumecnt_by_date1.append(tot_vol1)



# 	# 分日期画
# 	plt.figure(col+'tollgate volume by (date)',figsize=(12,8))
# 	plt.xlabel('datetime') ; plt.ylabel('volume')
# 	plt.plot( pd.date_range('20160919','20161017',freq='D') , volumecnt_by_date,'o-') 
# 	plt.plot( pd.date_range('20160919','20161017',freq='D') , volumecnt_by_date1,'o-'); plt.legend(['17-19','15-17'])

# 	plt.figure(col +'tollgate volume by (date , window)',figsize=(12,8))
# 	plt.savefig('./by_time/volumecnt result/pm(17-19) by (date win)/'+col +' tollgate volume by (date , window)'+'.png')
# 	plt.figure(col +'tollgate volume by (date)',figsize=(12,8))
# 	plt.savefig('./by_time/volumecnt result/pm(17-19) by (date)/'+col +' tollgate volume by (date)'+'.png')

# plt.close('all')

# # ---------- 24h --------------
# for i,col in enumerate(['1_0','1_1','2_0','3_0','3_1']):
# 	# 每天流量
# 	tollgate_id = int(col.split('_')[0])
# 	direction = int(col.split('_')[1])

# 	volumecnt_by_date = []
# 	for date in pd.date_range('20160919','20161017',freq='D'):

# 		# if i==2:break
# 		tot_vol = volume[(volume['tollgate_id']==tollgate_id) & (volume['direction']==direction) & \
# 				(pd.to_datetime(volume['time'].dt.date)== date)]['time'].count()
# 		volumecnt_by_date.append(tot_vol)
# 	# 分日期画
# 	plt.figure(col+'tollgate volume by (date)',figsize=(12,8))
# 	plt.plot( pd.date_range('20160919','20161017',freq='D') , volumecnt_by_date,'o-');plt.legend(['24h tol'])
# 	plt.xlabel('datetime') ; plt.ylabel('volume')

# 	plt.figure(col +'tollgate volume by (date)',figsize=(12,8))
# 	plt.savefig('./by_time/volumecnt result/24h by (date)/'+col +' tollgate volume by (date)'+'.png')

# plt.close('all')

