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

traj = pd.read_csv('../dataset/dataSets/training/trajectories(table 5)_training.csv')
traj['starting_time'] = pd.to_datetime(traj['starting_time'])
traj['hour'] = traj['starting_time'].dt.hour
# # ---------- am --------------
# record = np.zeros((6, 12*7 , 12))

# for weeknum,Mon in enumerate(pd.date_range('20160725','20161016',freq='7D')):
# 	Sun = Mon + pd.to_timedelta('6 day')

# 	for j,date in enumerate(pd.date_range(Mon,Sun,freq='D')):
# 		starttime = date + pd.to_timedelta('8 h')
# 		endtime = date + pd.to_timedelta('10 h')
# 		avgtime = cal_avg_travel_time(starttime , endtime ,traj)
# 		avgtime = fillna_by_fbmean(avgtime)
# 		avgtime.fillna(0,inplace=True)
# 		#-------------------
# 		starttime1 = date + pd.to_timedelta('6 h')
# 		endtime1 = date + pd.to_timedelta('8 h')
# 		avgtime1 = cal_avg_travel_time(starttime1 , endtime1 ,traj)
# 		avgtime1 = fillna_by_fbmean(avgtime1)
# 		avgtime1.fillna(0,inplace=True)
# 		#--------------------

# 		for k,col in enumerate(['A_2','A_3','B_1','B_3','C_1','C_3']):
# 			record[k,weeknum*7+j,0:6] = avgtime1[col]
# 			record[k,weeknum*7+j,6:12] = avgtime[col]

# 			# print record[k]

# 			nameinfo = col+'am'+'_week'+str(weeknum+1)  # 1_1am_week1
# 			plt.figure(nameinfo,figsize=(12,8))
# 			color = sns.color_palette('hls', 7)[j]

# 			plt.plot(avgtime['t1'].dt.time, avgtime[col],'o-',color = color,label=date.date())
# 			plt.plot(avgtime1['t1'].dt.time, avgtime1[col],'o-',color = color,label=date.date())

# 			plt.legend(loc=2)
# 			plt.savefig('./win plot by week/avg time/am/{0}/{1}.png'.format(col,nameinfo))
# 	# # 每周画一个
# 	# for k,col in enumerate(['A_2','A_3','B_1','B_3','C_1','C_3']):
# 	# 	nameinfo = col+'am'+'_week'+str(weeknum+1)+'_total'
# 	# 	plt.figure(nameinfo,figsize=(12,8))
# 	# 	plt.plot(avgtime1['t1'].dt.time, record[k,weeknum*7+1:weeknum*7+7,0:6].sum(0),'o-')
# 	# 	plt.plot(avgtime['t1'].dt.time, record[k,weeknum*7+1:weeknum*7+7,6:12].sum(0),'o-')
# 	# 	plt.savefig('./win plot by week/avg time/am/{0}/{1}.png'.format(col,nameinfo))

# plt.close()

# ---------- pm --------------
record = np.zeros((6, 12*7 , 12))

for weeknum,Mon in enumerate(pd.date_range('20160725','20161016',freq='7D')):
	Sun = Mon + pd.to_timedelta('6 day')

	for j,date in enumerate(pd.date_range(Mon,Sun,freq='D')):
		starttime = date + pd.to_timedelta('17 h')
		endtime = date + pd.to_timedelta('19 h')
		avgtime = cal_avg_travel_time(starttime , endtime ,traj)
		avgtime = fillna_by_fbmean(avgtime)
		avgtime.fillna(0,inplace=True)
		#-------------------
		starttime1 = date + pd.to_timedelta('15 h')
		endtime1 = date + pd.to_timedelta('17 h')
		avgtime1 = cal_avg_travel_time(starttime1 , endtime1 ,traj)
		avgtime1 = fillna_by_fbmean(avgtime1)
		avgtime1.fillna(0,inplace=True)
		#--------------------

		for k,col in enumerate(['A_2','A_3','B_1','B_3','C_1','C_3']):
			record[k,weeknum*7+j,0:6] = avgtime1[col]
			record[k,weeknum*7+j,6:12] = avgtime[col]

			# print record[k]

			nameinfo = col+'pm'+'_week'+str(weeknum+1)  # 1_1pm_week1
			plt.figure(nameinfo,figsize=(12,8))
			color = sns.color_palette('hls', 7)[j]

			plt.plot(avgtime['t1'].dt.time, avgtime[col],'o-',color = color,label=date.date())
			plt.plot(avgtime1['t1'].dt.time, avgtime1[col],'o-',color = color,label=date.date())

			plt.legend(loc=2)
			plt.savefig('./win plot by week/avg time/pm/{0}/{1}.png'.format(col,nameinfo))
	# # 每周画一个
	# for k,col in enumerate(['A_2','A_3','B_1','B_3','C_1','C_3']):
	# 	nameinfo = col+'am'+'_week'+str(weeknum+1)+'_total'
	# 	plt.figure(nameinfo,figsize=(12,8))
	# 	plt.plot(avgtime1['t1'].dt.time, record[k,weeknum*7+1:weeknum*7+7,0:6].sum(0),'o-')
	# 	plt.plot(avgtime['t1'].dt.time, record[k,weeknum*7+1:weeknum*7+7,6:12].sum(0),'o-')
	# 	plt.savefig('./win plot by week/avg time/am/{0}/{1}.png'.format(col,nameinfo))

plt.close()

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




















# # #########################################################
# #  tollgate volume
# volume = pd.read_csv('../dataset/dataSets/training/volume(table 6)_training.csv')
# volume['time'] = pd.to_datetime(volume['time'])
# volume['hour'] = volume['time'].dt.hour

# print volume.head()
# # ---------- pm --------------
# record = np.zeros((5, 4*7 , 12))

# for weeknum,Mon in enumerate(pd.date_range('20160919','20161016',freq='7D')):
# 	# if weeknum==1:break
# 	Sun = Mon + pd.to_timedelta('6 day')

# 	for j,date in enumerate(pd.date_range(Mon,Sun,freq='D')):
# 		starttime = date + pd.to_timedelta('8 h')
# 		endtime = date + pd.to_timedelta('10 h')
# 		volumecnt = cal_tollgate_volume(starttime , endtime ,volume)
# 		volumecnt.fillna(0,inplace=True)
# 		#-------------------
# 		starttime1 = date + pd.to_timedelta('6 h')
# 		endtime1 = date + pd.to_timedelta('8 h')
# 		volumecnt1 = cal_tollgate_volume(starttime1 , endtime1 ,volume)
# 		volumecnt1.fillna(0,inplace=True)
# 		#--------------------

# 		for k,col in enumerate(['1_0','1_1','2_0','3_0','3_1']):
# 			record[k,weeknum*7+j,0:6] = volumecnt1[col]
# 			record[k,weeknum*7+j,6:12] = volumecnt[col]

# 			# print record[k]

# 			nameinfo = col+'am'+'_week'+str(weeknum+1)  # 1_1am_week1
# 			plt.figure(nameinfo,figsize=(12,8))
# 			color = sns.color_palette('hls', 7)[j]

# 			plt.plot(volumecnt['t1'].dt.time, volumecnt[col],'o-',color = color,label=date.date())
# 			plt.plot(volumecnt1['t1'].dt.time, volumecnt1[col],'o-',color = color,label=date.date())

# 			plt.legend(loc=2)
# 			plt.savefig('./win plot by week/volume/am/{0}/{1}.png'.format(col,nameinfo))

# 	# 每周画一个
# 	for k,col in enumerate(['1_0','1_1','2_0','3_0','3_1']):
# 		nameinfo = col+'am'+'_week'+str(weeknum+1)+'_total'
# 		plt.figure(nameinfo,figsize=(12,8))
# 		plt.plot(volumecnt1['t1'].dt.time, record[k,weeknum*7+1:weeknum*7+7,0:6].sum(0),'o-')
# 		plt.plot(volumecnt['t1'].dt.time, record[k,weeknum*7+1:weeknum*7+7,6:12].sum(0),'o-')
# 		plt.savefig('./win plot by week/volume/am/{0}/{1}.png'.format(col,nameinfo))

# plt.close()



# # ---------- pm --------------
# record = np.zeros((5, 4*7 , 12))

# for weeknum,Mon in enumerate(pd.date_range('20160919','20161016',freq='7D')):
# 	# if weeknum==1:break
# 	Sun = Mon + pd.to_timedelta('6 day')

# 	for j,date in enumerate(pd.date_range(Mon,Sun,freq='D')):
# 		starttime = date + pd.to_timedelta('17 h')
# 		endtime = date + pd.to_timedelta('19 h')
# 		volumecnt = cal_tollgate_volume(starttime , endtime ,volume)
# 		volumecnt.fillna(0,inplace=True)
# 		#-------------------
# 		starttime1 = date + pd.to_timedelta('15 h')
# 		endtime1 = date + pd.to_timedelta('17 h')
# 		volumecnt1 = cal_tollgate_volume(starttime1 , endtime1 ,volume)
# 		volumecnt1.fillna(0,inplace=True)
# 		#--------------------

# 		for k,col in enumerate(['1_0','1_1','2_0','3_0','3_1']):
# 			record[k,weeknum*7+j,0:6] = volumecnt1[col]
# 			record[k,weeknum*7+j,6:12] = volumecnt[col]

# 			# print record[k]

# 			nameinfo = col+'pm'+'_week'+str(weeknum+1)  # 1_1pm_week1
# 			plt.figure(nameinfo,figsize=(12,8))
# 			color = sns.color_palette('hls', 7)[j]

# 			plt.plot(volumecnt['t1'].dt.time, volumecnt[col],'o-',color = color,label=date.date())
# 			plt.plot(volumecnt1['t1'].dt.time, volumecnt1[col],'o-',color = color,label=date.date())

# 			plt.legend(loc=1)
# 			plt.savefig('./win plot by week/volume/pm/{0}/{1}.png'.format(col,nameinfo))

# 	# 每周画一个
# 	for k,col in enumerate(['1_0','1_1','2_0','3_0','3_1']):
# 		nameinfo = col+'pm'+'_week'+str(weeknum+1)+'_total'
# 		plt.figure(nameinfo,figsize=(12,8))
# 		plt.plot(volumecnt1['t1'].dt.time, record[k,weeknum*7+1:weeknum*7+7,0:6].sum(0),'o-')
# 		plt.plot(volumecnt['t1'].dt.time, record[k,weeknum*7+1:weeknum*7+7,6:12].sum(0),'o-')
# 		plt.savefig('./win plot by week/volume/pm/{0}/{1}.png'.format(col,nameinfo))

# plt.close()



