#coding=utf-8
import numpy as np
import pandas as pd



def cal_avg_travel_time(starttime , endtime ,traj):

#[starttime , endtime)区间内，以20min窗（左闭右开）统计6条路线的 avg travel time
	print 'calulating avg travel time in time range [{0},{1})'.format(starttime,endtime)
	# traj = pd.read_csv('../dataset/dataSets/training/trajectories(table 5)_training.csv')
	traj['starting_time'] = pd.to_datetime(traj['starting_time'])

	result = pd.DataFrame(columns=['A_2','A_3','B_1','B_3','C_1','C_3','t1','t2'])
	win_start_list = pd.date_range(starttime, endtime, freq = '20min',closed = 'left')
	for j,t1 in enumerate(win_start_list):
		t2 = t1 + pd.to_timedelta('20 m')
		
		wdata = traj[(traj['starting_time']>=t1) & (traj['starting_time']<t2)]

		avgtime = wdata[['intersection_id' ,'tollgate_id','travel_time']].groupby(\
			['intersection_id','tollgate_id']).mean()
		avgtime = avgtime.T
		avgtime.columns = [muticol[0]+'_'+ str(muticol[1]) for muticol in avgtime.columns]
		avgtime['t1'] = t1
		avgtime['t2'] = t2

		result = pd.concat([result, avgtime],axis=0,ignore_index=True)
	return result

def fillna_by_fbmean(result):
	# 缺失值填充
	# result = pd.DataFrame({'c1':[np.nan,2,3],'c2':[1,np.nan,np.nan],'c3':[1,np.nan,3]})
	result_ff = result.fillna(method='ffill')
	result_bf = result.fillna(method='bfill')

	for col in ['A_2','A_3','B_1','B_3','C_1','C_3']:
		result_ff.loc[result_ff[col].isnull() , col] = result_bf.loc[result_ff[col].isnull() , col]
		result_bf.loc[result_bf[col].isnull() , col] = result_ff.loc[result_bf[col].isnull() , col]
		result[col] = 0.5*result_ff[col] + 0.5*result_bf[col]
	return result

