#coding=utf-8
import numpy as np
import pandas as pd

def cal_tollgate_volume(starttime , endtime ,volume):

#[starttime , endtime)区间内，以20min窗（左闭右开）统计5个 tollgate-direction pairs 的流量
	
	print 'calulating tollgate volume in time range [{0},{1})'.format(starttime,endtime)
	# volume = pd.read_csv('../dataset/dataSets/training/volume(table 6)_training.csv')
	volume['time'] = pd.to_datetime(volume['time'])

	result = pd.DataFrame(columns=['1_0','1_1','2_0','3_0','3_1','t1','t2'])
	win_start_list = pd.date_range(starttime, endtime, freq = '20min',closed = 'left')
	for j,t1 in enumerate(win_start_list):
		t2 = t1 + pd.to_timedelta('20 m')
			
		wdata = volume[(volume['time']>=t1) & (volume['time']<t2)]
		volumecnt = wdata[['tollgate_id' ,'direction','time']].groupby(\
				['tollgate_id','direction']).count()

		volumecnt = volumecnt.T
		volumecnt.columns = [str(muticol[0])+'_'+ str(muticol[1]) for muticol in volumecnt.columns]
		
		volumecnt['t1'] = t1
		volumecnt['t2'] = t2

		result = pd.concat([result, volumecnt],axis=0,ignore_index=True)
	return result

if __name__ == '__main__':
	starttime = pd.datetime(2016,9,19,15,0,0)
	endtime = pd.datetime(2016,9,19,17,0,0)
	filename = 'test.csv'

	cal_tollgate_volume(starttime , endtime , filename)