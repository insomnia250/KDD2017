#coding=utf-8
from __future__ import division 
import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas import Series
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from evaluation import *
import xgboost as xgb 
import cPickle
'''
dropcols = ['involume','outvolume','chokevolume','sign_choke','rate_choke',
	'involume_win0','involume_win1','involume_win2','involume_win3','involume_win4','involume_win5',
	'ddefault0','ddefault1','ddefault2','ddefault3','ddefault4','ddefault5',
	'ftavgtime_plus_ddefault0','ftavgtime_plus_ddefault1','ftavgtime_plus_ddefault2',
	'ftavgtime_plus_ddefault3','ftavgtime_plus_ddefault4','ftavgtime_plus_ddefault5',
	'lastcar_mean_defalut','lastcar_mid_defalut','lastlink_time','firstlink_time',
	'llt_over_len','llt_over_width','flt_over_len','flt_over_width']
# dropcols = []

# # －－－－－－－－－－avgtime－－－－－－－－－
#　整体建模
# load data
train_set  = pd.read_csv('../../samples/online_results/avgtime/train_set.csv').drop(dropcols,axis=1)
val_set = pd.read_csv('../../samples/online_results/avgtime/val_set.csv').drop(dropcols,axis=1)
test_set = pd.read_csv('../../samples/online_results/avgtime/test_set.csv').drop(dropcols,axis=1)

train_set[['intersection_id','tollgate_id']] = train_set[['intersection_id','tollgate_id']].astype(str)
val_set[['intersection_id','tollgate_id']] = val_set[['intersection_id','tollgate_id']].astype(str)
test_set[['intersection_id','tollgate_id']] = test_set[['intersection_id','tollgate_id']].astype(str)

submission = pd.DataFrame()
record = np.zeros((6,6)) ; detail_score = pd.DataFrame()


# 整理
for win_num in range(6):
	win_train_set = train_set[train_set['win_num']==win_num]
	win_val_set = val_set[val_set['win_num']==win_num]
	win_test_set = test_set[test_set['win_num']==win_num]

	winstart_train = win_train_set['winstart'].values
	winstart_val = win_val_set['winstart'].values
	winstart_test = win_test_set['winstart'].values

	win_train_set = win_train_set.drop(['winstart'],axis=1)
	win_val_set = win_val_set.drop(['winstart'],axis=1)
	win_test_set = win_test_set.drop(['winstart'],axis=1)

	y_train = win_train_set['tgavgtime'].values
	X_train = win_train_set.ix[:,1:]; 

	y_val = win_val_set['tgavgtime'].values
	X_val = win_val_set.ix[:,1:]  

	X_test = win_test_set.ix[:,1:]  

	iid_val = X_val['intersection_id'].values
	tid_val = X_val['tollgate_id'].values

	iid_test = X_test['intersection_id'].values
	tid_test = X_test['tollgate_id'].values

	X_all = pd.concat([X_train,X_val,X_test],axis=0,ignore_index=True)

	dum_iid = pd.get_dummies(X_all['intersection_id'] ,prefix = 'iid')
	dum_tid = pd.get_dummies(X_all['tollgate_id'] ,prefix = 'tid')
	X_all = X_all.drop(['intersection_id','tollgate_id'],axis=1)
	X_all = pd.concat([X_all,dum_iid,dum_tid],axis=1)

	l1 = X_train.shape[0]
	l2 = X_val.shape[0]
	l3 = X_test.shape[0]

	X_train = X_all.loc[0:l1-1,:]
	X_val = X_all.loc[l1:l1+l2-1,:]
	X_test = X_all.loc[l1+l2:,:]

	print len(X_train),len(X_val),len(X_test)

	params={
	'booster':'gbtree',
	'objective':'reg:linear',
	'gamma':0.1,
	'max_depth':9,
	# 'lambda':70,
	'subsample':0.7,
	'colsample_bytree':0.3,
	'min_child_weight':0.3, 
	'eta': 0.04,
	'seed':69,
	'silent':1,
	}

	dtrain = xgb.DMatrix(X_train, y_train)
	# # 交叉验证,obj=obj_function305 329
	# reg = xgb.cv(params, dtrain, num_boost_round=500 , nfold=5, feval=eva_function ,verbose_eval=True,obj=obj_function)

	# 训练obj=obj_function,
	num_boost_round = 200  #71
	watchlist  = [(dtrain,'train')]
	reg = xgb.train(params, dtrain, num_boost_round,obj=obj_function)#,evals=watchlist)

	# # # 打印特征重要性
	# featureColumns = X_train.columns
	# print_ft_impts(featureColumns,reg)

	# 去掉训练集中的模型反感的点
	preds_train = reg.predict(xgb.DMatrix( X_train))
	train_result = pd.DataFrame()
	train_result['preds'] = preds_train
	train_result['true'] = y_train
	train_result['loss'] = np.abs( (preds_train-y_train)/y_train )

	idx = (train_result['loss'] < train_result['loss'].quantile(0.95)).values  #
	X_train = X_train[idx];X_train.index = range(X_train.shape[0]) 
	y_train = y_train[idx]
	print len(X_train),len(y_train)
	dtrain = xgb.DMatrix(X_train, y_train)
	# reg = xgb.cv(params, dtrain, num_boost_round=520 , nfold=5, feval=eva_function ,verbose_eval=True,obj=obj_function)
	reg = xgb.train(params, dtrain, 263,obj=obj_function)#,evals=watchlist)

	#线下测试
	preds_val = reg.predict( xgb.DMatrix( X_val) )
	# 计分明细
	for col_num,col in enumerate(['A_2','A_3','B_1','B_3','C_1','C_3']):
		iid = col.split('_')[0]
		tid = col.split('_')[1]
		preds = preds_val[(iid_val==iid) & (tid_val==tid)]
		labels = y_val[(iid_val==iid) & (tid_val==tid)]
		record[col_num][win_num] = cal_MAPE(preds,labels)
	detail_score['win_num'+str(win_num)] = record[:,win_num]

	# 线上
	preds_test = reg.predict( xgb.DMatrix(X_test) )
	result_test=pd.DataFrame()
	result_test['avg_travel_time'] = preds_test
	result_test['intersection_id'] = iid_test
	result_test['tollgate_id'] = tid_test
	winend_test = (pd.to_datetime(winstart_test) + pd.to_timedelta('20 m')).astype(str)
	result_test['time_window'] = '['+winstart_test+','+ winend_test + ')'


	submission = pd.concat([submission,result_test[['intersection_id','tollgate_id','time_window','avg_travel_time']] ],
		axis=0,ignore_index=True)

submission['avg_travel_time'] = submission['avg_travel_time'].astype(float)
print submission.info()
submission.to_csv('./results/avgtime/multi_avgtime_maxdepth9_delete5percent.csv',index=False)

detail_score.index = ['A_2','A_3','B_1','B_3','C_1','C_3']
print detail_score
print detail_score.mean(0)
print detail_score.mean(1)
print 'val score:',detail_score.mean().mean()

############################################################################







'''





dropcols = []
dropcols = ['ddefault0','ddefault1','ddefault2','ddefault3','ddefault4','ddefault5',
	'ftvolume_plus_ddefault0','ftvolume_plus_ddefault1','ftvolume_plus_ddefault2',
	'ftvolume_plus_ddefault3','ftvolume_plus_ddefault4','ftvolume_plus_ddefault5']

#　整体建模
# load data
train_set  = pd.read_csv('../../samples/results/volume/train_set.csv').drop(dropcols,axis=1)
val_set = pd.read_csv('../../samples/results/volume/val_set.csv').drop(dropcols,axis=1)
test_set = pd.read_csv('../../samples/results/volume/test_set.csv').drop(dropcols,axis=1)

train_set[['tid','direction']] = train_set[['tid','direction']].astype(str)
val_set[['tid','direction']] = val_set[['tid','direction']].astype(str)
test_set[['tid','direction']] = test_set[['tid','direction']].astype(str)

submission = pd.DataFrame()
record = np.zeros((5,6)) ; detail_score = pd.DataFrame()


# 整理
for win_num in range(6):
	win_train_set = train_set[train_set['win_num']==win_num]
	win_val_set = val_set[val_set['win_num']==win_num]
	win_test_set = test_set[test_set['win_num']==win_num]

	winstart_train = win_train_set['winstart'].values
	winstart_val = win_val_set['winstart'].values
	winstart_test = win_test_set['winstart'].values

	win_train_set = win_train_set.drop(['winstart'],axis=1)
	win_val_set = win_val_set.drop(['winstart'],axis=1)
	win_test_set = win_test_set.drop(['winstart'],axis=1)

	y_train = win_train_set['tgvolume']; y_train = y_train.replace(0,1).values
	X_train = win_train_set.ix[:,1:]; 

	y_val = win_val_set['tgvolume']; y_val = y_val.replace(0,1).values
	X_val = win_val_set.ix[:,1:]

	X_test = win_test_set.ix[:,1:]

	tid_val = X_val['tid'].values
	direction_val = X_val['direction'].values

	tid_test = X_test['tid'].values
	direction_test = X_test['direction'].values

	X_all = pd.concat([X_train,X_val,X_test],axis=0,ignore_index=True)

	dum_tid = pd.get_dummies(X_all['tid'] ,prefix = 'tid')
	dum_direction = pd.get_dummies(X_all['direction'] ,prefix = 'direction')
	X_all = X_all.drop(['tid','direction'],axis=1)
	X_all = pd.concat([X_all,dum_tid,dum_direction],axis=1)

	l1 = X_train.shape[0]
	l2 = X_val.shape[0]
	l3 = X_test.shape[0]

	X_train = X_all.loc[0:l1-1,:]
	X_val = X_all.loc[l1:l1+l2-1,:]
	X_test = X_all.loc[l1+l2:,:]

	print l1,l2,l3

	params={
	'booster':'gbtree',
	'objective':'reg:linear',
	'gamma':0.1,
	'max_depth':9,  #8
	# 'lambda':10,
	'subsample':0.7,
	'colsample_bytree':0.3,
	'min_child_weight':0.3, 
	'eta': 0.04,
	'seed':69,
	'silent':1,
	}
	dtrain = xgb.DMatrix(X_train, y_train)
	# # # # 交叉验证,obj=obj_function
	# reg = xgb.cv(params, dtrain, num_boost_round=520 , nfold=5, feval=eva_function ,verbose_eval=True)

	# 训练obj=obj_function,
	num_boost_round = 93  #88
	watchlist  = [(dtrain,'train')]
	reg = xgb.train(params, dtrain, num_boost_round)#,evals=watchlist)

	# 去掉训练集中的模型反感的点
	preds_train = reg.predict(xgb.DMatrix( X_train))
	train_result = pd.DataFrame()
	train_result['preds'] = preds_train
	train_result['true'] = y_train
	train_result['loss'] = np.abs( (preds_train-y_train)/y_train )

	idx = (train_result['loss'] < train_result['loss'].quantile(1)).values
	X_train = X_train[idx];X_train.index = range(X_train.shape[0]) 
	y_train = y_train[idx]
	print len(X_train),len(y_train)
	dtrain = xgb.DMatrix(X_train, y_train)
	# reg = xgb.cv(params, dtrain, num_boost_round=520 , nfold=5, feval=eva_function ,verbose_eval=True)
	reg = xgb.train(params, dtrain, 93)#,evals=watchlist)

	#线下测试
	preds_val = reg.predict( xgb.DMatrix( X_val) )
	# 计分明细
	for col_num,col in enumerate(['1_0','1_1','2_0','3_0','3_1']):
		tid = col.split('_')[0]
		direction = col.split('_')[1]
		preds = preds_val[(tid_val==tid) & (direction_val==direction)]
		labels = y_val[(tid_val==tid) & (direction_val==direction)]
		record[col_num][win_num] = cal_MAPE(preds,labels)
	detail_score['win_num'+str(win_num)] = record[:,win_num]

	# 线上
	preds_test = reg.predict( xgb.DMatrix( X_test) )
	result_test=pd.DataFrame()
	result_test['volume'] = preds_test
	result_test['tollgate_id'] = tid_test
	result_test['direction'] = direction_test
	winend_test = (pd.to_datetime(winstart_test) + pd.to_timedelta('20 m')).astype(str)
	result_test['time_window'] = '['+winstart_test+','+ winend_test + ')'

	submission = pd.concat([submission,result_test[['tollgate_id','time_window','direction','volume']] ],
		axis=0,ignore_index=True)

submission['volume'] = submission['volume'].astype(int)
# print submission.info()
# submission.to_csv('./results/volume/multi_volume_maxdepth9_delete0percent.csv',index=False)

detail_score.index = ['1_0','1_1','2_0','3_0','3_1']
print detail_score
print detail_score.mean(0)
print detail_score.mean(1)
print 'val score:',detail_score.mean().mean()
