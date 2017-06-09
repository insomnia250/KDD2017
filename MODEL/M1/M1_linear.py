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
from sklearn.ensemble import GradientBoostingRegressor
import cPickle



# －－－－－－－－－－avgtime－－－－－－－－－
# 分路线建模
# load data
submission = pd.DataFrame()
tot_score = []
for col in ['A_2','A_3','B_1','B_3','C_1','C_3']:
	iid = col.split('_')[0]
	tid = col.split('_')[1]

	train_set = pd.read_csv('../../samples/results/avgtime/{0}/train_set.csv'.format(col))
	val_set = pd.read_csv('../../samples/results/avgtime/{0}/offlinetest_set.csv'.format(col))
	test_set = pd.read_csv('../../samples/results/avgtime/{0}/onlinetest_set.csv'.format(col))

	winstart_train = train_set['winstart']
	winstart_val = val_set['winstart']
	winstart_test = test_set['winstart']

	train_set = train_set.drop(['winstart'],axis=1)
	val_set = val_set.drop(['winstart'],axis=1)
	test_set = test_set.drop(['winstart'],axis=1)

	y_train = train_set['tgavgtime']
	y_train = y_train.fillna(y_train.mean(0))
	X_train = train_set.ix[:,1:]; 
	X_train = X_train.fillna(X_train.mean(0))

	y_val = val_set['tgavgtime']
	y_val = y_val.fillna(y_train.mean(0))
	X_val = val_set.ix[:,1:]  
	X_val = X_val.fillna(X_train.mean(0))

	X_test = test_set.ix[:,1:]  
	X_test = X_test.fillna(X_train.mean(0))
	# from sklearn.linear_model import Ridge,RidgeCV
	# reg = RidgeCV(alphas=[0.01, 0.1,1.0, 10.0,100],normalize=True)
	# reg.fit(X_train,y_train)

	from sklearn.linear_model import Lasso,LassoCV
	reg = LassoCV(alphas=[0.01, 0.1,1.0, 10.0,100],normalize=True)
	reg.fit(X_train,y_train)
	print X_train.info()
	print reg.coef_

	# from sklearn.ensemble import RandomForestRegressor
	# reg = RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=7, min_samples_split=2, 
	# 					min_samples_leaf=1, min_weight_fraction_leaf=0.0,random_state=69)
	# reg.fit(X_train,y_train)

	#线下测试
	preds_val = reg.predict(X_val)

	result=pd.DataFrame()
	result['avg_travel_time'] = preds_val
	result['labels'] = y_val
	result['intersection_id'] = iid
	result['tollgate_id'] = tid
	winend_val = (pd.to_datetime(winstart_val) + pd.to_timedelta('20 m')).astype(str)
	result['time_window'] = '['+winstart_val+','+ winend_val + ')'
	
	print '{0}_{1}: {2}'.format(iid,tid,cal_MAPE(preds_val,y_val))
	tot_score.append(cal_MAPE(preds_val,y_val))
	# 线上
	preds_test = reg.predict(X_test)

	result_test=pd.DataFrame()
	result_test['avg_travel_time'] = preds_test
	result_test['intersection_id'] = iid
	result_test['tollgate_id'] = tid
	winend_test = (pd.to_datetime(winstart_test) + pd.to_timedelta('20 m')).astype(str)
	result_test['time_window'] = '['+winstart_test+','+ winend_test + ')'
	print winstart_test
	submission = pd.concat([submission,result_test[['intersection_id','tollgate_id','time_window','avg_travel_time']] ],
		axis=0,ignore_index=True)

print 'tot score:{0}'.format( np.mean(tot_score) )
submission['avg_travel_time'] = submission['avg_travel_time'].astype(float)
print submission.info()
submission.to_csv('./results/avgtime/avgtime_Lasso_sep.csv',index=False)


print '＝＝＝＝＝＝＝＝＝＝＝＝'
#　整体建模
# load data
linkwise_ft = pd.read_csv('../../features/linkwise/avgtime_linkwise_ft.csv')
linkwise_ft['tollgate_id'] = linkwise_ft['tollgate_id'].astype(str)

train_set  = pd.DataFrame()
val_set = pd.DataFrame()
test_set = pd.DataFrame()

submission = pd.DataFrame()
tot_score = []
for col in ['A_2','A_3','B_1','B_3','C_1','C_3']:
	iid = col.split('_')[0]
	tid = col.split('_')[1]

	part_train_set = pd.read_csv('../../samples/results/avgtime/{0}/train_set.csv'.format(col))
	part_val_set = pd.read_csv('../../samples/results/avgtime/{0}/offlinetest_set.csv'.format(col))
	part_test_set = pd.read_csv('../../samples/results/avgtime/{0}/onlinetest_set.csv'.format(col))

	part_train_set['intersection_id'] = iid
	part_train_set['tollgate_id'] = tid

	part_val_set['intersection_id'] = iid
	part_val_set['tollgate_id'] = tid
	
	part_test_set['intersection_id'] = iid
	part_test_set['tollgate_id'] = tid

	part_train_set = part_train_set.fillna(part_train_set.mean(0))
	part_val_set = part_val_set.fillna(part_train_set.mean(0))
	part_test_set = part_test_set.fillna(part_train_set.mean(0))


	part_train_set = pd.merge(part_train_set, linkwise_ft,how='left',on=['intersection_id','tollgate_id'])
	part_val_set = pd.merge(part_val_set, linkwise_ft,how='left',on=['intersection_id','tollgate_id'])
	part_test_set = pd.merge(part_test_set, linkwise_ft,how='left',on=['intersection_id','tollgate_id'])

	train_set = pd.concat([train_set,part_train_set],axis=0,ignore_index=True)
	val_set = pd.concat([val_set,part_val_set],axis=0,ignore_index=True)
	test_set = pd.concat([test_set,part_test_set],axis=0,ignore_index=True)

# 整理
winstart_train = train_set['winstart']
winstart_val = val_set['winstart']
winstart_test = test_set['winstart']

train_set = train_set.drop(['winstart'],axis=1)
val_set = val_set.drop(['winstart'],axis=1)
test_set = test_set.drop(['winstart'],axis=1)

y_train = train_set['tgavgtime']
X_train = train_set.ix[:,1:]; 

y_val = val_set['tgavgtime']
X_val = val_set.ix[:,1:]  

X_test = test_set.ix[:,1:]  

iid_val = X_val['intersection_id']
tid_val = X_val['tollgate_id']

iid_test = X_test['intersection_id']
tid_test = X_test['tollgate_id']

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
print l1,l2,l3
from sklearn.linear_model import Ridge,RidgeCV
reg = RidgeCV(alphas=[0.00001,0.0001,0.01, 0.1,1.0, 10.0],normalize=True)
# from sklearn.ensemble import RandomForestRegressor
# reg = RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=7, min_samples_split=2, 
# 						min_samples_leaf=1, min_weight_fraction_leaf=0.0,random_state=69)

# from sklearn.linear_model import Lasso,LassoCV
# reg = LassoCV(alphas=[0.00001,0.0001,0.01, 0.1,1.0, 10.0],normalize=True)
reg.fit(X_train,y_train)



#线下测试
preds_val = reg.predict(X_val)

result=pd.DataFrame()
result['avg_travel_time'] = preds_val
result['labels'] = y_val
result['intersection_id'] = iid_val
result['tollgate_id'] = tid_val
winend_val = (pd.to_datetime(winstart_val) + pd.to_timedelta('20 m')).astype(str)
result['time_window'] = '['+winstart_val+','+ winend_val + ')'

print 'tot score: {0}'.format(cal_MAPE(preds_val,y_val))

# for col in ['A_2','A_3','B_1','B_3','C_1','C_3']:
# 	iid = col.split('_')[0]
# 	tid = col.split('_')[1]

# 	preds = result[(iid_val==iid) & (tid_val==tid)]['avg_travel_time'] 
# 	labels = result[(iid_val==iid) & (tid_val==tid)]['labels']
# 	print '{0}_{1}: {2}'.format(iid,tid,cal_MAPE(preds,labels))


# 线上
preds_test = reg.predict(X_test)
result_test=pd.DataFrame()
result_test['avg_travel_time'] = preds_test
result_test['intersection_id'] = iid_test
result_test['tollgate_id'] = tid_test
winend_test = (pd.to_datetime(winstart_test) + pd.to_timedelta('20 m')).astype(str)
result_test['time_window'] = '['+winstart_test+','+ winend_test + ')'



submission = result_test[['intersection_id','tollgate_id','time_window','avg_travel_time']]
submission['avg_travel_time'] = submission['avg_travel_time'].astype(float)

print submission.info()
submission.to_csv('./results/avgtime/avgtime_Ridge_all.csv',index=False)
#####################################################################





#####################################################################
# # －－－－－－－－－－volume－－－－－－－－－
# # 分路线建模
# # load data
# submission = pd.DataFrame()
# tot_score = []
# for col in ['1_0','1_1','2_0','3_0','3_1']:
# 	tid = col.split('_')[0]
# 	direction = col.split('_')[1]

# 	train_set = pd.read_csv('../../samples/results/volume/{0}/train_set.csv'.format(col))
# 	val_set = pd.read_csv('../../samples/results/volume/{0}/offlinetest_set.csv'.format(col))
# 	test_set = pd.read_csv('../../samples/results/volume/{0}/onlinetest_set.csv'.format(col))

# 	winstart_train = train_set['winstart']
# 	winstart_val = val_set['winstart']
# 	winstart_test = test_set['winstart']

# 	train_set = train_set.drop(['winstart'],axis=1)
# 	val_set = val_set.drop(['winstart'],axis=1)
# 	test_set = test_set.drop(['winstart'],axis=1)

# 	y_train = train_set['tgvolume']
# 	y_train = y_train.fillna(y_train.mean(0))
# 	X_train = train_set.ix[:,1:]; 
# 	X_train = X_train.fillna(X_train.mean(0))

# 	y_val = val_set['tgvolume']
# 	y_val = y_val.fillna(y_train.mean(0))
# 	X_val = val_set.ix[:,1:]  
# 	X_val = X_val.fillna(X_train.mean(0))

# 	X_test = test_set.ix[:,1:]  
# 	X_test = X_test.fillna(X_train.mean(0))


# 	from sklearn.ensemble import RandomForestRegressor
# 	reg = RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=7, min_samples_split=2, 
# 						min_samples_leaf=1, min_weight_fraction_leaf=0.0,random_state=69)
# 	reg.fit(X_train,y_train)

# 	#线下测试
# 	preds_val = reg.predict(X_val)

# 	result_val=pd.DataFrame()
# 	result_val['volume'] = preds_val
# 	result_val['labels'] = y_val
# 	result_val['tollgate_id'] = tid
# 	result_val['direction'] = direction
# 	winend_val = (pd.to_datetime(winstart_val) + pd.to_timedelta('20 m')).astype(str)
# 	result_val['time_window'] = '['+winstart_val+','+ winend_val + ')'

# 	print '{0}_{1}: {2}'.format(tid,direction,cal_MAPE(preds_val,y_val))
# 	tot_score.append(cal_MAPE(preds_val,y_val))
	
# 	# 线上
# 	preds_test = reg.predict(X_test)
# 	result_test=pd.DataFrame()
# 	result_test['volume'] = preds_test
# 	result_test['tollgate_id'] = tid
# 	result_test['direction'] = direction
# 	winend_test = (pd.to_datetime(winstart_test) + pd.to_timedelta('20 m')).astype(str)
# 	result_test['time_window'] = '['+winstart_test+','+ winend_test + ')'


# 	submission = pd.concat([submission,result_test[['tollgate_id','time_window','direction','volume']] ],
# 		axis=0,ignore_index=True)

# print 'tot score:{0}'.format( np.mean(tot_score) )
# submission['volume'] = submission['volume'].astype(int)
# print submission.info()
# submission.to_csv('./results/volume/volume_RF_sep.csv',index=False)

# print '＝＝＝＝＝＝＝＝＝＝＝＝'

# #　整体建模
# # load data
# train_set  = pd.DataFrame()
# val_set = pd.DataFrame()
# test_set = pd.DataFrame()

# submission = pd.DataFrame()
# tot_score = []
# for col in ['1_0','1_1','2_0','3_0','3_1']:
# 	tid = col.split('_')[0]
# 	direction = col.split('_')[1]

# 	part_train_set = pd.read_csv('../../samples/results/volume/{0}/train_set.csv'.format(col))
# 	part_val_set = pd.read_csv('../../samples/results/volume/{0}/offlinetest_set.csv'.format(col))
# 	part_test_set = pd.read_csv('../../samples/results/volume/{0}/onlinetest_set.csv'.format(col))

# 	part_train_set['tid'] = tid
# 	part_train_set['direction'] = direction

# 	part_val_set['tid'] = tid
# 	part_val_set['direction'] = direction

# 	part_test_set['tid'] = tid
# 	part_test_set['direction'] = direction

# 	part_train_set = part_train_set.fillna(part_train_set.mean(0))
# 	part_val_set = part_val_set.fillna(part_train_set.mean(0))
# 	part_test_set = part_test_set.fillna(part_train_set.mean(0))

# 	train_set = pd.concat([train_set,part_train_set],axis=0,ignore_index=True)
# 	val_set = pd.concat([val_set,part_val_set],axis=0,ignore_index=True)
# 	test_set = pd.concat([test_set,part_test_set],axis=0,ignore_index=True)

# # 整理
# winstart_train = train_set['winstart']
# winstart_val = val_set['winstart']
# winstart_test = test_set['winstart']

# train_set = train_set.drop(['winstart'],axis=1)
# val_set = val_set.drop(['winstart'],axis=1)
# test_set = test_set.drop(['winstart'],axis=1)

# y_train = train_set['tgvolume']
# X_train = train_set.ix[:,1:]; 

# y_val = val_set['tgvolume']
# X_val = val_set.ix[:,1:]  

# X_test = test_set.ix[:,1:]  

# tid_val = X_val['tid']
# direction_val = X_val['direction']

# tid_test = X_test['tid']
# direction_test = X_test['direction']

# X_all = pd.concat([X_train,X_val,X_test],axis=0,ignore_index=True)

# dum_tid = pd.get_dummies(X_all['tid'] ,prefix = 'tid')
# dum_direction = pd.get_dummies(X_all['direction'] ,prefix = 'direction')
# X_all = X_all.drop(['tid','direction'],axis=1)
# X_all = pd.concat([X_all,dum_tid,dum_direction],axis=1)

# l1 = X_train.shape[0]
# l2 = X_val.shape[0]
# l3 = X_test.shape[0]

# X_train = X_all.loc[0:l1-1,:]
# X_val = X_all.loc[l1:l1+l2-1,:]
# X_test = X_all.loc[l1+l2:,:]

# print len(X_train),len(X_val),len(X_test)
# print l1,l2,l3
# # from sklearn.linear_model import Ridge,RidgeCV

# # reg = RidgeCV(alphas=[0.00001,0.0001,0.01, 0.1,1.0, 10.0],normalize=True)
# # reg.fit(X_train,y_train)
# from sklearn.ensemble import RandomForestRegressor
# reg = RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=7, min_samples_split=2, 
# 					min_samples_leaf=1, min_weight_fraction_leaf=0.0,random_state=69)
# reg.fit(X_train,y_train)

# #线下测试
# preds_val = reg.predict(X_val)

# result=pd.DataFrame()
# result['volume'] = preds_val
# result['labels'] = y_val
# result['tollgate_id'] = tid_val
# result['direction'] = direction_val
# winend_val = (pd.to_datetime(winstart_val) + pd.to_timedelta('20 m')).astype(str)
# result['time_window'] = '['+winstart_val+','+ winend_val + ')'

# print 'tot score: {0}'.format(cal_MAPE(preds_val,y_val))

# # for col in ['1_0','1_1','2_0','3_0','3_1']:
# # 	tid = col.split('_')[0]
# # 	direction = col.split('_')[1]

# # 	preds = result[(tid_val==tid) & (direction_val==direction)]['volume'] 
# # 	labels = result[(tid_val==tid) & (direction_val==direction)]['labels']
# # 	print '{0}_{1}: {2}'.format(tid,direction,cal_MAPE(preds,labels))

# # 线上
# preds_test = reg.predict(X_test)
# result_test=pd.DataFrame()
# result_test['volume'] = preds_test
# result_test['tollgate_id'] = tid_test
# result_test['direction'] = direction_test
# winend_test = (pd.to_datetime(winstart_test) + pd.to_timedelta('20 m')).astype(str)
# result_test['time_window'] = '['+winstart_test+','+ winend_test + ')'



# submission = result_test[['tollgate_id','time_window','direction','volume']]
# submission['volume'] = submission['volume'].astype(int)

# print submission.info()
# submission.to_csv('./results/volume/volume_RF_all.csv',index=False)




