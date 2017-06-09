#coding=utf-8
from __future__ import division 
import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas import Series
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import xgboost as xgb 
'''
dropcols = ['involume','outvolume','chokevolume','sign_choke','rate_choke',
	'involume_win0','involume_win1','involume_win2','involume_win3','involume_win4','involume_win5',
	'ddefault0','ddefault1','ddefault2','ddefault3','ddefault4','ddefault5',
	'ftavgtime_plus_ddefault0','ftavgtime_plus_ddefault1','ftavgtime_plus_ddefault2',
	'ftavgtime_plus_ddefault3','ftavgtime_plus_ddefault4','ftavgtime_plus_ddefault5',
	'lastcar_mean_defalut','lastcar_mid_defalut','lastlink_time','firstlink_time',
	'llt_over_len','llt_over_width','flt_over_len','flt_over_width']

linkwise_ft = pd.read_csv('../features/linkwise/avgtime_linkwise_ft.csv')[['intersection_id',
		'tollgate_id','length_tot','linknum','R']]
linkwise_ft['tollgate_id'] = linkwise_ft['tollgate_id'].astype(str)

train_set  = pd.DataFrame()
val_set = pd.DataFrame()
test_set = pd.DataFrame()
# quantiles
for col in ['A_2','A_3','B_1','B_3','C_1','C_3']:
	print 'concating ',col
	iid = col.split('_')[0]
	tid = col.split('_')[1]

	part_train_set = pd.read_csv('./online_results/avgtime/{0}/train_set.csv'.format(col))
	part_val_set = pd.read_csv('./online_results/avgtime/{0}/offlinetest_set.csv'.format(col))
	part_test_set = pd.read_csv('./online_results/avgtime/{0}/onlinetest_set.csv'.format(col))

	# virtual_part_train_set = pd.read_csv('./results/avgtime/{0}/virtual_train_set.csv'.format(col))
	# part_train_set = pd.concat([part_train_set , virtual_part_train_set],axis=0,ignore_index=True)


	part_train_set.drop(dropcols,axis=1,inplace=True)
	part_val_set.drop(dropcols,axis=1,inplace=True)
	part_test_set.drop(dropcols,axis=1,inplace=True)

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


ftcolumns = train_set.drop(['tgavgtime'],axis=1).columns

quantile1 = train_set[['win_num','tgavgtime']].groupby(['win_num'],sort=True).quantile(0.1).reset_index()
quantile2 = train_set[['win_num','tgavgtime']].groupby(['win_num'],sort=True).quantile(0.2).reset_index()
quantile3 = train_set[['win_num','tgavgtime']].groupby(['win_num'],sort=True).quantile(0.4).reset_index()
quantile4 = train_set[['win_num','tgavgtime']].groupby(['win_num'],sort=True).quantile(0.6).reset_index()
quantile5 = train_set[['win_num','tgavgtime']].groupby(['win_num'],sort=True).quantile(0.8).reset_index()
quantile6 = train_set[['win_num','tgavgtime']].groupby(['win_num'],sort=True).quantile(0.9).reset_index()

quantiles = DataFrame({'win_num':quantile1['win_num'],'quantile1':quantile1['tgavgtime']})
quantiles['quantile2'] = quantile2['tgavgtime']
quantiles['quantile3'] = quantile3['tgavgtime']
quantiles['quantile4'] = quantile4['tgavgtime']
quantiles['quantile5'] = quantile5['tgavgtime']
quantiles['quantile6'] = quantile6['tgavgtime']

print quantiles.head()

def set_clflabel(train_set, quantiles):
	train_set = pd.merge(train_set, quantiles,how='left',on='win_num')

	train_set['clf_label1'] = 0
	train_set.loc[train_set['tgavgtime']>train_set['quantile1'],'clf_label1'] = 1
	train_set['clf_label2'] = 0
	train_set.loc[train_set['tgavgtime']>train_set['quantile2'],'clf_label2'] = 1
	train_set['clf_label3'] = 0
	train_set.loc[train_set['tgavgtime']>train_set['quantile3'],'clf_label3'] = 1
	train_set['clf_label4'] = 0
	train_set.loc[train_set['tgavgtime']>train_set['quantile4'],'clf_label4'] = 1
	train_set['clf_label5'] = 0
	train_set.loc[train_set['tgavgtime']>train_set['quantile5'],'clf_label5'] = 1
	train_set['clf_label6'] = 0
	train_set.loc[train_set['tgavgtime']>train_set['quantile6'],'clf_label6'] = 1

	return train_set[['tgavgtime','clf_label1','clf_label2','clf_label3','clf_label4','clf_label5','clf_label6']+list(ftcolumns)]

# trainset
train_df = set_clflabel(train_set, quantiles)
train_df.to_csv('avgtime_clf_train.csv',index=False)

# offlinetestset
val_df = set_clflabel(val_set, quantiles)
val_df.to_csv('avgtime_clf_val.csv',index=False)

# onlinetestset
test_df = test_set[list(ftcolumns)]
test_df.to_csv('avgtime_clf_test.csv',index=False)

##################################################
'''
dropcols = ['ddefault0','ddefault1','ddefault2','ddefault3','ddefault4','ddefault5',
	'ftvolume_plus_ddefault0','ftvolume_plus_ddefault1','ftvolume_plus_ddefault2',
	'ftvolume_plus_ddefault3','ftvolume_plus_ddefault4','ftvolume_plus_ddefault5']

# linkwise_ft = pd.read_csv('../features/linkwise/avgtime_linkwise_ft.csv')[['intersection_id',
# 		'tollgate_id','length_tot','linknum','R']]
# linkwise_ft['tollgate_id'] = linkwise_ft['tollgate_id'].astype(str)

train_set  = pd.DataFrame()
val_set = pd.DataFrame()
test_set = pd.DataFrame()
# quantiles
for col in ['1_0','1_1','2_0','3_0','3_1']:
	tid = col.split('_')[0]
	direction = col.split('_')[1]


	part_train_set = pd.read_csv('./online_results/volume/{0}/train_set.csv'.format(col))
	part_val_set = pd.read_csv('./online_results/volume/{0}/offlinetest_set.csv'.format(col))
	part_test_set = pd.read_csv('./online_results/volume/{0}/onlinetest_set.csv'.format(col))

	# virtual_part_train_set = pd.read_csv('./results/volume/{0}/virtual_train_set.csv'.format(col))
	# part_train_set = pd.concat([part_train_set , virtual_part_train_set],axis=0,ignore_index=True)


	part_train_set.drop(dropcols,axis=1,inplace=True)
	part_val_set.drop(dropcols,axis=1,inplace=True)
	part_test_set.drop(dropcols,axis=1,inplace=True)

	part_train_set['tid'] = tid
	part_train_set['direction'] = direction

	part_val_set['tid'] = tid
	part_val_set['direction'] = direction

	part_test_set['tid'] = tid
	part_test_set['direction'] = direction

	part_train_set = part_train_set.fillna(part_train_set.mean(0))
	part_val_set = part_val_set.fillna(part_train_set.mean(0))
	part_test_set = part_test_set.fillna(part_train_set.mean(0))


	train_set = pd.concat([train_set,part_train_set],axis=0,ignore_index=True)
	val_set = pd.concat([val_set,part_val_set],axis=0,ignore_index=True)
	test_set = pd.concat([test_set,part_test_set],axis=0,ignore_index=True)


ftcolumns = train_set.drop(['tgvolume'],axis=1).columns

quantile1 = train_set[['win_num','tgvolume']].groupby(['win_num'],sort=True).quantile(0.1).reset_index()
quantile2 = train_set[['win_num','tgvolume']].groupby(['win_num'],sort=True).quantile(0.2).reset_index()
quantile3 = train_set[['win_num','tgvolume']].groupby(['win_num'],sort=True).quantile(0.4).reset_index()
quantile4 = train_set[['win_num','tgvolume']].groupby(['win_num'],sort=True).quantile(0.6).reset_index()
quantile5 = train_set[['win_num','tgvolume']].groupby(['win_num'],sort=True).quantile(0.8).reset_index()
quantile6 = train_set[['win_num','tgvolume']].groupby(['win_num'],sort=True).quantile(0.9).reset_index()

quantiles = DataFrame({'win_num':quantile1['win_num'],'quantile1':quantile1['tgvolume']})
quantiles['quantile2'] = quantile2['tgvolume']
quantiles['quantile3'] = quantile3['tgvolume']
quantiles['quantile4'] = quantile4['tgvolume']
quantiles['quantile5'] = quantile5['tgvolume']
quantiles['quantile6'] = quantile6['tgvolume']

print quantiles.head()

def set_clflabel(train_set, quantiles):
	train_set = pd.merge(train_set, quantiles,how='left',on='win_num')

	train_set['clf_label1'] = 0
	train_set.loc[train_set['tgvolume']>train_set['quantile1'],'clf_label1'] = 1
	train_set['clf_label2'] = 0
	train_set.loc[train_set['tgvolume']>train_set['quantile2'],'clf_label2'] = 1
	train_set['clf_label3'] = 0
	train_set.loc[train_set['tgvolume']>train_set['quantile3'],'clf_label3'] = 1
	train_set['clf_label4'] = 0
	train_set.loc[train_set['tgvolume']>train_set['quantile4'],'clf_label4'] = 1
	train_set['clf_label5'] = 0
	train_set.loc[train_set['tgvolume']>train_set['quantile5'],'clf_label5'] = 1
	train_set['clf_label6'] = 0
	train_set.loc[train_set['tgvolume']>train_set['quantile6'],'clf_label6'] = 1

	return train_set[['tgvolume','clf_label1','clf_label2','clf_label3','clf_label4','clf_label5','clf_label6']+list(ftcolumns)]

# trainset
train_df = set_clflabel(train_set, quantiles)
train_df.to_csv('volume_clf_train.csv',index=False)

# offlinetestset
val_df = set_clflabel(val_set, quantiles)
val_df.to_csv('volume_clf_val.csv',index=False)

# onlinetestset
test_df = test_set[list(ftcolumns)]
test_df.to_csv('volume_clf_test.csv',index=False)
