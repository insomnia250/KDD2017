#coding=utf-8
from __future__ import division 
import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas import Series

'''
# load data
linkwise_ft = pd.read_csv('../features/linkwise/avgtime_linkwise_ft.csv')[['intersection_id',
		'tollgate_id','length_tot','linknum','R']]
linkwise_ft['tollgate_id'] = linkwise_ft['tollgate_id'].astype(str)

train_set  = pd.DataFrame()
val_set = pd.DataFrame()
test_set = pd.DataFrame()

for col in ['A_2','A_3','B_1','B_3','C_1','C_3']:
	print 'concating ',col
	iid = col.split('_')[0]
	tid = col.split('_')[1]

	part_train_set = pd.read_csv('./online_results/avgtime/{0}/train_set.csv'.format(col))
	part_val_set = pd.read_csv('./online_results/avgtime/{0}/offlinetest_set.csv'.format(col))
	part_test_set = pd.read_csv('./online_results/avgtime/{0}/onlinetest_set.csv'.format(col))

	# virtual_part_train_set = pd.read_csv('./results/avgtime/{0}/virtual_train_set.csv'.format(col))
	# part_train_set = pd.concat([part_train_set , virtual_part_train_set],axis=0,ignore_index=True)

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

train_set.to_csv('./online_results/avgtime/train_set.csv',index=False)
val_set.to_csv('./online_results/avgtime/val_set.csv',index=False)
test_set.to_csv('./online_results/avgtime/test_set.csv',index=False)
################################################################################






'''

train_set  = pd.DataFrame()
val_set = pd.DataFrame()
test_set = pd.DataFrame()

submission = pd.DataFrame()
record = np.zeros((5,6)) ; detail_score = pd.DataFrame()
for col in ['1_0','1_1','2_0','3_0','3_1']:
	print 'concating ',col
	tid = col.split('_')[0]
	direction = col.split('_')[1]

	part_train_set = pd.read_csv('./online_results/volume/{0}/train_set.csv'.format(col))
	part_val_set = pd.read_csv('./online_results/volume/{0}/offlinetest_set.csv'.format(col))
	part_test_set = pd.read_csv('./online_results/volume/{0}/onlinetest_set.csv'.format(col))

	virtual_part_train_set = pd.read_csv('./online_results/volume/{0}/virtual_train_set.csv'.format(col))
	part_train_set = pd.concat([part_train_set , virtual_part_train_set],axis=0,ignore_index=True)

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

train_set.to_csv('./online_results/volume/train_set.csv',index=False)
val_set.to_csv('./online_results/volume/val_set.csv',index=False)
test_set.to_csv('./online_results/volume/test_set.csv',index=False)

##################################################################################

