#coding=utf-8
from __future__ import division  
import numpy as np  
from sklearn.cross_validation import KFold
from sklearn import model_selection 
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier  
from sklearn.linear_model import LogisticRegression  
import pandas as pd
from pandas import DataFrame
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from evaluation import *

def logloss(attempt, actual, epsilon=1.0e-15):  
    """Logloss, i.e. the score of the bioresponse competition. 
    """  
    attempt = np.clip(attempt, epsilon, 1.0-epsilon)  
    return - np.mean(actual * np.log(attempt) + (1.0 - actual) * np.log(1.0 - attempt)) 


np.random.seed(0) # seed to shuffle the train set  

# n_folds = 10  
n_folds = 5  
verbose = True  
shuffle = True  
'''
train_set = pd.read_csv('../../samples/avgtime_clf_train.csv')
val_set = pd.read_csv('../../samples/avgtime_clf_val.csv')
test_set = pd.read_csv('../../samples/avgtime_clf_test.csv')

# for win_num in range(6):
win_num = 5
print '#################################'
print win_num
win_train_set = train_set[train_set['win_num']==win_num]
win_val_set = val_set[val_set['win_num']==win_num]
win_test_set = test_set[test_set['win_num']==win_num]

win_train_set.index = range(win_train_set.shape[0])
win_val_set.index = range(win_val_set.shape[0])
win_test_set.index = range(win_test_set.shape[0])

winstart_train = win_train_set['winstart'].values
winstart_val = win_val_set['winstart'].values
winstart_test = win_test_set['winstart'].values

win_train_set = win_train_set.drop(['winstart'],axis=1)
win_val_set = win_val_set.drop(['winstart'],axis=1)
win_test_set = win_test_set.drop(['winstart'],axis=1)

y_train = win_train_set.ix[:,1:7] ; y_train_true = win_train_set.ix[:,0]
X_train = win_train_set.ix[:,7:]

y_val = win_val_set.ix[:,1:7] ; y_val_true = win_val_set.ix[:,0]
X_val = win_val_set.ix[:,7:]  

X_test = win_test_set

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
X_val = X_all.loc[l1:l1+l2-1,:]  ;X_val.index=range(X_val.shape[0])
X_test = X_all.loc[l1+l2:,:]		;X_test.index=range(X_test.shape[0])
print X_test.info()

# if shuffle:
# 	idx = np.random.permutation(y_train.shape[0])
# 	X_train = X_train[idx]  
# 	y_train = y_train[idx]  


skf = list(KFold(len(y_train), n_folds))  

params={
    'booster':'gbtree',
    'objective':'binary:logistic',
    'gamma':0.1,
    'max_depth':8,
    #'lambda':250,
    'subsample':0.7,
    'colsample_bytree':0.3,
    'min_child_weight':0.3, 
    'eta': 0.04,
    'seed':69,
    'silent':1,
    'eval_metric ':'auc'
    }

# num_boost_round = np.array([[158,167,166,125,144],
# 							[116,232,178,144,124],
# 							[147,144,177,121,189],
# 							[151,161,243,159,164],
# 							[186,188,107,103,104],
# 							[143,155,129,123,137]])
# num_boost_round = np.array([[130,138,156,149,188],
# 							[176,201,130,180,148],
# 							[232,127,176,196,159],
# 							[318,184,167,193,191],
# 							[195,176,157,196,196],
# 							[218,189,241,222,174]])
num_boost_round = np.array([[170,197,147,140,159],
							[149,171,162,151,163],
							[154,128,131,236,200],
							[157,199,123,179,175],
							[192,260,153,188,107],
							[147,161,245,169,224]])

# num_boost_round=1

clfsname = ['clf1' , 'clf2' , 'clf3' , 'clf4','clf5','clf6']  

print "Creating train and test sets for blending."  
  
dataset_blend_train = np.zeros((X_train.shape[0], len(clfsname)))  
dataset_blend_val = np.zeros((X_val.shape[0], len(clfsname)))  
dataset_blend_test = np.zeros((X_test.shape[0],len(clfsname)))

for j, name in enumerate(clfsname): 
	print '==============================================================================================='
	print j, name 
	dataset_blend_val_j = np.zeros((X_val.shape[0], len(skf)))
	dataset_blend_test_j = np.zeros((X_test.shape[0], len(skf)))  
	for i, (trainpart, apart) in enumerate(skf):  
		# print "Fold", i  
		X_trainpart = X_train.ix[trainpart]  
		y_trainpart = y_train.ix[trainpart , j]  
		X_apart = X_train.ix[apart]  
		y_apart = y_train.ix[apart , j]

		dtrain = xgb.DMatrix( X_trainpart, y_trainpart)
		# # 交叉验证
		# bst = xgb.cv(params, dtrain, num_boost_round=500, nfold=5 , metrics ='auc',verbose_eval=True,early_stopping_rounds=50 )
		print '\n'
		bst = xgb.train(params, dtrain, num_boost_round=150)

		y_submission = bst.predict( xgb.DMatrix( X_apart))
		temp = pd.DataFrame({'label':y_apart,'prob':y_submission})

		dataset_blend_train[apart, j] = y_submission 

		dataset_blend_val_j[:, i] = bst.predict( xgb.DMatrix(X_val) )
		dataset_blend_test_j[:,i] = bst.predict( xgb.DMatrix(X_test) ) # online
	dataset_blend_val[:,j] = dataset_blend_val_j.mean(1)
	dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)

	# print("log loss : %0.8f" % (logloss(dataset_blend_val[:,j], y_val.ix[:,j])))  
	print "auc: %0.8f" % (roc_auc_score(y_val.ix[:,j], dataset_blend_val[:,j]))
print  
print "Blending."  
dataset_blend_train = pd.DataFrame(dataset_blend_train)
dataset_blend_train = pd.concat([dataset_blend_train,X_train],axis=1)

dataset_blend_val = pd.DataFrame(dataset_blend_val)
dataset_blend_val = pd.concat([dataset_blend_val,X_val],axis=1)

dataset_blend_test = pd.DataFrame(dataset_blend_test)
dataset_blend_test = pd.concat([dataset_blend_test,X_test],axis=1)
params2={
    'booster':'gbtree',
    'objective':'reg:linear',
    'gamma':0.1,
    'max_depth':8,
    # 'lambda':250,
    'subsample':0.7,
    'colsample_bytree':0.3,
    'min_child_weight':0.3, 
    'eta': 0.04,
    'seed':69,
    'silent':1,
    }
num_boost_round2 = 70
dtrain2 = xgb.DMatrix( dataset_blend_train, y_train_true)
# 交叉验证,obj=obj_function305 329
reg = xgb.cv(params2, dtrain2, num_boost_round=500 , nfold=5, feval=eva_function ,verbose_eval=True)

bst2 = xgb.train(params2, dtrain2, num_boost_round2)

y_submission = bst2.predict( xgb.DMatrix( dataset_blend_val ) )


print "blend result"  
print("val loss : %0.5f" % (cal_MAPE(y_submission , y_val_true) ))

# 线上
online_submission = bst2.predict( xgb.DMatrix( dataset_blend_test ) )

result_test=pd.DataFrame()
result_test['avg_travel_time'] = online_submission
result_test['intersection_id'] = iid_test
result_test['tollgate_id'] = tid_test
winend_test = (pd.to_datetime(winstart_test) + pd.to_timedelta('20 m')).astype(str)
result_test['time_window'] = '['+winstart_test+','+ winend_test + ')'

submission_win = result_test[['intersection_id','tollgate_id','time_window','avg_travel_time']]
submission_win['avg_travel_time'] = submission_win['avg_travel_time'].astype(float)

base_submission = pd.read_csv('../M1/results/avgtime/multi_avgtime_XGBobj_all_[lastcar_holiday].csv')
submission = pd.concat([submission_win, base_submission],axis=0,ignore_index=True)
submission = submission.drop_duplicates(['intersection_id','tollgate_id','time_window'],keep = 'first')
submission.to_csv('M2_avgtime.csv')

'''
############################################

train_set = pd.read_csv('../../samples/volume_clf_train.csv')
val_set = pd.read_csv('../../samples/volume_clf_val.csv')
test_set = pd.read_csv('../../samples/volume_clf_test.csv')

# for win_num in range(6):
win_num = 5
print '#################################'
print win_num
win_train_set = train_set[train_set['win_num']==win_num]
win_val_set = val_set[val_set['win_num']==win_num]
win_test_set = test_set[test_set['win_num']==win_num]

win_train_set.index = range(win_train_set.shape[0])
win_val_set.index = range(win_val_set.shape[0])
win_test_set.index = range(win_test_set.shape[0])

winstart_train = win_train_set['winstart'].values
winstart_val = win_val_set['winstart'].values
winstart_test = win_test_set['winstart'].values

win_train_set = win_train_set.drop(['winstart'],axis=1)
win_val_set = win_val_set.drop(['winstart'],axis=1)
win_test_set = win_test_set.drop(['winstart'],axis=1)

y_train = win_train_set.ix[:,1:7] ; y_train_true = win_train_set.ix[:,0].replace(0,1)
X_train = win_train_set.ix[:,7:]

y_val = win_val_set.ix[:,1:7] ; y_val_true = win_val_set.ix[:,0].replace(0,1)
X_val = win_val_set.ix[:,7:]  

X_test = win_test_set

tid_val = X_val['tid'].values
direction_val = X_val['direction'].values

tid_test = X_test['tid'].values
direction_test = X_test['direction'].values

X_all = pd.concat([X_train,X_val,X_test],axis=0,ignore_index=True)

dum_iid = pd.get_dummies(X_all['tid'] ,prefix = 'tid')
dum_tid = pd.get_dummies(X_all['direction'] ,prefix = 'direction')
X_all = X_all.drop(['tid','direction'],axis=1)
X_all = pd.concat([X_all,dum_iid,dum_tid],axis=1)

l1 = X_train.shape[0]
l2 = X_val.shape[0]
l3 = X_test.shape[0]

X_train = X_all.loc[0:l1-1,:]
X_val = X_all.loc[l1:l1+l2-1,:]  ;X_val.index=range(X_val.shape[0])
X_test = X_all.loc[l1+l2:,:]		;X_test.index=range(X_test.shape[0])


# if shuffle:
# 	idx = np.random.permutation(y_train.shape[0])
# 	X_train = X_train[idx]  
# 	y_train = y_train[idx]  


skf = list(KFold(len(y_train), n_folds))  

params={
    'booster':'gbtree',
    'objective':'binary:logistic',
    'gamma':0.1,
    'max_depth':8,
    #'lambda':250,
    'subsample':0.7,
    'colsample_bytree':0.3,
    'min_child_weight':0.3, 
    'eta': 0.04,
    'seed':69,
    'silent':1,
    'eval_metric ':'auc'
    }

# num_boost_round = np.array([[158,167,166,125,144],
# 							[116,232,178,144,124],
# 							[147,144,177,121,189],
# 							[151,161,243,159,164],
# 							[186,188,107,103,104],
# 							[143,155,129,123,137]])
# num_boost_round = np.array([[130,138,156,149,188],
# 							[176,201,130,180,148],
# 							[232,127,176,196,159],
# 							[318,184,167,193,191],
# 							[195,176,157,196,196],
# 							[218,189,241,222,174]])
num_boost_round = np.array([[170,197,147,140,159],
							[149,171,162,151,163],
							[154,128,131,236,200],
							[157,199,123,179,175],
							[192,260,153,188,107],
							[147,161,245,169,224]])

# num_boost_round=1

clfsname = ['clf1' , 'clf2' , 'clf3' , 'clf4','clf5','clf6']  

print "Creating train and test sets for blending."  
  
dataset_blend_train = np.zeros((X_train.shape[0], len(clfsname)))  
dataset_blend_val = np.zeros((X_val.shape[0], len(clfsname)))  
dataset_blend_test = np.zeros((X_test.shape[0],len(clfsname)))

for j, name in enumerate(clfsname): 
	print '==============================================================================================='
	print j, name 
	dataset_blend_val_j = np.zeros((X_val.shape[0], len(skf)))
	dataset_blend_test_j = np.zeros((X_test.shape[0], len(skf)))  
	for i, (trainpart, apart) in enumerate(skf):  
		# print "Fold", i  
		X_trainpart = X_train.ix[trainpart]  
		y_trainpart = y_train.ix[trainpart , j]  
		X_apart = X_train.ix[apart]  
		y_apart = y_train.ix[apart , j]

		dtrain = xgb.DMatrix( X_trainpart, y_trainpart)
		# # 交叉验证
		# bst = xgb.cv(params, dtrain, num_boost_round=500, nfold=5 , metrics ='auc',verbose_eval=True,early_stopping_rounds=50 )
		print '\n'
		bst = xgb.train(params, dtrain, num_boost_round=70)

		# y_submission = bst.predict( xgb.DMatrix( X_apart))
		y_submission = bst.predict( xgb.DMatrix( X_apart))
		temp = pd.DataFrame({'label':y_apart,'prob':y_submission})

		dataset_blend_train[apart, j] = y_submission 

		dataset_blend_val_j[:, i] = bst.predict( xgb.DMatrix(X_val) )
		dataset_blend_test_j[:,i] = bst.predict( xgb.DMatrix(X_test) ) # online
	dataset_blend_val[:,j] = dataset_blend_val_j.mean(1)
	dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)

	# print("log loss : %0.8f" % (logloss(dataset_blend_val[:,j], y_val.ix[:,j])))  
	# print "auc: %0.8f" % (roc_auc_score(y_val.ix[:,j], dataset_blend_val[:,j]))
print  
print "Blending."  
dataset_blend_train = pd.DataFrame(dataset_blend_train)
dataset_blend_train = pd.concat([dataset_blend_train,X_train],axis=1)

dataset_blend_val = pd.DataFrame(dataset_blend_val)
dataset_blend_val = pd.concat([dataset_blend_val,X_val],axis=1)

dataset_blend_test = pd.DataFrame(dataset_blend_test)
dataset_blend_test = pd.concat([dataset_blend_test,X_test],axis=1)

params2={
    'booster':'gbtree',
    'objective':'reg:linear',
    'gamma':0.1,
    'max_depth':8,
    # 'lambda':250,
    'subsample':0.7,
    'colsample_bytree':0.3,
    'min_child_weight':0.3, 
    'eta': 0.04,
    'seed':69,
    'silent':1,
    }
num_boost_round2 = 500
dtrain2 = xgb.DMatrix( dataset_blend_train, y_train_true)
# print dataset_blend_train

# 交叉验证,obj=obj_function305 329
reg = xgb.cv(params2, dtrain2, num_boost_round=500 , nfold=5, feval=eva_function ,verbose_eval=True)

bst2 = xgb.train(params2, dtrain2, num_boost_round2)

y_submission = bst2.predict( xgb.DMatrix( dataset_blend_val ) )



print "blend result"  
print("val loss : %0.5f" % (cal_MAPE(y_submission , y_val_true) ))

# 线上
online_submission = bst2.predict( xgb.DMatrix( dataset_blend_test ) )
result_test=pd.DataFrame()
result_test['volume'] = online_submission
result_test['tollgate_id'] = tid_test
result_test['direction'] = direction_test
winend_test = (pd.to_datetime(winstart_test) + pd.to_timedelta('20 m')).astype(str)
result_test['time_window'] = '['+winstart_test+','+ winend_test + ')'

submission_win = result_test[['tollgate_id','time_window','direction','volume']]
submission_win['volume'] = submission_win['volume'].astype(int)

base_submission = pd.read_csv('../M1/results/volume/multi_volume_XGB_all_virtual.csv')
submission = pd.concat([submission_win, base_submission],axis=0,ignore_index=True)
submission = submission.drop_duplicates(['tollgate_id','direction','time_window'],keep = 'first')
submission.to_csv('M2_volume.csv')