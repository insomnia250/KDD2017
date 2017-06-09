#coding=utf-8
from __future__ import division
import numpy as np
import pandas as pd
from evaluation import * 
from keras.initializers import RandomUniform
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization

dropcols = []
dropcols = ['ddefault0','ddefault1','ddefault2','ddefault3','ddefault4','ddefault5',
	'ftvolume_plus_ddefault0','ftvolume_plus_ddefault1','ftvolume_plus_ddefault2',
	'ftvolume_plus_ddefault3','ftvolume_plus_ddefault4','ftvolume_plus_ddefault5']
#　整体建模
# load data
train_set  = pd.DataFrame()
val_set = pd.DataFrame()
test_set = pd.DataFrame()

submission = pd.DataFrame()
record = np.zeros((5,6)) ; detail_score = pd.DataFrame()
for col in ['1_0','1_1','2_0','3_0','3_1']:
	tid = col.split('_')[0]
	direction = col.split('_')[1]

	part_train_set = pd.read_csv('../../samples/results/volume/{0}/train_set.csv'.format(col))
	part_val_set = pd.read_csv('../../samples/results/volume/{0}/offlinetest_set.csv'.format(col))
	part_test_set = pd.read_csv('../../samples/results/volume/{0}/onlinetest_set.csv'.format(col))

	virtual_part_train_set = pd.read_csv('../../samples/results/volume/{0}/virtual_train_set.csv'.format(col))
	part_train_set = pd.concat([part_train_set , virtual_part_train_set],axis=0,ignore_index=True)

	part_train_set = part_train_set.drop(dropcols,axis=1)
	part_val_set = part_val_set.drop(dropcols,axis=1)
	part_test_set = part_test_set.drop(dropcols,axis=1)


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

# 整理
# for win_num in range(6):
win_num = 2
print win_num
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

# #归一化
scaler = {};
import sklearn.preprocessing as preprocessing
ColumnList = X_all.columns
# 每列归一化
for i,col in enumerate(ColumnList):
	scaler[col] = preprocessing.StandardScaler()
	scaler[col].fit(X_all[col].reshape(-1,1))
	X_all.loc[:,col] = scaler[col].transform(X_all[col].reshape(-1,1))


l1 = X_train.shape[0]
l2 = X_val.shape[0]
l3 = X_test.shape[0]

X_train = X_all.loc[0:l1-1,:]
X_val = X_all.loc[l1:l1+l2-1,:]
X_test = X_all.loc[l1+l2:,:]

print l1,l2,l3
#########
# model
early_stopping = EarlyStopping(monitor='val_loss', patience=1000)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-4)

model = Sequential()
model.add(Dense(50, input_dim=X_train.shape[1]))
model.add(Activation('tanh'))
# model.add(Dropout(0.5))

model.add(Dense(30))
model.add(Activation('tanh'))
# model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('linear'))
# model.add(Dropout(0.5))

model.compile(loss='mape',
          optimizer=adam,
          metrics=['mape'])
# 训练obj=obj_function,
model.fit(X_train.as_matrix(),y_train,verbose=True,
      epochs=7500,batch_size=200,validation_data=(X_val.as_matrix(), y_val) , callbacks=[early_stopping]
      )

#线下测试
preds_val = model.predict( X_val.as_matrix())[:,0]

# 计分明细
for col_num,col in enumerate(['1_0','1_1','2_0','3_0','3_1']):
	tid = col.split('_')[0]
	direction = col.split('_')[1]
	preds = preds_val[(tid_val==tid) & (direction_val==direction)]
	labels = y_val[(tid_val==tid) & (direction_val==direction)]
	record[col_num][win_num] = cal_MAPE(preds,labels)
detail_score['win_num'+str(win_num)] = record[:,win_num]

# 线上
preds_test = model.predict(X_test.as_matrix())[:,0]
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
# submission.to_csv('./results/volume/multi_volume_XGB_all_virtual.csv',index=False)


detail_score.index = ['1_0','1_1','2_0','3_0','3_1']
print detail_score
print detail_score.mean(0)
print detail_score.mean(1)
print 'val score:',detail_score.mean().mean()


